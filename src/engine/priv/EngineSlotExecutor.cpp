#include "EngineSlotExecutor.hpp"

#include "EngineExecution.hpp"

#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>

#include <chrono>
#include <utility>

namespace irt::engine::priv {

SlotExecutor::SlotExecutor(const PipelinePlan &pipeline, StageQueue &gpu_queue, StageQueue &postprocess_queue,
                           TicketPool &ticket_pool, std::vector<std::unique_ptr<Slot>> slots,
                           EngineTestOptions test_options, FailureStateReader state_reader,
                           BatchFailureHandler fail_batch, FaultRecorder record_fault,
                           FailureTransition transition_failed, BatchFinishedHandler finish_batch)
    : pipeline_(pipeline)
    , gpu_queue_(gpu_queue)
    , postprocess_queue_(postprocess_queue)
    , ticket_pool_(ticket_pool)
    , slots_(std::move(slots))
    , test_options_(test_options)
    , state_reader_(std::move(state_reader))
    , fail_batch_(std::move(fail_batch))
    , record_fault_(std::move(record_fault))
    , transition_failed_(std::move(transition_failed))
    , finish_batch_(std::move(finish_batch))
{
    if (slots_.empty() || !state_reader_ || !fail_batch_ || !record_fault_ || !transition_failed_ || !finish_batch_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "SlotExecutor requires slots and all lifecycle callbacks");
    }
    for (const auto &slot : slots_)
    {
        if (!slot)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "SlotExecutor cannot own an empty slot");
        }
        idle_slots_.push_back(slot.get());
    }
}

SlotExecutor::~SlotExecutor() noexcept
{
    stop();
}

void SlotExecutor::startDispatcher()
{
    std::lock_guard lock(lifecycle_mutex_);
    if (dispatcher_started_)
    {
        return;
    }
    if (stop_requested_.load())
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "SlotExecutor has already stopped");
    }
    dispatcher_started_ = true;
    try
    {
        dispatcher_ = std::thread([this] { dispatcherLoop(); });
    }
    catch (...)
    {
        dispatcher_started_ = false;
        throw;
    }
}

void SlotExecutor::startCompletionPoller()
{
    std::lock_guard lock(lifecycle_mutex_);
    if (completion_poller_started_)
    {
        return;
    }
    if (stop_requested_.load())
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "SlotExecutor has already stopped");
    }
    completion_poller_started_ = true;
    try
    {
        completion_poller_ = std::thread([this] { completionPollerLoop(); });
    }
    catch (...)
    {
        completion_poller_started_ = false;
        throw;
    }
}

void SlotExecutor::stop() noexcept
{
    std::lock_guard lifecycle_lock(lifecycle_mutex_);
    stop_requested_.store(true);
    gpu_queue_.close();
    slot_condition_.notify_all();
    completion_condition_.notify_all();

    if (dispatcher_.joinable() && dispatcher_.get_id() != std::this_thread::get_id())
    {
        dispatcher_.join();
    }

    {
        std::lock_guard lock(completion_mutex_);
        dispatcher_closed_ = true;
    }
    completion_condition_.notify_all();

    if (completion_poller_.joinable() && completion_poller_.get_id() != std::this_thread::get_id())
    {
        completion_poller_.join();
    }

    for (auto &slot : slots_)
    {
        if (slot)
        {
            synchronizeSlot(*slot);
            slot->context.reset();
        }
    }
}

void SlotExecutor::notifyFailure() noexcept
{
    failure_requested_.store(true);
    slot_condition_.notify_all();
}

SlotExecutor::Snapshot SlotExecutor::snapshot() const
{
    std::lock_guard lifecycle_lock(lifecycle_mutex_);
    std::lock_guard slot_lock(slot_mutex_);
    size_t active_slot_count = 0;
    for (const auto &slot : slots_)
    {
        std::lock_guard lock(slot->mutex);
        if (slot->active_batch)
        {
            ++active_slot_count;
        }
    }
    return {slots_.size(), idle_slots_.size(), active_slot_count, dispatcher_.joinable(),
            completion_poller_.joinable()};
}

Slot *SlotExecutor::acquireIdleSlot()
{
    std::unique_lock lock(slot_mutex_);
    slot_condition_.wait(lock, [this]
                          { return failure_requested_.load() || !idle_slots_.empty(); });
    if (failure_requested_.load() || idle_slots_.empty())
    {
        return nullptr;
    }
    auto *slot = idle_slots_.front();
    idle_slots_.pop_front();
    return slot;
}

void SlotExecutor::returnSlot(Slot &slot)
{
    {
        std::lock_guard lock(slot_mutex_);
        idle_slots_.push_back(&slot);
    }
    slot_condition_.notify_one();
}

void SlotExecutor::synchronizeSlot(Slot &slot) const noexcept
{
    cudaSetDevice(slot.device_id);
    if (slot.h2d_stream != nullptr)
    {
        cudaStreamSynchronize(slot.h2d_stream);
    }
    if (slot.compute_stream != nullptr)
    {
        cudaStreamSynchronize(slot.compute_stream);
    }
    if (slot.d2h_stream != nullptr)
    {
        cudaStreamSynchronize(slot.d2h_stream);
    }
}

void SlotExecutor::dispatcherLoop()
{
    for (;;)
    {
        auto batch = gpu_queue_.waitPop();
        if (!batch)
        {
            break;
        }
        if (state_reader_())
        {
            ticket_pool_.releaseInput(batch->input_ticket);
            batch->input_ticket = nullptr;
            fail_batch_(batch, std::make_exception_ptr(
                                  irt::Exception(irt::Status::INVALID_OPERATION, "Inference engine failed")));
            finish_batch_(batch);
            continue;
        }

        auto *output_ticket = ticket_pool_.acquireOutput();
        if (output_ticket == nullptr)
        {
            ticket_pool_.releaseInput(batch->input_ticket);
            batch->input_ticket = nullptr;
            fail_batch_(batch, std::make_exception_ptr(
                                  irt::Exception(irt::Status::INVALID_OPERATION, "Pinned output pool is stopping")));
            finish_batch_(batch);
            continue;
        }
        batch->output_ticket = output_ticket;

        auto *slot = acquireIdleSlot();
        if (slot == nullptr)
        {
            ticket_pool_.releaseInput(batch->input_ticket);
            ticket_pool_.releaseOutput(batch->output_ticket);
            batch->input_ticket  = nullptr;
            batch->output_ticket = nullptr;
            fail_batch_(batch, std::make_exception_ptr(
                                  irt::Exception(irt::Status::INVALID_OPERATION, "Inference engine failed")));
            finish_batch_(batch);
            continue;
        }

        try
        {
            irt::model::setCudaDevice(slot->device_id);
            dispatchGpuBatch(*slot, batch, pipeline_);
            completion_condition_.notify_one();
        }
        catch (...)
        {
            const auto error = std::current_exception();
            record_fault_(FaultStage::GpuDispatch, slot->device_id,
                          batch->requests.empty() ? nullptr : batch->requests.front(), error, state_reader_());
            synchronizeSlot(*slot);
            ticket_pool_.releaseInput(batch->input_ticket);
            ticket_pool_.releaseOutput(batch->output_ticket);
            batch->input_ticket  = nullptr;
            batch->output_ticket = nullptr;
            fail_batch_(batch, error);
            finish_batch_(batch);
            returnSlot(*slot);
        }
    }

    {
        std::lock_guard lock(completion_mutex_);
        dispatcher_closed_ = true;
    }
    completion_condition_.notify_all();
}

void SlotExecutor::completionPollerLoop()
{
    for (;;)
    {
        bool active = false;
        for (const auto &slot_owner : slots_)
        {
            auto    &slot = *slot_owner;
            BatchPtr batch;
            bool     input_released = false;
            {
                std::lock_guard lock(slot.mutex);
                batch          = slot.active_batch;
                input_released = slot.input_released;
            }
            if (!batch)
            {
                continue;
            }
            irt::model::setCudaDevice(slot.device_id);
            active = true;

            if (!input_released)
            {
                const auto status = cudaEventQuery(slot.h2d_done);
                if (status == cudaSuccess)
                {
                    ticket_pool_.releaseInput(batch->input_ticket);
                    batch->input_ticket = nullptr;
                    std::lock_guard lock(slot.mutex);
                    slot.input_released = true;
                }
                else if (status != cudaErrorNotReady)
                {
                    if (isFatalCudaError(status))
                    {
                        transition_failed_();
                    }
                    synchronizeSlot(slot);
                    const auto error = std::make_exception_ptr(
                        irt::Exception(irt::Status::ERROR_INTERNAL, "H2D completion query failed"));
                    ticket_pool_.releaseInput(batch->input_ticket);
                    ticket_pool_.releaseOutput(batch->output_ticket);
                    batch->input_ticket  = nullptr;
                    batch->output_ticket = nullptr;
                    fail_batch_(batch, error);
                    record_fault_(FaultStage::CompletionPoll, slot.device_id,
                                  batch->requests.empty() ? nullptr : batch->requests.front(), error,
                                  isFatalCudaError(status));
                    {
                        std::lock_guard lock(slot.mutex);
                        slot.active_batch.reset();
                    }
                    finish_batch_(batch);
                    returnSlot(slot);
                    continue;
                }
            }

            const auto status = cudaEventQuery(slot.d2h_done);
            if (status == cudaSuccess)
            {
                if (test_options_.completion_fatal
                    && !completion_fault_injected_.exchange(true, std::memory_order_acq_rel))
                {
                    if (test_options_.completion_fatal_delay_ms > 0)
                    {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(test_options_.completion_fatal_delay_ms));
                    }
                    const auto error = std::make_exception_ptr(
                        irt::Exception(irt::Status::ERROR_INTERNAL, "Injected completion poller failure"));
                    transition_failed_();
                    synchronizeSlot(slot);
                    ticket_pool_.releaseInput(batch->input_ticket);
                    ticket_pool_.releaseOutput(batch->output_ticket);
                    batch->input_ticket  = nullptr;
                    batch->output_ticket = nullptr;
                    fail_batch_(batch, error);
                    record_fault_(FaultStage::CompletionPoll, slot.device_id,
                                  batch->requests.empty() ? nullptr : batch->requests.front(), error, true);
                    {
                        std::lock_guard lock(slot.mutex);
                        slot.active_batch.reset();
                    }
                    finish_batch_(batch);
                    returnSlot(slot);
                    continue;
                }
                batch->gpu_finished = std::chrono::steady_clock::now();
                if (batch->input_ticket != nullptr)
                {
                    ticket_pool_.releaseInput(batch->input_ticket);
                    batch->input_ticket = nullptr;
                }
                {
                    std::lock_guard lock(slot.mutex);
                    slot.active_batch.reset();
                }
                returnSlot(slot);
                if (!postprocess_queue_.push(batch))
                {
                    ticket_pool_.releaseOutput(batch->output_ticket);
                    batch->output_ticket = nullptr;
                    fail_batch_(batch, std::make_exception_ptr(
                                          irt::Exception(irt::Status::INVALID_OPERATION,
                                                         "Postprocess queue is closed")));
                    finish_batch_(batch);
                }
            }
            else if (status != cudaErrorNotReady)
            {
                if (isFatalCudaError(status))
                {
                    transition_failed_();
                }
                synchronizeSlot(slot);
                const auto error = std::make_exception_ptr(
                    irt::Exception(irt::Status::ERROR_INTERNAL, "D2H completion query failed"));
                ticket_pool_.releaseInput(batch->input_ticket);
                ticket_pool_.releaseOutput(batch->output_ticket);
                batch->input_ticket  = nullptr;
                batch->output_ticket = nullptr;
                fail_batch_(batch, error);
                record_fault_(FaultStage::CompletionPoll, slot.device_id,
                              batch->requests.empty() ? nullptr : batch->requests.front(), error,
                              isFatalCudaError(status));
                {
                    std::lock_guard lock(slot.mutex);
                    slot.active_batch.reset();
                }
                finish_batch_(batch);
                returnSlot(slot);
            }
        }

        bool dispatcher_closed = false;
        {
            std::lock_guard lock(completion_mutex_);
            dispatcher_closed = dispatcher_closed_;
        }
        if (dispatcher_closed && !active)
        {
            return;
        }
        std::unique_lock lock(completion_mutex_);
        completion_condition_.wait_for(lock, std::chrono::milliseconds(1));
    }
}

} // namespace irt::engine::priv
