#include "EngineStageWorkers.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <chrono>
#include <utility>

namespace irt::engine::priv {

EngineStageWorkers::EngineStageWorkers(const EngineConfig &config, std::shared_ptr<const PipelinePlan> pipeline,
                                       StageQueue &prepare_queue, StageQueue &gpu_queue,
                                       StageQueue &postprocess_queue, TicketPool &ticket_pool,
                                       CompletionOrder &completion_order, EngineMetricsFault &metrics_fault,
                                       const int fixed_batch_size, const EngineTestOptions test_options,
                                       FailureStateReader state_reader, FaultRecorder record_fault,
                                       BatchFinishedHandler batch_finished)
    : config_(config)
    , pipeline_(std::move(pipeline))
    , prepare_queue_(prepare_queue)
    , gpu_queue_(gpu_queue)
    , postprocess_queue_(postprocess_queue)
    , ticket_pool_(ticket_pool)
    , completion_order_(completion_order)
    , metrics_fault_(metrics_fault)
    , fixed_batch_size_(fixed_batch_size)
    , test_options_(test_options)
    , state_reader_(std::move(state_reader))
    , record_fault_(std::move(record_fault))
    , batch_finished_(std::move(batch_finished))
{
    if (!pipeline_ || !state_reader_ || !record_fault_ || !batch_finished_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "EngineStageWorkers requires a pipeline and lifecycle callbacks");
    }
}

EngineStageWorkers::~EngineStageWorkers() noexcept
{
    stopPrepare();
    stopPostprocess();
}

void EngineStageWorkers::startPrepare(const size_t worker_count, WorkerCreationHook before_create)
{
    std::lock_guard lock(lifecycle_mutex_);
    if (!prepare_workers_.empty())
    {
        return;
    }

    try
    {
        prepare_workers_.reserve(worker_count);
        for (size_t index = 0; index < worker_count; ++index)
        {
            if (before_create)
            {
                before_create(index);
            }
            prepare_workers_.emplace_back([this, operators = pipeline_->createOperators()]() mutable
                                           { prepareLoop(std::move(operators)); });
        }
    }
    catch (...)
    {
        prepare_queue_.close();
        ticket_pool_.abort();
        joinWorkers(prepare_workers_);
        throw;
    }
}

void EngineStageWorkers::startPostprocess(const size_t worker_count, WorkerCreationHook before_create)
{
    std::lock_guard lock(lifecycle_mutex_);
    if (!postprocess_workers_.empty())
    {
        return;
    }

    try
    {
        postprocess_workers_.reserve(worker_count);
        for (size_t index = 0; index < worker_count; ++index)
        {
            if (before_create)
            {
                before_create(index);
            }
            postprocess_workers_.emplace_back([this, operators = pipeline_->createOperators()]() mutable
                                               { postprocessLoop(std::move(operators)); });
        }
    }
    catch (...)
    {
        postprocess_queue_.close();
        joinWorkers(postprocess_workers_);
        throw;
    }
}

void EngineStageWorkers::stopPrepare(const bool abort_pool) noexcept
{
    std::lock_guard lock(lifecycle_mutex_);
    prepare_queue_.close();
    if (abort_pool)
    {
        ticket_pool_.abort();
    }
    joinWorkers(prepare_workers_);
}

void EngineStageWorkers::stopPostprocess() noexcept
{
    std::lock_guard lock(lifecycle_mutex_);
    postprocess_queue_.close();
    joinWorkers(postprocess_workers_);
}

void EngineStageWorkers::joinWorkers(std::vector<std::thread> &workers) noexcept
{
    for (auto &worker : workers)
    {
        if (worker.joinable() && worker.get_id() != std::this_thread::get_id())
        {
            worker.join();
        }
    }
}

EngineStageWorkers::Snapshot EngineStageWorkers::snapshot() const
{
    std::lock_guard lock(lifecycle_mutex_);
    Snapshot result;
    result.prepare_worker_count       = prepare_workers_.size();
    result.postprocess_worker_count   = postprocess_workers_.size();
    result.joinable_prepare_workers   = static_cast<size_t>(std::count_if(
        prepare_workers_.begin(), prepare_workers_.end(), [](const std::thread &worker) { return worker.joinable(); }));
    result.joinable_postprocess_workers = static_cast<size_t>(std::count_if(
        postprocess_workers_.begin(), postprocess_workers_.end(),
        [](const std::thread &worker) { return worker.joinable(); }));
    return result;
}

bool EngineStageWorkers::removeInvalidBeforeSubmit(BatchState &batch)
{
    const auto now = std::chrono::steady_clock::now();
    auto       first_invalid = std::remove_if(
        batch.requests.begin(), batch.requests.end(), [&](const RequestPtr &request)
        {
            if (request->cancelled.load())
            {
                completion_order_.completeFailure(request, cancelledError(), FailureKind::Cancelled);
                return true;
            }
            if (request->deadline <= now)
            {
                completion_order_.completeFailure(request, timeoutError(), FailureKind::TimedOut);
                return true;
            }
            return false;
        });
    batch.requests.erase(first_invalid, batch.requests.end());
    return !batch.requests.empty();
}

bool EngineStageWorkers::prepareBatch(const BatchPtr &batch)
{
    if (!removeInvalidBeforeSubmit(*batch))
    {
        finishBatch(batch);
        return false;
    }
    batch->actual_batch = static_cast<int>(batch->requests.size());
    if (batch->actual_batch < config_.min_batch_size)
    {
        failBatch(batch, std::make_exception_ptr(
                             irt::Exception(irt::Status::NOT_READY, "Not enough requests to satisfy the minimum batch size")));
        finishBatch(batch);
        return false;
    }
    if (fixed_batch_size_ > 0)
    {
        if (batch->actual_batch != fixed_batch_size_ && config_.static_batch_policy == StaticBatchPolicy::Reject)
        {
            failBatch(batch, std::make_exception_ptr(
                                 irt::Exception(irt::Status::NOT_READY, "Fixed TensorRT batch requires padding")));
            finishBatch(batch);
            return false;
        }
        batch->execution_batch = fixed_batch_size_;
    }
    else
    {
        batch->execution_batch = batch->actual_batch;
    }
    return true;
}

void EngineStageWorkers::prepareLoop(std::vector<std::unique_ptr<IOperator>> operators)
{
    for (;;)
    {
        auto batch = prepare_queue_.waitPop();
        if (!batch)
        {
            return;
        }
        if (state_reader_())
        {
            failBatch(batch, failedError());
            finishBatch(batch);
            continue;
        }
        if (!prepareBatch(batch))
        {
            continue;
        }

        try
        {
            auto *ticket = ticket_pool_.acquireInput();
            if (ticket == nullptr)
            {
                throw irt::Exception(irt::Status::INVALID_OPERATION, "Pinned input pool is stopping");
            }
            batch->input_ticket           = ticket;
            batch->cpu_preprocess_started = std::chrono::steady_clock::now();
            ResultMap ignored;
            executePipelineStage(operators, pipeline_->nodes(), PipelineStage::CPU_PREPROCESS, *batch,
                                 config_.device_id, nullptr, ticket->tensors, ignored);
            batch->cpu_preprocess_finished = std::chrono::steady_clock::now();
            if (!gpu_queue_.push(batch))
            {
                ticket_pool_.releaseInput(batch->input_ticket);
                batch->input_ticket = nullptr;
                failBatch(batch, std::make_exception_ptr(
                                      irt::Exception(irt::Status::INVALID_OPERATION, "GPU queue is closed")));
                finishBatch(batch);
            }
        }
        catch (...)
        {
            const auto error = std::current_exception();
            record_fault_(FaultStage::CpuPreprocess, -1,
                          batch->requests.empty() ? nullptr : batch->requests.front(), error, false);
            ticket_pool_.releaseInput(batch->input_ticket);
            batch->input_ticket = nullptr;
            failBatch(batch, error);
            finishBatch(batch);
        }
    }
}

void EngineStageWorkers::postprocessLoop(std::vector<std::unique_ptr<IOperator>> operators)
{
    for (;;)
    {
        auto batch = postprocess_queue_.waitPop();
        if (!batch)
        {
            return;
        }
        try
        {
            batch->cpu_postprocess_started = std::chrono::steady_clock::now();
            processPostprocessBatch(
                operators, batch, *pipeline_,
                [this](const RequestPtr &request, InferenceResult result)
                {
                    const bool duplicate = test_options_.duplicate_completion;
                    InferenceResult duplicate_result = result;
                    completion_order_.completeSuccess(request, std::move(result));
                    if (duplicate)
                    {
                        completion_order_.completeSuccess(request, std::move(duplicate_result));
                    }
                },
                [this](const RequestPtr &request, std::exception_ptr error, FailureKind kind)
                {
                    completion_order_.completeFailure(request, error, kind);
                    if (test_options_.duplicate_completion)
                    {
                        completion_order_.completeFailure(request, error, kind);
                    }
                });
            batch->cpu_postprocess_finished = std::chrono::steady_clock::now();
        }
        catch (...)
        {
            const auto error = std::current_exception();
            record_fault_(FaultStage::CpuPostprocess, batch->device_id,
                          batch->requests.empty() ? nullptr : batch->requests.front(), error, false);
            failBatch(batch, error);
            batch->cpu_postprocess_finished = std::chrono::steady_clock::now();
        }
        ticket_pool_.releaseOutput(batch->output_ticket);
        batch->output_ticket = nullptr;
        recordBatchMetrics(*batch);
        finishBatch(batch);
    }
}

void EngineStageWorkers::failBatch(const BatchPtr &batch, const std::exception_ptr error)
{
    if (!batch)
    {
        return;
    }
    for (const auto &request : batch->requests)
    {
        const bool cancelled = request->cancelled.load();
        completion_order_.completeFailure(request, cancelled ? cancelledError() : error,
                                          cancelled ? FailureKind::Cancelled : FailureKind::Failed);
    }
}

void EngineStageWorkers::finishBatch(const BatchPtr &batch)
{
    if (metrics_fault_.finishBatch(batch) && batch_finished_)
    {
        batch_finished_();
    }
}

void EngineStageWorkers::recordBatchMetrics(const BatchState &batch)
{
    metrics_fault_.recordBatchLatency(batch);
}

} // namespace irt::engine::priv
