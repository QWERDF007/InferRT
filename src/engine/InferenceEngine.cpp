#include "priv/EngineCompletionOrder.hpp"
#include "priv/EngineMetricsFault.hpp"
#include "priv/EngineScheduler.hpp"
#include "priv/EngineSlot.hpp"
#include "priv/EngineSlotExecutor.hpp"
#include "priv/EngineStageWorkers.hpp"
#include "priv/EngineTicketPool.hpp"
#include "priv/EngineTestHooks.hpp"
#include "priv/EngineTypes.hpp"
#include "priv/EngineRuntimePlan.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/InferenceEngine.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace irt::engine {
using namespace priv;

namespace {

const char *startFaultPointName(const priv::StartFaultPoint point) noexcept
{
    switch (point)
    {
    case priv::StartFaultPoint::RuntimePlan:
        return "runtime plan";
    case priv::StartFaultPoint::Slot:
        return "slot";
    case priv::StartFaultPoint::TicketPool:
        return "ticket pool";
    case priv::StartFaultPoint::PrepareWorker:
        return "prepare worker";
    case priv::StartFaultPoint::PostprocessWorker:
        return "postprocess worker";
    case priv::StartFaultPoint::GpuDispatcher:
        return "GPU dispatcher";
    case priv::StartFaultPoint::CompletionPoller:
        return "completion poller";
    case priv::StartFaultPoint::Scheduler:
        return "scheduler";
    case priv::StartFaultPoint::None:
        break;
    }
    return "unknown";
}

void maybeInjectStartResourceFault(const priv::EngineTestOptions &options,
                                   const priv::StartFaultPoint point, const size_t index)
{
    if (options.start_fault_point == point && options.start_fault_index == index)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL,
                             "Injected start resource creation failure at %s index %zu",
                             startFaultPointName(point), index);
    }
}

} // namespace

class InferenceEngine::Impl
{
public:
    struct SubmittedRequest
    {
        uint64_t                     id{0};
        std::future<InferenceResult> future;
    };

    Impl(EngineConfig config, std::shared_ptr<const PipelinePlan> pipeline)
        : config_(std::move(config))
        , pipeline_(std::move(pipeline))
        , metrics_fault_(config_.fault_history_capacity)
        , completion_order_(metrics_fault_)
        , test_options_(priv::GetEngineTestOptions())
    {
        if (!pipeline_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "InferenceEngine requires a PipelinePlan");
        }
    }

    ~Impl()
    {
        shutdown();
    }

    void             start();
    SubmittedRequest submit(const cv::Mat &image, TensorInputMap inputs, RequestOptions options);
    InferenceResult  infer(const cv::Mat &image, std::chrono::milliseconds timeout);
    bool             cancel(uint64_t request_id);
    void             shutdown();

    [[nodiscard]] size_t                   pendingRequests() const;
    [[nodiscard]] EngineMetricsSnapshot    metrics() const;
    [[nodiscard]] EngineState              state() const noexcept;
    [[nodiscard]] const EngineConfig      &config() const noexcept;
    [[nodiscard]] std::vector<FaultRecord> faults() const;

private:
    struct StartResources;
    struct WorkerResources
    {
        EngineStageWorkers *stage_workers{nullptr};
        EngineScheduler    *scheduler{nullptr};
        TicketPool         *ticket_pool{nullptr};
        SlotExecutor       *slot_executor{nullptr};
    };

    void stopWorkersAndJoin(WorkerResources resources, bool abort_pools);
    void               transitionFailed();
    [[nodiscard]] bool isFailed() const;
    void               recordFault(FaultStage stage, int device_id, const RequestPtr &request, std::exception_ptr error,
                                   bool fatal = false);

    EngineConfig                                                        config_;
    std::shared_ptr<const PipelinePlan>                                 pipeline_;
    EngineMetricsFault                                                   metrics_fault_;
    CompletionOrder                                                       completion_order_;
    std::unordered_map<int, std::shared_ptr<irt::IExecutionPlan>> runtime_plans_;
    int                                                                 fixed_batch_size_{0};
    size_t                                                              max_inflight_batches_{0};

    mutable std::mutex                                                     shutdown_mutex_;
    mutable std::mutex                                                     mutex_;
    std::unique_ptr<EngineScheduler>                                      scheduler_;
    std::unique_ptr<SlotExecutor>                                        slot_executor_;
    std::unique_ptr<EngineStageWorkers>                                  stage_workers_;
    size_t                                                                 device_arena_bytes_{0};
    std::unordered_map<int, size_t>                                        device_arena_bytes_by_device_;
    size_t                                                                 pinned_memory_capacity_bytes_{0};
    EngineState                                                            state_{EngineState::Created};

    StageQueue              prepare_queue_;
    StageQueue              gpu_queue_;
    StageQueue              postprocess_queue_;

    std::unique_ptr<TicketPool>                ticket_pool_;
    priv::EngineTestOptions                    test_options_{};
};

struct InferenceEngine::Impl::StartResources
{
    Impl                                                                &impl;
    std::unordered_map<int, std::shared_ptr<irt::IExecutionPlan>> runtime_plans;
    std::vector<std::unique_ptr<Slot>>                                  slots;
    std::unique_ptr<TicketPool>                                         ticket_pool;
    std::unique_ptr<EngineStageWorkers>                                 stage_workers;
    std::unique_ptr<SlotExecutor>                                      slot_executor;
    std::unique_ptr<EngineScheduler>                                    scheduler;
    size_t                                                              device_arena_bytes{0};
    std::unordered_map<int, size_t>                                     device_arena_bytes_by_device;
    size_t                                                              pinned_memory_capacity_bytes{0};
    int                                                                 fixed_batch_size{0};
    size_t                                                              max_inflight_batches{0};
    bool                                                                committed{false};

    explicit StartResources(Impl &engine_impl) : impl(engine_impl) {}

    ~StartResources()
    {
        if (!committed)
        {
            impl.stopWorkersAndJoin(
                {stage_workers.get(), scheduler.get(), ticket_pool.get(), slot_executor.get()},
                true);

            stage_workers.reset();
            slot_executor.reset();
            ticket_pool.reset();
            runtime_plans.clear();
            device_arena_bytes_by_device.clear();
        }
    }

    void commit()
    {
        committed                          = true;
        impl.runtime_plans_                = std::move(runtime_plans);
        impl.ticket_pool_                  = std::move(ticket_pool);
        impl.stage_workers_                = std::move(stage_workers);
        impl.scheduler_                    = std::move(scheduler);
        impl.slot_executor_                = std::move(slot_executor);
        impl.device_arena_bytes_           = device_arena_bytes;
        impl.device_arena_bytes_by_device_ = std::move(device_arena_bytes_by_device);
        impl.pinned_memory_capacity_bytes_ = pinned_memory_capacity_bytes;
        impl.fixed_batch_size_             = fixed_batch_size;
        impl.max_inflight_batches_         = max_inflight_batches;
    }
};

void InferenceEngine::Impl::start()
{
    std::unique_lock shutdown_lock(shutdown_mutex_);
    {
        std::lock_guard lock(mutex_);
        if (state_ == EngineState::Running || state_ == EngineState::Starting || state_ == EngineState::Draining)
        {
            return;
        }
        if (state_ == EngineState::Failed || state_ == EngineState::Stopped)
        {
            throw irt::Exception(irt::Status::INVALID_OPERATION,
                                 "A stopped or failed InferenceEngine must be recreated");
        }
        state_ = EngineState::Starting;

        device_arena_bytes_ = 0;
        device_arena_bytes_by_device_.clear();
        pinned_memory_capacity_bytes_ = 0;
    }

    StartResources resources{*this};

    try
    {
        config_.validate();

        std::vector<int> device_ids = config_.device_ids;
        if (device_ids.empty())
        {
            device_ids.push_back(config_.device_id);
        }

        for (const int device_id : device_ids)
        {
            if (!resources.runtime_plans.contains(device_id))
            {
                maybeInjectStartResourceFault(test_options_, priv::StartFaultPoint::RuntimePlan,
                                               resources.runtime_plans.size());
                resources.runtime_plans.emplace(
                    device_id, priv::CreateEngineRuntimePlan(config_, device_id, pipeline_.get()));
            }
        }
        if (test_options_.start_fault_stage == 1)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Injected start failure at step 1 (runtime plans)");
        }

        resources.fixed_batch_size
            = resources.runtime_plans.at(device_ids.front())->capabilities().fixed_batch_size;
        for (const int device_id : device_ids)
        {
            if (resources.runtime_plans.at(device_id)->capabilities().fixed_batch_size != resources.fixed_batch_size)
            {
                throw irt::Exception(irt::Status::ERROR_INTERNAL,
                                     "TensorRT runtime plans disagree on fixed batch size");
            }
        }
        if (resources.fixed_batch_size > 0 && config_.max_batch_size > resources.fixed_batch_size)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Configured max batch %d exceeds fixed TensorRT batch %d", config_.max_batch_size,
                                 resources.fixed_batch_size);
        }

        resources.slots.reserve(config_.execution_slots);
        for (size_t index = 0; index < config_.execution_slots; ++index)
        {
            const int device_id = device_ids[index % device_ids.size()];
            auto      slot      = std::make_unique<Slot>(device_id);
            slot->runtime_plan  = resources.runtime_plans.at(device_id);
            priv::initializeEngineSlot(*slot, config_, pipeline_, resources.fixed_batch_size,
                                       resources.device_arena_bytes, resources.device_arena_bytes_by_device);
            maybeInjectStartResourceFault(test_options_, priv::StartFaultPoint::Slot, index);
            resources.slots.push_back(std::move(slot));
        }
        if (test_options_.start_fault_stage == 2)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Injected start failure at step 2 (slots)");
        }

        maybeInjectStartResourceFault(test_options_, priv::StartFaultPoint::TicketPool, 0);
        resources.ticket_pool = std::make_unique<TicketPool>(config_, pipeline_, resources.slots.front()->output_capacity_bytes);
        resources.pinned_memory_capacity_bytes = resources.ticket_pool->capacityBytes();
        resources.max_inflight_batches
            = config_.execution_slots + config_.cpu_preprocess_workers + config_.cpu_postprocess_workers;
        resources.stage_workers = std::make_unique<EngineStageWorkers>(
            config_, pipeline_, prepare_queue_, gpu_queue_, postprocess_queue_, *resources.ticket_pool,
            completion_order_, metrics_fault_, resources.fixed_batch_size, test_options_,
            [this] { return isFailed(); },
            [this](const FaultStage stage, const int device_id, const RequestPtr &request,
                   const std::exception_ptr error, const bool fatal) {
                recordFault(stage, device_id, request, error, fatal);
            },
            [this] {
                if (scheduler_)
                {
                    scheduler_->notifyBatchFinished();
                }
            });
        auto *stage_workers = resources.stage_workers.get();
        resources.slot_executor = std::make_unique<SlotExecutor>(
            *pipeline_, gpu_queue_, postprocess_queue_, *resources.ticket_pool, std::move(resources.slots), test_options_,
            [this] { return isFailed(); },
            [stage_workers](const BatchPtr &batch, const std::exception_ptr error)
            { stage_workers->failBatch(batch, error); },
            [this](const FaultStage stage, const int device_id, const RequestPtr &request,
                   const std::exception_ptr error, const bool fatal) {
                recordFault(stage, device_id, request, error, fatal);
            },
            [this] { transitionFailed(); },
            [stage_workers](const BatchPtr &batch) { stage_workers->finishBatch(batch); });
        if (test_options_.start_fault_stage == 3)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Injected start failure at step 3 (tickets)");
        }

        resources.stage_workers->startPrepare(
            config_.cpu_preprocess_workers,
            [this](const size_t index)
            { maybeInjectStartResourceFault(test_options_, priv::StartFaultPoint::PrepareWorker, index); });
        if (test_options_.start_fault_stage == 4)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Injected start failure at step 4 (prepare workers)");
        }

        resources.stage_workers->startPostprocess(
            config_.cpu_postprocess_workers,
            [this](const size_t index)
            { maybeInjectStartResourceFault(test_options_, priv::StartFaultPoint::PostprocessWorker, index); });
        if (test_options_.start_fault_stage == 5)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Injected start failure at step 5 (postprocess workers)");
        }

        maybeInjectStartResourceFault(test_options_, priv::StartFaultPoint::GpuDispatcher, 0);
        resources.slot_executor->startDispatcher();
        if (test_options_.start_fault_stage == 6)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Injected start failure at step 6 (gpu dispatcher)");
        }

        maybeInjectStartResourceFault(test_options_, priv::StartFaultPoint::CompletionPoller, 0);
        resources.slot_executor->startCompletionPoller();
        if (test_options_.start_fault_stage == 7)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Injected start failure at step 7 (completion poller)");
        }

        maybeInjectStartResourceFault(test_options_, priv::StartFaultPoint::Scheduler, 0);
        resources.scheduler = std::make_unique<EngineScheduler>(
            config_, [this] { return state(); }, prepare_queue_, completion_order_, metrics_fault_,
            resources.max_inflight_batches);
        resources.scheduler->start();
        if (test_options_.start_fault_stage == 8)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Injected start failure at step 8 (scheduler)");
        }

        resources.commit();

        {
            std::lock_guard lock(mutex_);
            state_ = EngineState::Running;
        }
        scheduler_->notifyStateChanged();
    }
    catch (...)
    {
        const auto error = std::current_exception();
        recordFault(FaultStage::Start, -1, nullptr, error, true);
        {
            std::lock_guard lock(mutex_);
            state_ = EngineState::Failed;
        }
        if (resources.scheduler)
        {
            resources.scheduler->notifyStateChanged();
        }
        throw;
    }
}

InferenceEngine::Impl::SubmittedRequest InferenceEngine::Impl::submit(const cv::Mat &image, TensorInputMap inputs,
                                                                      RequestOptions options)
{
    EngineScheduler *scheduler = nullptr;
    {
        std::lock_guard lifecycle_lock(shutdown_mutex_);
        std::lock_guard state_lock(mutex_);
        if (state_ != EngineState::Running || !scheduler_)
        {
            throw irt::Exception(irt::Status::INVALID_OPERATION, "Inference engine is not accepting requests");
        }
        scheduler = scheduler_.get();
    }
    auto submitted = scheduler->submit(image, std::move(inputs), std::move(options));
    return {submitted.id, std::move(submitted.future)};
}

InferenceResult InferenceEngine::Impl::infer(const cv::Mat &image, const std::chrono::milliseconds timeout)
{
    auto submitted = submit(image, {}, {});
    auto future    = std::move(submitted.future);
    if (future.wait_for(timeout) != std::future_status::ready)
    {
        cancel(submitted.id);
        throw irt::Exception(irt::Status::NOT_READY, "Inference request timed out");
    }
    return future.get();
}

bool InferenceEngine::Impl::cancel(const uint64_t request_id)
{
    std::lock_guard lifecycle_lock(shutdown_mutex_);
    if (!scheduler_)
    {
        return false;
    }
    return scheduler_->cancel(request_id);
}

void InferenceEngine::Impl::transitionFailed()
{
    {
        std::lock_guard lock(mutex_);
        if (state_ == EngineState::Running || state_ == EngineState::Draining)
        {
            state_ = EngineState::Failed;
        }
    }
    if (ticket_pool_)
    {
        ticket_pool_->abort();
    }
    if (scheduler_)
    {
        scheduler_->notifyStateChanged();
    }
    if (slot_executor_)
    {
        slot_executor_->notifyFailure();
    }
}

void InferenceEngine::Impl::recordFault(const FaultStage stage, const int device_id, const RequestPtr &request,
                                        const std::exception_ptr error, const bool fatal)
{
    metrics_fault_.recordFault(stage, device_id, request, error, fatal);
}

bool InferenceEngine::Impl::isFailed() const
{
    std::lock_guard lock(mutex_);
    return state_ == EngineState::Failed;
}

void InferenceEngine::Impl::stopWorkersAndJoin(const WorkerResources resources, const bool abort_pools)
{
    // First stop the scheduler.  It owns the request -> prepare queue handoff;
    // closing downstream queues before it exits can strand a newly formed batch.
    if (resources.scheduler)
    {
        resources.scheduler->stop();
    }

    // Once no more batches can be produced, drain each stage in pipeline order.
    if (resources.stage_workers)
    {
        resources.stage_workers->stopPrepare(abort_pools);
    }
    else if (abort_pools && resources.ticket_pool)
    {
        resources.ticket_pool->abort();
    }

    if (resources.slot_executor)
    {
        resources.slot_executor->stop();
    }

    if (resources.stage_workers)
    {
        resources.stage_workers->stopPostprocess();
    }
}

void InferenceEngine::Impl::shutdown()
{
    std::unique_lock shutdown_lock(shutdown_mutex_);
    {
        std::lock_guard lock(mutex_);
        if (state_ == EngineState::Stopped)
        {
            return;
        }
        if (state_ == EngineState::Created)
        {
            state_ = EngineState::Stopped;
            return;
        }
        if (state_ == EngineState::Running)
        {
            state_ = EngineState::Draining;
        }
    }

    bool abort_pools = false;
    {
        std::lock_guard lock(mutex_);
        abort_pools = state_ == EngineState::Failed;
    }
    stopWorkersAndJoin({stage_workers_.get(), scheduler_.get(), ticket_pool_.get(), slot_executor_.get()},
                       abort_pools);

    {
        std::lock_guard lock(mutex_);
        if (state_ != EngineState::Failed)
        {
            state_ = EngineState::Stopped;
        }

        metrics_fault_.clearInflightBatches();

        slot_executor_.reset();
        stage_workers_.reset();
        ticket_pool_.reset();
        runtime_plans_.clear();
        device_arena_bytes_by_device_.clear();
        device_arena_bytes_ = 0;
        pinned_memory_capacity_bytes_ = 0;

    }
    completion_order_.shutdown();
}

size_t InferenceEngine::Impl::pendingRequests() const
{
    std::lock_guard lifecycle_lock(shutdown_mutex_);
    return scheduler_ ? scheduler_->pendingRequests() : 0;
}

EngineMetricsSnapshot InferenceEngine::Impl::metrics() const
{
    // Executor and stage-worker ownership are published/cleared by start and
    // shutdown. Serialize the snapshot with those lifecycle transitions before
    // reading them, otherwise a concurrent metrics call could race with
    // destruction.
    std::unique_lock lifecycle_lock(shutdown_mutex_);
    const auto scheduler_snapshot = scheduler_ ? scheduler_->snapshot() : EngineScheduler::Snapshot{};
    const auto slot_snapshot      = slot_executor_ ? slot_executor_->snapshot() : SlotExecutor::Snapshot{};
    const auto stage_snapshot = stage_workers_ ? stage_workers_->snapshot() : EngineStageWorkers::Snapshot{};
    std::lock_guard state_lock(mutex_);
    const auto ticket_snapshot = ticket_pool_ ? ticket_pool_->snapshot() : TicketPool::Snapshot{};

    const auto prepare_queue_snapshot     = prepare_queue_.snapshot();
    const auto gpu_queue_snapshot         = gpu_queue_.snapshot();
    const auto postprocess_queue_snapshot = postprocess_queue_.snapshot();

    const size_t thread_count = stage_snapshot.prepare_worker_count + stage_snapshot.postprocess_worker_count
        + (slot_snapshot.dispatcher_joinable ? size_t{1} : size_t{0})
        + (slot_snapshot.completion_poller_joinable ? size_t{1} : size_t{0})
        + (scheduler_snapshot.joinable ? size_t{1} : size_t{0});
    const size_t joinable_thread_count = stage_snapshot.joinable_prepare_workers
        + stage_snapshot.joinable_postprocess_workers
        + (slot_snapshot.dispatcher_joinable ? size_t{1} : size_t{0})
        + (slot_snapshot.completion_poller_joinable ? size_t{1} : size_t{0})
        + (scheduler_snapshot.joinable ? size_t{1} : size_t{0});

    auto snapshot = metrics_fault_.snapshot();
    snapshot.queued_requests                = scheduler_snapshot.queued;
    snapshot.prepare_queue_high_watermark    = prepare_queue_snapshot.high_watermark;
    snapshot.gpu_queue_high_watermark        = gpu_queue_snapshot.high_watermark;
    snapshot.postprocess_queue_high_watermark = postprocess_queue_snapshot.high_watermark;
    snapshot.pinned_input_in_use             = ticket_snapshot.input_in_use;
    snapshot.pinned_input_high_watermark     = ticket_snapshot.input_high_watermark;
    snapshot.pinned_output_in_use            = ticket_snapshot.output_in_use;
    snapshot.pinned_output_high_watermark    = ticket_snapshot.output_high_watermark;
    snapshot.pinned_memory_capacity_bytes    = pinned_memory_capacity_bytes_;
    snapshot.device_arena_bytes              = device_arena_bytes_;
    snapshot.active_requests       = completion_order_.activeCount();
    snapshot.slot_count             = slot_snapshot.slot_count;
    snapshot.idle_slot_count        = slot_snapshot.idle_slot_count;
    snapshot.active_slot_count      = slot_snapshot.active_slot_count;
    snapshot.prepare_queue_size     = prepare_queue_snapshot.size;
    snapshot.gpu_queue_size         = gpu_queue_snapshot.size;
    snapshot.postprocess_queue_size = postprocess_queue_snapshot.size;
    snapshot.input_ticket_count     = ticket_snapshot.input_count;
    snapshot.output_ticket_count    = ticket_snapshot.output_count;
    snapshot.thread_count          = thread_count;
    snapshot.joinable_thread_count = joinable_thread_count;
    return snapshot;
}

std::vector<FaultRecord> InferenceEngine::Impl::faults() const
{
    return metrics_fault_.faults();
}

EngineState InferenceEngine::Impl::state() const noexcept
{
    std::lock_guard lock(mutex_);
    return state_;
}

const EngineConfig &InferenceEngine::Impl::config() const noexcept
{
    return config_;
}

RequestHandle::RequestHandle(const uint64_t request_id, std::future<InferenceResult> future)
    : request_id_(request_id)
    , future_(std::move(future))
{
}

uint64_t RequestHandle::id() const noexcept
{
    return request_id_;
}

bool RequestHandle::valid() const noexcept
{
    return request_id_ != 0 && future_.valid();
}

std::future<InferenceResult> RequestHandle::takeFuture() &&
{
    return std::move(future_);
}

InferenceEngine::InferenceEngine(EngineConfig config, std::shared_ptr<const PipelinePlan> pipeline)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(pipeline)))
{
}

InferenceEngine::~InferenceEngine() = default;

std::unique_ptr<InferenceEngine> InferenceEngine::create(EngineConfig                        config,
                                                         std::shared_ptr<const PipelinePlan> pipeline)
{
    return std::make_unique<InferenceEngine>(std::move(config), std::move(pipeline));
}

void InferenceEngine::start()
{
    impl_->start();
}

std::future<InferenceResult> InferenceEngine::submit(const cv::Mat &image)
{
    return std::move(impl_->submit(image, {}, {}).future);
}

std::future<InferenceResult> InferenceEngine::submit(const cv::Mat &image, TensorInputMap inputs)
{
    return std::move(impl_->submit(image, std::move(inputs), {}).future);
}

RequestHandle InferenceEngine::submit(const cv::Mat &image, RequestOptions options)
{
    auto submitted = impl_->submit(image, {}, std::move(options));
    return {submitted.id, std::move(submitted.future)};
}

RequestHandle InferenceEngine::submit(const cv::Mat &image, TensorInputMap inputs, RequestOptions options)
{
    auto submitted = impl_->submit(image, std::move(inputs), std::move(options));
    return {submitted.id, std::move(submitted.future)};
}

InferenceResult InferenceEngine::infer(const cv::Mat &image, const std::chrono::milliseconds timeout)
{
    return impl_->infer(image, timeout);
}

bool InferenceEngine::cancel(const uint64_t request_id)
{
    return impl_->cancel(request_id);
}

void InferenceEngine::shutdown()
{
    impl_->shutdown();
}

size_t InferenceEngine::pendingRequests() const
{
    return impl_->pendingRequests();
}

EngineMetricsSnapshot InferenceEngine::metrics() const
{
    return impl_->metrics();
}

std::vector<FaultRecord> InferenceEngine::faults() const
{
    return impl_->faults();
}

EngineState InferenceEngine::state() const noexcept
{
    return impl_->state();
}

const EngineConfig &InferenceEngine::config() const noexcept
{
    return impl_->config();
}

} // namespace irt::engine
