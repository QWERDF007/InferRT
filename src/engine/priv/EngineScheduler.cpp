#include "EngineScheduler.hpp"

#include <algorithm>
#include <chrono>
#include <inferrt/core/Exception.hpp>

namespace irt::engine::priv {

EngineScheduler::EngineScheduler(const EngineConfig &config, StateReader state_reader, StageQueue &prepare_queue,
                                 CompletionOrder &completions, EngineMetricsFault &metrics,
                                 const size_t max_inflight_batches)
    : config_(config)
    , state_reader_(std::move(state_reader))
    , prepare_queue_(prepare_queue)
    , completions_(completions)
    , metrics_(metrics)
    , max_inflight_batches_(max_inflight_batches)
{
    if (!state_reader_ || max_inflight_batches_ == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "EngineScheduler requires a state reader and positive in-flight capacity");
    }
}

EngineScheduler::~EngineScheduler() noexcept
{
    stop();
}

EngineScheduler::SubmittedRequest EngineScheduler::submit(const cv::Mat &image, TensorInputMap inputs,
                                                           RequestOptions options)
{
    if (image.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Input image is empty");
    }
    if (options.deadline.count() < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Request deadline must not be negative");
    }

    auto request                   = std::make_shared<Request>();
    request->image                 = image.clone();
    request->extra_inputs          = std::move(inputs);
    request->submitted             = std::chrono::steady_clock::now();
    request->priority              = options.priority;
    request->source_id             = options.source_id.empty() ? "default" : std::move(options.source_id);
    request->preserve_source_order = options.preserve_source_order;
    if (!options.compatibility_key.empty())
    {
        request->compatibility_key = std::move(options.compatibility_key);
    }
    if (options.deadline.count() > 0)
    {
        request->deadline = request->submitted + options.deadline;
    }
    auto future = request->promise.get_future();

    RequestPtr dropped;
    bool       drop_newest = false;
    {
        std::unique_lock lock(mutex_);
        if (state_reader_() != EngineState::Running)
        {
            throw irt::Exception(irt::Status::INVALID_OPERATION, "Inference engine is not accepting requests");
        }
        request->id = next_request_id_++;

        while (requests_.size() >= config_.queue_capacity)
        {
            switch (config_.queue_policy)
            {
            case QueuePolicy::Reject:
                metrics_.recordRequestRejected();
                throw irt::Exception(irt::Status::NOT_READY, "Inference request queue is full");
            case QueuePolicy::Block:
                space_condition_.wait(lock, [this]
                                       {
                                           return requests_.size() < config_.queue_capacity
                                               || state_reader_() != EngineState::Running;
                                       });
                if (state_reader_() != EngineState::Running)
                {
                    throw irt::Exception(irt::Status::INVALID_OPERATION,
                                         "Inference engine is not accepting requests");
                }
                break;
            case QueuePolicy::DropOldest:
                dropped = std::move(requests_.front());
                requests_.pop_front();
                break;
            case QueuePolicy::DropNewest:
                drop_newest = true;
                break;
            }
            if (dropped || drop_newest)
            {
                break;
            }
        }

        if (!drop_newest)
        {
            completions_.registerRequest(request);
            requests_.push_back(request);
            metrics_.recordRequestAccepted(requests_.size());
        }
    }

    if (dropped)
    {
        completions_.completeFailure(dropped, droppedError(), FailureKind::Dropped);
    }
    if (drop_newest)
    {
        // The request was rejected before admission, so it must not wait for
        // an earlier admitted request in the same source to complete.
        request->preserve_source_order = false;
        completions_.registerRequest(request);
        completions_.completeFailure(request, droppedError(), FailureKind::Dropped);
    }
    else
    {
        request_condition_.notify_one();
    }
    return {request->id, std::move(future)};
}

bool EngineScheduler::cancel(const uint64_t request_id)
{
    if (!completions_.cancel(request_id))
    {
        return false;
    }
    request_condition_.notify_one();
    return true;
}

void EngineScheduler::start()
{
    std::lock_guard lock(mutex_);
    if (started_)
    {
        return;
    }
    stop_requested_ = false;
    started_        = true;
    worker_         = std::thread([this] { run(); });
}

void EngineScheduler::stop() noexcept
{
    {
        std::lock_guard lock(mutex_);
        stop_requested_ = true;
    }
    request_condition_.notify_all();
    space_condition_.notify_all();
    batch_space_condition_.notify_all();
    if (worker_.joinable() && worker_.get_id() != std::this_thread::get_id())
    {
        worker_.join();
    }
}

void EngineScheduler::notifyStateChanged() noexcept
{
    request_condition_.notify_all();
    space_condition_.notify_all();
    batch_space_condition_.notify_all();
}

void EngineScheduler::notifyBatchFinished() noexcept
{
    batch_space_condition_.notify_one();
}

size_t EngineScheduler::pendingRequests() const
{
    std::lock_guard lock(mutex_);
    return requests_.size();
}

EngineScheduler::Snapshot EngineScheduler::snapshot() const
{
    std::lock_guard lock(mutex_);
    return {requests_.size(), worker_.joinable()};
}

void EngineScheduler::pruneQueuedRequestsLocked(std::vector<std::pair<RequestPtr, FailureKind>> &finished)
{
    const auto now = std::chrono::steady_clock::now();
    for (auto it = requests_.begin(); it != requests_.end();)
    {
        if ((*it)->cancelled.load())
        {
            finished.emplace_back(std::move(*it), FailureKind::Cancelled);
            it = requests_.erase(it);
        }
        else if ((*it)->deadline <= now)
        {
            finished.emplace_back(std::move(*it), FailureKind::TimedOut);
            it = requests_.erase(it);
        }
        else
        {
            ++it;
        }
    }
    if (!finished.empty())
    {
        metrics_.recordRequestQueueSize(requests_.size());
    }
}

RequestPtr EngineScheduler::chooseSeedLocked() const
{
    RequestPtr selected;
    for (const auto &request : requests_)
    {
        if (!selected || request->priority > selected->priority
            || (request->priority == selected->priority && request->submitted < selected->submitted))
        {
            selected = request;
        }
    }
    return selected;
}

size_t EngineScheduler::compatibleCountLocked(const Request &seed) const
{
    return static_cast<size_t>(std::count_if(requests_.begin(), requests_.end(),
                                             [&seed](const RequestPtr &request)
                                             {
                                                 return request->priority == seed.priority
                                                     && request->compatibility_key == seed.compatibility_key;
                                             }));
}

std::chrono::steady_clock::time_point EngineScheduler::batchDeadlineLocked(const Request &seed) const
{
    auto deadline = seed.submitted + config_.max_wait;
    for (const auto &request : requests_)
    {
        if (request->priority == seed.priority && request->compatibility_key == seed.compatibility_key)
        {
            deadline = std::min(deadline, request->submitted + config_.max_wait);
            deadline = std::min(deadline, request->deadline);
        }
    }
    return deadline;
}

size_t EngineScheduler::preferredBatchSize(const size_t compatible_count) const
{
    const size_t maximum = std::min(compatible_count, static_cast<size_t>(config_.max_batch_size));
    if (config_.preferred_batch_sizes.empty())
    {
        return maximum;
    }
    size_t preferred = 0;
    for (const int size : config_.preferred_batch_sizes)
    {
        if (size <= static_cast<int>(maximum))
        {
            preferred = std::max(preferred, static_cast<size_t>(size));
        }
    }
    return preferred == 0 ? maximum : preferred;
}

std::vector<RequestPtr> EngineScheduler::takeCompatibleBatchLocked(const Request &seed, const size_t count)
{
    std::vector<RequestPtr> batch;
    batch.reserve(count);
    for (auto it = requests_.begin(); it != requests_.end() && batch.size() < count;)
    {
        if ((*it)->priority == seed.priority && (*it)->compatibility_key == seed.compatibility_key)
        {
            batch.push_back(std::move(*it));
            it = requests_.erase(it);
        }
        else
        {
            ++it;
        }
    }
    metrics_.recordRequestQueueSize(requests_.size());
    return batch;
}

void EngineScheduler::failQueuedRequests(const std::vector<std::pair<RequestPtr, FailureKind>> &requests)
{
    for (const auto &[request, kind] : requests)
    {
        std::exception_ptr error;
        bool              fatal = false;
        switch (kind)
        {
        case FailureKind::Cancelled:
            error = cancelledError();
            break;
        case FailureKind::TimedOut:
            error = timeoutError();
            break;
        case FailureKind::Dropped:
            error = droppedError();
            break;
        case FailureKind::Failed:
            error = failedError();
            fatal = true;
            break;
        }
        metrics_.recordFault(FaultStage::Scheduler, -1, request, error, fatal);
        completions_.completeFailure(request, error, kind);
    }
}

void EngineScheduler::run()
{
    for (;;)
    {
        std::vector<std::pair<RequestPtr, FailureKind>> finished;
        BatchPtr                                        batch;
        bool                                            notify_space = false;
        bool                                            exit         = false;
        {
            std::unique_lock lock(mutex_);
            for (;;)
            {
                pruneQueuedRequestsLocked(finished);
                if (!finished.empty())
                {
                    notify_space = true;
                    break;
                }

                const auto state = state_reader_();
                if (state == EngineState::Failed)
                {
                    for (auto &request : requests_)
                    {
                        finished.emplace_back(std::move(request), FailureKind::Failed);
                    }
                    requests_.clear();
                    metrics_.recordRequestQueueSize(0);
                    notify_space = true;
                    exit = true;
                    break;
                }
                if (requests_.empty())
                {
                    if (stop_requested_ || state == EngineState::Failed || state == EngineState::Draining
                        || state == EngineState::Stopped)
                    {
                        exit = true;
                        break;
                    }
                    request_condition_.wait(lock, [this]
                                            {
                                                const auto state = state_reader_();
                                                return !requests_.empty() || stop_requested_ || state == EngineState::Failed
                                                    || state == EngineState::Draining || state == EngineState::Stopped;
                                            });
                    continue;
                }
                if (metrics_.snapshot().inflight_batches >= max_inflight_batches_)
                {
                    batch_space_condition_.wait(lock, [this]
                                                {
                                                    return state_reader_() == EngineState::Failed
                                                        || metrics_.snapshot().inflight_batches < max_inflight_batches_;
                                                });
                    continue;
                }

                const auto   seed             = chooseSeedLocked();
                const auto   compatible_count = compatibleCountLocked(*seed);
                const size_t target           = config_.preferred_batch_sizes.empty()
                                                   ? static_cast<size_t>(config_.max_batch_size)
                                                   : static_cast<size_t>(*std::max_element(
                                                         config_.preferred_batch_sizes.begin(),
                                                         config_.preferred_batch_sizes.end()));
                const auto deadline = batchDeadlineLocked(*seed);
                const bool draining = state == EngineState::Draining || stop_requested_;
                if (!draining && compatible_count < target && std::chrono::steady_clock::now() < deadline)
                {
                    request_condition_.wait_until(lock, deadline);
                    continue;
                }

                const size_t count = draining ? std::min(compatible_count, static_cast<size_t>(config_.max_batch_size))
                                              : preferredBatchSize(compatible_count);
                batch = std::make_shared<BatchState>();
                batch->requests  = takeCompatibleBatchLocked(*seed, count);
                batch->scheduled = std::chrono::steady_clock::now();
                if (!metrics_.tryStartBatch(max_inflight_batches_))
                {
                    for (auto it = batch->requests.rbegin(); it != batch->requests.rend(); ++it)
                    {
                        requests_.push_front(std::move(*it));
                    }
                    batch.reset();
                    continue;
                }
                notify_space = true;
                break;
            }
        }

        if (notify_space)
        {
            space_condition_.notify_all();
        }
        if (!finished.empty())
        {
            failQueuedRequests(finished);
        }
        if (exit)
        {
            return;
        }
        if (!batch)
        {
            continue;
        }
        if (!prepare_queue_.push(batch))
        {
            std::vector<std::pair<RequestPtr, FailureKind>> failed;
            for (const auto &request : batch->requests)
            {
                failed.emplace_back(request, FailureKind::Failed);
            }
            failQueuedRequests(failed);
            (void)metrics_.finishBatch(batch);
        }
    }
}

} // namespace irt::engine::priv
