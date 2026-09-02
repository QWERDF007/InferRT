#include "EngineMetricsFault.hpp"

#include <algorithm>
#include <chrono>

namespace irt::engine::priv {

EngineMetricsFault::EngineMetricsFault(const size_t fault_history_capacity) noexcept
    : fault_history_capacity_(fault_history_capacity)
{
}

bool EngineMetricsFault::tryStartBatch(const size_t max_inflight_batches)
{
    std::lock_guard lock(metrics_mutex_);
    if (metrics_.inflight_batches >= max_inflight_batches)
    {
        return false;
    }
    ++metrics_.inflight_batches;
    return true;
}

bool EngineMetricsFault::finishBatch(const BatchPtr &batch)
{
    if (!batch || batch->finalized.exchange(true))
    {
        return false;
    }
    std::lock_guard lock(metrics_mutex_);
    if (metrics_.inflight_batches > 0)
    {
        --metrics_.inflight_batches;
    }
    ++metrics_.completed_batches;
    return true;
}

void EngineMetricsFault::clearInflightBatches() noexcept
{
    std::lock_guard lock(metrics_mutex_);
    metrics_.inflight_batches = 0;
}

void EngineMetricsFault::recordBatchLatency(const BatchState &batch)
{
    auto earliest = batch.scheduled;
    for (const auto &request : batch.requests)
    {
        if (request)
        {
            earliest = std::min(earliest, request->submitted);
        }
    }

    std::lock_guard lock(metrics_mutex_);
    recordLatency(metrics_.queue_latency, elapsedMicroseconds(earliest, batch.cpu_preprocess_started));
    recordLatency(metrics_.cpu_preprocess_latency,
                  elapsedMicroseconds(batch.cpu_preprocess_started, batch.cpu_preprocess_finished));
    recordLatency(metrics_.gpu_latency, elapsedMicroseconds(batch.gpu_submitted, batch.gpu_finished));
    recordLatency(metrics_.cpu_postprocess_latency,
                  elapsedMicroseconds(batch.cpu_postprocess_started, batch.cpu_postprocess_finished));
}

void EngineMetricsFault::recordRequestAccepted(const size_t queued_requests)
{
    std::lock_guard lock(metrics_mutex_);
    ++metrics_.accepted_requests;
    metrics_.queued_requests = queued_requests;
    metrics_.queued_high_watermark = std::max(metrics_.queued_high_watermark, queued_requests);
}

void EngineMetricsFault::recordRequestQueueSize(const size_t queued_requests)
{
    std::lock_guard lock(metrics_mutex_);
    metrics_.queued_requests = queued_requests;
}

void EngineMetricsFault::recordRequestRejected()
{
    std::lock_guard lock(metrics_mutex_);
    ++metrics_.rejected_requests;
}

void EngineMetricsFault::recordRequestSuccess()
{
    std::lock_guard lock(metrics_mutex_);
    ++metrics_.completed_requests;
}

void EngineMetricsFault::recordRequestTerminal(const FailureKind kind)
{
    std::lock_guard lock(metrics_mutex_);
    switch (kind)
    {
    case FailureKind::Failed:
        ++metrics_.failed_requests;
        break;
    case FailureKind::Cancelled:
        ++metrics_.cancelled_requests;
        break;
    case FailureKind::TimedOut:
        ++metrics_.timed_out_requests;
        break;
    case FailureKind::Dropped:
        ++metrics_.dropped_requests;
        break;
    }
}

EngineMetricsSnapshot EngineMetricsFault::snapshot() const
{
    std::lock_guard lock(metrics_mutex_);
    return metrics_;
}

void EngineMetricsFault::recordFault(const FaultStage stage, const int device_id, const RequestPtr &request,
                                     const std::exception_ptr error, const bool fatal)
{
    if (fault_history_capacity_ == 0)
    {
        return;
    }

    FaultRecord record;
    record.timestamp       = std::chrono::system_clock::now();
    record.request_id      = request ? request->id : 0;
    record.source_id       = request ? request->source_id : std::string{};
    record.source_sequence = request ? request->source_sequence : 0;
    record.device_id       = device_id;
    record.stage           = stage;
    record.fatal           = fatal;
    record.status          = irt::Status::ERROR_INTERNAL;
    record.message         = "Unknown engine failure";
    try
    {
        if (error)
        {
            std::rethrow_exception(error);
        }
    }
    catch (const irt::Exception &exception)
    {
        record.status  = exception.code();
        record.message = exception.what();
    }
    catch (const std::bad_alloc &exception)
    {
        record.status  = irt::Status::ERROR_OUT_OF_MEMORY;
        record.message = exception.what();
    }
    catch (const std::exception &exception)
    {
        record.message = exception.what();
    }
    catch (...)
    {
    }

    std::lock_guard lock(fault_mutex_);
    while (faults_.size() >= fault_history_capacity_)
    {
        faults_.pop_front();
    }
    faults_.push_back(std::move(record));
}

std::vector<FaultRecord> EngineMetricsFault::faults() const
{
    std::lock_guard lock(fault_mutex_);
    return {faults_.begin(), faults_.end()};
}

} // namespace irt::engine::priv
