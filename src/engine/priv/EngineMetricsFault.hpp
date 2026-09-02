#pragma once

#include "EngineTypes.hpp"

#include <inferrt/engine/Export.h>

#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <vector>

namespace irt::engine::priv {

/** Owns request/batch metrics and the bounded fault history for one engine. */
class INFERRT_ENGINE_API EngineMetricsFault final
{
public:
    explicit EngineMetricsFault(size_t fault_history_capacity) noexcept;

    [[nodiscard]] bool tryStartBatch(size_t max_inflight_batches);
    [[nodiscard]] bool finishBatch(const BatchPtr &batch);
    void               clearInflightBatches() noexcept;
    void               recordBatchLatency(const BatchState &batch);

    void recordRequestAccepted(size_t queued_requests);
    void recordRequestQueueSize(size_t queued_requests);
    void recordRequestRejected();
    void recordRequestSuccess();
    void recordRequestTerminal(FailureKind kind);

    [[nodiscard]] EngineMetricsSnapshot snapshot() const;

    void                      recordFault(FaultStage stage, int device_id, const RequestPtr &request,
                                          std::exception_ptr error, bool fatal = false);
    [[nodiscard]] std::vector<FaultRecord> faults() const;

private:
    mutable std::mutex metrics_mutex_;
    EngineMetricsSnapshot metrics_;

    const size_t             fault_history_capacity_;
    mutable std::mutex       fault_mutex_;
    std::deque<FaultRecord>  faults_;
};

} // namespace irt::engine::priv
