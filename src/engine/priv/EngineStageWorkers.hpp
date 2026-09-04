#pragma once

#include "EngineCompletionOrder.hpp"
#include "EngineExecution.hpp"
#include "EngineStageQueue.hpp"
#include "EngineTestHooks.hpp"
#include "EngineTicketPool.hpp"

#include <inferrt/engine/EngineConfig.hpp>
#include <inferrt/engine/Export.h>
#include <inferrt/engine/Pipeline.hpp>

#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace irt::engine::priv {

/**
 * Owns the CPU stages between the scheduler and GPU slot executor.
 *
 * This module is the sole owner of prepare/postprocess worker threads and of
 * the stage-specific request completion logic.  InferenceEngine only
 * coordinates its lifecycle with the scheduler and slot executor.
 */
class INFERRT_ENGINE_API EngineStageWorkers final
{
public:
    struct Snapshot
    {
        size_t prepare_worker_count{0};
        size_t postprocess_worker_count{0};
        size_t joinable_prepare_workers{0};
        size_t joinable_postprocess_workers{0};
    };

    using FailureStateReader = std::function<bool()>;
    using FaultRecorder = std::function<void(FaultStage, int, const RequestPtr &, std::exception_ptr, bool)>;
    using BatchFinishedHandler = std::function<void()>;
    using WorkerCreationHook = std::function<void(size_t)>;

    EngineStageWorkers(const EngineConfig &config, std::shared_ptr<const PipelinePlan> pipeline,
                       StageQueue &prepare_queue, StageQueue &gpu_queue, StageQueue &postprocess_queue,
                       TicketPool &ticket_pool, CompletionOrder &completion_order, EngineMetricsFault &metrics_fault,
                       int fixed_batch_size, EngineTestOptions test_options,
                       FailureStateReader state_reader, FaultRecorder record_fault,
                       BatchFinishedHandler batch_finished);
    ~EngineStageWorkers() noexcept;

    EngineStageWorkers(const EngineStageWorkers &) = delete;
    EngineStageWorkers &operator=(const EngineStageWorkers &) = delete;

    void startPrepare(size_t worker_count, WorkerCreationHook before_create = {});
    void startPostprocess(size_t worker_count, WorkerCreationHook before_create = {});
    void stopPrepare(bool abort_pool = false) noexcept;
    void stopPostprocess() noexcept;

    void failBatch(const BatchPtr &batch, std::exception_ptr error);
    void finishBatch(const BatchPtr &batch);

    [[nodiscard]] Snapshot snapshot() const;

private:
    void prepareLoop(std::vector<std::unique_ptr<IOperator>> operators);
    void postprocessLoop(std::vector<std::unique_ptr<IOperator>> operators);

    [[nodiscard]] bool prepareBatch(const BatchPtr &batch);
    [[nodiscard]] bool removeInvalidBeforeSubmit(BatchState &batch);
    void               recordBatchMetrics(const BatchState &batch);

    void joinWorkers(std::vector<std::thread> &workers) noexcept;

    EngineConfig                     config_;
    std::shared_ptr<const PipelinePlan> pipeline_;
    StageQueue                      &prepare_queue_;
    StageQueue                      &gpu_queue_;
    StageQueue                      &postprocess_queue_;
    TicketPool                      &ticket_pool_;
    CompletionOrder                 &completion_order_;
    EngineMetricsFault              &metrics_fault_;
    const int                        fixed_batch_size_;
    EngineTestOptions                test_options_;
    FailureStateReader               state_reader_;
    FaultRecorder                    record_fault_;
    BatchFinishedHandler             batch_finished_;

    mutable std::mutex      lifecycle_mutex_;
    std::vector<std::thread> prepare_workers_;
    std::vector<std::thread> postprocess_workers_;
};

} // namespace irt::engine::priv
