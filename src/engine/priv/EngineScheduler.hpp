#pragma once

#include "EngineCompletionOrder.hpp"
#include "EngineStageQueue.hpp"

#include <inferrt/engine/Export.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace irt::engine::priv {

/** Owns ingress requests and the single thread that forms execution batches. */
class INFERRT_ENGINE_API EngineScheduler final
{
public:
    struct SubmittedRequest
    {
        uint64_t                     id{0};
        std::future<InferenceResult> future;
    };

    struct Snapshot
    {
        size_t queued{0};
        bool   joinable{false};
    };

    using StateReader = std::function<EngineState()>;

    EngineScheduler(const EngineConfig &config, StateReader state_reader, StageQueue &prepare_queue,
                    CompletionOrder &completions, EngineMetricsFault &metrics, size_t max_inflight_batches);
    ~EngineScheduler() noexcept;

    EngineScheduler(const EngineScheduler &)            = delete;
    EngineScheduler &operator=(const EngineScheduler &) = delete;

    [[nodiscard]] SubmittedRequest submit(const cv::Mat &image, TensorInputMap inputs, RequestOptions options);
    [[nodiscard]] bool              cancel(uint64_t request_id);

    void start();
    void stop() noexcept;
    void notifyStateChanged() noexcept;
    void notifyBatchFinished() noexcept;

    [[nodiscard]] size_t   pendingRequests() const;
    [[nodiscard]] Snapshot snapshot() const;

private:
    void run();
    void pruneQueuedRequestsLocked(std::vector<std::pair<RequestPtr, FailureKind>> &finished);
    [[nodiscard]] RequestPtr chooseSeedLocked() const;
    [[nodiscard]] size_t     compatibleCountLocked(const Request &seed) const;
    [[nodiscard]] std::chrono::steady_clock::time_point batchDeadlineLocked(const Request &seed) const;
    [[nodiscard]] size_t preferredBatchSize(size_t compatible_count) const;
    [[nodiscard]] std::vector<RequestPtr> takeCompatibleBatchLocked(const Request &seed, size_t count);
    void failQueuedRequests(const std::vector<std::pair<RequestPtr, FailureKind>> &requests);

    const EngineConfig &config_;
    StateReader         state_reader_;
    StageQueue         &prepare_queue_;
    CompletionOrder    &completions_;
    EngineMetricsFault &metrics_;
    const size_t        max_inflight_batches_;

    mutable std::mutex              mutex_;
    std::condition_variable         request_condition_;
    std::condition_variable         space_condition_;
    std::condition_variable batch_space_condition_;
    std::deque<RequestPtr>  requests_;
    uint64_t                next_request_id_{1};
    bool                    stop_requested_{false};
    bool                    started_{false};
    std::thread             worker_;
};

} // namespace irt::engine::priv
