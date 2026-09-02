#pragma once

#include "EngineTypes.hpp"

#include <algorithm>
#include <cstddef>
#include <condition_variable>
#include <deque>
#include <inferrt/engine/Export.h>
#include <mutex>

namespace irt::engine::priv {

/**
 * Owns one bounded pipeline handoff queue.
 *
 * The queue is intentionally unbounded at this private seam: the scheduler
 * owns ingress capacity and stage backpressure is represented by the number
 * of in-flight batches.  Closing drains already accepted batches, while all
 * later pushes are rejected.
 */
class INFERRT_ENGINE_API StageQueue final
{
public:
    struct Snapshot
    {
        size_t size{0};
        size_t high_watermark{0};
        bool   closed{false};
    };

    StageQueue() = default;
    ~StageQueue() noexcept;

    StageQueue(const StageQueue &)            = delete;
    StageQueue &operator=(const StageQueue &) = delete;

    [[nodiscard]] bool push(BatchPtr batch);
    [[nodiscard]] BatchPtr waitPop();
    void                  close() noexcept;

    [[nodiscard]] Snapshot snapshot() const;

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<BatchPtr>    batches_;
    size_t                  high_watermark_{0};
    bool                    closed_{false};
};

} // namespace irt::engine::priv
