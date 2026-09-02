#include "EngineStageQueue.hpp"

#include <utility>

namespace irt::engine::priv {

StageQueue::~StageQueue() noexcept
{
    close();
}

bool StageQueue::push(BatchPtr batch)
{
    if (!batch)
    {
        return false;
    }

    {
        std::lock_guard lock(mutex_);
        if (closed_)
        {
            return false;
        }
        batches_.push_back(std::move(batch));
        high_watermark_ = std::max(high_watermark_, batches_.size());
    }
    condition_.notify_one();
    return true;
}

BatchPtr StageQueue::waitPop()
{
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [this] { return closed_ || !batches_.empty(); });
    if (batches_.empty())
    {
        return nullptr;
    }
    auto batch = std::move(batches_.front());
    batches_.pop_front();
    return batch;
}

void StageQueue::close() noexcept
{
    {
        std::lock_guard lock(mutex_);
        closed_ = true;
    }
    condition_.notify_all();
}

StageQueue::Snapshot StageQueue::snapshot() const
{
    std::lock_guard lock(mutex_);
    return {batches_.size(), high_watermark_, closed_};
}

} // namespace irt::engine::priv
