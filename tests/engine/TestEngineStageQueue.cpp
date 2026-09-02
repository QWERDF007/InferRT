#include "../../src/engine/priv/EngineStageQueue.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <thread>

namespace {

using irt::engine::priv::BatchPtr;
using irt::engine::priv::BatchState;
using irt::engine::priv::StageQueue;

TEST(EngineStageQueueTest, CloseDrainsQueuedBatchesAndRejectsNewBatches)
{
    StageQueue queue;
    const auto first  = std::make_shared<BatchState>();
    const auto rejected = std::make_shared<BatchState>();

    ASSERT_TRUE(queue.push(first));
    EXPECT_EQ(queue.snapshot().size, 1U);

    queue.close();

    EXPECT_EQ(queue.waitPop(), first);
    EXPECT_EQ(queue.waitPop(), nullptr);
    EXPECT_FALSE(queue.push(rejected));

    const auto snapshot = queue.snapshot();
    EXPECT_EQ(snapshot.size, 0U);
    EXPECT_EQ(snapshot.high_watermark, 1U);
}

TEST(EngineStageQueueTest, CloseUnblocksAWaitingConsumer)
{
    StageQueue queue;
    BatchPtr   result;
    std::thread consumer([&] { result = queue.waitPop(); });

    queue.close();
    consumer.join();

    EXPECT_EQ(result, nullptr);
}

} // namespace
