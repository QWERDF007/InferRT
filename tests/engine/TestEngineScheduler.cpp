#include "../../src/engine/priv/EngineCompletionOrder.hpp"
#include "../../src/engine/priv/EngineMetricsFault.hpp"
#include "../../src/engine/priv/EngineScheduler.hpp"
#include "../../src/engine/priv/EngineStageQueue.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>

namespace {

irt::engine::EngineConfig makeSchedulerConfig()
{
    irt::engine::EngineConfig config;
    config.min_batch_size = 1;
    config.opt_batch_size = 1;
    config.max_batch_size = 1;
    config.queue_capacity = 1;
    config.queue_policy   = irt::engine::QueuePolicy::DropNewest;
    return config;
}

TEST(EngineSchedulerTest, DropNewestCompletesTheRejectedFuture)
{
    auto config = makeSchedulerConfig();
    irt::engine::priv::StageQueue prepare_queue;
    irt::engine::priv::EngineMetricsFault metrics(4);
    irt::engine::priv::CompletionOrder completions(metrics);
    auto state = irt::engine::EngineState::Running;
    irt::engine::priv::EngineScheduler scheduler(
        config, [&state] { return state; }, prepare_queue, completions, metrics, 1);

    const cv::Mat image(2, 2, CV_8UC3, cv::Scalar(1, 2, 3));
    auto first  = scheduler.submit(image, {}, {});
    auto second = scheduler.submit(image, {}, {});
    auto second_future = std::move(second.future);

    ASSERT_EQ(second_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_THROW(second_future.get(), irt::Exception);
    EXPECT_EQ(metrics.snapshot().accepted_requests, 1U);
    EXPECT_EQ(metrics.snapshot().dropped_requests, 1U);

    completions.shutdown();
    auto first_future = std::move(first.future);
    ASSERT_EQ(first_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_THROW(first_future.get(), irt::Exception);
}

TEST(EngineSchedulerTest, SchedulerPublishesACompatibleBatchToThePrepareQueue)
{
    auto config = makeSchedulerConfig();
    config.queue_capacity = 4;
    irt::engine::priv::StageQueue prepare_queue;
    irt::engine::priv::EngineMetricsFault metrics(4);
    irt::engine::priv::CompletionOrder completions(metrics);
    auto state = irt::engine::EngineState::Running;
    irt::engine::priv::EngineScheduler scheduler(
        config, [&state] { return state; }, prepare_queue, completions, metrics, 1);
    scheduler.start();

    const cv::Mat image(2, 2, CV_8UC3, cv::Scalar(1, 2, 3));
    auto submitted = scheduler.submit(image, {}, {});
    const auto batch = prepare_queue.waitPop();
    ASSERT_NE(batch, nullptr);
    ASSERT_EQ(batch->requests.size(), 1U);
    EXPECT_EQ(batch->requests.front()->id, submitted.id);

    state = irt::engine::EngineState::Draining;
    scheduler.notifyStateChanged();
    scheduler.stop();
    completions.shutdown();
    auto future = std::move(submitted.future);
    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_THROW(future.get(), irt::Exception);
}

TEST(EngineSchedulerTest, SnapshotIsAvailableWhileSchedulerIsIdle)
{
    auto config = makeSchedulerConfig();
    irt::engine::priv::StageQueue prepare_queue;
    irt::engine::priv::EngineMetricsFault metrics(4);
    irt::engine::priv::CompletionOrder completions(metrics);
    auto state = irt::engine::EngineState::Running;
    irt::engine::priv::EngineScheduler scheduler(
        config, [&state] { return state; }, prepare_queue, completions, metrics, 1);
    scheduler.start();

    const auto snapshot = scheduler.snapshot();
    EXPECT_TRUE(snapshot.joinable);
    EXPECT_EQ(snapshot.queued, 0U);

    state = irt::engine::EngineState::Draining;
    scheduler.notifyStateChanged();
    scheduler.stop();
    completions.shutdown();
}

} // namespace
