#include "../../src/engine/priv/EngineMetricsFault.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>

namespace {

TEST(EngineMetricsFaultTest, CountsEachBatchOnceAndAggregatesLatency)
{
    irt::engine::priv::EngineMetricsFault metrics(2);
    ASSERT_TRUE(metrics.tryStartBatch(1));
    EXPECT_FALSE(metrics.tryStartBatch(1));

    auto request = std::make_shared<irt::engine::priv::Request>();
    const auto now = std::chrono::steady_clock::now();
    request->submitted                 = now - std::chrono::microseconds(20);
    auto batch = std::make_shared<irt::engine::priv::BatchState>();
    batch->requests.push_back(request);
    batch->scheduled                 = now - std::chrono::microseconds(10);
    batch->cpu_preprocess_started    = now - std::chrono::microseconds(8);
    batch->cpu_preprocess_finished   = now - std::chrono::microseconds(6);
    batch->gpu_submitted             = now - std::chrono::microseconds(5);
    batch->gpu_finished              = now - std::chrono::microseconds(3);
    batch->cpu_postprocess_started   = now - std::chrono::microseconds(2);
    batch->cpu_postprocess_finished = now;

    metrics.recordBatchLatency(*batch);
    EXPECT_TRUE(metrics.finishBatch(batch));
    EXPECT_FALSE(metrics.finishBatch(batch));
    EXPECT_EQ(metrics.snapshot().completed_batches, 1U);
    EXPECT_EQ(metrics.snapshot().inflight_batches, 0U);
}

TEST(EngineMetricsFaultTest, BoundsFaultHistoryAndTracksRequestTerminalStates)
{
    irt::engine::priv::EngineMetricsFault metrics(1);
    metrics.recordRequestTerminal(irt::engine::priv::FailureKind::Cancelled);
    metrics.recordRequestTerminal(irt::engine::priv::FailureKind::Failed);
    metrics.recordFault(irt::engine::FaultStage::Scheduler, -1, nullptr,
                        std::make_exception_ptr(irt::Exception(irt::Status::ERROR_INTERNAL, "latest")));

    const auto snapshot = metrics.snapshot();
    EXPECT_EQ(snapshot.cancelled_requests, 1U);
    EXPECT_EQ(snapshot.failed_requests, 1U);
    ASSERT_EQ(metrics.faults().size(), 1U);
    EXPECT_NE(metrics.faults().front().message.find("latest"), std::string::npos);
}

} // namespace
