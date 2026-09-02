#include "../../src/engine/priv/EngineCompletionOrder.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>

namespace {

irt::engine::priv::RequestPtr makeRequest(const char *source, const uint64_t id)
{
    auto request                  = std::make_shared<irt::engine::priv::Request>();
    request->id                   = id;
    request->source_id            = source;
    request->preserve_source_order = true;
    return request;
}

TEST(EngineCompletionOrderTest, DeliversSameSourceRequestsInSubmissionOrder)
{
    irt::engine::priv::EngineMetricsFault metrics(4);
    irt::engine::priv::CompletionOrder  completions(metrics);
    auto first  = makeRequest("camera", 1);
    auto second = makeRequest("camera", 2);
    auto first_future  = first->promise.get_future();
    auto second_future = second->promise.get_future();

    completions.registerRequest(first);
    completions.registerRequest(second);
    ASSERT_EQ(first->source_sequence, 0U);
    ASSERT_EQ(second->source_sequence, 1U);

    irt::engine::InferenceResult second_result;
    second_result.outputs["value"] = {2.0F};
    completions.completeSuccess(second, std::move(second_result));
    EXPECT_EQ(second_future.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);

    irt::engine::InferenceResult first_result;
    first_result.outputs["value"] = {1.0F};
    completions.completeSuccess(first, std::move(first_result));

    ASSERT_EQ(first_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    ASSERT_EQ(second_future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_EQ(first_future.get().outputs.at("value").front(), 1.0F);
    EXPECT_EQ(second_future.get().outputs.at("value").front(), 2.0F);
}

TEST(EngineCompletionOrderTest, DuplicateCompletionIsIgnoredAndCancellationIsTerminal)
{
    irt::engine::priv::EngineMetricsFault metrics(4);
    irt::engine::priv::CompletionOrder  completions(metrics);
    auto request = makeRequest("default", 3);
    auto future  = request->promise.get_future();
    completions.registerRequest(request);

    EXPECT_TRUE(completions.cancel(request->id));
    EXPECT_FALSE(completions.cancel(request->id));
    completions.completeSuccess(request, {});
    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_THROW(future.get(), irt::Exception);
    EXPECT_EQ(metrics.snapshot().cancelled_requests, 1U);
}

TEST(EngineCompletionOrderTest, CancellationOwnsRequestWhileRemovingItFromActiveSet)
{
    irt::engine::priv::EngineMetricsFault metrics(4);
    irt::engine::priv::CompletionOrder  completions(metrics);
    auto request = makeRequest("default", 4);
    auto future  = request->promise.get_future();
    completions.registerRequest(request);

    const auto request_id = request->id;
    request.reset();

    EXPECT_TRUE(completions.cancel(request_id));
    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_THROW(future.get(), irt::Exception);
    EXPECT_EQ(completions.activeCount(), 0U);
    EXPECT_EQ(metrics.snapshot().cancelled_requests, 1U);
}

} // namespace
