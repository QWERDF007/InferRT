#include "../../src/engine/priv/EngineTicketPool.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <thread>

namespace {

std::shared_ptr<const irt::engine::PipelinePlan> makeDeviceOnlyPlan()
{
    auto registry = std::make_shared<irt::engine::OperatorRegistry>();
    irt::engine::PipelineBuilder builder;
    builder.addTensor("input", {irt::engine::TensorDataType::F32,
                                 irt::engine::TensorLayout::NCHW,
                                 irt::engine::MemoryKind::DEVICE,
                                 2,
                                 2,
                                 3})
        .setModelInput("input");
    return builder.build(std::move(registry));
}

irt::engine::EngineConfig makePoolConfig()
{
    irt::engine::EngineConfig config;
    config.max_batch_size         = 1;
    config.execution_slots        = 1;
    config.cpu_preprocess_workers = 0;
    config.cpu_postprocess_workers = 0;
    config.pinned_input_tickets   = 1;
    config.pinned_output_tickets  = 1;
    return config;
}

TEST(EngineTicketPoolTest, AbortUnblocksWaitersAndTracksBorrowedTickets)
{
    irt::engine::priv::TicketPool pool(makePoolConfig(), makeDeviceOnlyPlan(), {});

    const auto initial = pool.snapshot();
    EXPECT_EQ(initial.input_count, 1U);
    EXPECT_EQ(initial.output_count, 1U);
    EXPECT_EQ(initial.input_in_use, 0U);
    EXPECT_EQ(initial.output_in_use, 0U);

    auto *input = pool.acquireInput();
    ASSERT_NE(input, nullptr);
    EXPECT_EQ(pool.snapshot().input_in_use, 1U);

    irt::engine::priv::InputTicket *aborted = nullptr;
    std::thread waiter([&] { aborted = pool.acquireInput(); });
    pool.abort();
    waiter.join();

    EXPECT_EQ(aborted, nullptr);
    pool.releaseInput(input);
    EXPECT_EQ(pool.snapshot().input_in_use, 0U);
    EXPECT_TRUE(pool.snapshot().aborting);
}

} // namespace
