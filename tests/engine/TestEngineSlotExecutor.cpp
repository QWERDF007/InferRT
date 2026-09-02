#include "../../src/engine/priv/EngineSlotExecutor.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <thread>
#include <utility>
#include <vector>

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
    config.max_batch_size          = 1;
    config.execution_slots         = 1;
    config.cpu_preprocess_workers  = 0;
    config.cpu_postprocess_workers = 0;
    config.pinned_input_tickets    = 1;
    config.pinned_output_tickets   = 1;
    return config;
}

TEST(EngineSlotExecutorTest, OwnsSlotsAndStopsDispatcherAndPollerIdempotently)
{
    irt::engine::priv::StageQueue gpu_queue;
    irt::engine::priv::StageQueue postprocess_queue;
    auto                          pipeline = makeDeviceOnlyPlan();
    irt::engine::priv::TicketPool pool(makePoolConfig(), pipeline, {});
    std::vector<std::unique_ptr<irt::engine::priv::Slot>> slots;
    slots.push_back(std::make_unique<irt::engine::priv::Slot>(0));

    irt::engine::priv::SlotExecutor executor(
        *pipeline, gpu_queue, postprocess_queue, pool, std::move(slots), {}, [] { return false; },
        [](const irt::engine::priv::BatchPtr &, std::exception_ptr) {},
        [](irt::engine::FaultStage, int, const irt::engine::priv::RequestPtr &, std::exception_ptr, bool) {},
        [] {}, [](const irt::engine::priv::BatchPtr &) {});

    executor.startDispatcher();
    executor.startCompletionPoller();
    const auto started = executor.snapshot();
    EXPECT_EQ(started.slot_count, 1U);
    EXPECT_EQ(started.idle_slot_count, 1U);
    EXPECT_EQ(started.active_slot_count, 0U);
    EXPECT_TRUE(started.dispatcher_joinable);
    EXPECT_TRUE(started.completion_poller_joinable);

    std::thread first_stop([&executor] { executor.stop(); });
    std::thread second_stop([&executor] { executor.stop(); });
    first_stop.join();
    second_stop.join();
    executor.stop();
    const auto stopped = executor.snapshot();
    EXPECT_EQ(stopped.slot_count, 1U);
    EXPECT_EQ(stopped.idle_slot_count, 1U);
    EXPECT_EQ(stopped.active_slot_count, 0U);
    EXPECT_FALSE(stopped.dispatcher_joinable);
    EXPECT_FALSE(stopped.completion_poller_joinable);
}

} // namespace
