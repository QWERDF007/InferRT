#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/BuiltinOperators.hpp>
#include <inferrt/engine/InferenceEngine.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "TestEngineHelpers.hpp"
#include "../../src/engine/priv/EngineTestHooks.hpp"

namespace {

class DelayPostOperator final : public irt::engine::IOperator
{
public:
    irt::engine::OperatorContract contract() const override
    {
        return {irt::engine::ExecutionKind::CPU, irt::engine::PipelineStage::CPU_POSTPROCESS, 1, 0, false, true, 0};
    }

    void execute(irt::engine::OperatorContext &context) override
    {
        if (context.image != nullptr && context.image->data[0] == 1)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
};

using irt::engine::test::buildIdentityEngine;

irt::engine::EngineConfig makeGpuConfig(const std::filesystem::path &engine_file)
{
    irt::engine::EngineConfig config;
    config.model_name              = "engine_gpu_test";
    config.engine_file             = engine_file;
    config.preprocess.input_width  = 2;
    config.preprocess.input_height = 2;
    config.min_batch_size          = 1;
    config.opt_batch_size          = 2;
    config.max_batch_size          = 4;
    config.max_wait                = std::chrono::microseconds(1000);
    config.execution_slots         = 2;
    config.cpu_preprocess_workers  = 2;
    config.cpu_postprocess_workers = 2;
    return config;
}

std::shared_ptr<const irt::engine::PipelinePlan> makePipeline(const bool two_inputs,
                                                              const bool delay_postprocess = false)
{
    using namespace irt::engine;
    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);
    if (delay_postprocess)
    {
        if (!registry->registerCreator("test.delay_post",
                                       DelayPostOperator{}.contract(),
                                       [](const NodeConfig &) { return std::make_unique<DelayPostOperator>(); }))
        {
            throw std::runtime_error("Failed to register delayed test operator");
        }
    }
    const TensorDesc host{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 2, 2, 3};
    const TensorDesc device{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 2, 2, 3};
    PipelineBuilder  builder;
    builder.addTensor("image_host", host)
        .addTensor("image_device", device)
        .addNode("cpu.image_to_tensor",
                 {
                     {},
                     {"image_host"},
                     CpuImageToTensorOptions{[]
                                             {
                                                 irt::PreprocessSpec spec;
                                                 spec.input_width    = 2;
                                                 spec.input_height   = 2;
                                                 spec.input_channels = 3;
                                                 spec.mean           = {0.0F, 0.0F, 0.0F};
                                                 spec.stddev         = {1.0F, 1.0F, 1.0F};
                                                 return spec;
                                             }()}
    })
        .addNode("cuda.upload", {{"image_host"}, {"image_device"}, {}})
        .bindModelInput("image_device", "image");
    if (two_inputs)
    {
        builder.addTensor("aux_host", host)
            .addTensor("aux_device", device)
            .addNode("cpu.copy_tensor", {{}, {"aux_host"}, CpuCopyTensorOptions{"aux"}})
            .addNode("cuda.upload", {{"aux_host"}, {"aux_device"}, {}})
            .bindModelInput("aux_device", "aux");
    }
    if (delay_postprocess)
    {
        builder.addNode("test.delay_post", {{"model.output"}, {}, {}});
    }
    return builder.build(std::move(registry));
}

bool hasCudaGpu()
{
    int        count  = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0)
    {
        return false;
    }
    return true;
}

TEST(InferenceEngineGpuTest, DynamicBatchMapsEveryResultToItsSourceSequence)
{
    if (!hasCudaGpu())
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }
    const auto engine_file = buildIdentityEngine(false);
    auto       engine      = irt::engine::InferenceEngine::create(makeGpuConfig(engine_file), makePipeline(false));
    engine->start();

    std::vector<std::future<irt::engine::InferenceResult>> futures;
    for (int index = 0; index < 4; ++index)
    {
        cv::Mat                     image(2, 2, CV_8UC3, cv::Scalar(index, index + 1, index + 2));
        irt::engine::RequestOptions options;
        options.source_id = "camera-a";
        futures.push_back(engine->submit(image, options).takeFuture());
    }
    for (int index = 0; index < 4; ++index)
    {
        const auto result = futures[static_cast<size_t>(index)].get();
        EXPECT_EQ(result.source_id, "camera-a");
        EXPECT_EQ(result.source_sequence, static_cast<uint64_t>(index));
        ASSERT_TRUE(result.outputs.contains("output"));
        EXPECT_FLOAT_EQ(result.outputs.at("output").front(), static_cast<float>(index + 2) / 255.0F);
    }
    EXPECT_TRUE(engine->faults().empty());
    engine->shutdown();
}

TEST(InferenceEngineGpuTest, BindsNamedAuxiliaryTensorInput)
{
    if (!hasCudaGpu())
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }
    const auto engine_file = buildIdentityEngine(true);
    auto       engine      = irt::engine::InferenceEngine::create(makeGpuConfig(engine_file), makePipeline(true));
    engine->start();

    cv::Mat image(2, 2, CV_8UC3, cv::Scalar(0, 0, 255));
    auto    result = engine
                      ->submit(image,
                               irt::engine::TensorInputMap{
                                   {"aux", std::vector<float>(12, 0.25F)}
    })
                      .get();
    ASSERT_TRUE(result.outputs.contains("output"));
    EXPECT_FLOAT_EQ(result.outputs.at("output").front(), 1.25F);
    engine->shutdown();
}

TEST(InferenceEngineGpuTest, KeepsSameSourceFutureDeliveryInSubmitOrder)
{
    if (!hasCudaGpu())
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }
    const auto engine_file = buildIdentityEngine(false);
    auto       engine = irt::engine::InferenceEngine::create(makeGpuConfig(engine_file), makePipeline(false, true));
    engine->start();

    irt::engine::RequestOptions first_options;
    first_options.source_id         = "ordered-camera";
    first_options.compatibility_key = "first";
    auto first = engine->submit(cv::Mat(2, 2, CV_8UC3, cv::Scalar(1, 0, 0)), first_options).takeFuture();
    irt::engine::RequestOptions second_options;
    second_options.source_id         = "ordered-camera";
    second_options.compatibility_key = "second";
    auto second = engine->submit(cv::Mat(2, 2, CV_8UC3, cv::Scalar(2, 0, 0)), second_options).takeFuture();

    EXPECT_EQ(second.wait_for(std::chrono::milliseconds(75)), std::future_status::timeout);
    EXPECT_EQ(first.get().source_sequence, 0U);
    EXPECT_EQ(second.get().source_sequence, 1U);
    engine->shutdown();
}

TEST(InferenceEngineGpuTest, FlushesPendingOrderedCompletionDuringShutdown)
{
    if (!hasCudaGpu())
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }
    const auto engine_file = buildIdentityEngine(false);
    auto       config      = makeGpuConfig(engine_file);
    config.execution_slots = 2;
    config.cpu_postprocess_workers = 2;
    auto engine = irt::engine::InferenceEngine::create(std::move(config), makePipeline(false, true));
    engine->start();

    irt::engine::RequestOptions first_options;
    first_options.source_id         = "shutdown-order-camera";
    first_options.compatibility_key = "first";
    auto first = engine->submit(cv::Mat(2, 2, CV_8UC3, cv::Scalar(1, 0, 0)), first_options).takeFuture();
    irt::engine::RequestOptions second_options;
    second_options.source_id         = "shutdown-order-camera";
    second_options.compatibility_key = "second";
    auto second = engine->submit(cv::Mat(2, 2, CV_8UC3, cv::Scalar(2, 0, 0)), second_options).takeFuture();

    std::thread shutdown_thread([&engine] { engine->shutdown(); });
    shutdown_thread.join();

    ASSERT_EQ(first.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    ASSERT_EQ(second.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_EQ(first.get().source_sequence, 0U);
    EXPECT_EQ(second.get().source_sequence, 1U);
    EXPECT_EQ(engine->state(), irt::engine::EngineState::Stopped);
    const auto metrics = engine->metrics();
    EXPECT_EQ(metrics.inflight_batches, 0U);
    EXPECT_EQ(metrics.pinned_input_in_use, 0U);
    EXPECT_EQ(metrics.pinned_output_in_use, 0U);
    EXPECT_EQ(metrics.active_requests, 0U);
    EXPECT_EQ(metrics.slot_count, 0U);
    EXPECT_EQ(metrics.idle_slot_count, 0U);
    EXPECT_EQ(metrics.active_slot_count, 0U);
    EXPECT_EQ(metrics.prepare_queue_size, 0U);
    EXPECT_EQ(metrics.gpu_queue_size, 0U);
    EXPECT_EQ(metrics.postprocess_queue_size, 0U);
    EXPECT_EQ(metrics.input_ticket_count, 0U);
    EXPECT_EQ(metrics.output_ticket_count, 0U);
    EXPECT_EQ(metrics.thread_count, 0U);
    EXPECT_EQ(metrics.joinable_thread_count, 0U);
}

TEST(InferenceEngineGpuTest, DuplicateCompletionFromProductionPollerIsExactlyOnce)
{
    if (!hasCudaGpu())
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }
    const auto engine_file = buildIdentityEngine(false);
    auto       config      = makeGpuConfig(engine_file);
    irt::engine::priv::EngineTestOptions duplicate_options;
    duplicate_options.duplicate_completion = true;
    irt::engine::priv::EngineTestOptionsGuard duplicate_options_guard(duplicate_options);
    config.execution_slots = 1;
    config.max_batch_size = 1;
    config.opt_batch_size = 1;
    auto engine = irt::engine::InferenceEngine::create(std::move(config), makePipeline(false));
    engine->start();

    std::vector<std::future<irt::engine::InferenceResult>> futures;
    for (int index = 0; index < 4; ++index)
    {
        futures.push_back(
            engine->submit(cv::Mat(2, 2, CV_8UC3, cv::Scalar(index, 0, 0)), irt::engine::RequestOptions{})
                .takeFuture());
    }
    for (auto &future : futures)
    {
        ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
        EXPECT_TRUE(future.get().outputs.contains("output"));
    }
    const auto metrics = engine->metrics();
    EXPECT_EQ(metrics.accepted_requests, 4U);
    EXPECT_EQ(metrics.completed_requests, 4U);
    EXPECT_EQ(metrics.failed_requests + metrics.cancelled_requests + metrics.timed_out_requests
                  + metrics.dropped_requests,
              0U);
    engine->shutdown();
}

TEST(InferenceEngineGpuTest, FatalCompletionAndConcurrentShutdownConvergeExactlyOnce)
{
    if (!hasCudaGpu())
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }
    const auto engine_file = buildIdentityEngine(false);
    auto       config      = makeGpuConfig(engine_file);
    config.execution_slots = 1;
    config.queue_capacity = 8;
    config.max_batch_size = 1;
    config.opt_batch_size = 1;
    irt::engine::priv::EngineTestOptions fatal_options;
    fatal_options.completion_fatal = true;
    fatal_options.completion_fatal_delay_ms = 25;
    irt::engine::priv::EngineTestOptionsGuard fatal_options_guard(fatal_options);
    auto engine = irt::engine::InferenceEngine::create(std::move(config), makePipeline(false));
    engine->start();

    std::vector<std::future<irt::engine::InferenceResult>> futures;
    for (int index = 0; index < 8; ++index)
    {
        futures.push_back(
            engine->submit(cv::Mat(2, 2, CV_8UC3, cv::Scalar(index, 0, 0)), irt::engine::RequestOptions{})
                .takeFuture());
    }
    std::thread shutdown_thread([&engine] { engine->shutdown(); });
    shutdown_thread.join();

    size_t terminal_count = 0;
    for (auto &future : futures)
    {
        ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
        try
        {
            (void)future.get();
        }
        catch (const irt::Exception &)
        {
        }
        ++terminal_count;
    }
    EXPECT_EQ(terminal_count, futures.size());
    const auto faults = engine->faults();
    const auto completion_fault = std::find_if(
        faults.begin(), faults.end(), [](const irt::engine::FaultRecord &fault) {
            return fault.stage == irt::engine::FaultStage::CompletionPoll && fault.fatal;
        });
    ASSERT_NE(completion_fault, faults.end());
    EXPECT_NE(completion_fault->message.find("Injected completion poller failure"), std::string::npos);
    const auto metrics = engine->metrics();
    EXPECT_EQ(metrics.inflight_batches, 0U);
    EXPECT_EQ(metrics.queued_requests, 0U);
    EXPECT_EQ(metrics.pinned_input_in_use, 0U);
    EXPECT_EQ(metrics.pinned_output_in_use, 0U);
    EXPECT_EQ(metrics.active_requests, 0U);
    EXPECT_EQ(metrics.slot_count, 0U);
    EXPECT_EQ(metrics.idle_slot_count, 0U);
    EXPECT_EQ(metrics.active_slot_count, 0U);
    EXPECT_EQ(metrics.prepare_queue_size, 0U);
    EXPECT_EQ(metrics.gpu_queue_size, 0U);
    EXPECT_EQ(metrics.postprocess_queue_size, 0U);
    EXPECT_EQ(metrics.input_ticket_count, 0U);
    EXPECT_EQ(metrics.output_ticket_count, 0U);
    EXPECT_EQ(metrics.thread_count, 0U);
    EXPECT_EQ(metrics.joinable_thread_count, 0U);
    EXPECT_EQ(metrics.accepted_requests,
              metrics.completed_requests + metrics.failed_requests + metrics.cancelled_requests
                  + metrics.timed_out_requests + metrics.dropped_requests);
}

TEST(InferenceEngineGpuTest, RecordsNamedInputPreprocessFailure)
{
    if (!hasCudaGpu())
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }
    const auto engine_file = buildIdentityEngine(true);
    auto       engine      = irt::engine::InferenceEngine::create(makeGpuConfig(engine_file), makePipeline(true));
    engine->start();

    irt::engine::RequestOptions options;
    options.source_id = "aux-camera";
    auto request      = engine->submit(cv::Mat(2, 2, CV_8UC3, cv::Scalar(0, 0, 0)), {}, options);
    EXPECT_THROW(std::move(request).takeFuture().get(), irt::Exception);
    const auto faults = engine->faults();
    ASSERT_FALSE(faults.empty());
    EXPECT_EQ(faults.back().stage, irt::engine::FaultStage::CpuPreprocess);
    EXPECT_EQ(faults.back().source_id, "aux-camera");
    EXPECT_EQ(faults.back().source_sequence, 0U);
    engine->shutdown();
}

TEST(InferenceEngineGpuTest, MemoryLimitFailureIsRecorded)
{
    if (!hasCudaGpu())
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }
    const auto engine_file           = buildIdentityEngine(false);
    auto       config                = makeGpuConfig(engine_file);
    config.pinned_memory_limit_bytes = 1;
    auto engine                      = irt::engine::InferenceEngine::create(std::move(config), makePipeline(false));
    EXPECT_THROW(engine->start(), irt::Exception);
    ASSERT_FALSE(engine->faults().empty());
    EXPECT_EQ(engine->faults().back().stage, irt::engine::FaultStage::Start);
}

TEST(InferenceEngineGpuTest, DeviceArenaLimitFailureIsRecorded)
{
    if (!hasCudaGpu())
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }
    const auto engine_file           = buildIdentityEngine(false);
    auto       config                = makeGpuConfig(engine_file);
    config.device_memory_limit_bytes = 1;
    auto engine                      = irt::engine::InferenceEngine::create(std::move(config), makePipeline(false));
    EXPECT_THROW(engine->start(), irt::Exception);
    ASSERT_FALSE(engine->faults().empty());
    EXPECT_EQ(engine->faults().back().stage, irt::engine::FaultStage::Start);
    EXPECT_EQ(engine->faults().back().status, irt::Status::ERROR_OUT_OF_MEMORY);
}

TEST(InferenceEngineGpuTest, UsesMultipleDevicesWhenAvailable)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2)
    {
        GTEST_SKIP() << "Requires at least two CUDA GPUs";
    }
    const auto engine_file = buildIdentityEngine(false);
    auto       config      = makeGpuConfig(engine_file);
    config.device_ids      = {0, 1};
    auto engine            = irt::engine::InferenceEngine::create(std::move(config), makePipeline(false));
    engine->start();
    std::vector<std::future<irt::engine::InferenceResult>> futures;
    for (int index = 0; index < 8; ++index)
    {
        futures.push_back(engine->submit(cv::Mat(2, 2, CV_8UC3, cv::Scalar(0, 0, index))));
    }
    for (auto &future : futures)
    {
        EXPECT_TRUE(future.get().outputs.contains("output"));
    }
    EXPECT_TRUE(engine->faults().empty());
    engine->shutdown();
}

} // namespace
