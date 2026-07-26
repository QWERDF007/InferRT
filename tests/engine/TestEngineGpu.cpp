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

namespace {

class TensorRtLogger final : public nvinfer1::ILogger
{
public:
    void log(Severity, const char *) noexcept override {}
};

struct TensorRtDeleter
{
    template<typename T>
    void operator()(T *object) const
    {
        delete object;
    }
};

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

std::filesystem::path buildIdentityEngine(const bool two_inputs)
{
    TensorRtLogger                                       logger;
    std::unique_ptr<nvinfer1::IBuilder, TensorRtDeleter> builder(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        throw std::runtime_error("Failed to create TensorRT builder");
    }
    std::unique_ptr<nvinfer1::INetworkDefinition, TensorRtDeleter> network(builder->createNetworkV2(0U));
    if (!network)
    {
        throw std::runtime_error("Failed to create TensorRT network");
    }
    auto *image = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{-1, 3, 2, 2});
    if (!image)
    {
        throw std::runtime_error("Failed to add image input");
    }

    nvinfer1::ITensor *output = nullptr;
    if (two_inputs)
    {
        auto *aux = network->addInput("aux", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{-1, 3, 2, 2});
        if (!aux)
        {
            throw std::runtime_error("Failed to add auxiliary input");
        }
        auto *sum = network->addElementWise(*image, *aux, nvinfer1::ElementWiseOperation::kSUM);
        if (!sum)
        {
            throw std::runtime_error("Failed to add TensorRT sum layer");
        }
        output = sum->getOutput(0);
    }
    else
    {
        auto *identity = network->addIdentity(*image);
        if (!identity)
        {
            throw std::runtime_error("Failed to add TensorRT identity layer");
        }
        output = identity->getOutput(0);
    }
    output->setName("output");
    network->markOutput(*output);

    std::unique_ptr<nvinfer1::IBuilderConfig, TensorRtDeleter> config(builder->createBuilderConfig());
    if (!config)
    {
        throw std::runtime_error("Failed to create TensorRT builder config");
    }
    auto *profile = builder->createOptimizationProfile();
    if (!profile)
    {
        throw std::runtime_error("Failed to create TensorRT optimization profile");
    }
    for (const char *name : two_inputs ? std::vector<const char *>{"image", "aux"} : std::vector<const char *>{"image"})
    {
        if (!profile->setDimensions(name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 2, 2})
            || !profile->setDimensions(name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{2, 3, 2, 2})
            || !profile->setDimensions(name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{4, 3, 2, 2}))
        {
            throw std::runtime_error("Failed to configure TensorRT optimization profile");
        }
    }
    if (config->addOptimizationProfile(profile) == -1)
    {
        throw std::runtime_error("Failed to add TensorRT optimization profile");
    }
    std::unique_ptr<nvinfer1::IHostMemory, TensorRtDeleter> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized)
    {
        throw std::runtime_error("Failed to serialize TensorRT test engine");
    }

    const auto path = std::filesystem::temp_directory_path()
                    / (two_inputs ? "inferrt_gpu_multi_input.engine" : "inferrt_gpu_identity.engine");
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.good())
    {
        throw std::runtime_error("Failed to open TensorRT test engine output");
    }
    file.write(static_cast<const char *>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
    if (!file.good())
    {
        throw std::runtime_error("Failed to write TensorRT test engine");
    }
    return path;
}

irt::engine::EngineConfig makeGpuConfig(const std::filesystem::path &engine_file)
{
    irt::engine::EngineConfig config;
    config.model_name              = "engine_gpu_test";
    config.engine_file             = engine_file;
    config.input_width             = 2;
    config.input_height            = 2;
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
                     CpuImageToTensorOptions{{0.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 1.0F}, false}
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
