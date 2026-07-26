#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/BuiltinOperators.hpp>
#include <inferrt/engine/EngineConfig.hpp>
#include <inferrt/engine/InferenceEngine.hpp>
#include <inferrt/engine/Pipeline.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>

namespace {

class TestPostOperator final : public irt::engine::IOperator
{
public:
    irt::engine::OperatorContract contract() const override
    {
        return {irt::engine::ExecutionKind::CPU, irt::engine::PipelineStage::CPU_POSTPROCESS, 1, 0, false, true, 0};
    }

    void execute(irt::engine::OperatorContext &context) override
    {
        context.results["postprocessed"] = {static_cast<float>(context.request_index)};
    }
};

class TestResultPostOperator final : public irt::engine::IOperator
{
public:
    irt::engine::OperatorContract contract() const override
    {
        return {irt::engine::ExecutionKind::CPU, irt::engine::PipelineStage::CPU_POSTPROCESS, 1, 0, false, true, 0};
    }

    void execute(irt::engine::OperatorContext &) override {}
};

class TestCudaResultOperator final : public irt::engine::IOperator
{
public:
    irt::engine::OperatorContract contract() const override
    {
        return {irt::engine::ExecutionKind::CUDA, irt::engine::PipelineStage::CUDA_POSTPROCESS, 1, 1, false, false, 0};
    }

    void execute(irt::engine::OperatorContext &) override {}
};

irt::engine::EngineConfig makeConfig()
{
    irt::engine::EngineConfig config;
    config.model_name     = "resnet18";
    config.engine_file    = "resnet18.engine";
    config.input_width    = 224;
    config.input_height   = 224;
    config.min_batch_size = 1;
    config.opt_batch_size = 2;
    config.max_batch_size = 4;
    return config;
}

TEST(EngineConfigTest, AcceptsValidConfiguration)
{
    EXPECT_NO_THROW(makeConfig().validate());
}

TEST(EngineConfigTest, RejectsInvalidBatchRange)
{
    auto config           = makeConfig();
    config.min_batch_size = 4;
    config.opt_batch_size = 2;
    config.max_batch_size = 1;
    EXPECT_THROW(config.validate(), irt::Exception);
}

TEST(EngineConfigTest, ValidatesQueuePolicy)
{
    auto config         = makeConfig();
    config.queue_policy = static_cast<irt::engine::QueuePolicy>(99);
    EXPECT_THROW(config.validate(), irt::Exception);
}

TEST(EngineConfigTest, ValidatesPipelineWorkersAndPreferredBatches)
{
    auto config                    = makeConfig();
    config.cpu_preprocess_workers  = 2;
    config.cpu_postprocess_workers = 3;
    config.preferred_batch_sizes   = {1, 2, 4};
    config.static_batch_policy     = irt::engine::StaticBatchPolicy::Pad;
    EXPECT_NO_THROW(config.validate());

    config.cpu_preprocess_workers = 0;
    EXPECT_THROW(config.validate(), irt::Exception);

    config.cpu_preprocess_workers = 1;
    config.preferred_batch_sizes  = {5};
    EXPECT_THROW(config.validate(), irt::Exception);
}

TEST(EngineConfigTest, ValidatesMultipleDeviceConfiguration)
{
    auto config       = makeConfig();
    config.device_ids = {0, 1};
    EXPECT_NO_THROW(config.validate());

    config.device_ids = {0, 0};
    EXPECT_THROW(config.validate(), irt::Exception);

    config.device_ids = {0, -1};
    EXPECT_THROW(config.validate(), irt::Exception);
}

TEST(EngineConfigTest, AcceptsCudaLetterboxConfiguration)
{
    auto config               = makeConfig();
    config.mean               = {0.0F, 0.0F, 0.0F};
    config.stddev             = {1.0F, 1.0F, 1.0F};
    config.preprocess_backend = irt::engine::PreprocessBackend::CUDA;
    config.letterbox          = true;
    config.source_width       = 768;
    config.source_height      = 576;

    EXPECT_NO_THROW(config.validate());
}

TEST(EngineConfigTest, AcceptsCudaResizeAndNormalizeConfiguration)
{
    auto config               = makeConfig();
    config.preprocess_backend = irt::engine::PreprocessBackend::CUDA;
    config.source_width       = 768;
    config.source_height      = 576;

    EXPECT_NO_THROW(config.validate());
}

TEST(EngineConfigTest, RejectsNonIdentityNormalizationForCudaLetterbox)
{
    auto config               = makeConfig();
    config.preprocess_backend = irt::engine::PreprocessBackend::CUDA;
    config.letterbox          = true;
    config.source_width       = 768;
    config.source_height      = 576;

    EXPECT_NO_THROW(config.validate());
    EXPECT_THROW(irt::engine::InferenceEngine::create(std::move(config)), irt::Exception);
}

TEST(EngineConfigTest, RejectsIncompleteCudaConfiguration)
{
    auto config               = makeConfig();
    config.preprocess_backend = irt::engine::PreprocessBackend::CUDA;
    EXPECT_NO_THROW(config.validate());
    EXPECT_THROW(irt::engine::InferenceEngine::create(std::move(config)), irt::Exception);
}

TEST(EngineConfigTest, ValidatesFeatureOnlyOutputContract)
{
    auto config                 = makeConfig();
    config.feature_only         = true;
    config.output_tensor_names  = {"x_norm_clstoken"};
    config.feature_tensor_names = {"x_norm_clstoken"};

    EXPECT_NO_THROW(config.validate());

    config.feature_tensor_names.clear();
    EXPECT_THROW(config.validate(), irt::Exception);
}

TEST(EngineConfigTest, LoadsYamlRelativeToItsDirectory)
{
    const auto directory = std::filesystem::temp_directory_path() / "inferrt_engine_config_test";
    std::filesystem::create_directories(directory);
    const auto config_path = directory / "engine.yaml";

    {
        std::ofstream output(config_path);
        output << "model:\n"
                  "  name: resnet18\n"
                  "  engine: models/resnet18.engine\n"
                  "  runtime: tensorrt:0\n"
                  "input:\n"
                  "  width: 224\n"
                  "  height: 224\n"
                  "batch:\n"
                  "  min: 1\n"
                  "  opt: 2\n"
                  "  max: 4\n"
                  "  max_wait_us: 2500\n"
                  "runtime:\n"
                  "  execution_slots: 2\n"
                  "  queue_capacity: 32\n"
                  "  queue_policy: drop_oldest\n";
    }

    const auto config = irt::engine::EngineConfig::load(config_path);
    EXPECT_EQ(config.engine_file, directory / "models/resnet18.engine");
    EXPECT_EQ(config.max_wait, std::chrono::microseconds(2500));
    EXPECT_EQ(config.execution_slots, 2U);
    EXPECT_EQ(config.queue_capacity, 32U);
    EXPECT_EQ(config.queue_policy, irt::engine::QueuePolicy::DropOldest);

    std::filesystem::remove(config_path);
    std::filesystem::remove(directory);
}

TEST(PipelinePlanTest, BuildsImmutableCodeDefinedDag)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);
    ASSERT_TRUE(registry->registerCreator("test.post",
                                          [](const NodeConfig &) { return std::make_unique<TestPostOperator>(); }));

    PipelineBuilder builder;
    builder
        .addTensor("host_input",
                   {
                       TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3
    })
        .addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addNode(
            "cpu.image_to_tensor",
            {{}, {"host_input"}, CpuImageToTensorOptions{{0.485F, 0.456F, 0.406F}, {0.229F, 0.224F, 0.225F}, false}})
        .addNode("cuda.upload", {{"host_input"}, {"model_input"}, {}})
        .addNode("test.post", {{"model.output"}, {}, {}})
        .addResult("model_input", "preprocessed")
        .setModelInput("model_input");

    const auto plan = builder.build(registry);
    EXPECT_TRUE(registry->frozen());
    EXPECT_FALSE(registry->registerCreator("test.other",
                                           [](const NodeConfig &) -> std::unique_ptr<IOperator> { return nullptr; }));
    ASSERT_EQ(plan->nodes().size(), 3U);
    EXPECT_EQ(plan->nodes()[0].contract.stage, PipelineStage::CPU_PREPROCESS);
    EXPECT_EQ(plan->nodes()[1].contract.stage, PipelineStage::H2D);
    EXPECT_EQ(plan->nodes()[2].contract.stage, PipelineStage::CPU_POSTPROCESS);
    ASSERT_EQ(plan->results().size(), 1U);
    EXPECT_EQ(plan->results().front().tensor_name, "model_input");
    EXPECT_EQ(plan->results().front().result_name, "preprocessed");
    EXPECT_EQ(plan->modelInput(), "model_input");
    ASSERT_EQ(plan->modelInputs().size(), 1U);
    EXPECT_EQ(plan->modelInputs().front().tensor_name, "model_input");
    EXPECT_EQ(plan->createOperators().size(), 3U);

    const auto engine = InferenceEngine::create(makeConfig(), plan);
    EXPECT_EQ(engine->config().model_name, "resnet18");
}

TEST(PipelinePlanTest, BindsMultipleNamedTensorRtInputsInCode)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);
    const TensorDesc host{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 2, 2, 3};
    const TensorDesc device{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 2, 2, 3};
    PipelineBuilder  builder;
    builder.addTensor("image_host", host)
        .addTensor("image_device", device)
        .addTensor("aux_host", host)
        .addTensor("aux_device", device)
        .addNode("cpu.image_to_tensor",
                 {
                     {},
                     {"image_host"},
                     CpuImageToTensorOptions{{0.0F, 0.0F, 0.0F}, {1.0F, 1.0F, 1.0F}, false}
    })
        .addNode("cpu.copy_tensor", {{}, {"aux_host"}, CpuCopyTensorOptions{"aux"}})
        .addNode("cuda.upload", {{"image_host"}, {"image_device"}, {}})
        .addNode("cuda.upload", {{"aux_host"}, {"aux_device"}, {}})
        .bindModelInput("image_device", "image")
        .bindModelInput("aux_device", "aux");

    const auto plan = builder.build(std::move(registry));
    ASSERT_EQ(plan->modelInputs().size(), 2U);
    EXPECT_EQ(plan->modelInputs()[0].tensor_name, "image_device");
    EXPECT_EQ(plan->modelInputs()[0].engine_tensor_name, "image");
    EXPECT_EQ(plan->modelInputs()[1].tensor_name, "aux_device");
    EXPECT_EQ(plan->modelInputs()[1].engine_tensor_name, "aux");
}

TEST(PipelinePlanTest, ExposesDeviceResultOnlyToCpuPostprocess)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);
    ASSERT_TRUE(registry->registerCreator(
        "test.result_post", [](const NodeConfig &) { return std::make_unique<TestResultPostOperator>(); }));
    ASSERT_TRUE(registry->registerCreator(
        "test.cuda_result", [](const NodeConfig &) { return std::make_unique<TestCudaResultOperator>(); }));

    PipelineBuilder builder;
    builder
        .addTensor("host_input",
                   {
                       TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3
    })
        .addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addTensor("keep", {TensorDataType::I64, TensorLayout::HWC, MemoryKind::DEVICE, 16, 1, 1})
        .addNode(
            "cpu.image_to_tensor",
            {{}, {"host_input"}, CpuImageToTensorOptions{{0.485F, 0.456F, 0.406F}, {0.229F, 0.224F, 0.225F}, false}})
        .addNode("cuda.upload", {{"host_input"}, {"model_input"}, {}})
        .addNode("test.cuda_result", {{"model.output"}, {"keep"}, {}})
        .addNode("test.result_post", {{"result.keep_indices"}, {}, {}})
        .addResult("keep", "keep_indices")
        .setModelInput("model_input");

    EXPECT_NO_THROW(builder.build(registry));
}

TEST(PipelinePlanTest, RejectsWrongMemoryAtStageBoundary)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);
    PipelineBuilder builder;
    builder.addTensor("device_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addNode("cuda.upload", {{"device_input"}, {"model_input"}, {}})
        .setModelInput("model_input");

    EXPECT_THROW(builder.build(registry), irt::Exception);
}

TEST(RequestHandleTest, DefaultHandleIsInvalid)
{
    irt::engine::RequestHandle handle;
    EXPECT_EQ(handle.id(), 0U);
    EXPECT_FALSE(handle.valid());
}

} // namespace
