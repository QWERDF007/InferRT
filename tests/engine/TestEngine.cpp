#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/BuiltinOperators.hpp>
#include <inferrt/engine/EngineConfig.hpp>
#include <inferrt/engine/InferenceEngine.hpp>
#include <inferrt/engine/Pipeline.hpp>

#include "TestEngineHelpers.hpp"
#include "../../src/engine/priv/EngineTestHooks.hpp"

#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <thread>
#include <utility>
#include <vector>

namespace {

irt::PreprocessSpec makePreprocessSpec(const int width = 224, const int height = 224,
                                       std::vector<float> mean = {0.485F, 0.456F, 0.406F},
                                       std::vector<float> stddev = {0.229F, 0.224F, 0.225F},
                                       const bool letterbox = false)
{
    irt::PreprocessSpec spec;
    spec.input_width    = width;
    spec.input_height   = height;
    spec.input_channels = 3;
    spec.mean           = std::move(mean);
    spec.stddev         = std::move(stddev);
    spec.padding_mode   = letterbox ? irt::PaddingMode::Letterbox : irt::PaddingMode::DirectResize;
    return spec;
}

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

class ScratchOperator final : public irt::engine::IOperator
{
public:
    irt::engine::OperatorContract contract() const override
    {
        return {irt::engine::ExecutionKind::CPU, irt::engine::PipelineStage::CPU_PREPROCESS, 0, 0, false, false, 64};
    }

    void execute(irt::engine::OperatorContext &context) override
    {
        if (context.scratch == nullptr || context.scratch_bytes != 64)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Scratch contract was not materialized");
        }
        static_cast<unsigned char *>(context.scratch)[0] = 0x5a;
    }
};

class WrongContractOperator final : public irt::engine::IOperator
{
public:
    irt::engine::OperatorContract contract() const override
    {
        return {irt::engine::ExecutionKind::CPU, irt::engine::PipelineStage::CPU_POSTPROCESS, 0, 0, false, true, 0};
    }

    void execute(irt::engine::OperatorContext &) override {}
};

irt::engine::EngineConfig makeConfig()
{
    irt::engine::EngineConfig config;
    config.model_name     = "resnet18";
    config.engine_file    = "resnet18.engine";
    config.preprocess.input_width    = 224;
    config.preprocess.input_height   = 224;
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
    config.preprocess.mean            = {0.0F, 0.0F, 0.0F};
    config.preprocess.stddev          = {1.0F, 1.0F, 1.0F};
    config.preprocess.backend          = irt::engine::PreprocessBackend::CUDA;
    config.preprocess.padding_mode     = irt::PaddingMode::Letterbox;
    config.preprocess.source_width     = 768;
    config.preprocess.source_height    = 576;

    EXPECT_NO_THROW(config.validate());
}

TEST(EngineConfigTest, MaterializesPreprocessSpec)
{
    auto config               = makeConfig();
    config.preprocess.mean            = {0.1F, 0.2F, 0.3F};
    config.preprocess.stddev          = {0.9F, 0.8F, 0.7F};
    config.preprocess.backend          = irt::engine::PreprocessBackend::CUDA;
    config.preprocess.padding_mode     = irt::PaddingMode::Letterbox;
    config.preprocess.source_width     = 1280;
    config.preprocess.source_height    = 720;

    const auto spec = config.preprocessSpec();
    EXPECT_EQ(spec.input_width, config.preprocess.input_width);
    EXPECT_EQ(spec.input_height, config.preprocess.input_height);
    EXPECT_EQ(spec.input_channels, config.preprocess.input_channels);
    EXPECT_EQ(spec.backend, irt::PreprocessBackend::CUDA);
    EXPECT_EQ(spec.padding_mode, irt::PaddingMode::Letterbox);
    EXPECT_EQ(spec.source_width, 1280);
    EXPECT_EQ(spec.source_height, 720);
    EXPECT_EQ(spec.mean, (std::vector<float>{0.1F, 0.2F, 0.3F}));
    EXPECT_EQ(spec.stddev, (std::vector<float>{0.9F, 0.8F, 0.7F}));
}

TEST(EngineConfigTest, PreprocessSpecIsTheSingleConfigurationSource)
{
    irt::engine::EngineConfig config;
    config.preprocess.input_width    = 320;
    config.preprocess.input_height   = 192;
    config.preprocess.input_channels = 3;
    config.preprocess.mean           = {0.1F, 0.2F, 0.3F};
    config.preprocess.stddev         = {0.9F, 0.8F, 0.7F};
    config.preprocess.padding_mode   = irt::PaddingMode::Letterbox;
    config.preprocess.backend        = irt::PreprocessBackend::CUDA;

    const auto &spec = config.preprocessSpec();
    EXPECT_EQ(spec.input_width, 320);
    EXPECT_EQ(spec.input_height, 192);
    EXPECT_EQ(spec.mean, (std::vector<float>{0.1F, 0.2F, 0.3F}));
    EXPECT_EQ(spec.stddev, (std::vector<float>{0.9F, 0.8F, 0.7F}));
    EXPECT_EQ(spec.padding_mode, irt::PaddingMode::Letterbox);
    EXPECT_EQ(spec.backend, irt::PreprocessBackend::CUDA);
}

TEST(EngineConfigTest, ValidatesMaterializedPreprocessSpec)
{
    auto config         = makeConfig();
    config.preprocess.source_width = 640;
    EXPECT_THROW(config.validate(), irt::Exception);

    config.preprocess.source_height = 480;
    config.preprocess.stddev[1]     = 1.0e-8F;
    EXPECT_THROW(config.validate(), irt::Exception);
}

TEST(EngineConfigTest, AcceptsCudaResizeAndNormalizeConfiguration)
{
    auto config               = makeConfig();
    config.preprocess.backend       = irt::engine::PreprocessBackend::CUDA;
    config.preprocess.source_width  = 768;
    config.preprocess.source_height = 576;

    EXPECT_NO_THROW(config.validate());
}

TEST(EngineConfigTest, AcceptsNonIdentityNormalizationForCudaLetterbox)
{
    auto config               = makeConfig();
    config.preprocess.backend       = irt::engine::PreprocessBackend::CUDA;
    config.preprocess.padding_mode  = irt::PaddingMode::Letterbox;
    config.preprocess.source_width  = 768;
    config.preprocess.source_height = 576;

    EXPECT_NO_THROW(config.validate());
    EXPECT_NO_THROW(irt::engine::InferenceEngine::create(std::move(config)));
}

TEST(EngineConfigTest, RejectsIncompleteCudaConfiguration)
{
    auto config               = makeConfig();
    config.preprocess.backend = irt::engine::PreprocessBackend::CUDA;
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

TEST(EngineConfigTest, LoadsCompletePreprocessSpecFromYaml)
{
    const auto directory   = std::filesystem::temp_directory_path() / "inferrt_engine_preprocess_config_test";
    const auto config_path = directory / "engine.yaml";
    std::filesystem::create_directories(directory);
    {
        std::ofstream output(config_path);
        output << "model:\n"
                  "  name: gray_model\n"
                  "  engine: gray.engine\n"
                  "  runtime: tensorrt:1\n"
                  "input:\n"
                  "  width: 320\n"
                  "  height: 192\n"
                  "  channels: 1\n"
                  "preprocess:\n"
                  "  backend: cuda\n"
                  "  src_color: gray\n"
                  "  dst_color: gray\n"
                  "  interpolation: nearest\n"
                  "  padding_mode: center_crop\n"
                  "  padding_alignment: top_left\n"
                  "  source_channels: 1\n"
                  "  source_width: 640\n"
                  "  source_height: 480\n"
                  "  pad_value: 7.5\n"
                  "  pad_after_normalize: false\n"
                  "  scale: 0.5\n"
                  "  normalize:\n"
                  "    mean: [1.0]\n"
                  "    stddev: [2.0]\n"
                  "  output:\n"
                  "    layout: nchw\n"
                  "    dtype: f32\n"
                  "batch:\n"
                  "  min: 1\n"
                  "  opt: 1\n"
                  "  max: 1\n";
    }

    const auto config = irt::engine::EngineConfig::load(config_path);
    const auto &spec   = config.preprocessSpec();
    EXPECT_EQ(config.device_id, 1);
    EXPECT_EQ(spec.input_width, 320);
    EXPECT_EQ(spec.input_height, 192);
    EXPECT_EQ(spec.input_channels, 1);
    EXPECT_EQ(spec.source_channels, 1);
    EXPECT_EQ(spec.src_color, irt::ColorFormat::GRAY);
    EXPECT_EQ(spec.dst_color, irt::ColorFormat::GRAY);
    EXPECT_EQ(spec.interpolation, irt::Interpolation::Nearest);
    EXPECT_EQ(spec.padding_mode, irt::PaddingMode::CenterCrop);
    EXPECT_EQ(spec.padding_alignment, irt::PaddingAlignment::TopLeft);
    EXPECT_EQ(spec.source_width, 640);
    EXPECT_EQ(spec.source_height, 480);
    EXPECT_FLOAT_EQ(spec.pad_value, 7.5F);
    EXPECT_FALSE(spec.pad_after_normalize);
    EXPECT_FLOAT_EQ(spec.scale, 0.5F);
    EXPECT_EQ(spec.output_layout, irt::TensorLayout::NCHW);
    EXPECT_EQ(spec.output_dtype, irt::TensorDataType::F32);
    EXPECT_EQ(spec.mean, std::vector<float>({1.0F}));
    EXPECT_EQ(spec.stddev, std::vector<float>({2.0F}));

    std::filesystem::remove(config_path);
    std::filesystem::remove(directory);
}

TEST(EngineConfigTest, RejectsConflictingLegacyLetterboxFlag)
{
    const auto directory   = std::filesystem::temp_directory_path() / "inferrt_engine_preprocess_conflict_test";
    const auto config_path = directory / "engine.yaml";
    std::filesystem::create_directories(directory);
    {
        std::ofstream output(config_path);
        output << "model:\n"
                  "  name: resnet18\n"
                  "  engine: resnet18.engine\n"
                  "input:\n"
                  "  width: 224\n"
                  "  height: 224\n"
                  "preprocess:\n"
                  "  padding_mode: center_crop\n"
                  "  letterbox: true\n"
                  "batch:\n"
                  "  min: 1\n";
    }

    EXPECT_THROW(irt::engine::EngineConfig::load(config_path), irt::Exception);
    std::filesystem::remove(config_path);
    std::filesystem::remove(directory);
}

TEST(BuiltinOperatorTest, CopiesNonContinuousImageByRows)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);

    PipelineBuilder builder;
    builder.addTensor("host_image", {TensorDataType::U8, TensorLayout::HWC, MemoryKind::HOST, 2, 2, 3})
        .addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 2, 2, 3})
        .addNode("cpu.copy_image", {{}, {"host_image"}, {}})
        .setModelInput("model_input");
    const auto plan = builder.build(std::move(registry));
    auto       operators = plan->createOperators();
    ASSERT_EQ(operators.size(), 1U);

    cv::Mat backing(3, 3, CV_8UC3, cv::Scalar(0, 0, 0));
    backing.at<cv::Vec3b>(1, 1) = cv::Vec3b(1, 2, 3);
    backing.at<cv::Vec3b>(1, 2) = cv::Vec3b(4, 5, 6);
    backing.at<cv::Vec3b>(2, 1) = cv::Vec3b(7, 8, 9);
    backing.at<cv::Vec3b>(2, 2) = cv::Vec3b(10, 11, 12);
    const cv::Mat roi = backing(cv::Rect(1, 1, 2, 2));
    ASSERT_FALSE(roi.isContinuous());

    std::vector<uint8_t> output(2U * 2U * 3U, 0U);
    TensorViewMap         tensors;
    tensors.emplace("host_image",
                    TensorView{output.data(),
                               TensorDesc{TensorDataType::U8, TensorLayout::HWC, MemoryKind::HOST, 2, 2, 3},
                               output.size(), 1, output.size(), "host_image"});
    ResultMap      results;
    TensorInputMap inputs;
    OperatorContext context{0, 1, 0, nullptr, &roi, tensors, results, &inputs, nullptr, 0};

    ASSERT_NO_THROW(operators.front()->executeWithContract(context, plan->nodes().front().contract));
    const std::vector<uint8_t> expected{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    EXPECT_EQ(output, expected);
}

TEST(LegacyPipelineTest, MaterializesCenterCropStage)
{
    irt::engine::EngineConfig config = makeConfig();
    config.preprocess.backend       = irt::PreprocessBackend::CUDA;
    config.preprocess.source_width  = 4;
    config.preprocess.source_height = 2;
    config.preprocess.input_width   = 2;
    config.preprocess.input_height  = 2;
    config.preprocess.padding_mode  = irt::PaddingMode::CenterCrop;

    const auto plan = irt::engine::priv::CreateLegacyPipelineForTest(config);
    ASSERT_NE(plan, nullptr);

    const auto crop_node = std::find_if(plan->nodes().begin(), plan->nodes().end(),
                                        [](const irt::engine::PipelinePlan::Node &node) {
                                            return node.type == "cuda.center_crop";
                                        });
    ASSERT_NE(crop_node, plan->nodes().end());
    ASSERT_EQ(crop_node->config.inputs.size(), 1U);
    ASSERT_EQ(crop_node->config.outputs.size(), 1U);
    EXPECT_EQ(crop_node->config.inputs.front(), "resized_source");
    EXPECT_EQ(crop_node->config.outputs.front(), "cropped_source");

    const auto &tensors = plan->tensors();
    ASSERT_TRUE(tensors.contains("resized_source"));
    ASSERT_TRUE(tensors.contains("cropped_source"));
    EXPECT_EQ(tensors.at("resized_source").width(), 4);
    EXPECT_EQ(tensors.at("resized_source").height(), 2);
    EXPECT_EQ(tensors.at("cropped_source").width(), 2);
    EXPECT_EQ(tensors.at("cropped_source").height(), 2);
}

TEST(BuiltinOperatorTest, CenterCropCopiesCenteredPixels)
{
    irt::engine::EngineConfig config = makeConfig();
    config.preprocess.backend        = irt::PreprocessBackend::CUDA;
    config.preprocess.source_width   = 4;
    config.preprocess.source_height  = 2;
    config.preprocess.input_width    = 2;
    config.preprocess.input_height   = 2;
    config.preprocess.padding_mode   = irt::PaddingMode::CenterCrop;

    const auto plan = irt::engine::priv::CreateLegacyPipelineForTest(config);
    ASSERT_NE(plan, nullptr);
    const auto crop_node = std::find_if(plan->nodes().begin(), plan->nodes().end(),
                                        [](const irt::engine::PipelinePlan::Node &node) {
                                            return node.type == "cuda.center_crop";
                                        });
    ASSERT_NE(crop_node, plan->nodes().end());
    const auto node_index = static_cast<size_t>(std::distance(plan->nodes().begin(), crop_node));
    auto       operators  = plan->createOperators();
    ASSERT_EQ(operators.size(), plan->nodes().size());

    const auto &source_desc      = plan->tensors().at("resized_source");
    const auto &destination_desc = plan->tensors().at("cropped_source");
    std::vector<uint8_t> source(source_desc.byteSize());
    for (size_t index = 0; index < source.size(); ++index)
    {
        source[index] = static_cast<uint8_t>(index);
    }
    std::vector<uint8_t> expected{3U, 4U, 5U, 6U, 7U, 8U, 15U, 16U, 17U, 18U, 19U, 20U};
    std::vector<uint8_t> output(destination_desc.byteSize(), 0U);

    uint8_t *d_source = nullptr;
    uint8_t *d_output = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_source), source.size()), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&d_output), output.size()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_source, source.data(), source.size(), cudaMemcpyHostToDevice), cudaSuccess);

    irt::engine::TensorViewMap tensors;
    tensors.emplace("resized_source",
                    irt::TensorView{d_source, source_desc, source.size(), 1, source.size(), "resized_source"});
    tensors.emplace("cropped_source", irt::TensorView{d_output, destination_desc, output.size(), 1, output.size(),
                                                       "cropped_source"});
    irt::engine::ResultMap      results;
    irt::engine::TensorInputMap inputs;
    irt::engine::OperatorContext context{0, 1, 0, nullptr, nullptr, tensors, results, &inputs, nullptr, 0};

    ASSERT_NO_THROW(operators[node_index]->executeWithContract(context, plan->nodes()[node_index].contract));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(output.data(), d_output, output.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(output, expected);

    cudaFree(d_source);
    cudaFree(d_output);
}

TEST(LegacyPipelineTest, ConvertsChannelsBeforeLetterbox)
{
    irt::engine::EngineConfig config = makeConfig();
    config.preprocess.backend        = irt::PreprocessBackend::CUDA;
    config.preprocess.source_width   = 4;
    config.preprocess.source_height  = 2;
    config.preprocess.input_width    = 2;
    config.preprocess.input_height   = 2;
    config.preprocess.input_channels = 3;
    config.preprocess.source_channels = 4;
    config.preprocess.src_color       = irt::ColorFormat::BGRA;
    config.preprocess.dst_color       = irt::ColorFormat::RGB;
    config.preprocess.padding_mode    = irt::PaddingMode::Letterbox;

    const auto plan = irt::engine::priv::CreateLegacyPipelineForTest(config);
    ASSERT_NE(plan, nullptr);

    const auto color_node = std::find_if(plan->nodes().begin(), plan->nodes().end(),
                                         [](const irt::engine::PipelinePlan::Node &node) {
                                             return node.type == "cvcuda.cvt_color";
                                         });
    const auto letterbox_node = std::find_if(plan->nodes().begin(), plan->nodes().end(),
                                             [](const irt::engine::PipelinePlan::Node &node) {
                                                 return node.type == "cvcuda.letterbox";
                                             });
    ASSERT_NE(color_node, plan->nodes().end());
    ASSERT_NE(letterbox_node, plan->nodes().end());
    ASSERT_EQ(color_node->config.outputs.front(), "converted_source");
    ASSERT_EQ(letterbox_node->config.inputs.front(), "converted_source");

    const auto &tensors = plan->tensors();
    ASSERT_TRUE(tensors.contains("converted_source"));
    EXPECT_EQ(tensors.at("converted_source").width(), 4);
    EXPECT_EQ(tensors.at("converted_source").height(), 2);
    EXPECT_EQ(tensors.at("converted_source").channels(), 3);

    const auto &letterbox_options = std::any_cast<const irt::engine::LetterBoxOptions &>(letterbox_node->config.parameters);
    EXPECT_EQ(letterbox_options.preprocess.src_color, irt::ColorFormat::RGB);
    EXPECT_EQ(letterbox_options.preprocess.source_channels, 3);
}

TEST(PipelinePlanTest, BuildsImmutableCodeDefinedDag)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);
    ASSERT_TRUE(registry->registerCreator("test.post", TestPostOperator{}.contract(),
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
            {{}, {"host_input"}, CpuImageToTensorOptions{makePreprocessSpec()}})
        .addNode("cuda.upload", {{"host_input"}, {"model_input"}, {}})
        .addNode("test.post", {{"model.output"}, {}, {}})
        .addResult("model_input", "preprocessed")
        .setModelInput("model_input");

    const auto plan = builder.build(registry);
    EXPECT_TRUE(registry->frozen());
    EXPECT_FALSE(registry->registerCreator("test.other",
                                           OperatorContract{ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 0, 0, false, true, 0},
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
                     CpuImageToTensorOptions{makePreprocessSpec(224, 224, {0.0F, 0.0F, 0.0F},
                                                                  {1.0F, 1.0F, 1.0F})}
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
        "test.result_post", TestResultPostOperator{}.contract(),
        [](const NodeConfig &) { return std::make_unique<TestResultPostOperator>(); }));
    ASSERT_TRUE(registry->registerCreator(
        "test.cuda_result", TestCudaResultOperator{}.contract(),
        [](const NodeConfig &) { return std::make_unique<TestCudaResultOperator>(); }));

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
            {{}, {"host_input"}, CpuImageToTensorOptions{makePreprocessSpec()}})
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

TEST(PipelinePlanTest, RejectsUnknownOperatorType)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);
    PipelineBuilder builder;
    builder.addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addNode("non_existent_op", {{}, {"model_input"}, {}})
        .setModelInput("model_input");

    try
    {
        [[maybe_unused]] auto plan = builder.build(registry);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}

TEST(PipelinePlanTest, RejectsOperatorWithNullCreatorResult)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    OperatorContract contract{ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 0, 0, false, true, 0};
    ASSERT_TRUE(registry->registerCreator("null.op", contract, [](const NodeConfig &) -> std::unique_ptr<IOperator> { return nullptr; }));
    PipelineBuilder builder;
    builder.addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addNode("null.op", {{}, {}, {}})
        .setModelInput("model_input");

    auto plan = builder.build(registry);
    try
    {
        [[maybe_unused]] auto ops = plan->createOperators();
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INTERNAL);
    }
}

TEST(PipelinePlanTest, RejectsOperatorWithThrowingCreator)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    OperatorContract contract{ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 0, 0, false, true, 0};
    ASSERT_TRUE(registry->registerCreator("throw.op", contract, [](const NodeConfig &) -> std::unique_ptr<IOperator> {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Custom creator failed");
    }));
    PipelineBuilder builder;
    builder.addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addNode("throw.op", {{}, {}, {}})
        .setModelInput("model_input");

    auto plan = builder.build(registry);
    try
    {
        [[maybe_unused]] auto ops = plan->createOperators();
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}

TEST(PipelinePlanTest, RejectsNodeWithIncompatibleInputCount)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);
    PipelineBuilder builder;
    builder.addTensor("host_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addTensor("extra_host", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addNode("cuda.upload", {{"host_input", "extra_host"}, {"model_input"}, {}})
        .setModelInput("model_input");

    try
    {
        [[maybe_unused]] auto plan = builder.build(registry);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}

TEST(PipelinePlanTest, RejectsNodeWithInvalidParametersType)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);
    PipelineBuilder builder;
    builder.addTensor("host_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addNode("cpu.image_to_tensor", {{}, {"host_input"}, std::string("wrong_parameter_type")})
        .addNode("cuda.upload", {{"host_input"}, {"model_input"}, {}})
        .setModelInput("model_input");

    try
    {
        [[maybe_unused]] auto plan = builder.build(registry);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}

TEST(PipelinePlanTest, RejectsAliasingForNonInPlaceOperatorBeforeCreator)
{
    using namespace irt::engine;

    bool creator_called = false;
    auto registry       = std::make_shared<OperatorRegistry>();
    const OperatorContract contract{ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 1, 1, false, true, 0};
    ASSERT_TRUE(registry->registerCreator(
        "test.alias", contract, [&](const NodeConfig &) {
            creator_called = true;
            return std::make_unique<TestPostOperator>();
        }));

    PipelineBuilder builder;
    builder.addTensor("host", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 2, 2, 3})
        .addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 2, 2, 3})
        .addNode("test.alias", {{"host"}, {"host"}, {}})
        .setModelInput("model_input");

    EXPECT_THROW(builder.build(registry), irt::Exception);
    EXPECT_FALSE(creator_called);
}

TEST(PipelinePlanTest, RejectsCreatorContractMismatch)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    const OperatorContract declared{ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 0, 0, false, true, 0};
    ASSERT_TRUE(registry->registerCreator(
        "test.mismatch", declared, [](const NodeConfig &) { return std::make_unique<WrongContractOperator>(); }));

    PipelineBuilder builder;
    builder.addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 2, 2, 3})
        .addNode("test.mismatch", {{}, {}, {}})
        .setModelInput("model_input");
    const auto plan = builder.build(registry);

    EXPECT_THROW(plan->createOperators(), irt::Exception);
}

TEST(PipelinePlanTest, MaterializesOperatorScratchWorkspace)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    ASSERT_TRUE(registry->registerCreator(
        "test.scratch", ScratchOperator{}.contract(), [](const NodeConfig &) { return std::make_unique<ScratchOperator>(); }));

    PipelineBuilder builder;
    builder.addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 2, 2, 3})
        .addNode("test.scratch", {{}, {}, {}})
        .setModelInput("model_input");
    const auto plan = builder.build(registry);
    auto       operators = plan->createOperators();
    ASSERT_EQ(operators.size(), 1U);
    ASSERT_NE(operators.front()->scratchData(), nullptr);
    EXPECT_EQ(operators.front()->scratchBytes(), 64U);

    TensorViewMap tensors;
    ResultMap     results;
    TensorInputMap inputs;
    cv::Mat image;
    OperatorContext context{0, 1, 0, nullptr, &image, tensors, results, &inputs,
                            operators.front()->scratchData(), operators.front()->scratchBytes()};
    EXPECT_NO_THROW(operators.front()->executeWithContract(context, plan->nodes().front().contract));
}

TEST(PipelinePlanTest, RejectsGraphCycle)
{
    using namespace irt::engine;

    class TestLoopOp final : public IOperator
    {
    public:
        OperatorContract contract() const override
        {
            return {ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 1, 1, false, false, 0};
        }
        void execute(OperatorContext &) override {}
    };

    auto registry = std::make_shared<OperatorRegistry>();
    ASSERT_TRUE(registry->registerCreator("test.loop", TestLoopOp{}.contract(), [](const NodeConfig &) { return std::make_unique<TestLoopOp>(); }));

    PipelineBuilder builder;
    builder.addTensor("t1", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addTensor("t2", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addTensor("model_input", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addNode("test.loop", {{"t1"}, {"t2"}, {}})
        .addNode("test.loop", {{"t2"}, {"t1"}, {}})
        .setModelInput("model_input");

    try
    {
        [[maybe_unused]] auto plan = builder.build(registry);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}

TEST(InferenceEngineLifecycleTest, CannotStartStoppedEngine)
{
    using namespace irt::engine;

    auto engine = InferenceEngine::create(makeConfig());
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->state(), EngineState::Created);
    engine->shutdown();
    EXPECT_EQ(engine->state(), EngineState::Stopped);

    try
    {
        engine->start();
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::INVALID_OPERATION);
    }
}

TEST(InferenceEngineLifecycleTest, ConcurrentShutdownIsIdempotent)
{
    using namespace irt::engine;

    auto engine = InferenceEngine::create(makeConfig());
    ASSERT_NE(engine, nullptr);

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i)
    {
        threads.emplace_back([&] { engine->shutdown(); });
    }
    for (auto &t : threads)
    {
        t.join();
    }
    EXPECT_EQ(engine->state(), EngineState::Stopped);
}

TEST(InferenceEngineLifecycleTest, CanRecreateEngineAfterFailedStart)
{
    using namespace irt::engine;

    auto config = makeConfig();
    config.engine_file = "non_existent_engine.engine";

    auto engine1 = InferenceEngine::create(config);
    ASSERT_NE(engine1, nullptr);
    EXPECT_THROW(engine1->start(), irt::Exception);
    EXPECT_EQ(engine1->state(), EngineState::Failed);

    const auto faults = engine1->faults();
    EXPECT_FALSE(faults.empty());
    EXPECT_EQ(faults.front().stage, FaultStage::Start);

    auto engine2 = InferenceEngine::create(config);
    ASSERT_NE(engine2, nullptr);
    EXPECT_EQ(engine2->state(), EngineState::Created);
}

TEST(TensorViewTest, RespectsCapacityBounds)
{
    using namespace irt::engine;

    float buffer[64];
    TensorDesc desc{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 4, 4, 1};
    TensorView view{buffer, desc, 4 * sizeof(float), 4};

    EXPECT_NE(view.dataForRequest(0), nullptr);
    EXPECT_NE(view.dataForRequest(3), nullptr);
    EXPECT_EQ(view.dataForRequest(4), nullptr);
    EXPECT_EQ(view.dataForRequest(-1), nullptr);
    EXPECT_EQ(view.dataForRequest(100), nullptr);
}

TEST(TensorDescTest, ChecksMultiplicationOverflowAndNegativeDimensions)
{
    using namespace irt::engine;

    TensorDesc bad_width{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, -1, 224, 3};
    EXPECT_THROW(bad_width.bytesPerRequest(), irt::Exception);

    TensorDesc huge_desc{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 100000000, 100000000, 100000000};
    EXPECT_THROW(huge_desc.bytesPerRequest(), irt::Exception);
}

TEST(PipelinePlanTest, CreatorIsNotCalledWhenOperatorTypeIsUnknown)
{
    using namespace irt::engine;

    bool custom_creator_called = false;
    auto registry = std::make_shared<OperatorRegistry>();
    OperatorContract contract{ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 1, 1, false, true, 0};
    ASSERT_TRUE(registry->registerCreator("test.valid_op", contract, [&](const NodeConfig &) -> std::unique_ptr<IOperator> {
        custom_creator_called = true;
        return nullptr;
    }));

    PipelineBuilder builder;
    builder.addTensor("host_in", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addTensor("model_in", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addNode("test.unknown_op", {{}, {"host_in"}, {}})
        .addNode("test.valid_op", {{"host_in"}, {"model_in"}, {}})
        .setModelInput("model_in");

    EXPECT_THROW(builder.build(registry), irt::Exception);
    EXPECT_FALSE(custom_creator_called);
}

TEST(PipelinePlanTest, CreatorIsNotCalledWhenInputCountIsInvalid)
{
    using namespace irt::engine;

    bool creator_called = false;
    auto registry = std::make_shared<OperatorRegistry>();
    OperatorContract contract{ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 1, 1, false, true, 0};
    ASSERT_TRUE(registry->registerCreator("test.unary_op", contract, [&](const NodeConfig &) -> std::unique_ptr<IOperator> {
        creator_called = true;
        return nullptr;
    }));

    PipelineBuilder builder;
    builder.addTensor("in1", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addTensor("in2", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addTensor("out", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addNode("test.unary_op", {{"in1", "in2"}, {"out"}, {}})
        .setModelInput("out");

    EXPECT_THROW(builder.build(registry), irt::Exception);
    EXPECT_FALSE(creator_called);
}

TEST(PipelinePlanTest, CreatorIsNotCalledWhenDAGHasCycles)
{
    using namespace irt::engine;

    bool creator_called = false;
    auto registry = std::make_shared<OperatorRegistry>();
    OperatorContract contract{ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 1, 1, false, true, 0};
    ASSERT_TRUE(registry->registerCreator("test.step_op", contract, [&](const NodeConfig &) -> std::unique_ptr<IOperator> {
        creator_called = true;
        return nullptr;
    }));

    PipelineBuilder builder;
    builder.addTensor("t1", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addTensor("t2", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addNode("test.step_op", {{"t1"}, {"t2"}, {}})
        .addNode("test.step_op", {{"t2"}, {"t1"}, {}})
        .setModelInput("t2");

    EXPECT_THROW(builder.build(registry), irt::Exception);
    EXPECT_FALSE(creator_called);
}

TEST(TensorViewTest, RejectsZeroAndNegativeCapacity)
{
    using namespace irt::engine;

    float buffer[64];
    TensorDesc desc{TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 4, 4, 1};
    TensorView zero_cap_view{buffer, desc, 4 * sizeof(float), 0};
    EXPECT_EQ(zero_cap_view.dataForRequest(0), nullptr);

    TensorView neg_cap_view{buffer, desc, 4 * sizeof(float), -2};
    EXPECT_EQ(neg_cap_view.dataForRequest(0), nullptr);
}

TEST(EngineLifecycleTest, EngineStartFailureRollsBackCleanlyAndLeavesEngineInFailedState)
{
    using namespace irt::engine;

    EngineConfig config = makeConfig();
    config.engine_file = "non_existent_model_for_failure_test.engine";
    config.execution_slots = 1;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);

    PipelineBuilder builder;
    builder.addTensor("in", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addTensor("model_in", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addNode("cuda.upload", {{"in"}, {"model_in"}, {}})
        .setModelInput("model_in");
    auto plan = builder.build(registry);

    auto engine = InferenceEngine::create(config, plan);
    EXPECT_EQ(engine->state(), EngineState::Created);

    EXPECT_THROW(engine->start(), irt::Exception);
    EXPECT_EQ(engine->state(), EngineState::Failed);

    // Starting a failed engine throws
    EXPECT_THROW(engine->start(), irt::Exception);

    // Submitting to a failed engine throws
    cv::Mat dummy(224, 224, CV_8UC3, cv::Scalar(128, 128, 128));
    EXPECT_THROW(engine->submit(dummy), irt::Exception);

    // Metrics are cleanly 0
    auto metrics = engine->metrics();
    EXPECT_EQ(metrics.queued_requests, 0U);
    EXPECT_EQ(metrics.accepted_requests, 0U);
}

TEST(EngineLifecycleTest, ConcurrentShutdownStress)
{
    using namespace irt::engine;

    EngineConfig config = makeConfig();
    config.engine_file = "non_existent_model_for_stress_test.engine";
    config.execution_slots = 1;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);

    PipelineBuilder builder;
    builder.addTensor("in", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addTensor("model_in", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addNode("cuda.upload", {{"in"}, {"model_in"}, {}})
        .setModelInput("model_in");
    auto plan = builder.build(registry);

    auto engine = InferenceEngine::create(config, plan);

    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i)
    {
        threads.emplace_back([&engine] {
            engine->shutdown();
        });
    }
    for (auto &t : threads)
    {
        t.join();
    }
    EXPECT_EQ(engine->state(), EngineState::Stopped);
}

TEST(EngineLifecycleTest, StartFailurePinnedMemoryLimitRollback)
{
    using namespace irt::engine;

    const auto engine_file = irt::engine::test::buildIdentityEngine(false);

    EngineConfig config;
    config.model_name = "test_memory_engine";
    config.engine_file = engine_file;
    config.preprocess.input_width = 2;
    config.preprocess.input_height = 2;
    config.min_batch_size = 1;
    config.opt_batch_size = 1;
    config.max_batch_size = 1;
    config.execution_slots = 1;
    config.pinned_memory_limit_bytes = 1; // 极小限制导致 start 阶段票据分配直接抛出异常

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);

    PipelineBuilder builder;
    builder.addTensor("in", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 2, 2, 3})
        .addTensor("image", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 2, 2, 3})
        .addNode("cuda.upload", {{"in"}, {"image"}, {}})
        .setModelInput("image");
    auto plan = builder.build(registry);

    auto engine = InferenceEngine::create(config, plan);
    EXPECT_EQ(engine->state(), EngineState::Created);

    EXPECT_THROW(engine->start(), irt::Exception);
    EXPECT_EQ(engine->state(), EngineState::Failed);
    EXPECT_THROW(engine->start(), irt::Exception);

    // 验证销毁 engine 时无任何资源泄漏和崩溃
    engine.reset();
}

TEST(PipelinePlanTest, RejectsDuplicateResultName)
{
    using namespace irt::engine;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);

    PipelineBuilder builder;
    builder.addTensor("host_in", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 224, 224, 3})
        .addTensor("model_in", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 224, 224, 3})
        .addNode("cuda.upload", {{"host_in"}, {"model_in"}, {}})
        .setModelInput("model_in")
        .addResult("res1", "host_in");

    // 重复添加同名 result
    EXPECT_THROW(builder.addResult("res1", "host_in"), irt::Exception);
}

class EngineStartFaultInjectionTest : public ::testing::TestWithParam<int>
{
};

INSTANTIATE_TEST_SUITE_P(AllStartFaultStages, EngineStartFaultInjectionTest, ::testing::Range(1, 9));

TEST_P(EngineStartFaultInjectionTest, CleanRollbackAndInvariantsOnInjectedFailure)
{
    using namespace irt::engine;

    const int fault_stage = GetParam();
    const auto engine_file = irt::engine::test::buildIdentityEngine(false);

    EngineConfig config;
    config.model_name = "test_fault_engine";
    config.engine_file = engine_file;
    config.preprocess.input_width = 2;
    config.preprocess.input_height = 2;
    config.min_batch_size = 1;
    config.opt_batch_size = 2;
    config.max_batch_size = 4;
    config.execution_slots = 2;
    config.cpu_preprocess_workers = 2;
    config.cpu_postprocess_workers = 2;
    irt::engine::priv::EngineTestOptions test_options;
    test_options.start_fault_stage = fault_stage;
    irt::engine::priv::EngineTestOptionsGuard test_options_guard(test_options);

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);

    PipelineBuilder builder;
    builder.addTensor("in", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 2, 2, 3})
        .addTensor("image", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 2, 2, 3})
        .addNode("cuda.upload", {{"in"}, {"image"}, {}})
        .setModelInput("image");
    auto plan = builder.build(registry);

    auto engine = InferenceEngine::create(config, plan);
    ASSERT_NE(engine, nullptr);
    EXPECT_EQ(engine->state(), EngineState::Created);

    // 启动必须抛出异常
    EXPECT_THROW(engine->start(), irt::Exception);

    // 引擎状态确定性变为 Failed
    EXPECT_EQ(engine->state(), EngineState::Failed);

    // 启动失败后再次 start 必须直接拒绝
    EXPECT_THROW(engine->start(), irt::Exception);

    // 启动失败后 submit 必须直接拒绝
    cv::Mat dummy(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    EXPECT_THROW(engine->submit(dummy), irt::Exception);

    // 故障记录中必须有 Start 阶段的 fatal 记录，并保留原始阶段与状态
    const auto faults = engine->faults();
    ASSERT_FALSE(faults.empty());
    EXPECT_EQ(faults.back().stage, FaultStage::Start);
    EXPECT_TRUE(faults.back().fatal);
    EXPECT_EQ(faults.back().status, irt::Status::ERROR_INTERNAL);

    // 验证资源不变量：pending request 为 0
    EXPECT_EQ(engine->pendingRequests(), 0U);
    const auto metrics = engine->metrics();
    EXPECT_EQ(metrics.queued_requests, 0U);
    EXPECT_EQ(metrics.inflight_batches, 0U);
    EXPECT_EQ(metrics.pinned_input_in_use, 0U);
    EXPECT_EQ(metrics.pinned_output_in_use, 0U);
    EXPECT_EQ(metrics.pinned_memory_capacity_bytes, 0U);
    EXPECT_EQ(metrics.device_arena_bytes, 0U);
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

    // 析构与回收：销毁后不产生任何悬挂线程或崩溃
    engine.reset();
}

class EngineStartCreationFaultTest : public ::testing::TestWithParam<int>
{
};

INSTANTIATE_TEST_SUITE_P(AllStartCreationPoints, EngineStartCreationFaultTest, ::testing::Range(1, 9));

TEST_P(EngineStartCreationFaultTest, RollsBackAfterEachResourceCreationPoint)
{
    using namespace irt::engine;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }

    const auto engine_file = irt::engine::test::buildIdentityEngine(false);

    EngineConfig config;
    config.model_name              = "test_creation_fault_engine";
    config.engine_file             = engine_file;
    config.preprocess.input_width  = 2;
    config.preprocess.input_height = 2;
    config.min_batch_size          = 1;
    config.opt_batch_size          = 1;
    config.max_batch_size          = 1;
    config.execution_slots         = 2;
    config.cpu_preprocess_workers  = 2;
    config.cpu_postprocess_workers = 2;

    irt::engine::priv::EngineTestOptions options;
    options.start_fault_point = static_cast<irt::engine::priv::StartFaultPoint>(GetParam());
    options.start_fault_index = 0;
    irt::engine::priv::EngineTestOptionsGuard options_guard(options);

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);
    PipelineBuilder builder;
    builder.addTensor("in", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 2, 2, 3})
        .addTensor("image", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 2, 2, 3})
        .addNode("cuda.upload", {{"in"}, {"image"}, {}})
        .setModelInput("image");

    auto engine = InferenceEngine::create(config, builder.build(std::move(registry)));
    ASSERT_NE(engine, nullptr);
    EXPECT_THROW(engine->start(), irt::Exception);
    EXPECT_EQ(engine->state(), EngineState::Failed);

    const auto faults = engine->faults();
    ASSERT_FALSE(faults.empty());
    EXPECT_EQ(faults.back().stage, FaultStage::Start);
    EXPECT_EQ(faults.back().status, irt::Status::ERROR_INTERNAL);
    EXPECT_NE(faults.back().message.find("resource creation"), std::string::npos);

    const auto metrics = engine->metrics();
    EXPECT_EQ(metrics.active_requests, 0U);
    EXPECT_EQ(metrics.slot_count, 0U);
    EXPECT_EQ(metrics.idle_slot_count, 0U);
    EXPECT_EQ(metrics.active_slot_count, 0U);
    EXPECT_EQ(metrics.prepare_queue_size, 0U);
    EXPECT_EQ(metrics.gpu_queue_size, 0U);
    EXPECT_EQ(metrics.postprocess_queue_size, 0U);
    EXPECT_EQ(metrics.input_ticket_count, 0U);
    EXPECT_EQ(metrics.output_ticket_count, 0U);
    EXPECT_EQ(metrics.pinned_input_in_use, 0U);
    EXPECT_EQ(metrics.pinned_output_in_use, 0U);
    EXPECT_EQ(metrics.pinned_memory_capacity_bytes, 0U);
    EXPECT_EQ(metrics.device_arena_bytes, 0U);
    EXPECT_EQ(metrics.thread_count, 0U);
    EXPECT_EQ(metrics.joinable_thread_count, 0U);
    EXPECT_EQ(metrics.accepted_requests,
              metrics.completed_requests + metrics.failed_requests + metrics.cancelled_requests
                  + metrics.timed_out_requests + metrics.dropped_requests);

    EXPECT_NO_THROW(engine->shutdown());
    EXPECT_NO_THROW(engine->shutdown());
}

TEST(EngineStartCreationFaultTest, RollsBackAfterASecondSlotOrWorkerWasCreated)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }

    const auto engine_file = irt::engine::test::buildIdentityEngine(false);
    irt::engine::EngineConfig config;
    config.model_name              = "test_second_creation_fault_engine";
    config.engine_file             = engine_file;
    config.preprocess.input_width  = 2;
    config.preprocess.input_height = 2;
    config.min_batch_size          = 1;
    config.opt_batch_size          = 1;
    config.max_batch_size          = 1;
    config.execution_slots         = 2;
    config.cpu_preprocess_workers  = 2;
    config.cpu_postprocess_workers = 2;

    auto registry = std::make_shared<irt::engine::OperatorRegistry>();
    irt::engine::registerBuiltinOperators(*registry);
    irt::engine::PipelineBuilder builder;
    builder.addTensor("in", {irt::engine::TensorDataType::F32, irt::engine::TensorLayout::NCHW,
                               irt::engine::MemoryKind::HOST, 2, 2, 3})
        .addTensor("image", {irt::engine::TensorDataType::F32, irt::engine::TensorLayout::NCHW,
                               irt::engine::MemoryKind::DEVICE, 2, 2, 3})
        .addNode("cuda.upload", {{"in"}, {"image"}, {}})
        .setModelInput("image");
    const auto pipeline = builder.build(std::move(registry));

    for (const auto point : {irt::engine::priv::StartFaultPoint::Slot,
                             irt::engine::priv::StartFaultPoint::PrepareWorker,
                             irt::engine::priv::StartFaultPoint::PostprocessWorker})
    {
        irt::engine::priv::EngineTestOptions options;
        options.start_fault_point = point;
        options.start_fault_index = 1;
        irt::engine::priv::EngineTestOptionsGuard guard(options);

        auto engine = irt::engine::InferenceEngine::create(config, pipeline);
        ASSERT_NE(engine, nullptr);
        EXPECT_THROW(engine->start(), irt::Exception);
        EXPECT_EQ(engine->state(), irt::engine::EngineState::Failed);

        const auto metrics = engine->metrics();
        EXPECT_EQ(metrics.slot_count, 0U);
        EXPECT_EQ(metrics.input_ticket_count, 0U);
        EXPECT_EQ(metrics.output_ticket_count, 0U);
        EXPECT_EQ(metrics.thread_count, 0U);
        EXPECT_EQ(metrics.joinable_thread_count, 0U);
        EXPECT_NO_THROW(engine->shutdown());
    }
}

TEST(EngineLifecycleTest, FailedStartDoesNotPoisonSubsequentEngine)
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
    {
        GTEST_SKIP() << "CUDA GPU is unavailable";
    }

    const auto engine_file = irt::engine::test::buildIdentityEngine(false);
    auto       config      = makeConfig();
    config.model_name      = "failed_start_recovery";
    config.engine_file     = engine_file;
    config.preprocess.input_width  = 2;
    config.preprocess.input_height = 2;
    config.max_batch_size  = 1;
    config.opt_batch_size  = 1;
    config.execution_slots = 1;

    auto registry = std::make_shared<irt::engine::OperatorRegistry>();
    irt::engine::registerBuiltinOperators(*registry);
    irt::engine::PipelineBuilder builder;
    builder.addTensor("host", {irt::engine::TensorDataType::F32, irt::engine::TensorLayout::NCHW,
                                irt::engine::MemoryKind::HOST, 2, 2, 3})
        .addTensor("device", {irt::engine::TensorDataType::F32, irt::engine::TensorLayout::NCHW,
                               irt::engine::MemoryKind::DEVICE, 2, 2, 3})
        .addNode("cpu.image_to_tensor", {{}, {"host"},
                                          irt::engine::CpuImageToTensorOptions{config.preprocess}})
        .addNode("cuda.upload", {{"host"}, {"device"}, {}})
        .setModelInput("device");
    auto pipeline = builder.build(std::move(registry));

    {
        irt::engine::priv::EngineTestOptions options;
        options.start_fault_stage = 8;
        irt::engine::priv::EngineTestOptionsGuard guard(options);
        auto failed_engine = irt::engine::InferenceEngine::create(config, pipeline);
        EXPECT_THROW(failed_engine->start(), irt::Exception);
        EXPECT_EQ(failed_engine->state(), irt::engine::EngineState::Failed);
        const auto metrics = failed_engine->metrics();
        EXPECT_EQ(metrics.thread_count, 0U);
        EXPECT_EQ(metrics.joinable_thread_count, 0U);
        EXPECT_EQ(metrics.slot_count, 0U);
        EXPECT_EQ(metrics.input_ticket_count, 0U);
        EXPECT_EQ(metrics.output_ticket_count, 0U);
        failed_engine->shutdown();
        failed_engine->shutdown();
    }

    auto recovered_engine = irt::engine::InferenceEngine::create(config, pipeline);
    ASSERT_NO_THROW(recovered_engine->start());
    auto future = recovered_engine->submit(cv::Mat(2, 2, CV_8UC3, cv::Scalar(9, 8, 7)));
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_TRUE(future.get().outputs.contains("output"));

    std::thread first_shutdown([&] { recovered_engine->shutdown(); });
    std::thread second_shutdown([&] { recovered_engine->shutdown(); });
    first_shutdown.join();
    second_shutdown.join();
    EXPECT_EQ(recovered_engine->state(), irt::engine::EngineState::Stopped);
    const auto metrics = recovered_engine->metrics();
    EXPECT_EQ(metrics.thread_count, 0U);
    EXPECT_EQ(metrics.joinable_thread_count, 0U);
    EXPECT_EQ(metrics.slot_count, 0U);
    EXPECT_EQ(metrics.input_ticket_count, 0U);
    EXPECT_EQ(metrics.output_ticket_count, 0U);
    EXPECT_EQ(metrics.active_requests, 0U);
}

TEST(EnginePromiseTest, ExactlyOnceFulfillmentUnderConcurrentCompletion)
{
    using namespace irt::engine;

    // 验证多线程竞争 fulfillCompletions 时 promise 只会被 set 一次且不发生 std::future_error
    std::promise<InferenceResult> promise;
    auto future = promise.get_future();

    std::atomic<bool> fulfilled{false};
    std::atomic<int> success_count{0};
    std::atomic<int> exception_count{0};

    auto try_fulfill = [&]() {
        bool expected = false;
        if (fulfilled.compare_exchange_strong(expected, true))
        {
            try
            {
                InferenceResult res;
                res.request_id = 42;
                promise.set_value(std::move(res));
                ++success_count;
            }
            catch (...)
            {
                ++exception_count;
            }
        }
    };

    std::vector<std::thread> workers;
    for (int i = 0; i < 16; ++i)
    {
        workers.emplace_back(try_fulfill);
    }
    for (auto &t : workers)
    {
        t.join();
    }

    EXPECT_EQ(success_count.load(), 1);
    EXPECT_EQ(exception_count.load(), 0);
    ASSERT_TRUE(future.valid());
    auto result = future.get();
    EXPECT_EQ(result.request_id, 42U);
}

TEST(EngineLifecycleTest, RealEngineQueueDropAndCancellationPromiseFulfillment)
{
    using namespace irt::engine;
    const auto engine_file = irt::engine::test::buildIdentityEngine(false);

    EngineConfig config;
    config.model_name = "test_queue_engine";
    config.engine_file = engine_file;
    config.preprocess.input_width = 2;
    config.preprocess.input_height = 2;
    config.min_batch_size = 1;
    config.opt_batch_size = 1;
    config.max_batch_size = 1;
    config.queue_capacity = 2;
    config.queue_policy = QueuePolicy::DropOldest;
    config.execution_slots = 1;

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);

    PipelineBuilder builder;
    builder.addTensor("in", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST, 2, 2, 3})
        .addTensor("image", {TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE, 2, 2, 3})
        .addNode("cuda.upload", {{"in"}, {"image"}, {}})
        .setModelInput("image");
    auto plan = builder.build(registry);

    auto engine = InferenceEngine::create(config, plan);
    ASSERT_NE(engine, nullptr);
    EXPECT_NO_THROW(engine->start());
    EXPECT_EQ(engine->state(), EngineState::Running);

    cv::Mat dummy(2, 2, CV_8UC3, cv::Scalar(128, 128, 128));

    // 连续提交请求触发队列调度
    auto h1 = engine->submit(dummy, RequestOptions{});
    auto h2 = engine->submit(dummy, RequestOptions{});
    auto h3 = engine->submit(dummy, RequestOptions{});
    auto h4 = engine->submit(dummy, RequestOptions{});

    // 取消其中一个请求
    (void)engine->cancel(h3.id());

    auto f1 = std::move(h1).takeFuture();
    auto f2 = std::move(h2).takeFuture();
    auto f3 = std::move(h3).takeFuture();
    auto f4 = std::move(h4).takeFuture();

    // 正常关闭引擎
    engine->shutdown();
    EXPECT_EQ(engine->state(), EngineState::Stopped);

    // 验证所有请求的 future 都已妥善收敛（正常完成或抛出异常），无挂起未完成的 promise
    size_t terminal_count = 0;
    auto check_future = [&terminal_count](std::future<InferenceResult> &f) {
        if (f.valid())
        {
            try
            {
                [[maybe_unused]] auto res = f.get();
                ++terminal_count;
            }
            catch (const irt::Exception &)
            {
                // 预期可能由于丢弃/取消/关闭抛出异常，符合设计
                ++terminal_count;
            }
        }
    };

    check_future(f1);
    check_future(f2);
    check_future(f3);
    check_future(f4);

    EXPECT_EQ(terminal_count, 4U);
    EXPECT_EQ(engine->pendingRequests(), 0U);
    const auto metrics = engine->metrics();
    EXPECT_EQ(metrics.queued_requests, 0U);
    EXPECT_EQ(metrics.inflight_batches, 0U);
    EXPECT_EQ(metrics.pinned_input_in_use, 0U);
    EXPECT_EQ(metrics.pinned_output_in_use, 0U);
    EXPECT_EQ(metrics.pinned_memory_capacity_bytes, 0U);
    EXPECT_EQ(metrics.device_arena_bytes, 0U);
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
    EXPECT_EQ(metrics.completed_requests + metrics.failed_requests + metrics.cancelled_requests
                  + metrics.timed_out_requests + metrics.dropped_requests,
              metrics.accepted_requests);
}

} // namespace
