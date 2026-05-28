#include "TestModelCommon.hpp"

#include <gtest/gtest.h>

#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <vector>

using test::model::ExpectIrtExceptionCode;
using test::model::MakeNullBuffers;

namespace {

std::filesystem::path ProjectRoot()
{
    auto source = std::filesystem::path(__FILE__);
    if (source.is_relative())
    {
        source = std::filesystem::current_path() / source;
    }
    return source.parent_path().parent_path().parent_path();
}

std::filesystem::path SampleResNet50Onnx()
{
    return ProjectRoot() / "samples" / "model" / "onnx" / "resnet50.onnx";
}

void RunOnnxGraphBackend(irt::model::ModelBackend backend, bool feature_only)
{
    const auto onnx_path = SampleResNet50Onnx();
    if (!std::filesystem::exists(onnx_path))
    {
        GTEST_SKIP() << "sample ONNX file not found: " << onnx_path.string();
    }

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setBackend(backend);
    config->setDevice(irt::model::ModelDevice::CPU);
    if (feature_only)
    {
        config->setFeatureOnly(true);
        config->setFeatureTensorNames({"output"});
        config->setOutputTensorNames({"output"});
    }

    auto model = irt::model::CreateModel("onnx", std::move(config));
    ASSERT_NE(model, nullptr);
    model->buildOrLoad(onnx_path.string());

    const auto input_names = model->ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
    const auto output_names = model->ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
    EXPECT_EQ(model->backend(), backend);
    EXPECT_EQ(model->device(), irt::model::ModelDevice::CPU);
    ASSERT_EQ(input_names.size(), 1U);
    ASSERT_EQ(output_names.size(), 1U);
    EXPECT_EQ(model->tensorDataType(input_names.front()), nvinfer1::DataType::kFLOAT);
    EXPECT_EQ(model->tensorDataType(output_names.front()), nvinfer1::DataType::kFLOAT);

    const auto input_dims = model->tensorShape(input_names.front());
    const auto output_dims = model->tensorShape(output_names.front());
    EXPECT_EQ(input_dims.nbDims, 4);
    EXPECT_EQ(input_dims.d[0], 1);
    EXPECT_EQ(input_dims.d[1], 3);
    EXPECT_EQ(input_dims.d[2], 224);
    EXPECT_EQ(input_dims.d[3], 224);
    EXPECT_EQ(output_dims.nbDims, 2);
    EXPECT_EQ(output_dims.d[0], 1);
    EXPECT_EQ(output_dims.d[1], 1000);

    std::vector<float> input(irt::model::elementCount(input_dims), 0.01F);
    std::vector<float> output(irt::model::elementCount(output_dims), 0.0F);
    std::vector<void *> buffers{input.data(), output.data()};

    if (feature_only)
    {
        model->forwardFeatures(buffers);
    }
    else
    {
        model->infer(buffers);
    }

    EXPECT_TRUE(std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value); }));
    EXPECT_TRUE(std::any_of(output.begin(), output.end(), [](float value) { return std::abs(value) > 1.0e-7F; }));
}

void ExpectInvalidOutputRejected(irt::model::ModelBackend backend)
{
    const auto onnx_path = SampleResNet50Onnx();
    if (!std::filesystem::exists(onnx_path))
    {
        GTEST_SKIP() << "sample ONNX file not found: " << onnx_path.string();
    }

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setBackend(backend);
    config->setDevice(irt::model::ModelDevice::CPU);
    config->setOutputTensorNames({"missing_output"});

    auto model = irt::model::CreateModel("onnx", std::move(config));
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->buildOrLoad(onnx_path.string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

} // namespace

/**
 * @brief ONNX 模型应使用 `.onnx` 作为权重扩展名。
 */
TEST(ONNXModelPropertiesTest, UsesOnnxWeightsExtension)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->name(), "ONNX");
    EXPECT_EQ(model->wtsExtension(), ".onnx");
    EXPECT_EQ(model->engineExtension(), ".engine");
}

/**
 * @brief ONNX 模型不支持通过手写网络方式构建，应返回非法操作错误。
 */
TEST(ONNXModelBuildNetworkTest, ManualBuildNetworkThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->buildNetwork(nullptr, irt::model::WeightsMap{}); },
                           irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief engine 未初始化时，查询输入张量名称应抛出非法操作异常。
 */
TEST(ONNXModelRuntimeQueryTest, IOTensorNamesWithoutEngineThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->ioTensorNames(nvinfer1::TensorIOMode::kINPUT); },
                           irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief engine 未初始化时，查询张量形状应抛出非法操作异常。
 */
TEST(ONNXModelRuntimeQueryTest, TensorShapeWithoutEngineThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->tensorShape("input"); }, irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief engine 未初始化时，查询张量数据类型应抛出非法操作异常。
 */
TEST(ONNXModelRuntimeQueryTest, TensorDataTypeWithoutEngineThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->tensorDataType("input"); }, irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief context 未初始化时，设置运行时输入形状应抛出非法操作异常。
 */
TEST(ONNXModelRuntimeQueryTest, SetTensorShapeWithoutContextThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    const nvinfer1::Dims4 dims(1, 3, 224, 224);
    ExpectIrtExceptionCode([&] { model->setTensorShape("input", dims); },
                           irt::Status::ERROR_INVALID_OPERATION);
}

TEST(ONNXRuntimeBackendTest, RuntimeQueriesBeforeLoadThrowInvalidOperation)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setBackend(irt::model::ModelBackend::ONNXRuntime);
    config->setDevice(irt::model::ModelDevice::CPU);

    auto model = irt::model::CreateModel("onnx", std::move(config));
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->ioTensorNames(nvinfer1::TensorIOMode::kINPUT); },
                           irt::Status::ERROR_INVALID_OPERATION);
    ExpectIrtExceptionCode([&] { model->infer(MakeNullBuffers(2)); },
                           irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief 构建不存在的 ONNX 文件时，应返回非法参数错误。
 */
TEST(ONNXModelLifecycleTest, BuildWithNonExistentOnnxThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.onnx"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

TEST(ONNXRuntimeBackendTest, BuildWithNonExistentOnnxThrowsInvalidArgument)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setBackend(irt::model::ModelBackend::ONNXRuntime);
    config->setDevice(irt::model::ModelDevice::CPU);

    auto model = irt::model::CreateModel("onnx", std::move(config));
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.onnx"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief buildOrLoad 在 ONNX 文件不存在时，也应返回非法参数错误。
 */
TEST(ONNXRuntimeBackendTest, BuildsAndRunsCpuInferenceFromSampleOnnx)
{
    RunOnnxGraphBackend(irt::model::ModelBackend::ONNXRuntime, false);
}

TEST(ONNXRuntimeBackendTest, ForwardFeaturesRunsConfiguredGraphOutput)
{
    RunOnnxGraphBackend(irt::model::ModelBackend::ONNXRuntime, true);
}

TEST(ONNXRuntimeBackendTest, RejectsOutputNameThatIsNotGraphOutput)
{
    ExpectInvalidOutputRejected(irt::model::ModelBackend::ONNXRuntime);
}

TEST(OpenVINOBackendTest, BuildsAndRunsCpuInferenceFromSampleOnnx)
{
    RunOnnxGraphBackend(irt::model::ModelBackend::OpenVINO, false);
}

TEST(OpenVINOBackendTest, ForwardFeaturesRunsConfiguredGraphOutput)
{
    RunOnnxGraphBackend(irt::model::ModelBackend::OpenVINO, true);
}

TEST(OpenVINOBackendTest, RejectsOutputNameThatIsNotGraphOutput)
{
    ExpectInvalidOutputRejected(irt::model::ModelBackend::OpenVINO);
}

TEST(ONNXModelLifecycleTest, BuildOrLoadWithNonExistentOnnxThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->buildOrLoad("/non/existent/path/model.onnx"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief ONNX 通道暂不支持内部特征层选择，build 应在进入文件解析前拒绝该配置。
 */
TEST(ONNXModelLifecycleTest, BuildWithFeatureTensorSelectionThrowsInvalidOperation)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"layer4"});

    auto model = irt::model::CreateModel("onnx", std::move(config));
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.onnx"); },
                           irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief ONNX 通道暂不支持 featureOnly 裁剪网络，build 应返回非法操作错误。
 */
TEST(ONNXModelLifecycleTest, BuildWithFeatureOnlyThrowsInvalidOperation)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"layer4"});
    config->setFeatureOnly(true);

    auto model = irt::model::CreateModel("onnx", std::move(config));
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.onnx"); },
                           irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief ONNX 通道在 buildOrLoad 路径中也应拒绝 featureOnly 配置。
 */
TEST(ONNXModelLifecycleTest, BuildOrLoadWithFeatureOnlyThrowsInvalidOperation)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"layer4"});
    config->setFeatureOnly(true);

    auto model = irt::model::CreateModel("onnx", std::move(config));
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->buildOrLoad("/non/existent/path/model.onnx"); },
                           irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief context 未初始化时执行 ONNX 推理，应抛出非法操作异常。
 */
TEST(ONNXModelInferTest, InferWithoutContextThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->infer(MakeNullBuffers(2)); },
                           irt::Status::ERROR_INVALID_OPERATION);
}
