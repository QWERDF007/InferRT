#include "TestModelCommon.hpp"

#include <gtest/gtest.h>

#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>

#include <memory>
#include <vector>

using test::model::ExpectIrtExceptionCode;
using test::model::MakeNullBuffers;

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

/**
 * @brief buildOrLoad 在 ONNX 文件不存在时，也应返回非法参数错误。
 */
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
