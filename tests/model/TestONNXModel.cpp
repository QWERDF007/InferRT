#include <gtest/gtest.h>

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>

#include <vector>

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

    EXPECT_THROW({ model->buildNetwork(nullptr, irt::model::WeightsMap{}); }, irt::Exception);

    try
    {
        model->buildNetwork(nullptr, irt::model::WeightsMap{});
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_OPERATION);
    }
}

/**
 * @brief engine 未初始化时，查询输入张量名称应抛出非法操作异常。
 */
TEST(ONNXModelRuntimeQueryTest, IOTensorNamesWithoutEngineThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    EXPECT_THROW({ model->ioTensorNames(nvinfer1::TensorIOMode::kINPUT); }, irt::Exception);
}

/**
 * @brief engine 未初始化时，查询张量形状应抛出非法操作异常。
 */
TEST(ONNXModelRuntimeQueryTest, TensorShapeWithoutEngineThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    EXPECT_THROW({ model->tensorShape("input"); }, irt::Exception);
}

/**
 * @brief engine 未初始化时，查询张量数据类型应抛出非法操作异常。
 */
TEST(ONNXModelRuntimeQueryTest, TensorDataTypeWithoutEngineThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    EXPECT_THROW({ model->tensorDataType("input"); }, irt::Exception);
}

/**
 * @brief context 未初始化时，设置运行时输入形状应抛出非法操作异常。
 */
TEST(ONNXModelRuntimeQueryTest, SetTensorShapeWithoutContextThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    const nvinfer1::Dims4 dims(1, 3, 224, 224);
    EXPECT_THROW({ model->setTensorShape("input", dims); }, irt::Exception);
}

/**
 * @brief 构建不存在的 ONNX 文件时，应返回非法参数错误。
 */
TEST(ONNXModelLifecycleTest, BuildWithNonExistentOnnxThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    EXPECT_THROW({ model->build("/non/existent/path/model.onnx"); }, irt::Exception);

    try
    {
        model->build("/non/existent/path/model.onnx");
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}

/**
 * @brief buildOrLoad 在 ONNX 文件不存在时，也应返回非法参数错误。
 */
TEST(ONNXModelLifecycleTest, BuildOrLoadWithNonExistentOnnxThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    EXPECT_THROW({ model->buildOrLoad("/non/existent/path/model.onnx"); }, irt::Exception);

    try
    {
        model->buildOrLoad("/non/existent/path/model.onnx");
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}

/**
 * @brief context 未初始化时执行 ONNX 推理，应抛出非法操作异常。
 */
TEST(ONNXModelInferTest, InferWithoutContextThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("onnx");
    ASSERT_NE(model, nullptr);

    std::vector<void *> buffers(2, nullptr);

    EXPECT_THROW({ model->infer(buffers); }, irt::Exception);

    try
    {
        model->infer(buffers);
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_OPERATION);
    }
}
