#include "TestModelCommon.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>

using test::model::RegisteredModelsTest;
using test::model::kRegisteredModels;

INSTANTIATE_TEST_SUITE_P(KnownModels, RegisteredModelsTest, ::testing::ValuesIn(kRegisteredModels));

/**
 * @brief 所有内置模型都应使用统一的默认权重扩展名和引擎扩展名
 */
TEST_P(RegisteredModelsTest, DefaultExtensionsMatchExpectedValues)
{
    const auto &param = GetParam();

    auto model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->wtsExtension(), ".wts");
    EXPECT_EQ(model->engineExtension(), ".engine");
}

/**
 * @brief 所有内置模型默认日志级别都应为 WARNING
 */
TEST_P(RegisteredModelsTest, DefaultLogLevelIsWarning)
{
    const auto &param = GetParam();

    auto model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kWARNING);
}

/**
 * @brief 多次设置日志级别后，当前值应始终与最后一次设置保持一致
 */
TEST(IModelDefaultPropertiesTest, SetLogLevelTakesEffect)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    model->setLogLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kVERBOSE);

    model->setLogLevel(nvinfer1::ILogger::Severity::kERROR);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kERROR);

    model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kINFO);
}

/**
 * @brief 在 logger 尚未初始化时设置日志级别，也应正确保存在模型内部
 */
TEST(IModelLogLevelTest, SetLogLevelBeforeLoggerInitPersistsValue)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    model->setLogLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kVERBOSE);
}

/**
 * @brief 未构建 engine 时调用 save，应抛出 ERROR_INVALID_OPERATION
 */
TEST(IModelSaveTest, SaveWithoutEngineThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    EXPECT_THROW({ model->save("output.engine"); }, irt::Exception);

    try
    {
        model->save("output.engine");
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_OPERATION);
    }
}

/**
 * @brief 加载不存在的 engine 文件时，应抛出 ERROR_INVALID_ARGUMENT
 */
TEST(IModelLoadTest, LoadNonExistentFileThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    EXPECT_THROW({ model->load("/non/existent/path/model.engine"); }, irt::Exception);

    try
    {
        model->load("/non/existent/path/model.engine");
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}

/**
 * @brief 使用不存在的权重文件构建模型时，应抛出 ERROR_INVALID_ARGUMENT
 */
TEST(IModelBuildTest, BuildWithNonExistentWeightsThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    EXPECT_THROW({ model->build("/non/existent/path/model.wts"); }, irt::Exception);

    try
    {
        model->build("/non/existent/path/model.wts");
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}

/**
 * @brief buildOrLoad 在权重文件不存在时，也应沿用同样的错误码约定
 */
TEST(IModelBuildOrLoadTest, BuildOrLoadNonExistentWeightsThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    EXPECT_THROW({ model->buildOrLoad("/non/existent/path/model.wts"); }, irt::Exception);

    try
    {
        model->buildOrLoad("/non/existent/path/model.wts");
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), irt::Status::ERROR_INVALID_ARGUMENT);
    }
}
