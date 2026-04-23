
#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/core/Status.h>

namespace t = ::testing;


// ============================================================================
// CreateModel 工厂函数测试
// ============================================================================

/**
 * @brief 验证 CreateModel("alexnet") 返回有效的 AlexNet 实例
 */
TEST(IModelCreateTest, CreateAlexNetReturnsValidInstance)
{
    auto *model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->name(), "AlexNet");
    delete model;
}

/**
 * @brief 验证 CreateModel 对未实现的模型名称返回 nullptr
 */
TEST(IModelCreateTest, UnknownModelNameReturnsNullptr)
{
    auto *model = irt::model::CreateModel("unknown_model");
    EXPECT_EQ(model, nullptr);
}

/**
 * @brief 验证 CreateModel 对空字符串名称返回 nullptr
 */
TEST(IModelCreateTest, EmptyNameReturnsNullptr)
{
    auto *model = irt::model::CreateModel("");
    EXPECT_EQ(model, nullptr);
}

/**
 * @brief 验证 CreateModel 的模型名称匹配区分大小写
 */
TEST(IModelCreateTest, ModelNameIsCaseSensitive)
{
    // "AlexNet" (大写 A) 不应匹配 "alexnet" (小写 a)
    auto *model = irt::model::CreateModel("AlexNet");
    EXPECT_EQ(model, nullptr);
}

// ============================================================================
// IModel 虚方法默认值测试
// ============================================================================

// TEST(IModelDefaultsTest, alexnet_default_extensions)
// {
//     auto *model = irt::model::CreateModel("alexnet");
//     ASSERT_NE(model, nullptr);

//     EXPECT_EQ(model->wtsExtension(), ".wts");
//     EXPECT_EQ(model->engineExtension(), ".engine");

//     delete model;
// }

/**
 * @brief 验证 AlexNet 默认日志级别为 kWARNING
 */
TEST(IModelDefaultPropertiesTest, DefaultLogLevelIsWarning)
{
    auto *model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kWARNING);

    delete model;
}

/**
 * @brief 验证 setLogLevel 可修改日志级别并立即生效
 */
TEST(IModelDefaultPropertiesTest, SetLogLevelTakesEffect)
{
    auto *model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    model->setLogLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kVERBOSE);

    model->setLogLevel(nvinfer1::ILogger::Severity::kERROR);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kERROR);

    delete model;
}

// ============================================================================
// IModel::save 异常测试 - 未初始化引擎时调用 save
// ============================================================================

/**
 * @brief 验证未构建引擎时调用 save 抛出 ERROR_INVALID_OPERATION 异常
 */
TEST(IModelSaveTest, SaveWithoutEngineThrowsInvalidOperation)
{
    auto *model = irt::model::CreateModel("alexnet");
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

    delete model;
}

// ============================================================================
// IModel::load 异常测试 - 加载不存在的引擎文件
// ============================================================================

/**
 * @brief 验证加载不存在的引擎文件时抛出 irt::Exception 异常
 */
TEST(IModelLoadTest, LoadNonExistentFileThrowsException)
{
    auto *model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    EXPECT_THROW({ model->load("/non/existent/path/model.engine"); }, irt::Exception);

    // EXPECT_EQ(irt::GetLastError(), IRT_ERROR_INVALID_ARGUMENT);
    
    delete model;
}

// ============================================================================
// IModel::build 异常测试 - 加载不存在的权重文件
// ============================================================================

/**
 * @brief 验证使用不存在的权重文件构建引擎时抛出 irt::Exception 异常
 */
TEST(IModelBuildTest, BuildWithNonExistentWeightsThrowsException)
{
    auto *model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    EXPECT_THROW({ model->build("/non/existent/path/model.wts"); }, irt::Exception);

    // EXPECT_EQ(irt::GetLastError(), IRT_ERROR_INVALID_ARGUMENT);

    delete model;
}

// ============================================================================
// AlexNet::infer 异常测试 - 未初始化上下文时调用 infer
// ============================================================================

/**
 * @brief 验证未初始化上下文时调用 infer 抛出 ERROR_INVALID_OPERATION 异常
 */
TEST(AlexNetInferTest, InferWithoutContextThrowsException)
{
    auto *model = irt::model::CreateModel("alexnet");
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

    delete model;
}

/**
 * @brief 验证传入错误数量的 buffer 调用 infer 时抛出 irt::Exception 异常
 */
TEST(AlexNetInferTest, InferWithWrongBufferCountThrowsException)
{
    auto *model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    // AlexNet 要求 2 个 buffer（input + output）
    std::vector<void *> buffers(1, nullptr);

    // 即使 context 未初始化，buffer 数量检查可能先执行
    // 但实际实现中 context 检查在 buffer 检查之前
    // 所以这里仍然期望 ERROR_INVALID_OPERATION
    EXPECT_THROW({ model->infer(buffers); }, irt::Exception);

    delete model;
}

// ============================================================================
// IModel::buildOrLoad 异常测试 - 不存在的权重文件
// ============================================================================

/**
 * @brief 验证 buildOrLoad 传入不存在的权重文件路径时抛出 irt::Exception 异常
 */
TEST(IModelBuildOrLoadTest, BuildOrLoadNonExistentWeightsThrowsException)
{
    auto *model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    EXPECT_THROW({ model->buildOrLoad("/non/existent/path/model.wts"); }, irt::Exception);

    delete model;
}

// ============================================================================
// IModel setLogLevel 与 logger 联动测试
// ============================================================================

/**
 * @brief 验证在 logger 初始化之前设置 logLevel 仍能正确保存并生效
 */
TEST(IModelLogLevelTest, SetLogLevelBeforeLoggerInit)
{
    auto *model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    // 在 logger 初始化之前设置 log level
    model->setLogLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kVERBOSE);

    delete model;
}

/**
 * @brief 验证多次调用 setLogLevel 后每次都能正确更新并生效
 */
TEST(IModelLogLevelTest, SetLogLevelMultipleTimes)
{
    auto *model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    model->setLogLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kVERBOSE);

    model->setLogLevel(nvinfer1::ILogger::Severity::kERROR);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kERROR);

    model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kINFO);

    delete model;
}


