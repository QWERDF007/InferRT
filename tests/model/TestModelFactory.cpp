#include "TestModelCommon.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>

#include <algorithm>
#include <string>
#include <thread>
#include <vector>

using test::model::RegisteredModelsTest;
using test::model::kRegisteredModels;

namespace {

/**
 * @brief 模型工厂注册表的参数化测试基类。
 */
class ModelFactoryRegisteredModelsTest : public RegisteredModelsTest
{
};

std::unique_ptr<irt::model::IModel> MinimalCreator()
{
    return std::make_unique<irt::model::IModel>();
}

std::unique_ptr<irt::model::IModel> NullReturningCreator()
{
    return nullptr;
}

} // namespace

INSTANTIATE_TEST_SUITE_P(KnownModels, ModelFactoryRegisteredModelsTest, ::testing::ValuesIn(kRegisteredModels));

/**
 * @brief 未知模型名应返回空指针
 */
TEST(IModelCreateTest, UnknownModelNameReturnsNullptr)
{
    auto model = irt::model::CreateModel("unknown_model");
    EXPECT_EQ(model, nullptr);
}

/**
 * @brief 空字符串不应命中任何模型
 */
TEST(IModelCreateTest, EmptyNameReturnsNullptr)
{
    auto model = irt::model::CreateModel("");
    EXPECT_EQ(model, nullptr);
}

/**
 * @brief 仅包含空白字符的字符串不应命中任何模型
 */
TEST(IModelCreateTest, WhitespaceNameReturnsNullptr)
{
    auto model = irt::model::CreateModel("   ");
    EXPECT_EQ(model, nullptr);
}

/**
 * @brief isSupportedModel 应与 CreateModel 使用同一套大小写不敏感注册表。
 */
TEST_P(ModelFactoryRegisteredModelsTest, IsSupportedModelAcceptsRegisteredKeys)
{
    const auto &param = GetParam();

    EXPECT_TRUE(irt::model::isSupportedModel(param.key));
    EXPECT_TRUE(irt::model::isSupportedModel(param.mixed_case_key));
}

/**
 * @brief getRegisteredModelNames 应返回包含全部内置模型 key 的稳定清单。
 */
TEST(IModelRegisterTest, GetRegisteredModelNamesContainsKnownBuiltInModels)
{
    const auto names = irt::model::getRegisteredModelNames();

    for (const auto &param : kRegisteredModels)
    {
        EXPECT_NE(std::find(names.begin(), names.end(), std::string(param.key)), names.end()) << param.key;
    }
}

/**
 * @brief 使用标准注册 key 应能创建出正确的模型实例
 */
TEST_P(ModelFactoryRegisteredModelsTest, CreateModelReturnsValidInstanceForRegisteredKey)
{
    const auto &param = GetParam();

    auto model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(model->isValid());
    EXPECT_EQ(model->name(), param.display_name);
}

/**
 * @brief CreateModel 对大小写混合的 key 也应能正常识别
 */
TEST_P(ModelFactoryRegisteredModelsTest, CreateModelAcceptsMixedCaseKey)
{
    const auto &param = GetParam();

    auto model = irt::model::CreateModel(param.mixed_case_key);
    ASSERT_NE(model, nullptr);
    EXPECT_TRUE(model->isValid());
    EXPECT_EQ(model->name(), param.display_name);
}

/**
 * @brief 注册时拒绝空 creator。
 */
TEST(IModelRegisterTest, RejectsNullCreator)
{
    EXPECT_FALSE(irt::model::RegisterModel("null_creator_key", nullptr));
}

/**
 * @brief 注册时拒绝空名称或全空白名称。
 */
TEST(IModelRegisterTest, RejectsEmptyName)
{
    EXPECT_FALSE(irt::model::RegisterModel("", &MinimalCreator));
}

/**
 * @brief creator 返回 nullptr 时应抛出带上下文的异常，而非静默崩溃。
 */
TEST(IModelCreateTest, CreatorReturningNullIsReported)
{
    const std::string key = "null_returning_model_key";
    ASSERT_TRUE(irt::model::RegisterModel(key, &NullReturningCreator));
    EXPECT_THROW({
        irt::model::CreateModel(key);
    }, irt::Exception);
}

/**
 * @brief 新 key 注册成功后，应能通过工厂查询
 */
TEST(IModelRegisterTest, RegisterModelReturnsTrueForNewKey)
{
    const std::string key = "register_only_unique_key";
    EXPECT_TRUE(irt::model::RegisterModel(key, &MinimalCreator));
    EXPECT_TRUE(irt::model::isSupportedModel(key));
}

/**
 * @brief 模型注册表应对注册 key 做大小写归一化，避免大小写不同的重复条目。
 */
TEST(IModelRegisterTest, RegisterModelTreatsKeysCaseInsensitively)
{
    const std::string key = "register_only_case_key";
    ASSERT_TRUE(irt::model::RegisterModel(key, &MinimalCreator));
    EXPECT_FALSE(irt::model::RegisterModel("REGISTER_ONLY_CASE_KEY", &MinimalCreator));
    EXPECT_TRUE(irt::model::isSupportedModel("Register_Only_Case_Key"));
}

/**
 * @brief 对同一个 key 重复注册时，第二次应返回 false
 */
TEST(IModelRegisterTest, RegisterModelReturnsFalseForDuplicateKey)
{
    const std::string key = "register_only_duplicate_key";
    ASSERT_TRUE(irt::model::RegisterModel(key, &MinimalCreator));
    EXPECT_FALSE(irt::model::RegisterModel(key, &MinimalCreator));
}

/**
 * @brief 验证多线程并发查询与注册时线程安全（无数据竞争）。
 */
TEST(IModelThreadSafetyTest, ConcurrentLookupIsSafe)
{
    constexpr int kThreads = 8;
    constexpr int kIters   = 100;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t)
    {
        threads.emplace_back([t] {
            for (int i = 0; i < kIters; ++i)
            {
                EXPECT_TRUE(irt::model::isSupportedModel("resnet18"));
                EXPECT_TRUE(irt::model::isSupportedModel("YOLOv8"));
                EXPECT_FALSE(irt::model::isSupportedModel("non_existing_model_xyz"));
                auto names = irt::model::getRegisteredModelNames();
                EXPECT_FALSE(names.empty());
                if (t == 0 && i % 10 == 0)
                {
                    irt::model::RegisterModel("dynamic_test_key_" + std::to_string(i), &MinimalCreator);
                }
            }
        });
    }

    for (auto &th : threads)
    {
        th.join();
    }
}

TEST(IModelRegisterTest, RejectsAllWhitespaceName)
{
    EXPECT_FALSE(irt::model::RegisterModel("   \t\n  ", &MinimalCreator));
}

TEST(IModelHandleValidationTest, InvalidModelHandlesThrowOnAllMethods)
{
    irt::model::IModel invalid_model;
    EXPECT_FALSE(invalid_model.isValid());
    EXPECT_FALSE(static_cast<bool>(invalid_model));

    EXPECT_THROW(invalid_model.name(), irt::Exception);
    EXPECT_THROW(invalid_model.wtsExtension(), irt::Exception);
    EXPECT_THROW(invalid_model.engineExtension(), irt::Exception);
    EXPECT_THROW(invalid_model.logLevel(), irt::Exception);
    EXPECT_THROW(invalid_model.resolveExecutionStream(), irt::Exception);
    EXPECT_THROW(invalid_model.modelConfig(), irt::Exception);
    EXPECT_THROW(invalid_model.runtime(), irt::Exception);
    EXPECT_THROW(invalid_model.build("dummy.wts"), irt::Exception);
    EXPECT_THROW(invalid_model.load("dummy.engine"), irt::Exception);
    EXPECT_THROW(invalid_model.save("dummy.engine"), irt::Exception);
    EXPECT_THROW(invalid_model.buildOrLoad("dummy.wts"), irt::Exception);
    EXPECT_THROW(invalid_model.infer({}), irt::Exception);
    EXPECT_THROW(invalid_model.forwardFeatures({}), irt::Exception);
    EXPECT_THROW(invalid_model.setModelConfig(nullptr), irt::Exception);
    EXPECT_THROW(invalid_model.ioTensorNames(irt::TensorIOMode::Input), irt::Exception);
    EXPECT_THROW(invalid_model.tensorShape("input"), irt::Exception);
    EXPECT_THROW(invalid_model.tensorDataType("input"), irt::Exception);
    EXPECT_THROW(invalid_model.setTensorShape("input", irt::Shape{}), irt::Exception);
    EXPECT_THROW(invalid_model.setStream(0), irt::Exception);
    EXPECT_THROW(invalid_model.clearStream(), irt::Exception);
    EXPECT_THROW(invalid_model.setLogLevel(irt::model::LogLevel::Info), irt::Exception);
}

TEST(IModelConfigValidationTest, RejectsNegativeAndInvalidDimensions)
{
    irt::model::IModelConfig config;
    EXPECT_THROW(config.setInputShape(irt::Shape{-1, 3, 224, 224}), irt::Exception);
    EXPECT_THROW(config.setInputShape(irt::Shape{1, 0, 224, 224}), irt::Exception);
    EXPECT_THROW(config.setInputShapes({}), irt::Exception);
    EXPECT_THROW(config.setInputShapes({irt::Shape{1, 3, -1, 224}}), irt::Exception);
    EXPECT_THROW(config.setInputTensorNames({}), irt::Exception);
    EXPECT_THROW(config.setInputTensorNames({""}), irt::Exception);
    EXPECT_THROW(config.setInputTensorNames({"in", "in"}), irt::Exception);
    EXPECT_THROW(config.setOutputTensorNames({}), irt::Exception);
    EXPECT_THROW(config.setOutputTensorNames({""}), irt::Exception);
    EXPECT_THROW(config.setOutputTensorNames({"out", "out"}), irt::Exception);
    EXPECT_THROW(config.setFeatureTensorNames({""}), irt::Exception);
    EXPECT_THROW(config.setFeatureTensorNames({"f1", "f1"}), irt::Exception);
    EXPECT_THROW(config.setDynamicBatchRange(0, 4, 8), irt::Exception);
    EXPECT_THROW(config.setDynamicBatchRange(4, 2, 8), irt::Exception);
    EXPECT_THROW(config.setDynamicBatchRange(4, 4, 2), irt::Exception);
}
