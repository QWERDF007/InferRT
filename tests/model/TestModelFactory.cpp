#include "TestModelCommon.hpp"

#include <inferrt/model/IModel.h>

#include <algorithm>
#include <string>
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
    EXPECT_EQ(model->name(), param.display_name);
}

/**
 * @brief 新 key 注册成功后，应能通过工厂创建出对应实例
 */
TEST(IModelRegisterTest, RegisterModelReturnsTrueForNewKey)
{
    const std::string key = "register_only_unique_key";
    EXPECT_TRUE(irt::model::RegisterModel(key, nullptr));
}

/**
 * @brief 模型注册表应对注册 key 做大小写归一化，避免大小写不同的重复条目。
 */
TEST(IModelRegisterTest, RegisterModelTreatsKeysCaseInsensitively)
{
    const std::string key = "register_only_case_key";
    ASSERT_TRUE(irt::model::RegisterModel(key, nullptr));
    EXPECT_FALSE(irt::model::RegisterModel("REGISTER_ONLY_CASE_KEY", nullptr));
    EXPECT_TRUE(irt::model::isSupportedModel("Register_Only_Case_Key"));
}

/**
 * @brief 对同一个 key 重复注册时，第二次应返回 false
 */
TEST(IModelRegisterTest, RegisterModelReturnsFalseForDuplicateKey)
{
    const std::string key = "register_only_duplicate_key";
    ASSERT_TRUE(irt::model::RegisterModel(key, nullptr));
    EXPECT_FALSE(irt::model::RegisterModel(key, nullptr));
}
