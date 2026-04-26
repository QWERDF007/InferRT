#include "TestModelCommon.hpp"

#include <inferrt/model/IModel.h>

#include <string>

using test::model::CreateDummyModel;
using test::model::RegisteredModelsTest;
using test::model::kRegisteredModels;

INSTANTIATE_TEST_SUITE_P(KnownModels, RegisteredModelsTest, ::testing::ValuesIn(kRegisteredModels));

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
 * @brief 使用标准注册 key 应能创建出正确的模型实例
 */
TEST_P(RegisteredModelsTest, CreateModelReturnsValidInstanceForRegisteredKey)
{
    const auto &param = GetParam();

    auto model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->name(), param.display_name);
}

/**
 * @brief CreateModel 对大小写混合的 key 也应能正常识别
 */
TEST_P(RegisteredModelsTest, CreateModelAcceptsMixedCaseKey)
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
    const std::string key = "dummy_model_unique_key";
    EXPECT_TRUE(irt::model::RegisterModel(key, &CreateDummyModel));

    auto model = irt::model::CreateModel("DUMMY_MODEL_UNIQUE_KEY");
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->name(), "DummyModel");
}

/**
 * @brief 对同一个 key 重复注册时，第二次应返回 false
 */
TEST(IModelRegisterTest, RegisterModelReturnsFalseForDuplicateKey)
{
    const std::string key = "dummy_model_duplicate_key";
    ASSERT_TRUE(irt::model::RegisterModel(key, &CreateDummyModel));
    EXPECT_FALSE(irt::model::RegisterModel(key, &CreateDummyModel));
}
