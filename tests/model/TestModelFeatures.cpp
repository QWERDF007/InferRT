#include "TestModelCommon.hpp"

#include <memory>
#include <vector>

using test::model::ExpectIrtExceptionCode;
using test::model::MakeNullBuffers;
using test::model::RegisteredModelsTest;
using test::model::kRegisteredModels;

namespace {

/**
 * @brief 模型特征前向入口的参数化测试基类。
 */
class ModelFeaturesRegisteredModelsTest : public RegisteredModelsTest
{
};

/**
 * @brief 使用指定特征配置创建模型。
 * @param model_name 模型注册名。
 * @param feature_only 是否启用 featureOnly 模式。
 * @return 已注入特征配置的模型实例。
 */
std::unique_ptr<irt::model::IModel> createFeatureConfiguredModel(const std::string &model_name, bool feature_only)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"feature_for_runtime_guard"});
    config->setOutputTensorNames({"feature_for_runtime_guard"});
    config->setFeatureOnly(feature_only);
    return irt::model::CreateModel(model_name, std::move(config));
}

} // namespace

INSTANTIATE_TEST_SUITE_P(KnownModels, ModelFeaturesRegisteredModelsTest, ::testing::ValuesIn(kRegisteredModels));

/**
 * @brief 配置特征张量但未初始化特征执行上下文时，forwardFeatures 应抛出非法操作异常。
 */
TEST_P(ModelFeaturesRegisteredModelsTest, ForwardFeaturesWithoutFeatureContextThrowsInvalidOperation)
{
    const auto &param = GetParam();
    auto        model = createFeatureConfiguredModel(param.key, false);
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->forwardFeatures(MakeNullBuffers(2)); },
                           irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief featureOnly 模式下未初始化执行上下文时，forwardFeatures 仍应抛出非法操作异常。
 */
TEST_P(ModelFeaturesRegisteredModelsTest, ForwardFeaturesWithFeatureOnlyConfigWithoutContextThrowsInvalidOperation)
{
    const auto &param = GetParam();
    auto        model = createFeatureConfiguredModel(param.key, true);
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->forwardFeatures(MakeNullBuffers(2)); },
                           irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief featureOnly 模式下未初始化执行上下文时，infer 也应被同一运行时保护拦截。
 */
TEST_P(ModelFeaturesRegisteredModelsTest, InferWithFeatureOnlyConfigWithoutContextStillThrowsInvalidOperation)
{
    const auto &param = GetParam();
    auto        model = createFeatureConfiguredModel(param.key, true);
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->infer(MakeNullBuffers(2)); }, irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief 特征执行 context 缺失时，forwardFeatures 应先报运行时未就绪，而不是 buffer 数量错误。
 */
TEST_P(ModelFeaturesRegisteredModelsTest, ForwardFeaturesWithWrongBufferCountStillFailsBeforeBufferValidation)
{
    const auto &param = GetParam();
    auto        model = createFeatureConfiguredModel(param.key, false);
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->forwardFeatures(MakeNullBuffers(1)); },
                           irt::Status::ERROR_INVALID_OPERATION);
}
