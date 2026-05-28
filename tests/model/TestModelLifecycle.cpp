#include "TestModelCommon.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>

#include <memory>
#include <string>
#include <vector>

using test::model::RegisteredModelsTest;
using test::model::ExpectIrtExceptionCode;
using test::model::kRegisteredModels;

namespace {

/**
 * @brief 模型生命周期与默认配置的参数化测试基类。
 */
class ModelLifecycleRegisteredModelsTest : public RegisteredModelsTest
{
};

/**
 * @brief 使用新配置替换模型的输入/输出张量名称。
 * @param model 待更新的模型实例。
 * @param input_names 新输入张量名称列表。
 * @param output_names 新输出张量名称列表。
 */
void SetModelTensorNames(irt::model::IModel &model, std::vector<std::string> input_names,
                         std::vector<std::string> output_names)
{
    const auto &config = model.modelConfig();
    auto        new_config = std::make_unique<irt::model::IModelConfig>();
    new_config->setNumClasses(config.numClasses());
    new_config->setInputShapes(config.inputShapes());
    new_config->setInputTensorNames(std::move(input_names));
    new_config->setOutputTensorNames(std::move(output_names));
    new_config->setFeatureTensorNames(config.featureTensorNames());
    new_config->setFeatureOnly(config.featureOnly());
    new_config->setBackend(config.backend());
    new_config->setDevice(config.device());
    model.setModelConfig(std::move(new_config));
}

/**
 * @brief 仅替换模型输入张量名称，其他配置保持不变。
 * @param model 待更新的模型实例。
 * @param input_names 新输入张量名称列表。
 */
void SetModelInputTensorNames(irt::model::IModel &model, std::vector<std::string> input_names)
{
    SetModelTensorNames(model, std::move(input_names), model.modelConfig().outputTensorNames());
}

/**
 * @brief 仅替换模型输出张量名称，其他配置保持不变。
 * @param model 待更新的模型实例。
 * @param output_names 新输出张量名称列表。
 */
void SetModelOutputTensorNames(irt::model::IModel &model, std::vector<std::string> output_names)
{
    SetModelTensorNames(model, model.modelConfig().inputTensorNames(), std::move(output_names));
}

} // namespace

INSTANTIATE_TEST_SUITE_P(KnownModels, ModelLifecycleRegisteredModelsTest, ::testing::ValuesIn(kRegisteredModels));

/**
 * @brief 所有内置模型都应使用统一的默认权重扩展名和引擎扩展名。
 */
TEST_P(ModelLifecycleRegisteredModelsTest, DefaultExtensionsMatchExpectedValues)
{
    const auto &param = GetParam();

    auto model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);
    const std::string expected_wts_extension = std::string(param.key) == "onnx" ? ".onnx" : ".wts";
    EXPECT_EQ(model->wtsExtension(), expected_wts_extension);
    EXPECT_EQ(model->engineExtension(), ".engine");
}

/**
 * @brief 所有内置模型的默认日志级别都应为 WARNING。
 */
TEST_P(ModelLifecycleRegisteredModelsTest, DefaultLogLevelIsWarning)
{
    const auto &param = GetParam();

    auto model = irt::model::CreateModel(param.key);
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kWARNING);
}

/**
 * @brief 多次设置日志级别后，当前值应始终与最后一次设置保持一致。
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
 * @brief 通过 IModel 接口设置输入/输出张量名称后，应能原样读回配置值。
 */
TEST(IModelConfigTest, TensorNamesCanBeConfiguredThroughModelApi)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    const std::vector<std::string> input_names{"image", "aux"};
    const std::vector<std::string> output_names{"logits", "scores"};

    SetModelTensorNames(*model, input_names, output_names);

    EXPECT_EQ(model->modelConfig().inputTensorNames(), input_names);
    EXPECT_EQ(model->modelConfig().outputTensorNames(), output_names);
}

/**
 * @brief IModelConfig 默认值应与 ImageNet 分类模型约定保持一致。
 */
TEST(IModelConfigTest, DefaultConfigMatchesImageNetClassificationContract)
{
    const irt::model::IModelConfig config;

    EXPECT_EQ(config.numClasses(), 1000);
    EXPECT_EQ(config.inputTensorNames(), (std::vector<std::string>{"input"}));
    EXPECT_EQ(config.outputTensorNames(), (std::vector<std::string>{"output"}));
    EXPECT_TRUE(config.featureTensorNames().empty());
    EXPECT_FALSE(config.featureOnly());
    EXPECT_EQ(config.backend(), irt::model::ModelBackend::TensorRT);
    EXPECT_EQ(config.device(), irt::model::ModelDevice::GPU);
    ASSERT_EQ(config.inputShapes().size(), 1U);
    EXPECT_EQ(config.inputShape().nbDims, 4);
    EXPECT_EQ(config.inputShape().d[0], 1);
    EXPECT_EQ(config.inputShape().d[1], 3);
    EXPECT_EQ(config.inputShape().d[2], 224);
    EXPECT_EQ(config.inputShape().d[3], 224);
}

/**
 * @brief IModelConfig setter 应能覆盖类别数、输入形状和特征配置。
 */
TEST(IModelConfigTest, SettersUpdateAllPublicConfigFields)
{
    irt::model::IModelConfig config;
    config.setNumClasses(7);
    config.setInputShape(nvinfer1::Dims4{2, 3, 32, 32});
    config.setInputTensorNames({"image"});
    config.setOutputTensorNames({"logits", "aux"});
    config.setFeatureTensorNames({"layer1", "layer2"});
    config.setFeatureOnly(true);
    config.setBackend(irt::model::ModelBackend::ONNXRuntime);
    config.setDevice(irt::model::ModelDevice::CPU);

    EXPECT_EQ(config.numClasses(), 7);
    EXPECT_EQ(config.inputTensorNames(), (std::vector<std::string>{"image"}));
    EXPECT_EQ(config.outputTensorNames(), (std::vector<std::string>{"logits", "aux"}));
    EXPECT_EQ(config.featureTensorNames(), (std::vector<std::string>{"layer1", "layer2"}));
    EXPECT_TRUE(config.featureOnly());
    EXPECT_EQ(config.backend(), irt::model::ModelBackend::ONNXRuntime);
    EXPECT_EQ(config.device(), irt::model::ModelDevice::CPU);
    EXPECT_EQ(config.inputShape().d[0], 2);
    EXPECT_EQ(config.inputShape().d[2], 32);
}

/**
 * @brief CreateModel 应保留调用方传入的后端和设备配置。
 */
TEST(IModelConfigTest, CreateModelPreservesBackendAndDeviceFromConfig)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setBackend(irt::model::ModelBackend::ONNXRuntime);
    config->setDevice(irt::model::ModelDevice::CPU);

    auto model = irt::model::CreateModel("onnx", std::move(config));
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->backend(), irt::model::ModelBackend::ONNXRuntime);
    EXPECT_EQ(model->device(), irt::model::ModelDevice::CPU);
    EXPECT_EQ(model->modelConfig().backend(), irt::model::ModelBackend::ONNXRuntime);
    EXPECT_EQ(model->modelConfig().device(), irt::model::ModelDevice::CPU);
}

/**
 * @brief setInputShapes 应替换完整输入形状列表，并保持 inputShape 指向第一个输入。
 */
TEST(IModelConfigTest, SetInputShapesReplacesShapeListAndUpdatesFirstShape)
{
    irt::model::IModelConfig config;
    config.setInputShapes({nvinfer1::Dims4{1, 3, 64, 64}, nvinfer1::Dims4{1, 1, 16, 16}});

    ASSERT_EQ(config.inputShapes().size(), 2U);
    EXPECT_EQ(config.inputShape().d[2], 64);
    EXPECT_EQ(config.inputShapes()[1].d[1], 1);
}

/**
 * @brief setModelConfig 传入空指针时应恢复默认配置并清理旧的自定义值。
 */
TEST(IModelConfigTest, SetNullModelConfigResetsToDefaultConfig)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setNumClasses(12);
    config->setInputTensorNames({"custom_input"});
    config->setFeatureOnly(true);
    model->setModelConfig(std::move(config));

    model->setModelConfig(nullptr);

    EXPECT_EQ(model->modelConfig().numClasses(), 1000);
    EXPECT_EQ(model->modelConfig().inputTensorNames(), (std::vector<std::string>{"input"}));
    EXPECT_FALSE(model->modelConfig().featureOnly());
}

/**
 * @brief 通过 CreateModel 传入的自定义 config，应保留其中的输入/输出张量名称。
 */
TEST(IModelConfigTest, CreateModelPreservesCustomTensorNamesFromConfig)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputTensorNames({"custom_input_0", "custom_input_1"});
    config->setOutputTensorNames({"custom_output_0", "custom_output_1"});

    auto model = irt::model::CreateModel("alexnet", std::move(config));
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->modelConfig().inputTensorNames(), (std::vector<std::string>{"custom_input_0", "custom_input_1"}));
    EXPECT_EQ(model->modelConfig().outputTensorNames(), (std::vector<std::string>{"custom_output_0", "custom_output_1"}));
}

/**
 * @brief featureOnly 配置应保留层 key、导出张量名及 featureOnly 标志（层 key 与导出名可不同）。
 */
TEST(IModelConfigTest, FeatureOnlyConfigPreservesLayerKeysAndOutputNames)
{
    auto via_model_api = irt::model::CreateModel("resnet18");
    ASSERT_NE(via_model_api, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setNumClasses(via_model_api->modelConfig().numClasses());
    config->setInputShapes(via_model_api->modelConfig().inputShapes());
    config->setInputTensorNames(via_model_api->modelConfig().inputTensorNames());
    config->setFeatureTensorNames({"layer1", "layer4"});
    config->setOutputTensorNames({"feat_low", "feat_high"});
    config->setFeatureOnly(true);
    via_model_api->setModelConfig(std::move(config));

    EXPECT_TRUE(via_model_api->modelConfig().featureOnly());
    EXPECT_EQ(via_model_api->modelConfig().featureTensorNames(), (std::vector<std::string>{"layer1", "layer4"}));
    EXPECT_EQ(via_model_api->modelConfig().outputTensorNames(), (std::vector<std::string>{"feat_low", "feat_high"}));

    auto create_config = std::make_unique<irt::model::IModelConfig>();
    create_config->setFeatureTensorNames({"layer2"});
    create_config->setOutputTensorNames({"layer2"});
    create_config->setFeatureOnly(true);

    auto via_create = irt::model::CreateModel("resnet50", std::move(create_config));
    ASSERT_NE(via_create, nullptr);
    EXPECT_TRUE(via_create->modelConfig().featureOnly());
    EXPECT_EQ(via_create->modelConfig().featureTensorNames(), (std::vector<std::string>{"layer2"}));
    EXPECT_EQ(via_create->modelConfig().outputTensorNames(), (std::vector<std::string>{"layer2"}));
}

/**
 * @brief 在 logger 尚未初始化时设置日志级别，也应正确保存在模型内部。
 */
TEST(IModelLogLevelTest, SetLogLevelBeforeLoggerInitPersistsValue)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    model->setLogLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    EXPECT_EQ(model->logLevel(), nvinfer1::ILogger::Severity::kVERBOSE);
}

/**
 * @brief 未构建 engine 时调用 save，应抛出 ERROR_INVALID_OPERATION。
 */
TEST(IModelSaveTest, SaveWithoutEngineThrowsInvalidOperation)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->save("output.engine"); }, irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief 加载不存在的 engine 文件时，应抛出 ERROR_INVALID_ARGUMENT。
 */
TEST(IModelLoadTest, LoadNonExistentFileThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->load("/non/existent/path/model.engine"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 使用不存在的权重文件构建模型时，应抛出 ERROR_INVALID_ARGUMENT。
 */
TEST(IModelBuildTest, BuildWithNonExistentWeightsThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当输入张量名称列表为空时，build 应在配置校验阶段抛出 ERROR_INVALID_ARGUMENT。
 */
TEST(IModelBuildTest, BuildWithEmptyInputTensorNamesThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    SetModelInputTensorNames(*model, {});

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当输出张量名称列表为空时，build 应在配置校验阶段抛出 ERROR_INVALID_ARGUMENT。
 */
TEST(IModelBuildTest, BuildWithEmptyOutputTensorNamesThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    SetModelOutputTensorNames(*model, {});

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当输入张量名称中包含空字符串时，build 应拒绝该非法配置。
 */
TEST(IModelBuildTest, BuildWithEmptyInputTensorNameThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    SetModelInputTensorNames(*model, {""});

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当输出张量名称中包含空字符串时，build 应拒绝该非法配置。
 */
TEST(IModelBuildTest, BuildWithEmptyOutputTensorNameThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    SetModelOutputTensorNames(*model, {""});

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当特征张量名称中包含空字符串时，build 应拒绝该非法配置。
 */
TEST(IModelBuildTest, BuildWithEmptyFeatureTensorNameThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({""});
    model->setModelConfig(std::move(config));

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief featureOnly 时若输出张量名数量与特征层 key 不一致，build 应拒绝该非法配置。
 */
TEST(IModelBuildTest, BuildWithMismatchedOutputTensorCountInFeatureOnlyModeThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"pool1", "pool3"});
    config->setOutputTensorNames({"feat_only_one"});
    config->setFeatureOnly(true);
    model->setModelConfig(std::move(config));

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当启用 featureOnly 但未提供特征张量名称时，build 应拒绝该非法配置。
 */
TEST(IModelBuildTest, BuildWithFeatureOnlyAndNoFeatureTensorNamesThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureOnly(true);
    config->setOutputTensorNames({"layer1"});
    model->setModelConfig(std::move(config));

    ExpectIrtExceptionCode([&] { model->build("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief buildOrLoad 在权重文件不存在时，也应沿用同样的错误码约定。
 */
TEST(IModelBuildOrLoadTest, BuildOrLoadNonExistentWeightsThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    ExpectIrtExceptionCode([&] { model->buildOrLoad("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当输入张量名称列表为空时，buildOrLoad 也应沿用相同的配置校验规则。
 */
TEST(IModelBuildOrLoadTest, BuildOrLoadWithEmptyInputTensorNamesThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    SetModelInputTensorNames(*model, {});

    ExpectIrtExceptionCode([&] { model->buildOrLoad("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当输出张量名称列表为空时，buildOrLoad 也应沿用相同的配置校验规则。
 */
TEST(IModelBuildOrLoadTest, BuildOrLoadWithEmptyOutputTensorNamesThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    SetModelOutputTensorNames(*model, {});

    ExpectIrtExceptionCode([&] { model->buildOrLoad("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief featureOnly 时若输出张量名数量与特征层 key 不一致，buildOrLoad 应拒绝该非法配置。
 */
TEST(IModelBuildOrLoadTest, BuildOrLoadWithMismatchedOutputTensorCountInFeatureOnlyModeThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureTensorNames({"pool1", "pool3"});
    config->setOutputTensorNames({"feat_only_one"});
    config->setFeatureOnly(true);
    model->setModelConfig(std::move(config));

    ExpectIrtExceptionCode([&] { model->buildOrLoad("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 当启用 featureOnly 但未提供特征张量名称时，buildOrLoad 应拒绝该非法配置。
 */
TEST(IModelBuildOrLoadTest, BuildOrLoadWithFeatureOnlyAndNoFeatureTensorNamesThrowsInvalidArgument)
{
    auto model = irt::model::CreateModel("alexnet");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureOnly(true);
    config->setOutputTensorNames({"layer1"});
    model->setModelConfig(std::move(config));

    ExpectIrtExceptionCode([&] { model->buildOrLoad("/non/existent/path/model.wts"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}
