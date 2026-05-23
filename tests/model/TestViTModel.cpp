#include "TestModelCommon.hpp"

#include <inferrt/model/IModel.h>

#include <memory>
#include <string>
#include <vector>

using test::model::ExpectIrtExceptionCode;
using test::model::TempWeightsFile;

/**
 * @brief `vit` 兼容别名应创建 ViT-Base/16 224 结构。
 */
TEST(ViTModelFactoryTest, AliasCreatesBasePatch16Model)
{
    auto model = irt::model::CreateModel("vit");
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(model->name(), "ViTBasePatch16_224");
}

/**
 * @brief ViT 变体应保留通过模型配置注入的输入、输出和特征名称。
 */
TEST(ViTModelConfigTest, PreservesCustomTensorAndFeatureNames)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputTensorNames({"image"});
    config->setOutputTensorNames({"tokens"});
    config->setFeatureTensorNames({"blocks.0"});
    config->setFeatureOnly(true);

    auto model = irt::model::CreateModel("vit_base_patch16_224", std::move(config));
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->modelConfig().inputTensorNames(), (std::vector<std::string>{"image"}));
    EXPECT_EQ(model->modelConfig().outputTensorNames(), (std::vector<std::string>{"tokens"}));
    EXPECT_EQ(model->modelConfig().featureTensorNames(), (std::vector<std::string>{"blocks.0"}));
    EXPECT_TRUE(model->modelConfig().featureOnly());
}

/**
 * @brief 384 分辨率 ViT 变体应自动把默认输入尺寸规整为 1x3x384x384。
 */
TEST(ViTModelConfigTest, UsesVariantDefaultInputShapeFor384Model)
{
    auto model = irt::model::CreateModel("vit_base_patch16_384");
    ASSERT_NE(model, nullptr);

    const auto &shape = model->modelConfig().inputShape();
    EXPECT_EQ(shape.d[0], 1);
    EXPECT_EQ(shape.d[1], 3);
    EXPECT_EQ(shape.d[2], 384);
    EXPECT_EQ(shape.d[3], 384);
}

/**
 * @brief 384 变体只改默认尺寸，不应覆盖调用方显式传入的非默认输入尺寸。
 */
TEST(ViTModelConfigTest, PreservesExplicitNonDefaultInputShape)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(nvinfer1::Dims4{1, 3, 512, 512});

    auto model = irt::model::CreateModel("vit_base_patch16_384", std::move(config));
    ASSERT_NE(model, nullptr);

    const auto &shape = model->modelConfig().inputShape();
    EXPECT_EQ(shape.d[0], 1);
    EXPECT_EQ(shape.d[1], 3);
    EXPECT_EQ(shape.d[2], 512);
    EXPECT_EQ(shape.d[3], 512);
}

/**
 * @brief ViT 构建期应拒绝不能被 patch size 整除的输入尺寸。
 */
TEST(ViTModelBuildTest, BuildRejectsInputShapeNotDivisibleByPatchSize)
{
    auto model = irt::model::CreateModel("vit_base_patch16_224");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(nvinfer1::Dims4{1, 3, 225, 224});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_vit_test_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}
