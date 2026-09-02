#include "TestModelCommon.hpp"

#include <inferrt/model/IModel.h>

#include <memory>
#include <string>
#include <vector>

using test::model::ExpectIrtExceptionCode;
using test::model::TempWeightsFile;

namespace {

/**
 * @brief 断言模型当前使用指定的 NCHW 输入尺寸。
 */
void expectInputShape(const irt::model::IModel &model, int n, int c, int h, int w)
{
    const auto &shape = model.modelConfig().inputShape();
    EXPECT_EQ(shape[0], n);
    EXPECT_EQ(shape[1], c);
    EXPECT_EQ(shape[2], h);
    EXPECT_EQ(shape[3], w);
}

} // namespace

/**
 * @brief DINOv2 官方 key 应创建 518x518 的 ViT-S/14 主干。
 */
TEST(DINOModelFactoryTest, DINOv2OfficialKeyCreatesBackbone)
{
    auto model = irt::model::CreateModel("dinov2_vits14");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->name(), "DINOv2ViTS14");
    expectInputShape(*model, 1, 3, 518, 518);
}

/**
 * @brief DINOv2 timm key 应作为官方结构的别名注册。
 */
TEST(DINOModelFactoryTest, DINOv2TimmAliasCreatesSameBackbone)
{
    auto model = irt::model::CreateModel("vit_small_patch14_reg4_dinov2");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->name(), "DINOv2ViTS14Reg4");
    expectInputShape(*model, 1, 3, 518, 518);
}

/**
 * @brief DINOv3 官方 hub key 使用官方默认 224x224 输入。
 */
TEST(DINOModelFactoryTest, DINOv3OfficialKeyUsesOfficialInputSize)
{
    auto model = irt::model::CreateModel("dinov3_vitb16");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->name(), "DINOv3ViTB16");
    expectInputShape(*model, 1, 3, 224, 224);
}

/**
 * @brief DINOv3 timm key 使用 timm default_cfg 中的 256x256 输入。
 */
TEST(DINOModelFactoryTest, DINOv3TimmAliasUsesTimmInputSize)
{
    auto model = irt::model::CreateModel("vit_base_patch16_dinov3_qkvb");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->name(), "DINOv3ViTB16");
    expectInputShape(*model, 1, 3, 256, 256);
}

/**
 * @brief DINO 模型应保留调用方传入的张量名和 feature-only 配置。
 */
TEST(DINOModelConfigTest, PreservesCustomTensorAndFeatureNames)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputTensorNames({"image"});
    config->setOutputTensorNames({"cls_feature"});
    config->setFeatureTensorNames({"x_norm_clstoken"});
    config->setFeatureOnly(true);

    auto model = irt::model::CreateModel("dinov2_vitb14_reg", std::move(config));
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->modelConfig().inputTensorNames(), (std::vector<std::string>{"image"}));
    EXPECT_EQ(model->modelConfig().outputTensorNames(), (std::vector<std::string>{"cls_feature"}));
    EXPECT_EQ(model->modelConfig().featureTensorNames(), (std::vector<std::string>{"x_norm_clstoken"}));
    EXPECT_TRUE(model->modelConfig().featureOnly());
}

/**
 * @brief DINO 默认尺寸规整不应覆盖调用方显式设置的合法输入尺寸。
 */
TEST(DINOModelConfigTest, PreservesExplicitNonDefaultInputShape)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(irt::Shape{1, 3, 280, 280});

    auto model = irt::model::CreateModel("dinov2_vits14", std::move(config));
    ASSERT_NE(model, nullptr);

    expectInputShape(*model, 1, 3, 280, 280);
}

/**
 * @brief DINOv2 构建期应拒绝不能被 patch size 整除的输入尺寸。
 */
TEST(DINOModelBuildTest, DINOv2BuildRejectsInputShapeNotDivisibleByPatchSize)
{
    auto model = irt::model::CreateModel("dinov2_vits14");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(irt::Shape{1, 3, 519, 518});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_dinov2_test_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief DINOv3 构建期应拒绝非 RGB 输入，避免构建出与 patch 权重不匹配的网络。
 */
TEST(DINOModelBuildTest, DINOv3BuildRejectsInvalidChannelCount)
{
    auto model = irt::model::CreateModel("dinov3_vits16");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(irt::Shape{1, 1, 224, 224});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_dinov3_test_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief DINO TensorRT 手写网络应允许动态 batch 配置，并继续进入权重校验阶段。
 */
TEST(DINOModelBuildTest, DynamicBatchConfigIsAccepted)
{
    auto model = irt::model::CreateModel("dinov2_vits14");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(irt::Shape{2, 3, 518, 518});
    config->setDynamicBatchRange(1, 2, 4);
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_dinov2_dynamic_batch_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}
