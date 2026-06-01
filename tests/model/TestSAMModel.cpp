#include "TestModelCommon.hpp"

#include <inferrt/model/IModel.h>

#include <memory>
#include <string>
#include <vector>

using test::model::ExpectIrtExceptionCode;
using test::model::TempWeightsFile;

namespace {

/**
 * @brief 断言 SAM 系列默认输入输出契约。
 */
void expectSAMContract(const irt::model::IModel &model, int image_size, bool expect_default_outputs = true)
{
    EXPECT_EQ(model.modelConfig().inputTensorNames(),
              (std::vector<std::string>{"image", "point_coords", "point_labels", "mask_input", "has_mask_input"}));
    if (expect_default_outputs)
    {
        EXPECT_EQ(model.modelConfig().outputTensorNames(),
                  (std::vector<std::string>{"masks", "iou_predictions", "low_res_masks"}));
    }
    ASSERT_EQ(model.modelConfig().inputShapes().size(), 5);

    const auto &image_shape = model.modelConfig().inputShapes()[0];
    EXPECT_EQ(image_shape.d[0], 1);
    EXPECT_EQ(image_shape.d[1], 3);
    EXPECT_EQ(image_shape.d[2], image_size);
    EXPECT_EQ(image_shape.d[3], image_size);

    const auto &point_shape = model.modelConfig().inputShapes()[1];
    EXPECT_EQ(point_shape.d[0], 1);
    EXPECT_EQ(point_shape.d[1], 16);
    EXPECT_EQ(point_shape.d[2], 2);
    EXPECT_EQ(point_shape.d[3], 1);

    const auto &mask_shape = model.modelConfig().inputShapes()[3];
    EXPECT_EQ(mask_shape.d[0], 1);
    EXPECT_EQ(mask_shape.d[1], 1);
    EXPECT_EQ(mask_shape.d[2], 256);
    EXPECT_EQ(mask_shape.d[3], 256);
}

/**
 * @brief 生成只改变图像尺寸的完整 SAM 输入形状。
 */
std::vector<nvinfer1::Dims4> makeSAMInputShapes(int image_size, int batch = 1)
{
    return {
        nvinfer1::Dims4{batch,  3, image_size, image_size},
        nvinfer1::Dims4{batch, 16,          2,          1},
        nvinfer1::Dims4{batch, 16,          1,          1},
        nvinfer1::Dims4{batch,  1,        256,        256},
        nvinfer1::Dims4{batch,  1,          1,          1},
    };
}

} // namespace

/**
 * @brief SAM v1 默认 key 应创建 ViT-H 入口并使用 1024 输入。
 */
TEST(SAMModelFactoryTest, SAMDefaultKeyCreatesViTHContract)
{
    auto model = irt::model::CreateModel("sam");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->name(), "SAMViTH");
    expectSAMContract(*model, 1024);
    EXPECT_EQ(model->modelConfig().numClasses(), 3);
}

/**
 * @brief EdgeSAM key 应创建 RepViT-M1 入口并复用 SAM prompt 输入契约。
 */
TEST(SAMModelFactoryTest, EdgeSAMKeyCreatesRepViTContract)
{
    auto model = irt::model::CreateModel("edge_sam");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->name(), "EdgeSAM");
    expectSAMContract(*model, 1024);
    EXPECT_EQ(model->modelConfig().numClasses(), 3);
}

/**
 * @brief SAM2 默认 key 应创建 Hiera-L 入口并复用 SAM prompt 输入契约。
 */
TEST(SAMModelFactoryTest, SAM2DefaultKeyCreatesHieraLargeContract)
{
    auto model = irt::model::CreateModel("sam2");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->name(), "SAM2HieraLarge");
    expectSAMContract(*model, 1024, false);
}

/**
 * @brief SAM3 图像入口使用官方 1008 输入尺寸。
 */
TEST(SAMModelFactoryTest, SAM3DefaultKeyUsesImageModelInputSize)
{
    auto model = irt::model::CreateModel("sam3");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->name(), "SAM3Image");
    expectSAMContract(*model, 1008);
}

/**
 * @brief SAM 系列变体 key 应注册到模型工厂。
 */
TEST(SAMModelFactoryTest, RegistersSAMFamilyVariants)
{
    const std::vector<std::pair<std::string, std::string>> cases{
        {             "sam_vit_b",             "SAMViTB"},
        {             "sam_vit_l",             "SAMViTL"},
        {             "sam_vit_h",             "SAMViTH"},
        {              "edge_sam",             "EdgeSAM"},
        {       "sam2_hiera_tiny",       "SAM2HieraTiny"},
        {      "sam2_hiera_small",      "SAM2HieraSmall"},
        {  "sam2_hiera_base_plus",   "SAM2HieraBasePlus"},
        {      "sam2_hiera_large",      "SAM2HieraLarge"},
        {     "sam2_1_hiera_tiny",     "SAM2.1HieraTiny"},
        {    "sam2_1_hiera_small",    "SAM2.1HieraSmall"},
        {"sam2_1_hiera_base_plus", "SAM2.1HieraBasePlus"},
        {    "sam2_1_hiera_large",    "SAM2.1HieraLarge"},
        {            "sam3_image",           "SAM3Image"},
    };

    for (const auto &[key, display_name] : cases)
    {
        auto model = irt::model::CreateModel(key);
        ASSERT_NE(model, nullptr) << key;
        EXPECT_EQ(model->name(), display_name) << key;
    }
}

/**
 * @brief SAM feature-only 配置应保留调用方指定的特征和输出名。
 */
TEST(SAMModelConfigTest, PreservesFeatureOnlyConfig)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setFeatureOnly(true);
    config->setFeatureTensorNames({"image_embedding"});
    config->setOutputTensorNames({"image_embedding"});

    auto model = irt::model::CreateModel("sam_vit_b", std::move(config));
    ASSERT_NE(model, nullptr);

    EXPECT_TRUE(model->modelConfig().featureOnly());
    EXPECT_EQ(model->modelConfig().featureTensorNames(), (std::vector<std::string>{"image_embedding"}));
    EXPECT_EQ(model->modelConfig().outputTensorNames(), (std::vector<std::string>{"image_embedding"}));
    expectSAMContract(*model, 1024, false);
}

/**
 * @brief SAM 构建期应拒绝不符合官方固定输入尺寸的图像。
 */
TEST(SAMModelBuildTest, BuildRejectsInvalidImageSize)
{
    auto model = irt::model::CreateModel("sam_vit_b");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputTensorNames({"image", "point_coords", "point_labels", "mask_input", "has_mask_input"});
    config->setInputShapes(makeSAMInputShapes(640));
    config->setOutputTensorNames({"masks", "iou_predictions", "low_res_masks"});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_sam_test_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief SAM 构建期应拒绝缺失 prompt 输入的配置，避免运行时绑定顺序不明确。
 */
TEST(SAMModelBuildTest, BuildRejectsMissingPromptInputs)
{
    auto model = irt::model::CreateModel("sam2_hiera_tiny");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputTensorNames({"image"});
    config->setInputShape(nvinfer1::Dims4{1, 3, 1024, 1024});
    config->setOutputTensorNames({"masks", "iou_predictions", "low_res_masks"});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_sam2_test_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief SAM point 坐标输入必须保持官方 ``1x16x2x1`` 契约，防止 prompt 绑定顺序或布局错误。
 */
TEST(SAMModelBuildTest, BuildRejectsInvalidPointCoordinateShape)
{
    auto model = irt::model::CreateModel("sam_vit_b");
    ASSERT_NE(model, nullptr);

    auto shapes = makeSAMInputShapes(1024);
    shapes[1]   = nvinfer1::Dims4{1, 8, 2, 1};

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputTensorNames({"image", "point_coords", "point_labels", "mask_input", "has_mask_input"});
    config->setInputShapes(shapes);
    config->setOutputTensorNames({"masks", "iou_predictions", "low_res_masks"});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_sam_point_shape_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief SAM TensorRT 手写网络应允许所有输入共享同一个动态 batch。
 */
TEST(SAMModelBuildTest, DynamicBatchConfigIsAccepted)
{
    auto model = irt::model::CreateModel("edge_sam");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputTensorNames({"image", "point_coords", "point_labels", "mask_input", "has_mask_input"});
    config->setInputShapes(makeSAMInputShapes(1024, 2));
    config->setOutputTensorNames({"masks", "iou_predictions", "low_res_masks"});
    config->setDynamicBatchRange(1, 2, 4);
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_edge_sam_dynamic_batch_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief SAM 动态 batch 要求 prompt、mask 输入与 image 输入保持相同 batch。
 */
TEST(SAMModelBuildTest, DynamicBatchRejectsMismatchedPromptBatch)
{
    auto model = irt::model::CreateModel("sam_vit_b");
    ASSERT_NE(model, nullptr);

    auto shapes = makeSAMInputShapes(1024, 2);
    shapes[1]   = nvinfer1::Dims4{1, 16, 2, 1};

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputTensorNames({"image", "point_coords", "point_labels", "mask_input", "has_mask_input"});
    config->setInputShapes(shapes);
    config->setOutputTensorNames({"masks", "iou_predictions", "low_res_masks"});
    config->setDynamicBatchRange(1, 2, 4);
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_sam_mismatched_dynamic_batch_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 非 feature-only 的 SAM 分割路径固定输出 mask、IoU 与 low-res mask 三个张量。
 */
TEST(SAMModelBuildTest, BuildRejectsWrongOutputTensorCount)
{
    auto model = irt::model::CreateModel("sam2_1_hiera_tiny");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputTensorNames({"image", "point_coords", "point_labels", "mask_input", "has_mask_input"});
    config->setInputShapes(makeSAMInputShapes(1024));
    config->setOutputTensorNames({"masks", "iou_predictions"});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_sam_outputs_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief SAM v1 官方 TensorRT 路径必须使用完整 state_dict 导出的权重。
 */
TEST(SAMModelBuildTest, BuildRequiresOfficialSAMWeights)
{
    auto model = irt::model::CreateModel("sam_vit_b");
    ASSERT_NE(model, nullptr);

    const TempWeightsFile weights("inferrt_sam_build_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief EdgeSAM TensorRT 路径必须使用官方 EdgeSAM state_dict 导出的权重。
 */
TEST(SAMModelBuildTest, BuildEdgeSAMRequiresOfficialWeights)
{
    auto model = irt::model::CreateModel("edge_sam");
    ASSERT_NE(model, nullptr);

    const TempWeightsFile weights("inferrt_edge_sam_build_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief SAM2 官方 TensorRT 路径必须使用完整 state_dict 导出的权重。
 */
TEST(SAMModelBuildTest, BuildSAM2RequiresOfficialWeights)
{
    auto model = irt::model::CreateModel("sam2_hiera_tiny");
    ASSERT_NE(model, nullptr);

    const TempWeightsFile weights("inferrt_sam2_build_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief SAM3 在原生 image encoder 接入前不应静默映射为 SAM v1/SAM2。
 */
TEST(SAMModelBuildTest, BuildRejectsUnimplementedSAM3NativeBackbone)
{
    auto model = irt::model::CreateModel("sam3_image");
    ASSERT_NE(model, nullptr);

    const TempWeightsFile weights("inferrt_sam3_native_backbone_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_NOT_IMPLEMENTED);
}
