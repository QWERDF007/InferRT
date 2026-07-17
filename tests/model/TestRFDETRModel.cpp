#include "TestModelCommon.hpp"

#include <inferrt/model/IModel.h>

#include <memory>
#include <string>
#include <vector>

using test::model::ExpectIrtExceptionCode;
using test::model::TempWeightsFile;

namespace {

/**
 * @brief RF-DETR 变体默认配置测试样例。
 */
struct RFDETRVariantCase
{
    const char *key;          ///< 下划线注册 key。
    const char *hyphen_key;   ///< 官方连字符 key。
    const char *display_name; ///< 模型显示名称。
    int         resolution;   ///< 默认输入分辨率。
};

/**
 * @brief 断言模型当前使用指定 NCHW 输入尺寸。
 */
void expectInputShape(const irt::model::IModel &model, int n, int c, int h, int w)
{
    const auto &shape = model.modelConfig().inputShape();
    EXPECT_EQ(shape.d[0], n);
    EXPECT_EQ(shape.d[1], c);
    EXPECT_EQ(shape.d[2], h);
    EXPECT_EQ(shape.d[3], w);
}

/**
 * @brief 断言 RF-DETR 检测原生 TensorRT 图的默认 I/O 契约。
 */
void expectDetectionContract(const irt::model::IModel &model, int resolution)
{
    EXPECT_EQ(model.wtsExtension(), ".wts");
    EXPECT_EQ(model.modelConfig().inputTensorNames(), (std::vector<std::string>{"input"}));
    expectInputShape(model, 1, 3, resolution, resolution);
    EXPECT_EQ(model.modelConfig().numClasses(), 90);
    EXPECT_EQ(model.modelConfig().outputTensorNames(), (std::vector<std::string>{"dets", "labels"}));
}

/**
 * @brief 断言 RF-DETR 实例分割原生 TensorRT 图的默认 I/O 契约。
 */
void expectSegmentationContract(const irt::model::IModel &model, int resolution)
{
    EXPECT_EQ(model.wtsExtension(), ".wts");
    EXPECT_EQ(model.modelConfig().inputTensorNames(), (std::vector<std::string>{"input"}));
    expectInputShape(model, 1, 3, resolution, resolution);
    EXPECT_EQ(model.modelConfig().numClasses(), 90);
    EXPECT_EQ(model.modelConfig().outputTensorNames(), (std::vector<std::string>{"dets", "labels", "masks"}));
}

} // namespace

/**
 * @brief RF-DETR 检测变体应使用官方默认输入尺寸和检测输出名。
 */
TEST(RFDETRModelFactoryTest, DetectionVariantsUseOfficialDefaults)
{
    constexpr RFDETRVariantCase cases[] = {
        {            "rfdetr_base",             "rfdetr-base",            "RFDETRBase", 560},
        {            "rfdetr_nano",             "rfdetr-nano",            "RFDETRNano", 384},
        {           "rfdetr_small",            "rfdetr-small",           "RFDETRSmall", 512},
        {          "rfdetr_medium",           "rfdetr-medium",          "RFDETRMedium", 576},
        {           "rfdetr_large",            "rfdetr-large",           "RFDETRLarge", 704},
        {"rfdetr_large_deprecated", "rfdetr-large-deprecated", "RFDETRLargeDeprecated", 560},
    };

    for (const auto &test_case : cases)
    {
        for (const char *key : {test_case.key, test_case.hyphen_key})
        {
            auto model = irt::model::CreateModel(key);
            ASSERT_NE(model, nullptr) << key;
            EXPECT_EQ(model->name(), test_case.display_name) << key;
            expectDetectionContract(*model, test_case.resolution);
        }
    }
}

/**
 * @brief RF-DETR 实例分割变体应在检测输出外额外导出 masks。
 */
TEST(RFDETRModelFactoryTest, SegmentationVariantsUseOfficialDefaults)
{
    constexpr RFDETRVariantCase cases[] = {
        {"rfdetr_seg_preview", "rfdetr-seg-preview", "RFDETRSegPreview", 432},
        {   "rfdetr_seg_nano",    "rfdetr-seg-nano",    "RFDETRSegNano", 312},
        {  "rfdetr_seg_small",   "rfdetr-seg-small",   "RFDETRSegSmall", 384},
        { "rfdetr_seg_medium",  "rfdetr-seg-medium",  "RFDETRSegMedium", 432},
        {  "rfdetr_seg_large",   "rfdetr-seg-large",   "RFDETRSegLarge", 504},
        { "rfdetr_seg_xlarge",  "rfdetr-seg-xlarge",  "RFDETRSegXLarge", 624},
        {"rfdetr_seg_2xlarge", "rfdetr-seg-2xlarge", "RFDETRSeg2XLarge", 768},
    };

    for (const auto &test_case : cases)
    {
        for (const char *key : {test_case.key, test_case.hyphen_key})
        {
            auto model = irt::model::CreateModel(key);
            ASSERT_NE(model, nullptr) << key;
            EXPECT_EQ(model->name(), test_case.display_name) << key;
            expectSegmentationContract(*model, test_case.resolution);
        }
    }
}

/**
 * @brief RF-DETR Seg 2XLarge 应兼容上游权重文件中的 xxlarge 命名。
 */
TEST(RFDETRModelFactoryTest, SegmentationXXLargeAliasCreates2XLarge)
{
    for (const char *key : {"rfdetr_seg_xxlarge", "rfdetr-seg-xxlarge"})
    {
        auto model = irt::model::CreateModel(key);
        ASSERT_NE(model, nullptr) << key;
        EXPECT_EQ(model->name(), "RFDETRSeg2XLarge") << key;
        expectSegmentationContract(*model, 768);
    }
}

/**
 * @brief RF-DETR 默认规整不应覆盖调用方显式传入的张量名、输入尺寸、类别数和后端。
 */
TEST(RFDETRModelConfigTest, PreservesExplicitNativeConfig)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputTensorNames({"image"});
    config->setInputShape(nvinfer1::Dims4{2, 3, 640, 640});
    config->setNumClasses(7);
    config->setOutputTensorNames({"boxes", "logits"});
    config->setRuntime(irt::model::ModelRuntime::parse("onnxruntime:cpu"));

    auto model = irt::model::CreateModel("rfdetr_medium", std::move(config));
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->modelConfig().inputTensorNames(), (std::vector<std::string>{"image"}));
    expectInputShape(*model, 2, 3, 640, 640);
    EXPECT_EQ(model->modelConfig().numClasses(), 7);
    EXPECT_EQ(model->modelConfig().outputTensorNames(), (std::vector<std::string>{"boxes", "logits"}));
    EXPECT_EQ(model->modelConfig().runtime().backend(), irt::model::ModelRuntime::Backend::ONNXRuntime);
}

/**
 * @brief RF-DETR 手写 TensorRT 图入口应拒绝空 network 指针。
 */
TEST(RFDETRModelBuildTest, BuildNetworkRejectsNullNetwork)
{
    auto model = irt::model::CreateModel("rfdetr_nano");
    ASSERT_NE(model, nullptr);

    irt::model::WeightsMap weights;
    ExpectIrtExceptionCode([&] { model->buildNetwork(nullptr, weights); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief RF-DETR 构建期应拒绝不能被 patch_size*num_windows 整除的输入尺寸。
 */
TEST(RFDETRModelBuildTest, BuildRejectsInputShapeNotDivisibleByPatchWindow)
{
    auto model = irt::model::CreateModel("rfdetr_nano");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(nvinfer1::Dims4{1, 3, 385, 384});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_rfdetr_shape_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief RF-DETR 构建期应拒绝非 RGB 输入，避免 patch embedding 权重通道不匹配。
 */
TEST(RFDETRModelBuildTest, BuildRejectsInvalidChannelCount)
{
    auto model = irt::model::CreateModel("rfdetr_nano");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(nvinfer1::Dims4{1, 1, 384, 384});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_rfdetr_channels_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief RF-DETR 检测路径固定导出 dets 和 labels 两个张量。
 */
TEST(RFDETRModelBuildTest, BuildRejectsWrongDetectionOutputTensorCount)
{
    auto model = irt::model::CreateModel("rfdetr_nano");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(nvinfer1::Dims4{1, 3, 384, 384});
    config->setOutputTensorNames({"dets"});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_rfdetr_det_outputs_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief RF-DETR 分割路径固定导出 dets、labels 和 masks 三个张量。
 */
TEST(RFDETRModelBuildTest, BuildRejectsWrongSegmentationOutputTensorCount)
{
    auto model = irt::model::CreateModel("rfdetr_seg_small");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(nvinfer1::Dims4{1, 3, 384, 384});
    config->setOutputTensorNames({"dets", "labels"});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_rfdetr_seg_outputs_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief RF-DETR TensorRT 手写图应允许动态 batch 配置，并继续进入权重校验阶段。
 */
TEST(RFDETRModelBuildTest, DynamicBatchConfigIsAccepted)
{
    auto model = irt::model::CreateModel("rfdetr-nano");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(nvinfer1::Dims4{2, 3, 384, 384});
    config->setDynamicBatchRange(1, 2, 4);
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_rfdetr_dynamic_batch_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}
