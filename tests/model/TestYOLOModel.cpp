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
    EXPECT_EQ(shape.d[0], n);
    EXPECT_EQ(shape.d[1], c);
    EXPECT_EQ(shape.d[2], h);
    EXPECT_EQ(shape.d[3], w);
}

/**
 * @brief 断言 YOLO 默认检测输出张量名称。
 */
void expectDefaultYoloOutputs(const irt::model::IModel &model)
{
    EXPECT_EQ(model.modelConfig().outputTensorNames(), (std::vector<std::string>{"output0", "output1", "output2"}));
}

/**
 * @brief 断言 YOLOv8-Seg 默认分割输出张量名称。
 */
void expectDefaultYoloSegOutputs(const irt::model::IModel &model)
{
    EXPECT_EQ(model.modelConfig().outputTensorNames(),
              (std::vector<std::string>{"output0", "output1", "output2", "proto"}));
}

} // namespace

/**
 * @brief YOLOv5 兼容别名应创建 YOLOv5s 检测模型。
 */
TEST(YOLOModelFactoryTest, YOLOv5AliasCreatesSmallDetector)
{
    auto model = irt::model::CreateModel("yolov5");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->name(), "YOLOv5s");
    expectInputShape(*model, 1, 3, 640, 640);
    EXPECT_EQ(model->modelConfig().numClasses(), 80);
    expectDefaultYoloOutputs(*model);
}

/**
 * @brief YOLOv8-Seg 兼容别名应创建常用的 YOLOv8n 分割模型。
 */
TEST(YOLOModelFactoryTest, YOLOv8SegAliasCreatesNanoSegmenter)
{
    auto model = irt::model::CreateModel("yolov8_seg");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->name(), "YOLOv8nSeg");
    expectInputShape(*model, 1, 3, 640, 640);
    EXPECT_EQ(model->modelConfig().numClasses(), 80);
    expectDefaultYoloSegOutputs(*model);

    auto hyphen = irt::model::CreateModel("yolov8n-seg");
    ASSERT_NE(hyphen, nullptr);
    EXPECT_EQ(hyphen->name(), "YOLOv8nSeg");
    expectDefaultYoloSegOutputs(*hyphen);
}

/**
 * @brief YOLOv8 兼容别名应创建常用的 YOLOv8n 检测模型。
 */
TEST(YOLOModelFactoryTest, YOLOv8AliasCreatesNanoDetector)
{
    auto model = irt::model::CreateModel("yolov8");
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->name(), "YOLOv8n");
    expectInputShape(*model, 1, 3, 640, 640);
    EXPECT_EQ(model->modelConfig().numClasses(), 80);
    expectDefaultYoloOutputs(*model);
}

/**
 * @brief YOLOv5/YOLOv8 的各尺寸 key 应注册到统一模型工厂。
 */
TEST(YOLOModelFactoryTest, RegistersScaledDetectorVariants)
{
    const std::vector<std::pair<std::string, std::string>> cases{
        {    "yolov5n",    "YOLOv5n"},
        {    "yolov5s",    "YOLOv5s"},
        {    "yolov5m",    "YOLOv5m"},
        {    "yolov5l",    "YOLOv5l"},
        {    "yolov5x",    "YOLOv5x"},
        {    "yolov8n",    "YOLOv8n"},
        {    "yolov8s",    "YOLOv8s"},
        {    "yolov8m",    "YOLOv8m"},
        {    "yolov8l",    "YOLOv8l"},
        {    "yolov8x",    "YOLOv8x"},
        {"yolov8n_seg", "YOLOv8nSeg"},
        {"yolov8s_seg", "YOLOv8sSeg"},
        {"yolov8m_seg", "YOLOv8mSeg"},
        {"yolov8l_seg", "YOLOv8lSeg"},
        {"yolov8x_seg", "YOLOv8xSeg"},
        {"yolov8n-seg", "YOLOv8nSeg"},
        {"yolov8s-seg", "YOLOv8sSeg"},
        {"yolov8m-seg", "YOLOv8mSeg"},
        {"yolov8l-seg", "YOLOv8lSeg"},
        {"yolov8x-seg", "YOLOv8xSeg"},
    };

    for (const auto &[key, display_name] : cases)
    {
        auto model = irt::model::CreateModel(key);
        ASSERT_NE(model, nullptr) << key;
        EXPECT_EQ(model->name(), display_name) << key;
    }
}

/**
 * @brief YOLO 默认配置规整不应覆盖调用方显式传入的输入、类别数和输出名。
 */
TEST(YOLOModelConfigTest, PreservesExplicitDetectionConfig)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputTensorNames({"images"});
    config->setInputShape(nvinfer1::Dims4{1, 3, 320, 320});
    config->setNumClasses(3);
    config->setOutputTensorNames({"p3", "p4", "p5"});

    auto model = irt::model::CreateModel("yolov8s", std::move(config));
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->modelConfig().inputTensorNames(), (std::vector<std::string>{"images"}));
    expectInputShape(*model, 1, 3, 320, 320);
    EXPECT_EQ(model->modelConfig().numClasses(), 3);
    EXPECT_EQ(model->modelConfig().outputTensorNames(), (std::vector<std::string>{"p3", "p4", "p5"}));
}

/**
 * @brief YOLOv8-Seg 显式配置不应被默认分割配置覆盖。
 */
TEST(YOLOModelConfigTest, PreservesExplicitSegmentationConfig)
{
    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputTensorNames({"images"});
    config->setInputShape(nvinfer1::Dims4{1, 3, 320, 320});
    config->setNumClasses(3);
    config->setOutputTensorNames({"p3", "p4", "p5", "proto"});

    auto model = irt::model::CreateModel("yolov8s_seg", std::move(config));
    ASSERT_NE(model, nullptr);

    EXPECT_EQ(model->modelConfig().inputTensorNames(), (std::vector<std::string>{"images"}));
    expectInputShape(*model, 1, 3, 320, 320);
    EXPECT_EQ(model->modelConfig().numClasses(), 3);
    EXPECT_EQ(model->modelConfig().outputTensorNames(), (std::vector<std::string>{"p3", "p4", "p5", "proto"}));
}

/**
 * @brief YOLO 构建期应拒绝不能被 32 整除的输入尺寸。
 */
TEST(YOLOModelBuildTest, BuildRejectsInputShapeNotDivisibleByStride)
{
    auto model = irt::model::CreateModel("yolov5n");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(nvinfer1::Dims4{1, 3, 641, 640});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_yolov5_test_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief YOLO 构建期应拒绝非 RGB 输入通道，避免卷积权重形状错误。
 */
TEST(YOLOModelBuildTest, BuildRejectsInvalidChannelCount)
{
    auto model = irt::model::CreateModel("yolov8n");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(nvinfer1::Dims4{1, 1, 640, 640});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_yolov8_test_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief YOLO 检测模型只支持单输入，避免多输入配置进入 TensorRT 构图后才暴露错误。
 */
TEST(YOLOModelBuildTest, BuildRejectsMultipleInputShapes)
{
    auto model = irt::model::CreateModel("yolov8n");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShapes({
        nvinfer1::Dims4{1, 3, 640, 640},
        nvinfer1::Dims4{1, 3, 640, 640}
    });
    config->setOutputTensorNames({"output0", "output1", "output2"});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_yolov8_multi_input_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief YOLO 检测头固定导出 3 个尺度输出，输出名数量错误时应在构建前期拒绝。
 */
TEST(YOLOModelBuildTest, BuildRejectsWrongOutputTensorCount)
{
    auto model = irt::model::CreateModel("yolov5n");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(nvinfer1::Dims4{1, 3, 640, 640});
    config->setOutputTensorNames({"output0", "output1"});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_yolov5_outputs_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief YOLOv8-Seg 固定导出 3 个检测分支和 1 个 proto，输出名数量错误时应拒绝构建。
 */
TEST(YOLOModelBuildTest, BuildRejectsWrongSegmentationOutputTensorCount)
{
    auto model = irt::model::CreateModel("yolov8n_seg");
    ASSERT_NE(model, nullptr);

    auto config = std::make_unique<irt::model::IModelConfig>();
    config->setInputShape(nvinfer1::Dims4{1, 3, 640, 640});
    config->setOutputTensorNames({"output0", "output1", "output2"});
    model->setModelConfig(std::move(config));

    const TempWeightsFile weights("inferrt_yolov8_seg_outputs_");
    ExpectIrtExceptionCode([&] { model->build(weights.path().string()); }, irt::Status::ERROR_INVALID_ARGUMENT);
}
