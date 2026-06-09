#pragma once

#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.hpp>
#include <inferrt/model/IModel.h>

#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace test::model {

/**
 * @brief 已注册模型的测试样例。
 *
 * `key` 为注册表中的标准 key，`mixed_case_key` 用于验证大小写不敏感查找，
 * `display_name` 则对应模型实例 `name()` 的期望返回值。
 */
struct RegisteredModelCase
{
    const char *key;
    const char *mixed_case_key;
    const char *display_name;
};

/**
 * @brief 当前仓库中内置注册的模型清单。
 */
inline constexpr std::array<RegisteredModelCase, 132> kRegisteredModels = {
    {
     {"onnx", "ONNX", "ONNX"},
     {"alexnet", "AlexNet", "AlexNet"},
     {"googlenet", "GoogLeNet", "GoogLeNet"},
     {"mobilenet_v2", "MobileNet_V2", "MobileNetV2"},
     {"mobilenet_v3_large", "MobileNet_V3_Large", "MobileNetV3Large"},
     {"mobilenet_v3_small", "MobileNet_V3_Small", "MobileNetV3Small"},
     {"vgg11", "VGG11", "VGG11"},
     {"vgg13", "VGG13", "VGG13"},
     {"vgg16", "VGG16", "VGG16"},
     {"vgg19", "VGG19", "VGG19"},
     {"resnet18", "ResNet18", "ResNet18"},
     {"resnet34", "ResNet34", "ResNet34"},
     {"resnet50", "ResNet50", "ResNet50"},
     {"resnet101", "ResNet101", "ResNet101"},
     {"resnet152", "ResNet152", "ResNet152"},
     {"wide_resnet50_2", "Wide_ResNet50_2", "WideResNet50_2"},
     {"wide_resnet101_2", "Wide_ResNet101_2", "WideResNet101_2"},
     {"vit", "ViT", "ViTBasePatch16_224"},
     {"vit_tiny_patch16_224", "ViT_Tiny_Patch16_224", "ViTTinyPatch16_224"},
     {"vit_tiny_patch16_384", "ViT_Tiny_Patch16_384", "ViTTinyPatch16_384"},
     {"vit_small_patch32_224", "ViT_Small_Patch32_224", "ViTSmallPatch32_224"},
     {"vit_small_patch32_384", "ViT_Small_Patch32_384", "ViTSmallPatch32_384"},
     {"vit_small_patch16_224", "ViT_Small_Patch16_224", "ViTSmallPatch16_224"},
     {"vit_small_patch16_384", "ViT_Small_Patch16_384", "ViTSmallPatch16_384"},
     {"vit_small_patch8_224", "ViT_Small_Patch8_224", "ViTSmallPatch8_224"},
     {"vit_base_patch32_224", "ViT_Base_Patch32_224", "ViTBasePatch32_224"},
     {"vit_base_patch32_384", "ViT_Base_Patch32_384", "ViTBasePatch32_384"},
     {"vit_base_patch16_224", "ViT_Base_Patch16_224", "ViTBasePatch16_224"},
     {"vit_base_patch16_384", "ViT_Base_Patch16_384", "ViTBasePatch16_384"},
     {"vit_base_patch8_224", "ViT_Base_Patch8_224", "ViTBasePatch8_224"},
     {"vit_large_patch32_224", "ViT_Large_Patch32_224", "ViTLargePatch32_224"},
     {"vit_large_patch32_384", "ViT_Large_Patch32_384", "ViTLargePatch32_384"},
     {"vit_large_patch16_224", "ViT_Large_Patch16_224", "ViTLargePatch16_224"},
     {"vit_large_patch16_384", "ViT_Large_Patch16_384", "ViTLargePatch16_384"},
     {"vit_large_patch14_224", "ViT_Large_Patch14_224", "ViTLargePatch14_224"},
     {"vit_huge_patch14_224", "ViT_Huge_Patch14_224", "ViTHugePatch14_224"},
     {"vit_giant_patch14_224", "ViT_Giant_Patch14_224", "ViTGiantPatch14_224"},
     {"vit_gigantic_patch14_224", "ViT_Gigantic_Patch14_224", "ViTGiganticPatch14_224"},
     {"dinov2_vits14", "DINOv2_ViTS14", "DINOv2ViTS14"},
     {"dinov2_vitb14", "DINOv2_ViTB14", "DINOv2ViTB14"},
     {"dinov2_vitl14", "DINOv2_ViTL14", "DINOv2ViTL14"},
     {"dinov2_vitg14", "DINOv2_ViTG14", "DINOv2ViTG14"},
     {"dinov2_vits14_reg", "DINOv2_ViTS14_Reg", "DINOv2ViTS14Reg4"},
     {"dinov2_vitb14_reg", "DINOv2_ViTB14_Reg", "DINOv2ViTB14Reg4"},
     {"dinov2_vitl14_reg", "DINOv2_ViTL14_Reg", "DINOv2ViTL14Reg4"},
     {"dinov2_vitg14_reg", "DINOv2_ViTG14_Reg", "DINOv2ViTG14Reg4"},
     {"dinov2_vits14_reg4", "DINOv2_ViTS14_Reg4", "DINOv2ViTS14Reg4"},
     {"dinov2_vitb14_reg4", "DINOv2_ViTB14_Reg4", "DINOv2ViTB14Reg4"},
     {"dinov2_vitl14_reg4", "DINOv2_ViTL14_Reg4", "DINOv2ViTL14Reg4"},
     {"dinov2_vitg14_reg4", "DINOv2_ViTG14_Reg4", "DINOv2ViTG14Reg4"},
     {"vit_small_patch14_dinov2", "ViT_Small_Patch14_DINOv2", "DINOv2ViTS14"},
     {"vit_base_patch14_dinov2", "ViT_Base_Patch14_DINOv2", "DINOv2ViTB14"},
     {"vit_large_patch14_dinov2", "ViT_Large_Patch14_DINOv2", "DINOv2ViTL14"},
     {"vit_giant_patch14_dinov2", "ViT_Giant_Patch14_DINOv2", "DINOv2ViTG14"},
     {"vit_small_patch14_reg4_dinov2", "ViT_Small_Patch14_Reg4_DINOv2", "DINOv2ViTS14Reg4"},
     {"vit_base_patch14_reg4_dinov2", "ViT_Base_Patch14_Reg4_DINOv2", "DINOv2ViTB14Reg4"},
     {"vit_large_patch14_reg4_dinov2", "ViT_Large_Patch14_Reg4_DINOv2", "DINOv2ViTL14Reg4"},
     {"vit_giant_patch14_reg4_dinov2", "ViT_Giant_Patch14_Reg4_DINOv2", "DINOv2ViTG14Reg4"},
     {"dinov3_vits16", "DINOv3_ViTS16", "DINOv3ViTS16"},
     {"dinov3_vits16plus", "DINOv3_ViTS16Plus", "DINOv3ViTS16Plus"},
     {"dinov3_vitb16", "DINOv3_ViTB16", "DINOv3ViTB16"},
     {"dinov3_vitl16", "DINOv3_ViTL16", "DINOv3ViTL16"},
     {"dinov3_vitl16plus", "DINOv3_ViTL16Plus", "DINOv3ViTL16Plus"},
     {"dinov3_vith16plus", "DINOv3_ViTH16Plus", "DINOv3ViTH16Plus"},
     {"dinov3_vit7b16", "DINOv3_ViT7B16", "DINOv3ViT7B16"},
     {"vit_small_patch16_dinov3", "ViT_Small_Patch16_DINOv3", "DINOv3ViTS16"},
     {"vit_small_patch16_dinov3_qkvb", "ViT_Small_Patch16_DINOv3_QKVB", "DINOv3ViTS16"},
     {"vit_small_plus_patch16_dinov3", "ViT_Small_Plus_Patch16_DINOv3", "DINOv3ViTS16Plus"},
     {"vit_small_plus_patch16_dinov3_qkvb", "ViT_Small_Plus_Patch16_DINOv3_QKVB", "DINOv3ViTS16Plus"},
     {"vit_base_patch16_dinov3", "ViT_Base_Patch16_DINOv3", "DINOv3ViTB16"},
     {"vit_base_patch16_dinov3_qkvb", "ViT_Base_Patch16_DINOv3_QKVB", "DINOv3ViTB16"},
     {"vit_large_patch16_dinov3", "ViT_Large_Patch16_DINOv3", "DINOv3ViTL16"},
     {"vit_large_patch16_dinov3_qkvb", "ViT_Large_Patch16_DINOv3_QKVB", "DINOv3ViTL16"},
     {"vit_huge_plus_patch16_dinov3", "ViT_Huge_Plus_Patch16_DINOv3", "DINOv3ViTH16Plus"},
     {"vit_huge_plus_patch16_dinov3_qkvb", "ViT_Huge_Plus_Patch16_DINOv3_QKVB", "DINOv3ViTH16Plus"},
     {"vit_7b_patch16_dinov3", "ViT_7B_Patch16_DINOv3", "DINOv3ViT7B16"},
     {"rfdetr_base", "RFDETR_Base", "RFDETRBase"},
     {"rfdetr-base", "RFDETR-Base", "RFDETRBase"},
     {"rfdetr_nano", "RFDETR_Nano", "RFDETRNano"},
     {"rfdetr-nano", "RFDETR-Nano", "RFDETRNano"},
     {"rfdetr_small", "RFDETR_Small", "RFDETRSmall"},
     {"rfdetr-small", "RFDETR-Small", "RFDETRSmall"},
     {"rfdetr_medium", "RFDETR_Medium", "RFDETRMedium"},
     {"rfdetr-medium", "RFDETR-Medium", "RFDETRMedium"},
     {"rfdetr_large", "RFDETR_Large", "RFDETRLarge"},
     {"rfdetr-large", "RFDETR-Large", "RFDETRLarge"},
     {"rfdetr_large_deprecated", "RFDETR_Large_Deprecated", "RFDETRLargeDeprecated"},
     {"rfdetr-large-deprecated", "RFDETR-Large-Deprecated", "RFDETRLargeDeprecated"},
     {"rfdetr_seg_preview", "RFDETR_Seg_Preview", "RFDETRSegPreview"},
     {"rfdetr-seg-preview", "RFDETR-Seg-Preview", "RFDETRSegPreview"},
     {"rfdetr_seg_nano", "RFDETR_Seg_Nano", "RFDETRSegNano"},
     {"rfdetr-seg-nano", "RFDETR-Seg-Nano", "RFDETRSegNano"},
     {"rfdetr_seg_small", "RFDETR_Seg_Small", "RFDETRSegSmall"},
     {"rfdetr-seg-small", "RFDETR-Seg-Small", "RFDETRSegSmall"},
     {"rfdetr_seg_medium", "RFDETR_Seg_Medium", "RFDETRSegMedium"},
     {"rfdetr-seg-medium", "RFDETR-Seg-Medium", "RFDETRSegMedium"},
     {"rfdetr_seg_large", "RFDETR_Seg_Large", "RFDETRSegLarge"},
     {"rfdetr-seg-large", "RFDETR-Seg-Large", "RFDETRSegLarge"},
     {"rfdetr_seg_xlarge", "RFDETR_Seg_XLarge", "RFDETRSegXLarge"},
     {"rfdetr-seg-xlarge", "RFDETR-Seg-XLarge", "RFDETRSegXLarge"},
     {"rfdetr_seg_2xlarge", "RFDETR_Seg_2XLarge", "RFDETRSeg2XLarge"},
     {"rfdetr-seg-2xlarge", "RFDETR-Seg-2XLarge", "RFDETRSeg2XLarge"},
     {"rfdetr_seg_xxlarge", "RFDETR_Seg_XXLarge", "RFDETRSeg2XLarge"},
     {"rfdetr-seg-xxlarge", "RFDETR-Seg-XXLarge", "RFDETRSeg2XLarge"},
     {"sam", "SAM", "SAMViTH"},
     {"sam_vit_b", "SAM_ViT_B", "SAMViTB"},
     {"sam_vit_l", "SAM_ViT_L", "SAMViTL"},
     {"sam_vit_h", "SAM_ViT_H", "SAMViTH"},
     {"edge_sam", "Edge_SAM", "EdgeSAM"},
     {"sam2", "SAM2", "SAM2HieraLarge"},
     {"sam2_hiera_tiny", "SAM2_Hiera_Tiny", "SAM2HieraTiny"},
     {"sam2_hiera_small", "SAM2_Hiera_Small", "SAM2HieraSmall"},
     {"sam2_hiera_base_plus", "SAM2_Hiera_Base_Plus", "SAM2HieraBasePlus"},
     {"sam2_hiera_large", "SAM2_Hiera_Large", "SAM2HieraLarge"},
     {"sam2_1_hiera_tiny", "SAM2_1_Hiera_Tiny", "SAM2.1HieraTiny"},
     {"sam2_1_hiera_small", "SAM2_1_Hiera_Small", "SAM2.1HieraSmall"},
     {"sam2_1_hiera_base_plus", "SAM2_1_Hiera_Base_Plus", "SAM2.1HieraBasePlus"},
     {"sam2_1_hiera_large", "SAM2_1_Hiera_Large", "SAM2.1HieraLarge"},
     {"sam3", "SAM3", "SAM3Image"},
     {"sam3_image", "SAM3_Image", "SAM3Image"},
     {"yolov5", "YOLOv5", "YOLOv5s"},
     {"yolov5n", "YOLOv5N", "YOLOv5n"},
     {"yolov5s", "YOLOv5S", "YOLOv5s"},
     {"yolov5m", "YOLOv5M", "YOLOv5m"},
     {"yolov5l", "YOLOv5L", "YOLOv5l"},
     {"yolov5x", "YOLOv5X", "YOLOv5x"},
     {"yolov8", "YOLOv8", "YOLOv8n"},
     {"yolov8n", "YOLOv8N", "YOLOv8n"},
     {"yolov8s", "YOLOv8S", "YOLOv8s"},
     {"yolov8m", "YOLOv8M", "YOLOv8m"},
     {"yolov8l", "YOLOv8L", "YOLOv8l"},
     {"yolov8x", "YOLOv8X", "YOLOv8x"},
     }
};

/**
 * @brief 面向全部已注册模型的参数化测试基类。
 */
class RegisteredModelsTest : public ::testing::TestWithParam<RegisteredModelCase>
{
};

/**
 * @brief 构造用于未初始化 runtime 测试的空 buffer 列表。
 * @param count buffer 数量。
 * @return 指针均为空的 buffer 列表。
 */
inline std::vector<void *> MakeNullBuffers(size_t count)
{
    return std::vector<void *>(count, nullptr);
}

/**
 * @brief 最小 `.wts` 临时文件，用于触发模型构建期的配置校验。
 *
 * 文件只包含一个占位权重；输入尺寸、batch 或通道数等错误会在访问完整权重前抛出，
 * 因此可复用于 ViT、DINO 等手写 TensorRT 网络测试。
 */
class TempWeightsFile
{
public:
    /**
     * @brief 创建一个合法但不完整的 `.wts` 文件。
     */
    explicit TempWeightsFile(std::string prefix = "inferrt_model_test_")
    {
        static std::atomic<int> counter{0};
        path_ = std::filesystem::temp_directory_path()
              / std::filesystem::path(std::move(prefix)
                                      + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) + ".wts");
        std::ofstream(path_) << "1\nplaceholder 0\n";
    }

    /**
     * @brief 析构时删除临时权重文件。
     */
    ~TempWeightsFile()
    {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    /**
     * @brief 获取临时权重文件路径。
     */
    const std::filesystem::path &path() const noexcept
    {
        return path_;
    }

    TempWeightsFile(const TempWeightsFile &)            = delete;
    TempWeightsFile &operator=(const TempWeightsFile &) = delete;

private:
    std::filesystem::path path_; ///< 临时 `.wts` 文件路径。
};

/**
 * @brief 断言指定调用抛出 InferRT 异常且错误码符合预期。
 *
 * 该辅助函数用于减少测试中重复的 `try/catch` 样板代码，同时保留错误码校验。
 *
 * @tparam Fn 可调用对象类型。
 * @param fn 待执行的调用。
 * @param expected_code 期望的 InferRT 错误码。
 */
template<typename Fn>
void ExpectIrtExceptionCode(Fn &&fn, irt::Status expected_code)
{
    try
    {
        std::forward<Fn>(fn)();
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), expected_code);
    }
}

} // namespace test::model
