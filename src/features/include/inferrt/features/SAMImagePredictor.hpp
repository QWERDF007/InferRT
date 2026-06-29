#pragma once

/**
 * @file SAMImagePredictor.hpp
 * @brief SAM/SAM2 单图 prompt 分割预测器与 mask 后处理公共 API。
 */

#include <inferrt/features/Export.h>
#include <inferrt/model/IModelConfig.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace irt::features {

inline constexpr const char *kDefaultSAMImagePredictorModelName = "sam2_1_hiera_tiny"; ///< 默认 SAM 预测模型名。
inline constexpr int         kDefaultSAMImagePredictorMaxPoints = 16;                  ///< 默认最大 prompt 点数量。
inline constexpr float       kDefaultSAMMaskThreshold           = 0.0F;                ///< 默认 mask logit 阈值。

/**
 * @brief SAM 图像预处理几何模式。
 */
enum class SAMImageResizeMode
{
    Auto,              ///< 根据模型名自动选择预处理模式。
    ResizeLongestSide, ///< SAM v1/EdgeSAM 模式：最长边缩放、补齐正方形。
    StretchSquare,     ///< SAM2ImagePredictor 模式：输入直接拉伸为正方形。
};

/**
 * @brief prompt 坐标解释方式。
 */
enum class SAMPromptCoordinateMode
{
    ImagePixels,    ///< 坐标为原图像素坐标。
    NormalizedImage ///< 坐标为相对于原图宽高的 [0, 1] 归一化坐标。
};

/**
 * @brief SAM mask token 输出选择模式。
 */
enum class SAMMaskOutputMode
{
    Auto,      ///< 单点提示返回 multimask，多点/box/mask 输入返回 single-mask。
    Single,    ///< 返回官方 `multimask_output=false` 的 single-mask token。
    Multimask, ///< 返回官方 `multimask_output=true` 的 3 个候选 token。
    All,       ///< 返回模型的全部 raw mask token。
};

/**
 * @brief SAM 点 prompt。
 */
struct SAMPromptPoint
{
    float x{0.0F};  ///< 点的 x 坐标。
    float y{0.0F};  ///< 点的 y 坐标。
    int   label{1}; ///< 点标签，1 表示前景，0 表示背景。
};

/**
 * @brief SAM box prompt，坐标为左上和右下边界。
 */
struct SAMPromptBox
{
    float x1{0.0F}; ///< 左上角 x 坐标。
    float y1{0.0F}; ///< 左上角 y 坐标。
    float x2{1.0F}; ///< 右下角 x 坐标。
    float y2{1.0F}; ///< 右下角 y 坐标。
};

/**
 * @brief 一次 SAM 单图预测使用的 prompt 集合。
 */
struct SAMImagePrompt
{
    std::vector<SAMPromptPoint> points;                                                ///< 点 prompt 列表。
    std::optional<SAMPromptBox> box;                                                   ///< 可选 box prompt。
    SAMPromptCoordinateMode     coordinate_mode{SAMPromptCoordinateMode::ImagePixels}; ///< prompt 坐标解释方式。
    std::vector<float>          mask_input; ///< 可选上一轮低分辨率 mask logits，形状为 1xHxW。
};

/**
 * @brief SAM mask 后处理选项。
 */
struct SAMImagePredictOptions
{
    bool              return_logits{false}; ///< 为 true 时返回高分辨率 logits；为 false 时返回阈值化 0/1 mask。
    float             mask_threshold{kDefaultSAMMaskThreshold};  ///< logits 二值化阈值，默认 0。
    int               max_hole_area{0};                          ///< 填充面积不超过该值的低分辨率背景洞；0 表示关闭。
    int               max_sprinkle_area{0};                      ///< 移除面积不超过该值的低分辨率前景噪点；0 表示关闭。
    SAMMaskOutputMode mask_output_mode{SAMMaskOutputMode::Auto}; ///< single/multimask 输出选择策略。
};

/**
 * @brief SAMImagePredictor 配置。
 */
struct SAMImagePredictorConfig
{
    std::string              model_name{kDefaultSAMImagePredictorModelName};    ///< 内置 SAM/SAM2 模型名称。
    irt::model::ModelBackend model_backend{irt::model::ModelBackend::TensorRT}; ///< 模型运行时后端。
    irt::model::ModelDevice  model_device{irt::model::ModelDevice::GPU};        ///< 模型运行设备。
    SAMImageResizeMode       resize_mode{SAMImageResizeMode::Auto};             ///< 图像预处理和 mask 还原几何模式。
};

/**
 * @brief 将低分辨率 mask logits 还原到原图尺寸所需的几何信息。
 */
struct SAMMaskPostprocessGeometry
{
    int                original_width{0};                              ///< 原图宽度。
    int                original_height{0};                             ///< 原图高度。
    int                model_width{0};                                 ///< SAM 模型输入宽度。
    int                model_height{0};                                ///< SAM 模型输入高度。
    int                resized_width{0};                               ///< padding 前或拉伸后的图像宽度。
    int                resized_height{0};                              ///< padding 前或拉伸后的图像高度。
    SAMImageResizeMode resize_mode{SAMImageResizeMode::StretchSquare}; ///< mask 后处理几何模式。
};

/**
 * @brief SAM 单图预测结果。
 */
struct SAMImagePrediction
{
    int  mask_count{0};           ///< mask 数量。
    int  width{0};                ///< 高分辨率输出 mask 宽度，即原图宽度。
    int  height{0};               ///< 高分辨率输出 mask 高度，即原图高度。
    int  low_res_width{0};        ///< 低分辨率 logits 宽度。
    int  low_res_height{0};       ///< 低分辨率 logits 高度。
    bool masks_are_logits{false}; ///< masks 是否为 logits；否则为 0/1 float mask。

    std::vector<float>        masks;           ///< 高分辨率 mask，按 CxHxW 排列；内容由 masks_are_logits 决定。
    std::vector<std::uint8_t> binary_masks;    ///< 高分辨率二值 mask，按 CxHxW 排列，仅阈值化输出时填充。
    std::vector<float>        low_res_masks;   ///< 低分辨率 mask logits，按 CxHxW 排列，并 clamp 到 [-32, 32]。
    std::vector<float>        iou_predictions; ///< 每个 mask 对应的 IoU/质量预测。
};

/**
 * @brief SAM/SAM2 单图 prompt 分割预测器。
 *
 * 该类封装 InferRT SAM 五输入三输出契约，并按模型对应几何集成 mask 后处理。
 */
class INFERRT_FEATURES_API SAMImagePredictor
{
public:
    /**
     * @brief 构造 SAM 单图预测器。
     * @param config 预测器配置。
     */
    explicit SAMImagePredictor(SAMImagePredictorConfig config = {});

    /**
     * @brief 析构预测器并释放模型资源。
     */
    ~SAMImagePredictor();

    SAMImagePredictor(const SAMImagePredictor &)            = delete;
    SAMImagePredictor &operator=(const SAMImagePredictor &) = delete;

    /**
     * @brief 移动构造预测器。
     * @param other 被移动的预测器。
     */
    SAMImagePredictor(SAMImagePredictor &&other) noexcept;

    /**
     * @brief 移动赋值预测器。
     * @param other 被移动的预测器。
     * @return 当前预测器引用。
     */
    SAMImagePredictor &operator=(SAMImagePredictor &&other) noexcept;

    /**
     * @brief 构建或加载 SAM 模型。
     * @param weights_file 权重、engine 或图模型文件路径。
     */
    void load(const std::filesystem::path &weights_file);

    /**
     * @brief 对单张图像执行 prompt 分割预测。
     * @param image_path 输入图像路径。
     * @param prompt 点、框和可选上一轮 mask logits prompt。
     * @param options mask 后处理选项。
     * @return 原图尺寸预测结果和低分辨率 logits。
     */
    SAMImagePrediction predict(const std::filesystem::path &image_path, const SAMImagePrompt &prompt = {},
                               const SAMImagePredictOptions &options = {});

    /**
     * @brief 查询预测器是否已经加载模型。
     * @return 已加载可预测时返回 true。
     */
    bool isReady() const noexcept;

    /**
     * @brief 获取当前预测器配置。
     * @return 配置常量引用。
     */
    const SAMImagePredictorConfig &config() const noexcept;

    /**
     * @brief 对模型输出的低分辨率 mask logits 执行几何一致的 mask 后处理。
     * @param low_res_masks 低分辨率 mask logits，按 CxHxW 排列。
     * @param mask_count mask 数量。
     * @param low_res_height 低分辨率 logits 高度。
     * @param low_res_width 低分辨率 logits 宽度。
     * @param iou_predictions 模型输出的 IoU/质量预测。
     * @param geometry 还原到原图尺寸所需的几何信息。
     * @param options mask 后处理选项。
     * @return 原图尺寸预测结果和 clamp 后的低分辨率 logits。
     */
    static SAMImagePrediction postprocessMasks(const std::vector<float> &low_res_masks, int mask_count,
                                               int low_res_height, int low_res_width,
                                               const std::vector<float>         &iou_predictions,
                                               const SAMMaskPostprocessGeometry &geometry,
                                               const SAMImagePredictOptions     &options = {});

private:
    class Impl;                  ///< 私有实现类型。
    std::unique_ptr<Impl> impl_; ///< 私有实现对象。
};

} // namespace irt::features
