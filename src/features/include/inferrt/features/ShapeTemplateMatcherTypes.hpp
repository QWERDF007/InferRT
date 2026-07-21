#pragma once

/**
 * @file ShapeTemplateMatcherTypes.hpp
 * @brief 形状模板匹配各版本共享的数据类型。
 */

#include <inferrt/features/Export.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <filesystem>
#include <vector>

namespace irt::features {

inline constexpr int   kDefaultShapeTemplateNumFeatures     = 128;   ///< 默认每个模板最多保留的特征点数。
inline constexpr int   kDefaultShapeTemplateMinFeatures     = 4;     ///< 默认每个模板至少需要的有效特征点数。
inline constexpr float kDefaultShapeTemplateWeakThreshold   = 30.0f; ///< 默认源图梯度弱阈值。
inline constexpr float kDefaultShapeTemplateStrongThreshold = 60.0f; ///< 默认模板候选特征强阈值。
inline constexpr float kDefaultShapeTemplateMatchThreshold  = 80.0f; ///< 默认匹配分数阈值。
inline constexpr float kDefaultShapeTemplateNmsThreshold    = 0.3f;  ///< 默认 NMS IoU 阈值。

/** @brief 形状模板匹配器配置。 */
struct ShapeTemplateMatcherConfig
{
    int   num_features{kDefaultShapeTemplateNumFeatures};        ///< 每个模板最多保留的梯度方向特征点数。
    int   min_features{kDefaultShapeTemplateMinFeatures};        ///< 模板训练成功所需的最少有效特征点数。
    float weak_threshold{kDefaultShapeTemplateWeakThreshold};     ///< 源图量化梯度方向时使用的幅值阈值。
    float strong_threshold{kDefaultShapeTemplateStrongThreshold}; ///< 模板候选特征点的梯度幅值阈值。
    int   max_label_difference{1};                               ///< 环形方向 bin 容差，合法范围为 ``[0, 4]``。
    float match_threshold{kDefaultShapeTemplateMatchThreshold};   ///< 默认匹配分数阈值，合法范围为 ``[0, 100]``。
    float nms_threshold{kDefaultShapeTemplateNmsThreshold};       ///< NMS 的 IoU 阈值；小于 0 时关闭 NMS。
    int   max_results{0};                                        ///< 最多返回的匹配数量；0 表示不限制。
    int   scan_step{1};                                          ///< 滑窗扫描步长，单位为像素。
    int   max_parallelism{0};                                    ///< 模板扫描工作线程数；0 表示自动，1 表示串行。
    int   max_training_parallelism{0};                           ///< v1/v2 模板训练工作线程数；0 表示自动，1 表示串行；v0 始终使用原始串行路径；该运行时项不写入模板文件。
    float min_feature_distance{0.0f};                            ///< 贪心选点最小间距；0 表示按模板面积自动估计。
};

/**
 * @brief 单次匹配的运行时策略。
 *
 * 默认值会完整扫描模板和保存的空间扫描网格，因此与未传入该对象时的结果严格一致。
 * 非默认值用于在已知场景可接受召回率或定位精度损失时缩短匹配时间；它们不会写入模板文件。
 */
struct ShapeTemplateMatchOptions
{
    int template_stride{1}; ///< 每隔多少个模板变体扫描一次；1 表示扫描全部模板，``> 1`` 为近似模式。
    int scan_step{0};       ///< 0 表示使用模板配置中的 ``scan_step``；正数覆盖它，较大值为近似空间搜索。
};

/** @brief 生成旋转/缩放模板变体时使用的元数据。 */
struct ShapeTemplateVariant
{
    float angle_degrees{0.0f}; ///< 训练图像逆时针旋转角度，单位为度。
    float scale{1.0f};         ///< 训练图像缩放倍率。
};

/**
 * @brief 多模板变体训练中的一个输入项。
 *
 * @details 每个输入项可使用不同的图像和目标掩膜，但同一次
 * ``addTemplateVariantsBatch()`` 调用中的所有输入项共用一组角度/尺度变体。图像和掩膜只在
 * 调用期间被只读访问；非空掩膜必须与图像尺寸一致。
 */
struct ShapeTemplateTrainingInput
{
    cv::Mat image;       ///< 待训练的模板图像或从大图裁剪出的 ROI。
    cv::Mat object_mask; ///< 可选目标掩膜；为空时整张输入图有效。
};

/** @brief 模板中的一个量化梯度方向特征点。 */
struct ShapeTemplateFeature
{
    int   x{0};                ///< 相对模板左上角的 x 坐标。
    int   y{0};                ///< 相对模板左上角的 y 坐标。
    int   label{0};            ///< 量化方向标签，范围为 ``[0, 7]``。
    float angle_degrees{0.0f}; ///< 原始梯度方向角，单位为度。
};

/** @brief 已训练模板的元数据和选中的特征点集合。 */
struct ShapeTemplateInfo
{
    int   template_id{-1};     ///< 模板文件内的全局模板 ID。
    int   width{0};            ///< 裁剪后的模板特征包围盒宽度。
    int   height{0};           ///< 裁剪后的模板特征包围盒高度。
    int   tl_x{0};             ///< 特征点包围盒在训练图像中的左上角 x 坐标。
    int   tl_y{0};             ///< 特征点包围盒在训练图像中的左上角 y 坐标。
    float angle_degrees{0.0f}; ///< 模板变体对应的旋转角度元数据。
    float scale{1.0f};         ///< 模板变体对应的缩放倍率元数据。

    std::vector<ShapeTemplateFeature> features; ///< 模板保留的稀疏梯度方向特征点。
};

/** @brief 一个形状模板匹配结果。 */
struct ShapeTemplateMatch
{
    int   x{0};                ///< 源图坐标系下匹配框左上角 x 坐标。
    int   y{0};                ///< 源图坐标系下匹配框左上角 y 坐标。
    int   width{0};            ///< 匹配模板宽度。
    int   height{0};           ///< 匹配模板高度。
    float similarity{0.0f};    ///< 梯度方向一致性分数，范围为 ``[0, 100]``。
    int   template_id{-1};     ///< 模板文件内的全局模板 ID。
    float angle_degrees{0.0f}; ///< 命中模板的训练变体元数据。
    float scale{1.0f};         ///< 命中模板的训练变体元数据。
};

/** @brief 内置形状模板匹配实现版本。 */
enum class ShapeTemplateMatcherVersion
{
    V0, ///< 原始标量实现；训练与匹配均不使用 SIMD 优化。
    V1, ///< AVX2 加速匹配与严格等价的优化训练实现。
    V2, ///< AVX512F/BW 加速匹配与严格等价的优化训练实现。
};

} // namespace irt::features
