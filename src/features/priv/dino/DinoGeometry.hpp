#pragma once

/**
 * @file DinoGeometry.hpp
 * @brief 区域检索的几何契约：坐标校验、面积、交集、IoU、patch 权重与 NMS。
 */

#include <inferrt/features/Export.h>

#include "DinoTypes.hpp"

#include <inferrt/features/DinoRegionSearch.hpp>

#include <string>
#include <vector>

namespace irt::features::priv {

/** @brief 校验矩形坐标有限且有序；零面积、NaN、Inf 一律拒绝。 */
INFERRT_FEATURES_API void dinoValidateRect(const DinoRect &rect, const char *owner);

/** @brief 校验多边形为无自交简单多边形；失败时返回 false 并写入原因。 */
INFERRT_FEATURES_API bool dinoValidatePolygon(const std::vector<DinoPoint> &polygon, std::string &message);

/** @brief 多边形面积（顶点顺时针或逆时针均可）。 */
INFERRT_FEATURES_API double dinoPolygonArea(const std::vector<DinoPoint> &polygon);

/** @brief 多边形与轴对齐矩形的交集面积。 */
INFERRT_FEATURES_API double dinoPolygonRectArea(const std::vector<DinoPoint> &polygon, const DinoRect &rect);

/** @brief 两个轴对齐矩形的交集面积。 */
INFERRT_FEATURES_API double dinoRectIntersectionArea(const DinoRect &a, const DinoRect &b) noexcept;

/** @brief 交并比；任一矩形为空时返回 0。 */
INFERRT_FEATURES_API double dinoIoU(const DinoRect &a, const DinoRect &b) noexcept;

/** @brief 把公开 ROI 转成内部 ROI，并按 canonical 图像范围做有限夹紧。 */
INFERRT_FEATURES_API DinoRoi dinoToInternalRoi(const DinoSearchRoi &roi, int image_width, int image_height);

/** @brief 返回 ROI 包围盒的长边长度。 */
INFERRT_FEATURES_API double dinoRoiLongSide(const DinoRoi &roi) noexcept;

/** @brief 把矩形夹紧到 ``[0, W) x [0, H)``；不做大幅自动裁剪，仅用于生成输出坐标。 */
INFERRT_FEATURES_API DinoRect dinoClampRect(const DinoRect &rect, int image_width, int image_height) noexcept;

/** @brief 把矩形按比例扩边并夹紧到原图。 */
INFERRT_FEATURES_API DinoRect dinoExpandRect(const DinoRect &rect, double expand_ratio, int image_width, int image_height) noexcept;

/**
 * @brief 计算一个视图内每个 patch 的 ROI 覆盖权重。
 *
 * bbox 使用矩形覆盖面积，polygon 使用 patch 与多边形的相交面积；边界 patch 一律按面积
 * 判定，不按中心点落点。返回长度为 ``grid_height * grid_width`` 的权重数组，取值范围
 * ``[0, 1]``，单位为“占该 patch 的面积比例”。
 */
INFERRT_FEATURES_API std::vector<float> dinoPatchRoiWeights(const DinoViewPlan &plan, const DinoRoi &roi);

/** @brief 标准适用范围，来源于 profile 而非硬编码。 */
struct DinoValidatedRange
{
    int    min_image_edge{256};
    int    max_image_edge{4096};
    double min_target_short_px{64.0};
    double max_target_aspect{4.0};
};

/** @brief 判断 ROI 是否落在标准适用范围内；返回 false 时写入具体原因。 */
INFERRT_FEATURES_API bool dinoWithinValidatedProfile(const DinoRoi &roi, int image_width, int image_height,
                                const DinoValidatedRange &range, std::string &message);

/** @brief 按同图 IoU 阈值做最终框非极大值抑制，保留分数更高的框。 */
INFERRT_FEATURES_API void dinoNmsWithinImages(std::vector<DinoMatchResult> &results, double iou_threshold);

} // namespace irt::features::priv
