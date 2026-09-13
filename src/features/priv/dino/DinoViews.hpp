#pragma once

/**
 * @file DinoViews.hpp
 * @brief 多尺度图库视图与查询上下文视图的几何规划与光栅渲染。
 *
 * 视图几何是索引签名的一部分：``DinoViewPlan`` 保存 canonical -> 输入光栅的仿射变换、
 * 有效像素矩形与 patch 网格，后续所有坐标都从该变换还原，不做整数取整累加。
 */

#include <inferrt/features/Export.h>

#include "DinoTypes.hpp"

#include <inferrt/core/PreprocessSpec.hpp>

#include <opencv2/core.hpp>

#include <vector>

namespace irt::features::priv {

/** @brief 一个视图的模型输入光栅与几何规划。 */
struct DinoViewRaster
{
    DinoViewPlan       plan{};
    std::vector<float> chw{}; ///< ``3 x edge x edge`` 归一化输入，padding 为 0。
};

/** @brief 视图规划器：按 profile 生成图库视图与查询上下文视图。 */
class INFERRT_FEATURES_API DinoViewPlanner
{
public:
    DinoViewPlanner(int patch_size, int encoder_edge, double view_overlap, std::vector<int> tile_edges,
                    std::vector<double> roi_target_lengths);

    int patchSize() const noexcept
    {
        return patch_size_;
    }

    int encoderEdge() const noexcept
    {
        return encoder_edge_;
    }

    /** @brief 从任意 canonical 源矩形构造视图规划；源矩形可超出图像范围。 */
    DinoViewPlan makePlan(DinoRect source_rect, int image_width, int image_height, bool is_full_view,
                          int scale_index) const;

    /** @brief 规划图库视图：整图视图 + 各尺度重叠切片，重复视图去重。 */
    std::vector<DinoViewPlan> planGalleryViews(int image_width, int image_height) const;

    /** @brief 规划查询上下文视图：让 ROI 长边在输入中占不同长度。 */
    std::vector<DinoViewPlan> planQueryViews(const DinoRoi &roi, int image_width, int image_height) const;

private:
    int                 patch_size_{0};
    int                 encoder_edge_{0};
    double              view_overlap_{0.25};
    std::vector<int>    tile_edges_{};
    std::vector<double> roi_target_lengths_{};
};

/** @brief 返回视图归一化使用的预处理规格（颜色顺序、scale、mean/std 的唯一来源）。 */
INFERRT_FEATURES_API irt::PreprocessSpec dinoViewPreprocessSpec(int encoder_edge, int patch_size);

/** @brief 计算每个 patch 的有效像素面积占比，padding 与越界区域不计入。 */
INFERRT_FEATURES_API std::vector<float> dinoPatchValidArea(const DinoViewPlan &plan);

/** @brief 按视图几何从 canonical 图像渲染模型输入光栅。 */
INFERRT_FEATURES_API DinoViewRaster dinoRenderView(const cv::Mat &canonical_bgr, const DinoViewPlan &plan,
                             const irt::PreprocessSpec &spec);

} // namespace irt::features::priv
