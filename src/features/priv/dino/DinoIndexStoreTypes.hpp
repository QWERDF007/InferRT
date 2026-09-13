#pragma once

/**
 * @file DinoIndexStoreTypes.hpp
 * @brief 索引视图表与描述区间的存储视图类型。
 */

#include "DinoTypes.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace irt::features::priv {

/** @brief 视图表中一行的解码结果。 */
struct DinoIndexView
{
    int        image_index{-1};  ///< 该视图所属图像在 index.yaml images 中的下标。
    DinoRect   source_rect{};    ///< canonical 空间源区域。
    DinoAffine canonical_to_input{}; ///< canonical px -> 模型输入 px。
    DinoAffine input_to_canonical{}; ///< 模型输入 px -> canonical px。
    int        input_width{0};
    int        input_height{0};
    int        grid_height{0};
    int        grid_width{0};
    int        patch_size{0};
    bool       is_full_view{false};
    int        scale_index{-1};
};

/** @brief 一段连续的描述区间。 */
struct DinoDescriptorRange
{
    size_t begin{0};
    size_t count{0};
};

/**
 * @brief 紧凑描述块：INT8 码 + 每条描述子的还原因子（``scale * inv_norm``）。
 *
 * 局部通道的归约（CPU 与 GPU）都消费这一种形态；需要 FP32 的调用方（区域通道、
 * 窗口重打分）走 ``readRegionVectors`` / ``readLocalVectors``，它们内部由同一读取函数派生，
 * 因此解码逻辑只有一份。
 */
struct DinoCompactBlock
{
    const int8_t *codes{nullptr};   ///< ``count * dimension`` 个 INT8 码。
    const float  *factors{nullptr}; ///< 每条描述子一个还原因子。
    size_t        count{0};
    size_t        dimension{0};
};

/** @brief 紧凑读取的可复用暂存缓冲；避免每次读取都重新分配。 */
struct DinoCompactScratch
{
    std::vector<int8_t> codes{};
    std::vector<float>  factors{};
    std::vector<float>  scales{};
};

/** @brief 把索引视图表的一行还原为可用于坐标计算的视图规划。 */
inline DinoViewPlan dinoPlanFromIndexView(const DinoIndexView &view) noexcept
{
    DinoViewPlan plan;
    plan.source_rect        = view.source_rect;
    plan.canonical_to_input = view.canonical_to_input;
    plan.input_to_canonical = view.input_to_canonical;
    plan.input_width        = view.input_width;
    plan.input_height       = view.input_height;
    plan.patch_size         = view.patch_size;
    plan.grid_height        = view.grid_height;
    plan.grid_width         = view.grid_width;
    plan.is_full_view       = view.is_full_view;
    plan.scale_index        = view.scale_index;
    return plan;
}

/** @brief 把查询视图的 ROI 假设映射到候选窗口内的 canonical 坐标换算。 */
struct DinoWindowHypothesis
{
    double scale{1.0};       ///< 查询 ROI 到候选窗口的统一缩放。
    DinoPoint window_center{}; ///< 候选窗口中心（canonical）。
    DinoPoint roi_center{};    ///< 查询 ROI 中心（canonical）。

    /** @brief 把候选窗口内的 canonical 点换算回查询 ROI 坐标。 */
    DinoPoint toRoiSpace(const double x, const double y) const noexcept
    {
        return DinoPoint{(x - window_center.x) / scale + roi_center.x, (y - window_center.y) / scale + roi_center.y};
    }
};

} // namespace irt::features::priv
