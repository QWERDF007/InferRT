#pragma once

/**
 * @file DinoDescriptors.hpp
 * @brief 区域整体描述（多尺度窗口池化）、局部描述（误差受控相邻合并）与 INT8 编码。
 */

#include <inferrt/features/Export.h>

#include "DinoTypes.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace irt::features::priv {

/** @brief 区域窗口枚举配置。 */
struct DinoRegionWindowConfig
{
    std::vector<double> ratios{1.0, 0.5, 0.25};
    double              stride_ratio{0.5};
    double              min_valid_fraction{0.5};
};

/** @brief 描述生成配置。 */
struct DinoDescriptorBuildConfig
{
    DinoRegionWindowConfig window{};
    bool                   merge_enabled{true};
    double                 merge_epsilon{0.10};
    int                    max_leaf_side_patches{4};
};

/** @brief 单个视图的描述生成统计。 */
struct DinoViewDescriptorStats
{
    uint64_t original_patch_count{0}; ///< 视图内全部 patch 数。
    uint64_t valid_patch_count{0};    ///< 有效 patch 数。
    uint64_t region_count{0};
    uint64_t local_count{0};
    float    max_radius{0.0F};
    double   mean_radius{0.0};
};

/** @brief 一个视图的两类描述。 */
struct DinoViewDescriptors
{
    int                              view_id{0};
    std::vector<std::vector<float>>  region_vectors{};
    std::vector<DinoRegionDescriptor> region_meta{};
    std::vector<std::vector<float>>  local_vectors{};
    std::vector<DinoLocalLeaf>       local_meta{};
    DinoViewDescriptorStats          stats{};
};

/** @brief 枚举视图内的区域窗口。 */
INFERRT_FEATURES_API std::vector<DinoRegionDescriptor> dinoEnumerateRegionWindows(const DinoFeatureGrid &grid,
                                                             const DinoRegionWindowConfig &config,
                                                             int view_id);

/** @brief 对窗口执行面积加权池化并 L2 归一化。 */
INFERRT_FEATURES_API std::vector<float> dinoPoolRegionWindow(const DinoFeatureGrid &grid, const DinoRegionDescriptor &window);

/** @brief 生成一个视图的区域与局部描述。 */
INFERRT_FEATURES_API DinoViewDescriptors dinoBuildViewDescriptors(int view_id, const DinoFeatureGrid &grid,
                                             const DinoDescriptorBuildConfig &config);

/** @brief 按 ``s = max|r| / 127`` 量化单位向量，码字取 ties-to-even 舍入。 */
INFERRT_FEATURES_API DinoQuantizedVector dinoQuantizeInt8(const float *values, size_t dimension);

/** @brief 还原量化向量为 ``r_hat = s * z * inv_norm``。 */
INFERRT_FEATURES_API void dinoDequantizeInt8(const DinoQuantizedVector &quantized, float *output);

/** @brief 计算 ``||r - r_hat||2``；输入必须是同一条原始单位向量。 */
INFERRT_FEATURES_API float dinoQuantizationDelta(const float *values, const DinoQuantizedVector &quantized);

/** @brief 归一化一个向量；范数过小或维度为 0 时返回 false。 */
INFERRT_FEATURES_API bool dinoNormalizeVector(float *values, size_t dimension) noexcept;

} // namespace irt::features::priv
