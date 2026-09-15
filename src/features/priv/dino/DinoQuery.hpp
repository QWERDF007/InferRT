#pragma once

/**
 * @file DinoQuery.hpp
 * @brief 查询表示：多视图 ROI 描述与查询局部选择。
 */

#include "DinoBackbone.hpp"
#include "DinoIngest.hpp"
#include "DinoTypes.hpp"
#include "DinoViews.hpp"

#include <inferrt/features/DinoRegionSearch.hpp>

#include <string>
#include <vector>

namespace irt::features::priv {

/** @brief 一条查询局部描述。 */
struct DinoQueryToken
{
    int        cell{0};      ///< 所属格子 ID。
    int        slot{0};      ///< 该 token 在格子内的槽位（0/1）。
    float      weight{0.0F}; ///< 该 token 在格子内的权重。
    DinoPoint  source{};     ///< canonical 空间位置（patch 中心）。
    std::vector<float> vector{}; // original D for fine matching
    std::vector<float> coarse_vector{};
};

/** @brief 一个查询视图的 ROI 表示。 */
struct DinoQueryView
{
    DinoViewPlan              plan{};
    DinoFeatureGrid           grid{};
    DinoRect                  roi_bbox{};
    std::vector<float>        roi_vector{};      ///< ROI 面积加权平均描述。
    std::vector<float>        coarse_roi_vector{};
    std::vector<float>        roi_weights{};     ///< 逐 patch 的 ROI 覆盖权重，供精匹配复用。
    std::vector<DinoQueryToken> tokens{};
    int                       valid_cells{0};
};

/** @brief 查询图与 ROI 的完整表示。 */
struct DinoQuery
{
    DinoRoi roi{};
    std::vector<DinoQueryView> views{};
    int                        cells{4};
    int                        valid_cell_count{0};
    bool                       low_local_evidence{false};
    bool                       outside_validated_profile{false};
    std::string                profile_note{};
    std::string                selection_note{};
};

/**
 * @brief 构造查询表示。
 *
 * 围绕 ROI 中心生成自然上下文视图，每个视图给出一个 ROI 面积加权平均描述与最多
 * ``query_local_max_per_cell`` 个/格的局部描述。有效局部不足时只标记证据不足，不伪造描述。
 */
DinoQuery dinoBuildQuery(const DinoCanonicalImage &image, const DinoRoi &roi, DinoBackbone &backbone,
                         const DinoViewPlanner &planner, const DinoRegionSearchConfig &config,
                         size_t &model_forwards);

} // namespace irt::features::priv
