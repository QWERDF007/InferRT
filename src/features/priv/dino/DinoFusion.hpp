#pragma once

/**
 * @file DinoFusion.hpp
 * @brief 双路候选融合：独立配额、轮流取用、按空间重叠去重。
 */

#include <inferrt/features/Export.h>

#include "DinoTypes.hpp"

#include <inferrt/features/DinoRegionSearch.hpp>

#include <cstddef>
#include <vector>

namespace irt::features::priv {

/** @brief 融合结果与去重统计。 */
struct DinoFusionOutcome
{
    std::vector<DinoCandidate> candidates{};
    size_t region_taken{0};
    size_t local_taken{0};
    size_t duplicates_merged{0};
};

/**
 * @brief 融合两条通道的候选。
 *
 * 每条通道各保留独立额度并轮流取新候选；重复只计入 provenance，不重复消耗额度；
 * 某条通道耗尽时由另一条补足。同一图片的不同位置不会被合并。
 */
INFERRT_FEATURES_API DinoFusionOutcome dinoFuseCandidates(const std::vector<DinoCandidate> &region_candidates,
                                     const std::vector<DinoCandidate> &local_candidates,
                                     const DinoRegionSearchConfig &config);

} // namespace irt::features::priv
