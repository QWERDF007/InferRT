#pragma once

/**
 * @file DinoFineMatch.hpp
 * @brief 候选局部精算：原维度相似度图、连续ROI定位与紧裁DINO复核。
 */

#include "DinoBackbone.hpp"
#include "DinoFeatureCache.hpp"
#include "DinoIndexStore.hpp"
#include "DinoIngest.hpp"
#include "DinoQuery.hpp"
#include "DinoTime.hpp"
#include "DinoViews.hpp"

#include <inferrt/features/DinoRegionSearch.hpp>

#include <vector>

namespace irt::features::priv {

/** @brief 精匹配输出。 */
struct DinoFineMatchOutcome
{
    std::vector<DinoMatchResult> results{};
    std::vector<DinoMatchResult> localized_results{};
    std::vector<DinoMatchResult> verification_input{};
    size_t                       completed_candidates{0};
    size_t                       total_candidates{0};
    size_t                       model_forwards{0};
    bool                         incomplete{false};
    double                       extract_ms{0.0};  ///< 候选解码、光栅渲染与骨干前向耗时。
    double                       match_ms{0.0};    ///< 模板匹配、细化与打分耗时。
};

/** @brief 对全部候选执行精算并返回排序后的结果。 */
DinoFineMatchOutcome dinoFineMatch(const DinoIndexReader &reader, const DinoCanonicalImage &query_image,
                                   const DinoQuery &query, const std::vector<DinoCandidate> &candidates,
                                   const DinoRegionSearchConfig &config, DinoBackbone &backbone,
                                   const DinoViewPlanner &planner, DinoImageCache &image_cache,
                                   DinoFeatureGridCache &feature_cache, const std::string &extractor_signature,
                                   const DinoDeadline &deadline,
                                   const std::function<std::filesystem::path(int64_t)> &image_resolver = {});

} // namespace irt::features::priv
