/**
 * @file DinoFusion.cpp
 * @brief 双路候选融合实现。
 */

#include "DinoFusion.hpp"
#include "DinoGeometry.hpp"

#include <algorithm>

namespace irt::features::priv {

namespace {

double areaRatio(const DinoRect &a, const DinoRect &b) noexcept
{
    const double area_a = a.area();
    const double area_b = b.area();
    if (area_a <= 0.0 || area_b <= 0.0)
    {
        return 0.0;
    }
    return area_a > area_b ? area_a / area_b : area_b / area_a;
}

/** @brief 找到可合并的既有候选；不存在时返回 -1。 */
int findDuplicate(const std::vector<DinoCandidate> &accepted, const DinoCandidate &candidate,
                  const DinoRegionSearchConfig &config)
{
    for (size_t index = 0; index < accepted.size(); ++index)
    {
        const auto &existing = accepted[index];
        if (existing.image_id != candidate.image_id)
        {
            continue;
        }
        if (dinoIoU(existing.source_bbox, candidate.source_bbox) >= config.coarse_dedup_iou
            && areaRatio(existing.source_bbox, candidate.source_bbox) <= config.coarse_dedup_area_ratio)
        {
            return static_cast<int>(index);
        }
    }
    return -1;
}

void mergeInto(DinoCandidate &target, const DinoCandidate &source)
{
    if (source.score > target.score)
    {
        target.source_bbox = source.source_bbox;
        target.view_id = source.view_id;
        target.query_view_id = source.query_view_id;
    }
    target.from_region  = target.from_region || source.from_region;
    target.from_local   = target.from_local || source.from_local;
    if (source.from_region)
    {
        target.region_score = std::max(target.region_score, source.region_score);
    }
    if (source.from_local)
    {
        target.local_score = std::max(target.local_score, source.local_score);
    }
    target.score = std::max(target.region_score, target.local_score);
}

} // namespace

DinoFusionOutcome dinoFuseCandidates(const std::vector<DinoCandidate> &region_candidates,
                                     const std::vector<DinoCandidate> &local_candidates,
                                     const DinoRegionSearchConfig &config)
{
    DinoFusionOutcome outcome;
    const size_t per_channel_quota = config.coarse_k / 2U;

    size_t region_index = 0;
    size_t local_index  = 0;
    bool   turn_region  = true;

    const auto try_take = [&](const std::vector<DinoCandidate> &channel, size_t &index, size_t &taken) -> bool
    {
        if (index >= channel.size())
        {
            return false;
        }
        const auto &candidate = channel[index];
        const int   duplicate = findDuplicate(outcome.candidates, candidate, config);
        if (duplicate >= 0)
        {
            mergeInto(outcome.candidates[static_cast<size_t>(duplicate)], candidate);
            ++outcome.duplicates_merged;
            ++index;
            return true;
        }
        outcome.candidates.push_back(candidate);
        ++taken;
        ++index;
        return true;
    };

    while (outcome.candidates.size() < config.coarse_k)
    {
        const bool region_exhausted = region_index >= region_candidates.size();
        const bool local_exhausted  = local_index >= local_candidates.size();
        if (region_exhausted && local_exhausted)
        {
            break;
        }

        // 每条通道先各取满独立额度，之后由剩余通道补足。
        const bool region_has_quota = outcome.region_taken < per_channel_quota;
        const bool local_has_quota  = outcome.local_taken < per_channel_quota;

        if (turn_region)
        {
            if (!region_exhausted && (region_has_quota || !local_has_quota))
            {
                try_take(region_candidates, region_index, outcome.region_taken);
            }
            else if (!local_exhausted)
            {
                try_take(local_candidates, local_index, outcome.local_taken);
            }
        }
        else
        {
            if (!local_exhausted && (local_has_quota || !region_has_quota))
            {
                try_take(local_candidates, local_index, outcome.local_taken);
            }
            else if (!region_exhausted)
            {
                try_take(region_candidates, region_index, outcome.region_taken);
            }
        }
        turn_region = !turn_region;
    }

    return outcome;
}

} // namespace irt::features::priv
