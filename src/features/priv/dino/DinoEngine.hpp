#pragma once

/**
 * @file DinoEngine.hpp
 * @brief 本地索引建库和查询编排。
 */

#include "DinoIngest.hpp"
#include "DinoIndexStore.hpp"
#include "DinoQuery.hpp"
#include "DinoTime.hpp"

#include <inferrt/features/DinoRegionSearch.hpp>

#include <filesystem>
#include <vector>

namespace irt::features::priv {

/** @brief 查询的完整内部结果，评测与 CLI 都从这里派生对外契约。 */
struct DinoSearchDetail
{
    DinoSearchResponse           response{};
    std::vector<DinoCandidate>   coarse_candidates{};
    std::vector<DinoCandidate>   region_candidates{};
    std::vector<DinoCandidate>   local_candidates{};
    std::vector<DinoMatchResult> matches_before_truncation{};
    std::vector<DinoMatchResult> matches_after_truncation{};
    size_t                       self_excluded{0};
    std::string                  query_image_id{};
};

/** @brief 建立本地索引。 */
DinoBuildReport dinoBuildIndex(const std::filesystem::path &gallery_root, const DinoRegionSearchConfig &config,
                               const std::filesystem::path &index_root,
                               const DinoBuildProgressCallback &progress_callback);


/** @brief 查询入口。 */
DinoSearchDetail dinoSearchIndex(const std::filesystem::path &index_root, const DinoSearchRequest &request,
                                 const DinoRegionSearchConfig &config,
                                 const DinoSearchProgressCallback &progress_callback);


/** @brief 当前进程峰值 RSS 字节数；无法获取时返回 0。 */
uint64_t dinoPeakRssBytes();

} // namespace irt::features::priv
