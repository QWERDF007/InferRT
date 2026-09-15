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


/** @brief 建立本地索引。 */
DinoBuildReport dinoBuildIndex(const std::filesystem::path &gallery_root, const DinoRegionSearchConfig &config,
                               const std::filesystem::path &index_root,
                               const DinoBuildProgressCallback &progress_callback,
                               const DinoOperationControl &control = {});

/** @brief 从显式条目列表建立本地索引（核心建库入口）。 */
DinoBuildReport dinoBuildItems(const std::vector<DinoImageItem> &items, const DinoRegionSearchConfig &config,
                               const std::filesystem::path &index_root,
                               const DinoBuildProgressCallback &progress_callback,
                               const DinoOperationControl &control = {});

/** @brief 查询入口。 */
DinoSearchResponse dinoSearchIndex(const std::filesystem::path &index_root, const DinoSearchRequest &request,
                                   const DinoRegionSearchConfig &config,
                                   const DinoSearchProgressCallback &progress_callback,
                                   const DinoOperationControl &control = {});

/** @brief 释放当前进程持有的运行时资源。 */
void dinoReleaseRuntime(bool release_backbone = true);

/** @brief 当前进程峰值 RSS 字节数；无法获取时返回 0。 */
uint64_t dinoPeakRssBytes();

} // namespace irt::features::priv
