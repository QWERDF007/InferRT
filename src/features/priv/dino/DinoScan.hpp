#pragma once

/**
 * @file DinoScan.hpp
 * @brief 紧凑描述的分块精确扫描与双路粗选。
 */

#include "DinoIndexStore.hpp"
#include "DinoQuery.hpp"
#include "DinoTopK.hpp"
#include "DinoTime.hpp"

#include <inferrt/features/DinoRegionSearch.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace irt::features::priv {

/** @brief 扫描输出。 */
struct DinoScanOutcome
{
    std::vector<DinoCandidate> region_candidates{};
    std::vector<DinoCandidate> local_candidates{};
    size_t scanned_region_descriptors{0};
    size_t scanned_local_descriptors{0};
    size_t retained_local_views{0};
    size_t window_rescore_windows{0};
    bool   view_survival_truncated{false};
    bool   incomplete{false};
    double region_scan_ms{0.0};       ///< 区域通道扫描耗时。
    double local_scan_ms{0.0};        ///< 局部通道视图级聚合耗时。
    double window_rescore_ms{0.0};    ///< 局部通道窗口重打分耗时。
    const char *similarity_backend{"cpu:scalar"}; ///< 实际使用的相似度归约后端。
    bool   compact_scan{false};       ///< 是否走紧凑（INT8）扫描路径。
    uint64_t device_allocated_bytes{0};    ///< 归约器占用的设备显存峰值。
    uint64_t device_reserved_delta_bytes{0}; ///< 进程显存占用峰值增量。
};

/**
 * @brief 执行区域通道与局部通道的精确扫描。
 *
 * 区域通道对每个查询视图维护区域 Top-K；局部通道先按视图聚合查询格子分数，保留有限
 * 视图后在视图的区域窗口内重打分。全程不使用 ANN，也不持久保存每个匹配点。
 */
DinoScanOutcome dinoScan(const DinoIndexReader &reader, const DinoQuery &query,
                         const DinoRegionSearchConfig &config, const DinoDeadline &deadline);

} // namespace irt::features::priv
