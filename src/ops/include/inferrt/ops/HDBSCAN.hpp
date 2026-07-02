#pragma once

#include <inferrt/ops/Clustering.hpp>
#include <inferrt/ops/Export.h>

#include <cstdint>
#include <vector>

namespace irt::ops {

inline constexpr auto kDefaultHDBSCANMetric = ClusteringMetric::Euclidean; ///< 默认距离度量。

/**
 * @brief HDBSCAN 层次树簇选择方法。
 */
enum class HDBSCANClusterSelectionMethod
{
    Eom,  ///< Excess of Mass 簇选择方法。
    Leaf, ///< 叶子簇选择方法。
};

/**
 * @brief HDBSCAN 聚类配置。
 */
struct HDBSCANConfig final
{
    int64_t min_cluster_size{5};            ///< 形成簇所需的最小样本数。
    int64_t min_samples{0};                 ///< 核心距离邻居数，0 表示使用 min_cluster_size。
    double  cluster_selection_epsilon{0.0}; ///< 簇选择距离阈值。
    int64_t max_cluster_size{0};            ///< 最大簇大小，0 表示不限制。
    double  alpha{1.0};                     ///< robust single linkage 缩放系数。
    int64_t leaf_size{40};                  ///< 树搜索叶子节点大小。
    double  minkowski_p{2.0};               ///< Minkowski 距离阶数。
    bool    allow_single_cluster{false};    ///< 是否允许所有非噪声样本合并为单个簇。

    ClusteringAlgorithm           algorithm{ClusteringAlgorithm::KDTree};                       ///< 邻域搜索算法。
    ClusteringMetric              metric{kDefaultHDBSCANMetric};                                ///< 距离度量。
    HDBSCANClusterSelectionMethod cluster_selection_method{HDBSCANClusterSelectionMethod::Eom}; ///< 簇选择方法。
};

/**
 * @brief HDBSCAN 聚类结果。
 */
struct HDBSCANResult final
{
    std::vector<int64_t> labels;        ///< 每个样本的聚类标签，噪声点为 -1。
    std::vector<double>  probabilities; ///< 每个样本属于当前簇的稳定性概率。
};

/**
 * @brief 执行 HDBSCAN 聚类。
 * @param samples 输入样本矩阵，形状为 [num_samples, num_features]，按行主序存储。
 * @param num_samples 样本数量。
 * @param num_features 每个样本的特征维度。
 * @param config HDBSCAN 聚类配置。
 * @return HDBSCAN 聚类结果，包含每个样本的标签和稳定性概率。
 */
[[nodiscard]] INFERRT_OPS_API HDBSCANResult hdbscan(const float *samples, int64_t num_samples, int64_t num_features,
                                                    const HDBSCANConfig &config);

} // namespace irt::ops
