#pragma once

#include <inferrt/ops/Clustering.hpp>
#include <inferrt/ops/Export.h>

#include <cstdint>
#include <vector>

namespace irt::ops {

/**
 * @brief DBSCAN 聚类配置。
 */
struct DBSCANConfig final
{
    float   eps{0.5f};        ///< 邻域半径。
    int64_t min_samples{5};   ///< 判定核心点所需的最小邻域样本数。
    int64_t leaf_size{30};    ///< 树搜索叶子节点大小。
    double  minkowski_p{2.0}; ///< Minkowski 距离阶数。

    ClusteringAlgorithm algorithm{ClusteringAlgorithm::KDTree}; ///< 邻域搜索算法。
    ClusteringMetric    metric{ClusteringMetric::Euclidean};    ///< 距离度量。
};

/**
 * @brief DBSCAN 聚类结果。
 */
struct DBSCANResult final
{
    std::vector<int64_t> core_sample_indices; ///< 核心样本索引。
    std::vector<int64_t> labels;              ///< 每个样本的聚类标签，噪声点为 -1。
};

/**
 * @brief 执行 DBSCAN 聚类。
 * @param samples 输入样本矩阵，形状为 [num_samples, num_features]，按行主序存储。
 * @param num_samples 样本数量。
 * @param num_features 每个样本的特征维度。
 * @param config DBSCAN 聚类配置。
 * @return DBSCAN 聚类结果，包含核心样本索引和每个样本的标签。
 */
[[nodiscard]] INFERRT_OPS_API DBSCANResult dbscan(const float *samples, int64_t num_samples, int64_t num_features,
                                                  const DBSCANConfig &config);

} // namespace irt::ops
