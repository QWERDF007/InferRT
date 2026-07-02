#pragma once

#include <inferrt/ops/Clustering.hpp>

#include <cstdint>
#include <vector>

namespace irt::ops::detail {

/**
 * @brief 校验聚类输入样本矩阵。
 * @param samples 输入样本矩阵，形状为 [num_samples, num_features]，按行主序存储。
 * @param num_samples 样本数量。
 * @param num_features 每个样本的特征维度。
 * @return 无。
 */
void validateSampleMatrix(const float *samples, int64_t num_samples, int64_t num_features);

/**
 * @brief 校验邻域搜索算法配置。
 * @param algorithm 邻域搜索算法。
 * @param leaf_size 树搜索叶子节点大小。
 * @return 无。
 */
void validateNeighborSearchConfig(ClusteringAlgorithm algorithm, int64_t leaf_size);

/**
 * @brief 校验聚类距离度量配置。
 * @param metric 距离度量。
 * @param minkowski_p Minkowski 距离阶数。
 * @return 无。
 */
void validateMetricConfig(ClusteringMetric metric, double minkowski_p);

/**
 * @brief 计算两个样本之间的平方欧氏距离。
 * @param samples 输入样本矩阵，形状为 [num_samples, num_features]，按行主序存储。
 * @param lhs 左侧样本索引。
 * @param rhs 右侧样本索引。
 * @param num_features 每个样本的特征维度。
 * @return 两个样本之间的平方欧氏距离。
 */
[[nodiscard]] double squaredEuclideanDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features);

/**
 * @brief 计算两个样本之间的欧氏距离。
 * @param samples 输入样本矩阵，形状为 [num_samples, num_features]，按行主序存储。
 * @param lhs 左侧样本索引。
 * @param rhs 右侧样本索引。
 * @param num_features 每个样本的特征维度。
 * @return 两个样本之间的欧氏距离。
 */
[[nodiscard]] double euclideanDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features);

/**
 * @brief 按指定度量计算两个样本之间的距离。
 * @param samples 输入样本矩阵，形状为 [num_samples, num_features]，按行主序存储。
 * @param lhs 左侧样本索引。
 * @param rhs 右侧样本索引。
 * @param num_features 每个样本的特征维度。
 * @param metric 距离度量。
 * @param minkowski_p Minkowski 距离阶数。
 * @return 两个样本之间的距离。
 */
[[nodiscard]] double clusteringDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features,
                                        ClusteringMetric metric, double minkowski_p);

/**
 * @brief 查询每个样本在半径内的邻域样本。
 * @param samples 输入样本矩阵，形状为 [num_samples, num_features]，按行主序存储。
 * @param num_samples 样本数量。
 * @param num_features 每个样本的特征维度。
 * @param radius 邻域半径。
 * @param algorithm 邻域搜索算法。
 * @param leaf_size 树搜索叶子节点大小。
 * @param metric 距离度量。
 * @param minkowski_p Minkowski 距离阶数。
 * @return 每个样本的邻域索引列表。
 */
[[nodiscard]] std::vector<std::vector<int64_t>> radiusNeighborhoods(const float *samples, int64_t num_samples,
                                                                    int64_t num_features, double radius,
                                                                    ClusteringAlgorithm algorithm, int64_t leaf_size,
                                                                    ClusteringMetric metric, double minkowski_p);

/**
 * @brief 计算每个样本的第 k 近邻距离。
 * @param samples 输入样本矩阵，形状为 [num_samples, num_features]，按行主序存储。
 * @param num_samples 样本数量。
 * @param num_features 每个样本的特征维度。
 * @param kth 近邻序号。
 * @param algorithm 邻域搜索算法。
 * @param leaf_size 树搜索叶子节点大小。
 * @param metric 距离度量。
 * @param minkowski_p Minkowski 距离阶数。
 * @return 每个样本的第 k 近邻距离。
 */
[[nodiscard]] std::vector<double> kthNeighborDistances(const float *samples, int64_t num_samples, int64_t num_features,
                                                       int64_t kth, ClusteringAlgorithm algorithm, int64_t leaf_size,
                                                       ClusteringMetric metric, double minkowski_p);

} // namespace irt::ops::detail
