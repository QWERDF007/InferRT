#pragma once

#include <inferrt/ops/Clustering.hpp>

#include <cstdint>
#include <memory>
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
 * @brief 按指定度量计算两个样本之间用于搜索/比较的距离。
 *
 * 欧氏距离返回平方距离，Minkowski 距离返回 p 次方距离；这些单调变换可避免在邻域搜索中频繁开方。
 *
 * @param samples 输入样本矩阵，形状为 [num_samples, num_features]，按行主序存储。
 * @param lhs 左侧样本索引。
 * @param rhs 右侧样本索引。
 * @param num_features 每个样本的特征维度。
 * @param metric 距离度量。
 * @param minkowski_p Minkowski 距离阶数。
 * @return 用于搜索/比较的距离。
 */
[[nodiscard]] double clusteringSearchDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features,
                                              ClusteringMetric metric, double minkowski_p);

/**
 * @brief 将真实距离半径转换为搜索距离半径。
 * @param radius 真实距离半径。
 * @param metric 距离度量。
 * @param minkowski_p Minkowski 距离阶数。
 * @return 搜索距离半径。
 */
[[nodiscard]] double clusteringSearchRadius(double radius, ClusteringMetric metric, double minkowski_p);

/**
 * @brief 将搜索距离转换回真实距离。
 * @param search_distance 搜索距离。
 * @param metric 距离度量。
 * @param minkowski_p Minkowski 距离阶数。
 * @return 真实距离。
 */
[[nodiscard]] double clusteringOutputDistance(double search_distance, ClusteringMetric metric, double minkowski_p);

class RadiusNeighborhoodIndex
{
public:
    RadiusNeighborhoodIndex()                                  = default;
    RadiusNeighborhoodIndex(const RadiusNeighborhoodIndex &)            = delete;
    RadiusNeighborhoodIndex &operator=(const RadiusNeighborhoodIndex &) = delete;
    RadiusNeighborhoodIndex(RadiusNeighborhoodIndex &&)                 = delete;
    RadiusNeighborhoodIndex &operator=(RadiusNeighborhoodIndex &&)      = delete;
    virtual ~RadiusNeighborhoodIndex();

    virtual void radiusNeighbors(int64_t query, double radius, std::vector<int64_t> &result,
                                 bool sort_result) const = 0;
    [[nodiscard]] virtual int64_t radiusNeighborCount(int64_t query, double radius, int64_t stop_count) const = 0;
};

/**
 * @brief 创建可复用的半径邻域索引。
 * @param samples 输入样本矩阵，形状为 [num_samples, num_features]，按行主序存储。
 * @param num_samples 样本数量。
 * @param num_features 每个样本的特征维度。
 * @param algorithm 邻域搜索算法。
 * @param leaf_size 树搜索叶子节点大小。
 * @param metric 距离度量。
 * @param minkowski_p Minkowski 距离阶数。
 * @return 可复用的半径邻域索引。
 */
[[nodiscard]] std::unique_ptr<RadiusNeighborhoodIndex>
makeRadiusNeighborhoodIndex(const float *samples, int64_t num_samples, int64_t num_features,
                            ClusteringAlgorithm algorithm, int64_t leaf_size, ClusteringMetric metric,
                            double minkowski_p);

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

/**
 * @brief 计算每个样本的第 k 近邻搜索距离。
 *
 * 欧氏距离返回平方距离，Minkowski 距离返回 p 次方距离；其他度量返回原距离。
 *
 * @param samples 输入样本矩阵，形状为 [num_samples, num_features]，按行主序存储。
 * @param num_samples 样本数量。
 * @param num_features 每个样本的特征维度。
 * @param kth 近邻序号。
 * @param algorithm 邻域搜索算法。
 * @param leaf_size 树搜索叶子节点大小。
 * @param metric 距离度量。
 * @param minkowski_p Minkowski 距离阶数。
 * @return 每个样本的第 k 近邻搜索距离。
 */
[[nodiscard]] std::vector<double> kthNeighborSearchDistances(const float *samples, int64_t num_samples,
                                                             int64_t num_features, int64_t kth,
                                                             ClusteringAlgorithm algorithm, int64_t leaf_size,
                                                             ClusteringMetric metric, double minkowski_p);

} // namespace irt::ops::detail
