#pragma once

#include <cstdint>

namespace irt::ops {

/**
 * @brief 邻域搜索算法类型。
 */
enum class ClusteringAlgorithm
{
    Brute,    ///< 暴力全量距离搜索。
    KDTree,   ///< KDTree 邻域搜索。
    BallTree, ///< BallTree 邻域搜索。
};

/**
 * @brief 聚类距离度量类型。
 */
enum class ClusteringMetric
{
    Euclidean, ///< 欧氏距离。
    Cosine,    ///< 余弦距离，仅暴力搜索支持。
    Manhattan, ///< 曼哈顿距离。
    Chebyshev, ///< 切比雪夫距离。
    Minkowski, ///< Minkowski 距离，阶数由 minkowski_p 指定。
};

} // namespace irt::ops
