#pragma once

#include <cstdint>

namespace irt::ops {

inline constexpr int64_t kDefaultClusteringLeafSize = 30;

enum class ClusteringAlgorithm
{
    Brute,
    KDTree,
    BallTree,
};

enum class ClusteringMetric
{
    Euclidean,
    Cosine,
    Manhattan,
    Chebyshev,
    Minkowski,
};

} // namespace irt::ops
