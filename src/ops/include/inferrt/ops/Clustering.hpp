#pragma once

#include <cstdint>

namespace irt::ops {

inline constexpr int64_t kDefaultClusteringLeafSize = 30;

enum class ClusteringAlgorithm
{
    Auto,
    Brute,
    KDTree,
    BallTree,
};

} // namespace irt::ops
