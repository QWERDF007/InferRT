#pragma once

#include "ClusteringNeighborhood.hpp"

#include <memory>
#include <vector>

namespace irt::ops::detail {

void validateNeighborSearchMetric(ClusteringAlgorithm algorithm, ClusteringMetric metric, double minkowski_p);

[[nodiscard]] std::unique_ptr<RadiusNeighborhoodIndex>
makeKDTreeNeighborhoodIndex(const float *samples, int64_t num_samples, int64_t num_features, int64_t leaf_size,
                            ClusteringMetric metric, double minkowski_p);

[[nodiscard]] std::unique_ptr<RadiusNeighborhoodIndex>
makeBallTreeNeighborhoodIndex(const float *samples, int64_t num_samples, int64_t num_features, int64_t leaf_size,
                              ClusteringMetric metric, double minkowski_p);

void kdTreeKthSearchDistances(const float *samples, int64_t num_samples, int64_t num_features, int64_t leaf_size,
                             ClusteringMetric metric, double minkowski_p, int64_t kth, std::vector<double> &result);

void ballTreeKthSearchDistances(const float *samples, int64_t num_samples, int64_t num_features, int64_t leaf_size,
                               ClusteringMetric metric, double minkowski_p, int64_t kth, std::vector<double> &result);

} // namespace irt::ops::detail
