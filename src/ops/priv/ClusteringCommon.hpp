#pragma once

#include <inferrt/ops/Clustering.hpp>

#include <cstdint>
#include <vector>

namespace irt::ops::detail {

void validateSampleMatrix(const float *samples, int64_t num_samples, int64_t num_features);

void validateNeighborSearchConfig(ClusteringAlgorithm algorithm, int64_t leaf_size);

[[nodiscard]] double squaredEuclideanDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features);

[[nodiscard]] double euclideanDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features);

[[nodiscard]] ClusteringAlgorithm selectNeighborSearchAlgorithm(ClusteringAlgorithm requested, int64_t num_features);

[[nodiscard]] std::vector<std::vector<int64_t>> radiusNeighborhoods(const float *samples, int64_t num_samples,
                                                                    int64_t num_features, double radius,
                                                                    ClusteringAlgorithm algorithm, int64_t leaf_size);

[[nodiscard]] std::vector<double> kthNeighborDistances(const float *samples, int64_t num_samples, int64_t num_features,
                                                       int64_t kth, ClusteringAlgorithm algorithm, int64_t leaf_size);

} // namespace irt::ops::detail
