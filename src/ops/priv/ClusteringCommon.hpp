#pragma once

#include <inferrt/ops/Clustering.hpp>

#include <cstdint>

namespace irt::ops::detail {

/** Validate the row-major sample matrix shared by DBSCAN and HDBSCAN. */
void validateSampleMatrix(const float *samples, int64_t num_samples, int64_t num_features);

/** Validate the neighborhood algorithm and its tree leaf size. */
void validateNeighborSearchConfig(ClusteringAlgorithm algorithm, int64_t leaf_size);

/** Validate the metric and its optional Minkowski exponent. */
void validateMetricConfig(ClusteringMetric metric, double minkowski_p);

} // namespace irt::ops::detail
