#pragma once

#include "ClusteringCommon.hpp"

#include <vector>

namespace irt::ops::detail {

/** Calculate two samples' squared Euclidean distance. */
[[nodiscard]] double squaredEuclideanDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features);

/** Calculate two samples' Euclidean distance. */
[[nodiscard]] double euclideanDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features);

/** Calculate two samples' distance using the configured metric. */
[[nodiscard]] double clusteringDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features,
                                        ClusteringMetric metric, double minkowski_p);

/** Calculate the monotonic distance used for neighbor comparisons. */
[[nodiscard]] double clusteringSearchDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features,
                                              ClusteringMetric metric, double minkowski_p);

[[nodiscard]] std::vector<double> cosineInverseNorms(const float *samples, int64_t num_samples, int64_t num_features);

struct DistanceBlock4
{
    double first{0.0};
    double second{0.0};
    double third{0.0};
    double fourth{0.0};
};

class SearchDistanceCalculator final
{
public:
    SearchDistanceCalculator(const float *samples, int64_t num_features, ClusteringMetric metric, double minkowski_p,
                             const std::vector<double> *inverse_norms = nullptr);

    [[nodiscard]] double operator()(int64_t lhs, int64_t rhs) const;
    [[nodiscard]] bool canUseBlock4() const;
    [[nodiscard]] DistanceBlock4 block4(int64_t lhs, int64_t first_rhs) const;
    [[nodiscard]] DistanceBlock4 indexedBlock4(int64_t lhs, const int64_t *rhs_indices) const;
    [[nodiscard]] int withinRadiusMask4(int64_t lhs, int64_t first_rhs, double search_radius) const;
    [[nodiscard]] int indexedWithinRadiusMask4(int64_t lhs, const int64_t *rhs_indices, double search_radius) const;

private:
    const float               *samples_{nullptr};
    int64_t                    num_features_{0};
    ClusteringMetric           metric_{ClusteringMetric::Euclidean};
    double                     minkowski_p_{2.0};
    const std::vector<double> *inverse_norms_{nullptr};
};

/** Return the monotonic powered form used by Minkowski distance searches. */
[[nodiscard]] double clusteringMinkowskiPower(double value, double minkowski_p);

/** Return the real-distance root of a powered Minkowski distance. */
[[nodiscard]] double clusteringMinkowskiRoot(double value, double minkowski_p);

/** Convert a real radius to the monotonic search radius for a metric. */
[[nodiscard]] double clusteringSearchRadius(double radius, ClusteringMetric metric, double minkowski_p);

/** Convert a monotonic search distance back to its real metric distance. */
[[nodiscard]] double clusteringOutputDistance(double search_distance, ClusteringMetric metric, double minkowski_p);

} // namespace irt::ops::detail
