#pragma once

#include "ClusteringDistance.hpp"

#include <memory>
#include <vector>

namespace irt::ops::detail {

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

/** Create a reusable brute, KD-tree, or Ball-tree radius index. */
[[nodiscard]] std::unique_ptr<RadiusNeighborhoodIndex>
makeRadiusNeighborhoodIndex(const float *samples, int64_t num_samples, int64_t num_features,
                            ClusteringAlgorithm algorithm, int64_t leaf_size, ClusteringMetric metric,
                            double minkowski_p);

/** Query the radius neighborhood of every sample. */
[[nodiscard]] std::vector<std::vector<int64_t>> radiusNeighborhoods(const float *samples, int64_t num_samples,
                                                                     int64_t num_features, double radius,
                                                                     ClusteringAlgorithm algorithm, int64_t leaf_size,
                                                                     ClusteringMetric metric, double minkowski_p);

/** Calculate the real distance to the kth neighbor of every sample. */
[[nodiscard]] std::vector<double> kthNeighborDistances(const float *samples, int64_t num_samples, int64_t num_features,
                                                       int64_t kth, ClusteringAlgorithm algorithm, int64_t leaf_size,
                                                       ClusteringMetric metric, double minkowski_p);

/** Calculate the monotonic search distance to the kth neighbor of every sample. */
[[nodiscard]] std::vector<double> kthNeighborSearchDistances(const float *samples, int64_t num_samples,
                                                              int64_t num_features, int64_t kth,
                                                              ClusteringAlgorithm algorithm, int64_t leaf_size,
                                                              ClusteringMetric metric, double minkowski_p);

} // namespace irt::ops::detail
