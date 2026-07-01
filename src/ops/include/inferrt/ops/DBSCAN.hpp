#pragma once

#include <inferrt/ops/Clustering.hpp>
#include <inferrt/ops/Export.h>

#include <cstdint>
#include <vector>

namespace irt::ops {

inline constexpr float   kDefaultDBSCANEps        = 0.5f;
inline constexpr int64_t kDefaultDBSCANMinSamples = 5;
inline constexpr double  kDefaultDBSCANMinkowskiP = 2.0;
inline constexpr auto    kDefaultDBSCANMetric     = ClusteringMetric::Cosine;

struct DBSCANConfig final
{
    float               eps{kDefaultDBSCANEps};
    int64_t             min_samples{kDefaultDBSCANMinSamples};
    ClusteringAlgorithm algorithm{ClusteringAlgorithm::Auto};
    int64_t             leaf_size{kDefaultClusteringLeafSize};
    ClusteringMetric    metric{kDefaultDBSCANMetric};
    double              minkowski_p{kDefaultDBSCANMinkowskiP};
};

struct DBSCANResult final
{
    std::vector<int64_t> core_sample_indices;
    std::vector<int64_t> labels;
};

[[nodiscard]] INFERRT_OPS_API DBSCANResult dbscan(const float *samples, int64_t num_samples, int64_t num_features,
                                                  const DBSCANConfig &config);

} // namespace irt::ops
