#pragma once

#include <inferrt/ops/Clustering.hpp>
#include <inferrt/ops/Export.h>

#include <cstdint>
#include <vector>

namespace irt::ops {

inline constexpr int64_t kDefaultHDBSCANMinClusterSize          = 5;
inline constexpr int64_t kDefaultHDBSCANMinSamples              = 0;
inline constexpr double  kDefaultHDBSCANClusterSelectionEpsilon = 0.0;
inline constexpr int64_t kDefaultHDBSCANMaxClusterSize          = 0;
inline constexpr double  kDefaultHDBSCANAlpha                   = 1.0;
inline constexpr double  kDefaultHDBSCANMinkowskiP              = 2.0;
inline constexpr auto    kDefaultHDBSCANMetric                  = ClusteringMetric::Cosine;

enum class HDBSCANClusterSelectionMethod
{
    Eom,
    Leaf,
};

struct HDBSCANConfig final
{
    int64_t                       min_cluster_size{kDefaultHDBSCANMinClusterSize};
    int64_t                       min_samples{kDefaultHDBSCANMinSamples};
    double                        cluster_selection_epsilon{kDefaultHDBSCANClusterSelectionEpsilon};
    int64_t                       max_cluster_size{kDefaultHDBSCANMaxClusterSize};
    double                        alpha{1.0};
    ClusteringAlgorithm           algorithm{ClusteringAlgorithm::KDTree};
    int64_t                       leaf_size{40};
    ClusteringMetric              metric{kDefaultHDBSCANMetric};
    double                        minkowski_p{kDefaultHDBSCANMinkowskiP};
    HDBSCANClusterSelectionMethod cluster_selection_method{HDBSCANClusterSelectionMethod::Eom};
    bool                          allow_single_cluster{false};
};

struct HDBSCANResult final
{
    std::vector<int64_t> labels;
    std::vector<double>  probabilities;
};

[[nodiscard]] INFERRT_OPS_API HDBSCANResult hdbscan(const float *samples, int64_t num_samples, int64_t num_features,
                                                    const HDBSCANConfig &config);

} // namespace irt::ops
