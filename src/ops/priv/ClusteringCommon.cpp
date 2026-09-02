#include "ClusteringCommon.hpp"

#include <inferrt/core/Exception.hpp>

#include <cmath>

namespace irt::ops::detail {

void validateSampleMatrix(const float *samples, int64_t num_samples, int64_t num_features)
{
    if (num_samples < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_samples must be non-negative, got %lld",
                        static_cast<long long>(num_samples));
    }
    if (num_features <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_features must be positive, got %lld",
                        static_cast<long long>(num_features));
    }
    if (num_samples > 0 && samples == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "samples must not be null");
    }
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            if (!std::isfinite(samples[sample * num_features + feature]))
            {
                throw Exception(Status::ERROR_INVALID_ARGUMENT, "samples must be finite");
            }
        }
    }
}

void validateNeighborSearchConfig(ClusteringAlgorithm algorithm, int64_t leaf_size)
{
    switch (algorithm)
    {
    case ClusteringAlgorithm::Brute:
    case ClusteringAlgorithm::KDTree:
    case ClusteringAlgorithm::BallTree:
        break;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering algorithm");
    }

    if (leaf_size < 1)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "leaf_size must be positive, got %lld",
                        static_cast<long long>(leaf_size));
    }
}

void validateMetricConfig(ClusteringMetric metric, double minkowski_p)
{
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
    case ClusteringMetric::Cosine:
    case ClusteringMetric::Manhattan:
    case ClusteringMetric::Chebyshev:
    case ClusteringMetric::Minkowski:
        break;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }

    if (metric == ClusteringMetric::Minkowski && (!std::isfinite(minkowski_p) || minkowski_p <= 0.0))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "minkowski_p must be positive and finite");
    }
}

} // namespace irt::ops::detail
