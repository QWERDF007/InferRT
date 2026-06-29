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
}

double squaredEuclideanDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features)
{
    double       sum     = 0.0;
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        const float lhs_value = lhs_ptr[feature];
        const float rhs_value = rhs_ptr[feature];
        if (!std::isfinite(lhs_value) || !std::isfinite(rhs_value))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "samples must be finite");
        }
        const double diff = static_cast<double>(lhs_value) - static_cast<double>(rhs_value);
        sum += diff * diff;
    }
    return sum;
}

double euclideanDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features)
{
    return std::sqrt(squaredEuclideanDistance(samples, lhs, rhs, num_features));
}

} // namespace irt::ops::detail
