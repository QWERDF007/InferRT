#include "ClusteringCommon.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace irt::ops::detail {
namespace {

using NeighborHeapItem = std::pair<double, int64_t>;

[[nodiscard]] double sampleValue(const float *samples, int64_t sample, int64_t feature, int64_t num_features)
{
    return static_cast<double>(samples[sample * num_features + feature]);
}

[[nodiscard]] double squaredEuclideanDistanceUnchecked(const float *samples, int64_t lhs, int64_t rhs,
                                                       int64_t num_features)
{
    double       sum     = 0.0;
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        const double diff = static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature]);
        sum += diff * diff;
    }
    return sum;
}

[[nodiscard]] bool isMinkowskiP3(double minkowski_p)
{
    return std::abs(minkowski_p - 3.0) <= 1e-12;
}

[[nodiscard]] double minkowskiPower(double value, double minkowski_p)
{
    if (isMinkowskiP3(minkowski_p))
    {
        return value * value * value;
    }
    if (minkowski_p == 2.0)
    {
        return value * value;
    }
    if (minkowski_p == 1.0)
    {
        return value;
    }
    return std::pow(value, minkowski_p);
}

[[nodiscard]] double minkowskiRoot(double value, double minkowski_p)
{
    if (isMinkowskiP3(minkowski_p))
    {
        return std::cbrt(value);
    }
    if (minkowski_p == 2.0)
    {
        return std::sqrt(value);
    }
    if (minkowski_p == 1.0)
    {
        return value;
    }
    return std::pow(value, 1.0 / minkowski_p);
}

[[nodiscard]] double manhattanDistanceUnchecked(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features)
{
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;
    if (num_features == 3)
    {
        return std::abs(static_cast<double>(lhs_ptr[0]) - static_cast<double>(rhs_ptr[0]))
               + std::abs(static_cast<double>(lhs_ptr[1]) - static_cast<double>(rhs_ptr[1]))
               + std::abs(static_cast<double>(lhs_ptr[2]) - static_cast<double>(rhs_ptr[2]));
    }

    double sum = 0.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        sum += std::abs(static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature]));
    }
    return sum;
}

[[nodiscard]] double chebyshevDistanceUnchecked(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features)
{
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;
    if (num_features == 3)
    {
        const double diff0 = std::abs(static_cast<double>(lhs_ptr[0]) - static_cast<double>(rhs_ptr[0]));
        const double diff1 = std::abs(static_cast<double>(lhs_ptr[1]) - static_cast<double>(rhs_ptr[1]));
        const double diff2 = std::abs(static_cast<double>(lhs_ptr[2]) - static_cast<double>(rhs_ptr[2]));
        return std::max(diff0, std::max(diff1, diff2));
    }

    double max_diff = 0.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        max_diff = std::max(
            max_diff, std::abs(static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature])));
    }
    return max_diff;
}

[[nodiscard]] double minkowskiPoweredDistanceUnchecked(const float *samples, int64_t lhs, int64_t rhs,
                                                       int64_t num_features, double minkowski_p)
{
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;
    if (isMinkowskiP3(minkowski_p) && num_features == 3)
    {
        const double diff0 = std::abs(static_cast<double>(lhs_ptr[0]) - static_cast<double>(rhs_ptr[0]));
        const double diff1 = std::abs(static_cast<double>(lhs_ptr[1]) - static_cast<double>(rhs_ptr[1]));
        const double diff2 = std::abs(static_cast<double>(lhs_ptr[2]) - static_cast<double>(rhs_ptr[2]));
        return diff0 * diff0 * diff0 + diff1 * diff1 * diff1 + diff2 * diff2 * diff2;
    }
    if (minkowski_p == 2.0)
    {
        return squaredEuclideanDistanceUnchecked(samples, lhs, rhs, num_features);
    }
    if (minkowski_p == 1.0)
    {
        return manhattanDistanceUnchecked(samples, lhs, rhs, num_features);
    }

    double sum = 0.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        const double diff = std::abs(static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature]));
        sum += std::pow(diff, minkowski_p);
    }
    return sum;
}

[[nodiscard]] double cosineDistanceUnchecked(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features)
{
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;

    double dot      = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    if (num_features == 3)
    {
        const double lhs0 = static_cast<double>(lhs_ptr[0]);
        const double lhs1 = static_cast<double>(lhs_ptr[1]);
        const double lhs2 = static_cast<double>(lhs_ptr[2]);
        const double rhs0 = static_cast<double>(rhs_ptr[0]);
        const double rhs1 = static_cast<double>(rhs_ptr[1]);
        const double rhs2 = static_cast<double>(rhs_ptr[2]);
        dot               = lhs0 * rhs0 + lhs1 * rhs1 + lhs2 * rhs2;
        lhs_norm          = lhs0 * lhs0 + lhs1 * lhs1 + lhs2 * lhs2;
        rhs_norm          = rhs0 * rhs0 + rhs1 * rhs1 + rhs2 * rhs2;
    }
    else
    {
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const double lhs_value = static_cast<double>(lhs_ptr[feature]);
            const double rhs_value = static_cast<double>(rhs_ptr[feature]);
            dot += lhs_value * rhs_value;
            lhs_norm += lhs_value * lhs_value;
            rhs_norm += rhs_value * rhs_value;
        }
    }

    if (lhs_norm == 0.0 && rhs_norm == 0.0)
    {
        return 0.0;
    }
    if (lhs_norm == 0.0 || rhs_norm == 0.0)
    {
        return 1.0;
    }

    const double similarity = std::clamp(dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm)), -1.0, 1.0);
    return 1.0 - similarity;
}

[[nodiscard]] double squaredDistanceToCenterEuclidean(const float *samples, int64_t sample, int64_t num_features,
                                                      const std::vector<double> &center)
{
    double sum = 0.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        const double diff
            = sampleValue(samples, sample, feature, num_features) - center[static_cast<size_t>(feature)];
        sum += diff * diff;
    }
    return sum;
}

[[nodiscard]] double manhattanDistanceToCenter(const float *samples, int64_t sample, int64_t num_features,
                                               const std::vector<double> &center)
{
    const float *sample_ptr = samples + sample * num_features;
    if (num_features == 3)
    {
        return std::abs(static_cast<double>(sample_ptr[0]) - center[0])
               + std::abs(static_cast<double>(sample_ptr[1]) - center[1])
               + std::abs(static_cast<double>(sample_ptr[2]) - center[2]);
    }

    double sum = 0.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        sum += std::abs(static_cast<double>(sample_ptr[feature]) - center[static_cast<size_t>(feature)]);
    }
    return sum;
}

[[nodiscard]] double chebyshevDistanceToCenter(const float *samples, int64_t sample, int64_t num_features,
                                               const std::vector<double> &center)
{
    const float *sample_ptr = samples + sample * num_features;
    if (num_features == 3)
    {
        const double diff0 = std::abs(static_cast<double>(sample_ptr[0]) - center[0]);
        const double diff1 = std::abs(static_cast<double>(sample_ptr[1]) - center[1]);
        const double diff2 = std::abs(static_cast<double>(sample_ptr[2]) - center[2]);
        return std::max(diff0, std::max(diff1, diff2));
    }

    double max_diff = 0.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        max_diff = std::max(max_diff,
                            std::abs(static_cast<double>(sample_ptr[feature]) - center[static_cast<size_t>(feature)]));
    }
    return max_diff;
}

[[nodiscard]] double minkowskiPoweredDistanceToCenter(const float *samples, int64_t sample, int64_t num_features,
                                                      const std::vector<double> &center, double minkowski_p)
{
    const float *sample_ptr = samples + sample * num_features;
    if (isMinkowskiP3(minkowski_p) && num_features == 3)
    {
        const double diff0 = std::abs(static_cast<double>(sample_ptr[0]) - center[0]);
        const double diff1 = std::abs(static_cast<double>(sample_ptr[1]) - center[1]);
        const double diff2 = std::abs(static_cast<double>(sample_ptr[2]) - center[2]);
        return diff0 * diff0 * diff0 + diff1 * diff1 * diff1 + diff2 * diff2 * diff2;
    }

    double sum = 0.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        const double diff = std::abs(static_cast<double>(sample_ptr[feature]) - center[static_cast<size_t>(feature)]);
        sum += minkowskiPower(diff, minkowski_p);
    }
    return sum;
}

[[nodiscard]] double distanceToCenter(const float *samples, int64_t sample, int64_t num_features,
                                      const std::vector<double> &center, ClusteringMetric metric, double minkowski_p)
{
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
        return std::sqrt(squaredDistanceToCenterEuclidean(samples, sample, num_features, center));
    case ClusteringMetric::Manhattan:
        return manhattanDistanceToCenter(samples, sample, num_features, center);
    case ClusteringMetric::Chebyshev:
        return chebyshevDistanceToCenter(samples, sample, num_features, center);
    case ClusteringMetric::Minkowski:
        return minkowskiRoot(minkowskiPoweredDistanceToCenter(samples, sample, num_features, center, minkowski_p),
                             minkowski_p);
    case ClusteringMetric::Cosine:
    {
        double dot         = 0.0;
        double sample_norm = 0.0;
        double center_norm = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const double sample_value = sampleValue(samples, sample, feature, num_features);
            const double center_value = center[static_cast<size_t>(feature)];
            dot += sample_value * center_value;
            sample_norm += sample_value * sample_value;
            center_norm += center_value * center_value;
        }

        if (sample_norm == 0.0 && center_norm == 0.0)
        {
            return 0.0;
        }
        if (sample_norm == 0.0 || center_norm == 0.0)
        {
            return 1.0;
        }

        const double similarity = std::clamp(dot / (std::sqrt(sample_norm) * std::sqrt(center_norm)), -1.0, 1.0);
        return 1.0 - similarity;
    }
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

[[nodiscard]] double searchRadiusForMetric(double radius, ClusteringMetric metric, double minkowski_p)
{
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
        return radius * radius;
    case ClusteringMetric::Minkowski:
        return minkowskiPower(radius, minkowski_p);
    case ClusteringMetric::Manhattan:
    case ClusteringMetric::Chebyshev:
    case ClusteringMetric::Cosine:
        return radius;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

[[nodiscard]] double outputDistanceForMetric(double search_distance, ClusteringMetric metric, double minkowski_p)
{
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
        return std::sqrt(search_distance);
    case ClusteringMetric::Minkowski:
        return minkowskiRoot(search_distance, minkowski_p);
    case ClusteringMetric::Manhattan:
    case ClusteringMetric::Chebyshev:
    case ClusteringMetric::Cosine:
        return search_distance;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

[[nodiscard]] double distanceToBoundsSearch(const float *samples, int64_t sample, int64_t num_features,
                                            const std::vector<double> &min_bounds,
                                            const std::vector<double> &max_bounds, ClusteringMetric metric,
                                            double minkowski_p)
{
    const float *sample_ptr = samples + sample * num_features;
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
    {
        double sum = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const auto   index = static_cast<size_t>(feature);
            const double value = static_cast<double>(sample_ptr[feature]);
            const double diff  = value < min_bounds[index] ? min_bounds[index] - value
                                : value > max_bounds[index] ? value - max_bounds[index]
                                                            : 0.0;
            sum += diff * diff;
        }
        return sum;
    }
    case ClusteringMetric::Manhattan:
    {
        double sum = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const auto   index = static_cast<size_t>(feature);
            const double value = static_cast<double>(sample_ptr[feature]);
            sum += value < min_bounds[index] ? min_bounds[index] - value
                 : value > max_bounds[index] ? value - max_bounds[index]
                                             : 0.0;
        }
        return sum;
    }
    case ClusteringMetric::Chebyshev:
    {
        double max_diff = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const auto   index = static_cast<size_t>(feature);
            const double value = static_cast<double>(sample_ptr[feature]);
            const double diff  = value < min_bounds[index] ? min_bounds[index] - value
                                : value > max_bounds[index] ? value - max_bounds[index]
                                                            : 0.0;
            max_diff           = std::max(max_diff, diff);
        }
        return max_diff;
    }
    case ClusteringMetric::Minkowski:
    {
        double sum = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const auto   index = static_cast<size_t>(feature);
            const double value = static_cast<double>(sample_ptr[feature]);
            const double diff  = value < min_bounds[index] ? min_bounds[index] - value
                                : value > max_bounds[index] ? value - max_bounds[index]
                                                            : 0.0;
            sum += minkowskiPower(diff, minkowski_p);
        }
        return sum;
    }
    case ClusteringMetric::Cosine:
        return 0.0;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

[[nodiscard]] double maxDistanceToBoundsSearch(const float *samples, int64_t sample, int64_t num_features,
                                               const std::vector<double> &min_bounds,
                                               const std::vector<double> &max_bounds, ClusteringMetric metric,
                                               double minkowski_p)
{
    const float *sample_ptr = samples + sample * num_features;
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
    {
        double sum = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const auto   index = static_cast<size_t>(feature);
            const double value = static_cast<double>(sample_ptr[feature]);
            const double diff  = std::max(std::abs(value - min_bounds[index]), std::abs(value - max_bounds[index]));
            sum += diff * diff;
        }
        return sum;
    }
    case ClusteringMetric::Manhattan:
    {
        double sum = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const auto   index = static_cast<size_t>(feature);
            const double value = static_cast<double>(sample_ptr[feature]);
            sum += std::max(std::abs(value - min_bounds[index]), std::abs(value - max_bounds[index]));
        }
        return sum;
    }
    case ClusteringMetric::Chebyshev:
    {
        double max_diff = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const auto   index = static_cast<size_t>(feature);
            const double value = static_cast<double>(sample_ptr[feature]);
            max_diff = std::max(max_diff, std::max(std::abs(value - min_bounds[index]),
                                                   std::abs(value - max_bounds[index])));
        }
        return max_diff;
    }
    case ClusteringMetric::Minkowski:
    {
        double sum = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const auto   index = static_cast<size_t>(feature);
            const double value = static_cast<double>(sample_ptr[feature]);
            const double diff  = std::max(std::abs(value - min_bounds[index]), std::abs(value - max_bounds[index]));
            sum += minkowskiPower(diff, minkowski_p);
        }
        return sum;
    }
    case ClusteringMetric::Cosine:
        return std::numeric_limits<double>::infinity();
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

void computeBounds(const float *samples, const std::vector<int64_t> &indices, int64_t num_features,
                   std::vector<double> &min_bounds, std::vector<double> &max_bounds)
{
    min_bounds.assign(static_cast<size_t>(num_features), std::numeric_limits<double>::infinity());
    max_bounds.assign(static_cast<size_t>(num_features), -std::numeric_limits<double>::infinity());
    for (const int64_t index : indices)
    {
        const float *sample_ptr = samples + index * num_features;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const auto   bound_index = static_cast<size_t>(feature);
            const double value       = static_cast<double>(sample_ptr[feature]);
            min_bounds[bound_index]  = std::min(min_bounds[bound_index], value);
            max_bounds[bound_index]  = std::max(max_bounds[bound_index], value);
        }
    }
}

[[nodiscard]] int64_t widestDimensionFromBounds(const std::vector<double> &min_bounds,
                                                const std::vector<double> &max_bounds)
{
    int64_t best_dim   = 0;
    double  best_range = -1.0;
    for (size_t feature = 0; feature < min_bounds.size(); ++feature)
    {
        const double range = max_bounds[feature] - min_bounds[feature];
        if (range > best_range)
        {
            best_dim   = static_cast<int64_t>(feature);
            best_range = range;
        }
    }
    return best_dim;
}

[[nodiscard]] bool canUseBallLowerBound(ClusteringMetric metric, double minkowski_p)
{
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
    case ClusteringMetric::Manhattan:
    case ClusteringMetric::Chebyshev:
        return true;
    case ClusteringMetric::Minkowski:
        return minkowski_p >= 1.0;
    case ClusteringMetric::Cosine:
        return false;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

void validateNeighborSearchMetric(ClusteringAlgorithm algorithm, ClusteringMetric metric, double minkowski_p)
{
    validateMetricConfig(metric, minkowski_p);

    if (algorithm == ClusteringAlgorithm::Brute)
    {
        return;
    }

    if (algorithm != ClusteringAlgorithm::KDTree && algorithm != ClusteringAlgorithm::BallTree)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering algorithm");
    }

    switch (metric)
    {
    case ClusteringMetric::Euclidean:
    case ClusteringMetric::Manhattan:
    case ClusteringMetric::Chebyshev:
        return;
    case ClusteringMetric::Minkowski:
        if (minkowski_p < 1.0)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT,
                            "minkowski_p must be at least 1.0 for tree neighbor search");
        }
        return;
    case ClusteringMetric::Cosine:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "cosine metric is only supported by brute neighbor search");
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

class NeighborHeap final
{
public:
    explicit NeighborHeap(int64_t max_size)
        : max_size_(max_size)
    {
        items_.reserve(static_cast<size_t>(max_size));
    }

    void clear()
    {
        items_.clear();
    }

    [[nodiscard]] int64_t size() const
    {
        return static_cast<int64_t>(items_.size());
    }

    [[nodiscard]] double worstDistance() const
    {
        return items_.front().first;
    }

    void push(double distance, int64_t index)
    {
        if (static_cast<int64_t>(items_.size()) < max_size_)
        {
            items_.emplace_back(distance, index);
            if (static_cast<int64_t>(items_.size()) == max_size_)
            {
                std::make_heap(items_.begin(), items_.end());
            }
            return;
        }

        const auto &worst = items_.front();
        if (distance < worst.first || (distance == worst.first && index < worst.second))
        {
            std::pop_heap(items_.begin(), items_.end());
            items_.back() = {distance, index};
            std::push_heap(items_.begin(), items_.end());
        }
    }

private:
    int64_t                       max_size_{0};
    std::vector<NeighborHeapItem> items_;
};

class BruteRadiusIndex final : public RadiusNeighborhoodIndex
{
public:
    BruteRadiusIndex(const float *samples, int64_t num_samples, int64_t num_features, ClusteringMetric metric,
                     double minkowski_p)
        : samples_(samples)
        , num_samples_(num_samples)
        , num_features_(num_features)
        , metric_(metric)
        , minkowski_p_(minkowski_p)
    {
    }

    void radiusNeighbors(int64_t query, double radius, std::vector<int64_t> &result,
                         bool sort_result) const override
    {
        result.clear();
        result.reserve(static_cast<size_t>(num_samples_));

        const double search_radius = searchRadiusForMetric(radius, metric_, minkowski_p_);
        for (int64_t other = 0; other < num_samples_; ++other)
        {
            if (clusteringSearchDistance(samples_, query, other, num_features_, metric_, minkowski_p_) <= search_radius)
            {
                result.push_back(other);
            }
        }

        if (sort_result)
        {
            std::sort(result.begin(), result.end());
        }
    }

    [[nodiscard]] int64_t radiusNeighborCount(int64_t query, double radius, int64_t stop_count) const override
    {
        const int64_t limit         = stop_count > 0 ? stop_count : std::numeric_limits<int64_t>::max();
        const double  search_radius = searchRadiusForMetric(radius, metric_, minkowski_p_);

        int64_t count = 0;
        for (int64_t other = 0; other < num_samples_; ++other)
        {
            if (clusteringSearchDistance(samples_, query, other, num_features_, metric_, minkowski_p_) <= search_radius
                && ++count >= limit)
            {
                return count;
            }
        }
        return count;
    }

private:
    const float     *samples_{nullptr};
    int64_t          num_samples_{0};
    int64_t          num_features_{0};
    ClusteringMetric metric_{ClusteringMetric::Euclidean};
    double           minkowski_p_{2.0};
};

class KDTreeIndex final : public RadiusNeighborhoodIndex
{
public:
    KDTreeIndex(const float *samples, int64_t num_samples, int64_t num_features, int64_t leaf_size,
                ClusteringMetric metric, double minkowski_p)
        : samples_(samples)
        , num_features_(num_features)
        , leaf_size_(leaf_size)
        , metric_(metric)
        , minkowski_p_(minkowski_p)
    {
        std::vector<int64_t> indices(static_cast<size_t>(num_samples));
        std::iota(indices.begin(), indices.end(), int64_t{0});
        root_ = build(indices);
    }

    void radiusNeighbors(int64_t query, double radius, std::vector<int64_t> &result, bool sort_result) const override
    {
        result.clear();
        radiusSearch(root_, query, searchRadius(radius), result);
        if (sort_result)
        {
            std::sort(result.begin(), result.end());
        }
    }

    [[nodiscard]] int64_t radiusNeighborCount(int64_t query, double radius, int64_t stop_count) const override
    {
        const int64_t limit = stop_count > 0 ? stop_count : std::numeric_limits<int64_t>::max();
        int64_t       count = 0;
        radiusCountSearch(root_, query, searchRadius(radius), limit, count);
        return count;
    }

    void kthSearchDistances(int64_t kth, std::vector<double> &result) const
    {
        NeighborHeap heap(kth);
        for (int64_t sample = 0; sample < static_cast<int64_t>(result.size()); ++sample)
        {
            heap.clear();
            knnSearch(root_, sample, kth, heap);
            result[static_cast<size_t>(sample)] = heap.worstDistance();
        }
    }

private:
    static constexpr size_t kInvalidNode = std::numeric_limits<size_t>::max();

    struct Node
    {
        bool                 leaf{false};
        int64_t              split_dim{0};
        double               split_value{0.0};
        size_t               left{kInvalidNode};
        size_t               right{kInvalidNode};
        int64_t              size{0};
        std::vector<double>  min_bounds;
        std::vector<double>  max_bounds;
        std::vector<int64_t> indices;
    };

    [[nodiscard]] double searchRadius(double radius) const
    {
        return searchRadiusForMetric(radius, metric_, minkowski_p_);
    }

    [[nodiscard]] double searchDistance(int64_t lhs, int64_t rhs) const
    {
        return clusteringSearchDistance(samples_, lhs, rhs, num_features_, metric_, minkowski_p_);
    }

    [[nodiscard]] double outputDistance(double distance) const
    {
        return outputDistanceForMetric(distance, metric_, minkowski_p_);
    }

    [[nodiscard]] double minSearchDistanceToNode(int64_t query, const Node &node) const
    {
        return distanceToBoundsSearch(samples_, query, num_features_, node.min_bounds, node.max_bounds, metric_,
                                      minkowski_p_);
    }

    [[nodiscard]] bool nodeFullyInsideRadius(int64_t query, const Node &node, double search_radius) const
    {
        return maxDistanceToBoundsSearch(samples_, query, num_features_, node.min_bounds, node.max_bounds, metric_,
                                         minkowski_p_)
               <= search_radius;
    }

    void appendSubtree(size_t node_index, std::vector<int64_t> &result) const
    {
        const auto &node = nodes_[node_index];
        if (node.leaf)
        {
            result.insert(result.end(), node.indices.begin(), node.indices.end());
            return;
        }
        appendSubtree(node.left, result);
        appendSubtree(node.right, result);
    }

    [[nodiscard]] size_t build(std::vector<int64_t> &indices)
    {
        const size_t node_index = nodes_.size();
        nodes_.push_back({});
        auto &node = nodes_.back();
        node.size  = static_cast<int64_t>(indices.size());
        computeBounds(samples_, indices, num_features_, node.min_bounds, node.max_bounds);

        if (static_cast<int64_t>(indices.size()) <= leaf_size_)
        {
            node.leaf    = true;
            node.indices = std::move(indices);
            return node_index;
        }

        const int64_t split_dim = widestDimensionFromBounds(node.min_bounds, node.max_bounds);
        const size_t  mid       = indices.size() / 2;
        std::nth_element(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(mid), indices.end(),
                         [this, split_dim](int64_t lhs, int64_t rhs)
                         {
                             const double lhs_value = sampleValue(samples_, lhs, split_dim, num_features_);
                             const double rhs_value = sampleValue(samples_, rhs, split_dim, num_features_);
                             return lhs_value == rhs_value ? lhs < rhs : lhs_value < rhs_value;
                         });

        std::vector<int64_t> left(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(mid));
        std::vector<int64_t> right(indices.begin() + static_cast<std::ptrdiff_t>(mid), indices.end());

        node.split_dim   = split_dim;
        node.split_value = sampleValue(samples_, right.front(), split_dim, num_features_);

        const size_t left_child  = build(left);
        const size_t right_child = build(right);
        nodes_[node_index].left  = left_child;
        nodes_[node_index].right = right_child;
        return node_index;
    }

    void radiusSearch(size_t node_index, int64_t query, double search_radius, std::vector<int64_t> &result) const
    {
        const auto &node = nodes_[node_index];
        if (minSearchDistanceToNode(query, node) > search_radius)
        {
            return;
        }
        if (nodeFullyInsideRadius(query, node, search_radius))
        {
            appendSubtree(node_index, result);
            return;
        }

        if (node.leaf)
        {
            for (const int64_t index : node.indices)
            {
                if (searchDistance(query, index) <= search_radius)
                {
                    result.push_back(index);
                }
            }
            return;
        }

        const double query_value = sampleValue(samples_, query, node.split_dim, num_features_);
        const double diff        = query_value - node.split_value;
        const size_t near_child  = diff <= 0.0 ? node.left : node.right;
        const size_t far_child   = diff <= 0.0 ? node.right : node.left;

        radiusSearch(near_child, query, search_radius, result);
        radiusSearch(far_child, query, search_radius, result);
    }

    void radiusCountSearch(size_t node_index, int64_t query, double search_radius, int64_t stop_count,
                           int64_t &count) const
    {
        if (count >= stop_count)
        {
            return;
        }

        const auto &node = nodes_[node_index];
        if (minSearchDistanceToNode(query, node) > search_radius)
        {
            return;
        }
        if (nodeFullyInsideRadius(query, node, search_radius))
        {
            count += node.size;
            return;
        }

        if (node.leaf)
        {
            for (const int64_t index : node.indices)
            {
                if (searchDistance(query, index) <= search_radius && ++count >= stop_count)
                {
                    return;
                }
            }
            return;
        }

        const auto &left_node    = nodes_[node.left];
        const auto &right_node   = nodes_[node.right];
        const bool  left_is_near = minSearchDistanceToNode(query, left_node) <= minSearchDistanceToNode(query, right_node);
        const auto  first_child  = left_is_near ? node.left : node.right;
        const auto  second_child = left_is_near ? node.right : node.left;

        radiusCountSearch(first_child, query, search_radius, stop_count, count);
        if (count < stop_count)
        {
            radiusCountSearch(second_child, query, search_radius, stop_count, count);
        }
    }

    void knnSearch(size_t node_index, int64_t query, int64_t kth, NeighborHeap &heap) const
    {
        const auto &node = nodes_[node_index];
        if (heap.size() >= kth && minSearchDistanceToNode(query, node) > heap.worstDistance())
        {
            return;
        }

        if (node.leaf)
        {
            for (const int64_t index : node.indices)
            {
                heap.push(searchDistance(query, index), index);
            }
            return;
        }

        const auto &left_node    = nodes_[node.left];
        const auto &right_node   = nodes_[node.right];
        const bool  left_is_near = minSearchDistanceToNode(query, left_node) <= minSearchDistanceToNode(query, right_node);
        const auto  first_child  = left_is_near ? node.left : node.right;
        const auto  second_child = left_is_near ? node.right : node.left;
        knnSearch(first_child, query, kth, heap);
        knnSearch(second_child, query, kth, heap);
    }

    const float      *samples_{nullptr};
    int64_t           num_features_{0};
    int64_t           leaf_size_{0};
    ClusteringMetric  metric_{ClusteringMetric::Euclidean};
    double            minkowski_p_{2.0};
    std::vector<Node> nodes_;
    size_t            root_{kInvalidNode};
};

class BallTreeIndex final : public RadiusNeighborhoodIndex
{
public:
    BallTreeIndex(const float *samples, int64_t num_samples, int64_t num_features, int64_t leaf_size,
                  ClusteringMetric metric, double minkowski_p)
        : samples_(samples)
        , num_features_(num_features)
        , leaf_size_(leaf_size)
        , metric_(metric)
        , minkowski_p_(minkowski_p)
        , use_ball_lower_bound_(canUseBallLowerBound(metric, minkowski_p))
    {
        std::vector<int64_t> indices(static_cast<size_t>(num_samples));
        std::iota(indices.begin(), indices.end(), int64_t{0});
        root_ = build(indices);
    }

    void radiusNeighbors(int64_t query, double radius, std::vector<int64_t> &result, bool sort_result) const override
    {
        result.clear();
        radiusSearch(root_, query, searchRadius(radius), result);
        if (sort_result)
        {
            std::sort(result.begin(), result.end());
        }
    }

    [[nodiscard]] int64_t radiusNeighborCount(int64_t query, double radius, int64_t stop_count) const override
    {
        const int64_t limit = stop_count > 0 ? stop_count : std::numeric_limits<int64_t>::max();
        int64_t       count = 0;
        radiusCountSearch(root_, query, searchRadius(radius), limit, count);
        return count;
    }

    void kthSearchDistances(int64_t kth, std::vector<double> &result) const
    {
        NeighborHeap heap(kth);
        for (int64_t sample = 0; sample < static_cast<int64_t>(result.size()); ++sample)
        {
            heap.clear();
            knnSearch(root_, sample, kth, heap);
            result[static_cast<size_t>(sample)] = heap.worstDistance();
        }
    }

private:
    static constexpr size_t kInvalidNode = std::numeric_limits<size_t>::max();

    struct Node
    {
        bool                 leaf{false};
        size_t               left{kInvalidNode};
        size_t               right{kInvalidNode};
        std::vector<double>  center;
        double               radius{0.0};
        int64_t              size{0};
        std::vector<double>  min_bounds;
        std::vector<double>  max_bounds;
        std::vector<int64_t> indices;
    };

    [[nodiscard]] bool useSquaredEuclideanDistance() const
    {
        return metric_ == ClusteringMetric::Euclidean;
    }

    [[nodiscard]] bool usePoweredMinkowskiDistance() const
    {
        return metric_ == ClusteringMetric::Minkowski;
    }

    [[nodiscard]] double searchRadius(double radius) const
    {
        return searchRadiusForMetric(radius, metric_, minkowski_p_);
    }

    [[nodiscard]] double searchDistance(int64_t lhs, int64_t rhs) const
    {
        return clusteringSearchDistance(samples_, lhs, rhs, num_features_, metric_, minkowski_p_);
    }

    [[nodiscard]] double outputDistance(double distance) const
    {
        return outputDistanceForMetric(distance, metric_, minkowski_p_);
    }

    [[nodiscard]] size_t build(std::vector<int64_t> &indices)
    {
        const size_t node_index = nodes_.size();
        nodes_.push_back({});
        auto &node = nodes_.back();
        node.size  = static_cast<int64_t>(indices.size());
        computeBounds(samples_, indices, num_features_, node.min_bounds, node.max_bounds);

        node.center.assign(static_cast<size_t>(num_features_), 0.0);
        for (const int64_t index : indices)
        {
            for (int64_t feature = 0; feature < num_features_; ++feature)
            {
                node.center[static_cast<size_t>(feature)] += sampleValue(samples_, index, feature, num_features_);
            }
        }
        for (double &value : node.center)
        {
            value /= static_cast<double>(indices.size());
        }

        double radius = 0.0;
        for (const int64_t index : indices)
        {
            radius = std::max(radius,
                              distanceToCenter(samples_, index, num_features_, node.center, metric_, minkowski_p_));
        }
        node.radius = radius;

        if (static_cast<int64_t>(indices.size()) <= leaf_size_ || radius == 0.0)
        {
            node.leaf    = true;
            node.indices = std::move(indices);
            return node_index;
        }

        const int64_t split_dim = widestDimensionFromBounds(node.min_bounds, node.max_bounds);
        const size_t  mid       = indices.size() / 2;
        std::nth_element(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(mid), indices.end(),
                         [this, split_dim](int64_t lhs, int64_t rhs)
                         {
                             const double lhs_value = sampleValue(samples_, lhs, split_dim, num_features_);
                             const double rhs_value = sampleValue(samples_, rhs, split_dim, num_features_);
                             return lhs_value == rhs_value ? lhs < rhs : lhs_value < rhs_value;
                         });

        std::vector<int64_t> left(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(mid));
        std::vector<int64_t> right(indices.begin() + static_cast<std::ptrdiff_t>(mid), indices.end());
        const size_t         left_child  = build(left);
        const size_t         right_child = build(right);
        nodes_[node_index].left          = left_child;
        nodes_[node_index].right         = right_child;
        return node_index;
    }

    [[nodiscard]] double minSearchDistanceToNode(int64_t query, const Node &node) const
    {
        if (!use_ball_lower_bound_)
        {
            return 0.0;
        }

        double ball_lower = 0.0;
        if (useSquaredEuclideanDistance())
        {
            const double center_distance_sq
                = squaredDistanceToCenterEuclidean(samples_, query, num_features_, node.center);
            const double node_radius_sq = node.radius * node.radius;
            if (center_distance_sq > node_radius_sq)
            {
                const double lower_bound = std::sqrt(center_distance_sq) - node.radius;
                ball_lower               = lower_bound * lower_bound;
            }
        }
        else
        {
            const double center_distance
                = distanceToCenter(samples_, query, num_features_, node.center, metric_, minkowski_p_);
            if (center_distance > node.radius)
            {
                ball_lower = searchRadius(center_distance - node.radius);
            }
        }

        const double box_lower = distanceToBoundsSearch(samples_, query, num_features_, node.min_bounds,
                                                        node.max_bounds, metric_, minkowski_p_);
        return std::max(ball_lower, box_lower);
    }

    [[nodiscard]] bool nodeCanIntersectRadius(int64_t query, const Node &node, double search_radius) const
    {
        if (!use_ball_lower_bound_)
        {
            return true;
        }

        return minSearchDistanceToNode(query, node) <= search_radius;
    }

    [[nodiscard]] bool nodeFullyInsideRadius(int64_t query, const Node &node, double search_radius) const
    {
        return maxDistanceToBoundsSearch(samples_, query, num_features_, node.min_bounds, node.max_bounds, metric_,
                                         minkowski_p_)
               <= search_radius;
    }

    void appendSubtree(size_t node_index, std::vector<int64_t> &result) const
    {
        const auto &node = nodes_[node_index];
        if (node.leaf)
        {
            result.insert(result.end(), node.indices.begin(), node.indices.end());
            return;
        }
        appendSubtree(node.left, result);
        appendSubtree(node.right, result);
    }

    void radiusSearch(size_t node_index, int64_t query, double search_radius, std::vector<int64_t> &result) const
    {
        const auto &node = nodes_[node_index];
        if (!nodeCanIntersectRadius(query, node, search_radius))
        {
            return;
        }
        if (nodeFullyInsideRadius(query, node, search_radius))
        {
            appendSubtree(node_index, result);
            return;
        }

        if (node.leaf)
        {
            for (const int64_t index : node.indices)
            {
                if (searchDistance(query, index) <= search_radius)
                {
                    result.push_back(index);
                }
            }
            return;
        }

        radiusSearch(node.left, query, search_radius, result);
        radiusSearch(node.right, query, search_radius, result);
    }

    void radiusCountSearch(size_t node_index, int64_t query, double search_radius, int64_t stop_count,
                           int64_t &count) const
    {
        if (count >= stop_count)
        {
            return;
        }

        const auto &node = nodes_[node_index];
        if (!nodeCanIntersectRadius(query, node, search_radius))
        {
            return;
        }
        if (nodeFullyInsideRadius(query, node, search_radius))
        {
            count += node.size;
            return;
        }

        if (node.leaf)
        {
            for (const int64_t index : node.indices)
            {
                if (searchDistance(query, index) <= search_radius && ++count >= stop_count)
                {
                    return;
                }
            }
            return;
        }

        const auto &left_node    = nodes_[node.left];
        const auto &right_node   = nodes_[node.right];
        const bool  left_is_near
            = minSearchDistanceToNode(query, left_node) <= minSearchDistanceToNode(query, right_node);
        const auto  first_child  = left_is_near ? node.left : node.right;
        const auto  second_child = left_is_near ? node.right : node.left;
        radiusCountSearch(first_child, query, search_radius, stop_count, count);
        if (count < stop_count)
        {
            radiusCountSearch(second_child, query, search_radius, stop_count, count);
        }
    }

    void knnSearch(size_t node_index, int64_t query, int64_t kth, NeighborHeap &heap) const
    {
        const auto &node     = nodes_[node_index];
        const auto  min_dist = minSearchDistanceToNode(query, node);
        if (heap.size() >= kth && min_dist > heap.worstDistance())
        {
            return;
        }

        if (node.leaf)
        {
            for (const int64_t index : node.indices)
            {
                heap.push(searchDistance(query, index), index);
            }
            return;
        }

        const auto &left_node    = nodes_[node.left];
        const auto &right_node   = nodes_[node.right];
        const bool  left_is_near
            = minSearchDistanceToNode(query, left_node) <= minSearchDistanceToNode(query, right_node);
        const auto  first_child  = left_is_near ? node.left : node.right;
        const auto  second_child = left_is_near ? node.right : node.left;
        knnSearch(first_child, query, kth, heap);
        knnSearch(second_child, query, kth, heap);
    }

    const float      *samples_{nullptr};
    int64_t           num_features_{0};
    int64_t           leaf_size_{0};
    ClusteringMetric  metric_{ClusteringMetric::Euclidean};
    double            minkowski_p_{2.0};
    bool              use_ball_lower_bound_{true};
    std::vector<Node> nodes_;
    size_t            root_{kInvalidNode};
};

[[nodiscard]] std::vector<std::vector<int64_t>> bruteRadiusNeighborhoods(const float *samples, int64_t num_samples,
                                                                         int64_t num_features, double radius,
                                                                         ClusteringMetric metric, double minkowski_p)
{
    std::vector<std::vector<int64_t>> neighborhoods(static_cast<size_t>(num_samples));
    const double                      radius_sq = radius * radius;
    if (metric == ClusteringMetric::Euclidean)
    {
        for (int64_t sample = 0; sample < num_samples; ++sample)
        {
            neighborhoods[static_cast<size_t>(sample)].push_back(sample);
        }

        for (int64_t sample = 0; sample < num_samples; ++sample)
        {
            for (int64_t other = sample + 1; other < num_samples; ++other)
            {
                if (squaredEuclideanDistanceUnchecked(samples, sample, other, num_features) <= radius_sq)
                {
                    neighborhoods[static_cast<size_t>(sample)].push_back(other);
                    neighborhoods[static_cast<size_t>(other)].push_back(sample);
                }
            }
        }
        for (auto &neighbors : neighborhoods)
        {
            std::sort(neighbors.begin(), neighbors.end());
        }
        return neighborhoods;
    }

    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        auto &neighbors = neighborhoods[static_cast<size_t>(sample)];
        neighbors.reserve(static_cast<size_t>(num_samples));
        for (int64_t other = 0; other < num_samples; ++other)
        {
            if (clusteringDistance(samples, sample, other, num_features, metric, minkowski_p) <= radius)
            {
                neighbors.push_back(other);
            }
        }
    }
    return neighborhoods;
}

[[nodiscard]] std::vector<double> bruteKthNeighborSearchDistances(const float *samples, int64_t num_samples,
                                                                  int64_t num_features, int64_t kth,
                                                                  ClusteringMetric metric, double minkowski_p);

[[nodiscard]] std::vector<double> bruteKthNeighborDistances(const float *samples, int64_t num_samples,
                                                            int64_t num_features, int64_t kth, ClusteringMetric metric,
                                                            double minkowski_p)
{
    std::vector<double> result = bruteKthNeighborSearchDistances(samples, num_samples, num_features, kth, metric,
                                                                 minkowski_p);
    for (double &distance : result)
    {
        distance = outputDistanceForMetric(distance, metric, minkowski_p);
    }
    return result;
}

[[nodiscard]] std::vector<double> bruteKthNeighborSearchDistances(const float *samples, int64_t num_samples,
                                                                  int64_t num_features, int64_t kth,
                                                                  ClusteringMetric metric, double minkowski_p)
{
    std::vector<double> result(static_cast<size_t>(num_samples), 0.0);
    std::vector<double> row(static_cast<size_t>(num_samples), 0.0);
    const auto          nth = row.begin() + (kth - 1);
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        for (int64_t other = 0; other < num_samples; ++other)
        {
            row[static_cast<size_t>(other)]
                = clusteringSearchDistance(samples, sample, other, num_features, metric, minkowski_p);
        }
        std::nth_element(row.begin(), nth, row.end());
        result[static_cast<size_t>(sample)] = *nth;
    }
    return result;
}

} // namespace

RadiusNeighborhoodIndex::~RadiusNeighborhoodIndex() = default;

std::unique_ptr<RadiusNeighborhoodIndex>
makeRadiusNeighborhoodIndex(const float *samples, int64_t num_samples, int64_t num_features,
                            ClusteringAlgorithm algorithm, int64_t leaf_size, ClusteringMetric metric,
                            double minkowski_p)
{
    validateNeighborSearchMetric(algorithm, metric, minkowski_p);
    switch (algorithm)
    {
    case ClusteringAlgorithm::Brute:
        return std::make_unique<BruteRadiusIndex>(samples, num_samples, num_features, metric, minkowski_p);
    case ClusteringAlgorithm::KDTree:
        return std::make_unique<KDTreeIndex>(samples, num_samples, num_features, leaf_size, metric, minkowski_p);
    case ClusteringAlgorithm::BallTree:
        return std::make_unique<BallTreeIndex>(samples, num_samples, num_features, leaf_size, metric, minkowski_p);
    }

    throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering algorithm");
}

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

double clusteringDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features, ClusteringMetric metric,
                          double minkowski_p)
{
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
        return euclideanDistance(samples, lhs, rhs, num_features);
    case ClusteringMetric::Manhattan:
        return manhattanDistanceUnchecked(samples, lhs, rhs, num_features);
    case ClusteringMetric::Chebyshev:
        return chebyshevDistanceUnchecked(samples, lhs, rhs, num_features);
    case ClusteringMetric::Minkowski:
    {
        if (!std::isfinite(minkowski_p) || minkowski_p <= 0.0)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "minkowski_p must be positive and finite");
        }
        return minkowskiRoot(minkowskiPoweredDistanceUnchecked(samples, lhs, rhs, num_features, minkowski_p),
                             minkowski_p);
    }
    case ClusteringMetric::Cosine:
        return cosineDistanceUnchecked(samples, lhs, rhs, num_features);
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

double clusteringSearchDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features,
                                ClusteringMetric metric, double minkowski_p)
{
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
        return squaredEuclideanDistanceUnchecked(samples, lhs, rhs, num_features);
    case ClusteringMetric::Manhattan:
        return manhattanDistanceUnchecked(samples, lhs, rhs, num_features);
    case ClusteringMetric::Chebyshev:
        return chebyshevDistanceUnchecked(samples, lhs, rhs, num_features);
    case ClusteringMetric::Minkowski:
    {
        if (!std::isfinite(minkowski_p) || minkowski_p <= 0.0)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "minkowski_p must be positive and finite");
        }
        return minkowskiPoweredDistanceUnchecked(samples, lhs, rhs, num_features, minkowski_p);
    }
    case ClusteringMetric::Cosine:
        return cosineDistanceUnchecked(samples, lhs, rhs, num_features);
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

double clusteringSearchRadius(double radius, ClusteringMetric metric, double minkowski_p)
{
    return searchRadiusForMetric(radius, metric, minkowski_p);
}

double clusteringOutputDistance(double search_distance, ClusteringMetric metric, double minkowski_p)
{
    return outputDistanceForMetric(search_distance, metric, minkowski_p);
}

std::vector<std::vector<int64_t>> radiusNeighborhoods(const float *samples, int64_t num_samples, int64_t num_features,
                                                      double radius, ClusteringAlgorithm algorithm, int64_t leaf_size,
                                                      ClusteringMetric metric, double minkowski_p)
{
    if (num_samples == 0)
    {
        return {};
    }
    if (!std::isfinite(radius) || radius < 0.0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "radius must be finite and non-negative");
    }

    validateNeighborSearchMetric(algorithm, metric, minkowski_p);
    switch (algorithm)
    {
    case ClusteringAlgorithm::Brute:
        return bruteRadiusNeighborhoods(samples, num_samples, num_features, radius, metric, minkowski_p);
    case ClusteringAlgorithm::KDTree:
    {
        KDTreeIndex                       index(samples, num_samples, num_features, leaf_size, metric, minkowski_p);
        std::vector<std::vector<int64_t>> neighborhoods(static_cast<size_t>(num_samples));
        for (int64_t sample = 0; sample < num_samples; ++sample)
        {
            index.radiusNeighbors(sample, radius, neighborhoods[static_cast<size_t>(sample)], true);
        }
        return neighborhoods;
    }
    case ClusteringAlgorithm::BallTree:
    {
        BallTreeIndex                     index(samples, num_samples, num_features, leaf_size, metric, minkowski_p);
        std::vector<std::vector<int64_t>> neighborhoods(static_cast<size_t>(num_samples));
        for (int64_t sample = 0; sample < num_samples; ++sample)
        {
            index.radiusNeighbors(sample, radius, neighborhoods[static_cast<size_t>(sample)], true);
        }
        return neighborhoods;
    }
    }

    throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering algorithm");
}

std::vector<double> kthNeighborDistances(const float *samples, int64_t num_samples, int64_t num_features, int64_t kth,
                                         ClusteringAlgorithm algorithm, int64_t leaf_size, ClusteringMetric metric,
                                         double minkowski_p)
{
    auto result = kthNeighborSearchDistances(samples, num_samples, num_features, kth, algorithm, leaf_size, metric,
                                             minkowski_p);
    for (double &distance : result)
    {
        distance = outputDistanceForMetric(distance, metric, minkowski_p);
    }
    return result;
}

std::vector<double> kthNeighborSearchDistances(const float *samples, int64_t num_samples, int64_t num_features,
                                               int64_t kth, ClusteringAlgorithm algorithm, int64_t leaf_size,
                                               ClusteringMetric metric, double minkowski_p)
{
    if (num_samples == 0)
    {
        return {};
    }
    if (kth < 1 || kth > num_samples)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "kth must be positive and at most num_samples, got %lld",
                        static_cast<long long>(kth));
    }

    validateNeighborSearchMetric(algorithm, metric, minkowski_p);
    switch (algorithm)
    {
    case ClusteringAlgorithm::Brute:
        return bruteKthNeighborSearchDistances(samples, num_samples, num_features, kth, metric, minkowski_p);
    case ClusteringAlgorithm::KDTree:
    {
        KDTreeIndex         index(samples, num_samples, num_features, leaf_size, metric, minkowski_p);
        std::vector<double> result(static_cast<size_t>(num_samples), 0.0);
        index.kthSearchDistances(kth, result);
        return result;
    }
    case ClusteringAlgorithm::BallTree:
    {
        BallTreeIndex       index(samples, num_samples, num_features, leaf_size, metric, minkowski_p);
        std::vector<double> result(static_cast<size_t>(num_samples), 0.0);
        index.kthSearchDistances(kth, result);
        return result;
    }
    }

    throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering algorithm");
}

} // namespace irt::ops::detail
