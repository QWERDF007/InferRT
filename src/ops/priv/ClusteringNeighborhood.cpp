#include "ClusteringNeighborhood.hpp"
#include "ClusteringTrees.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace irt::ops::detail {
namespace {

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
        , distance_(samples, num_features, metric, minkowski_p)
    {
    }

    void radiusNeighbors(int64_t query, double radius, std::vector<int64_t> &result,
                         bool sort_result) const override
    {
        result.clear();
        result.reserve(static_cast<size_t>(num_samples_));

        const double search_radius = clusteringSearchRadius(radius, metric_, minkowski_p_);
        int64_t other = 0;
        if (distance_.canUseBlock4())
        {
            for (; other + 3 < num_samples_; other += 4)
            {
                const int mask = distance_.withinRadiusMask4(query, other, search_radius);
                if ((mask & 1) != 0)
                {
                    result.push_back(other);
                }
                if ((mask & 2) != 0)
                {
                    result.push_back(other + 1);
                }
                if ((mask & 4) != 0)
                {
                    result.push_back(other + 2);
                }
                if ((mask & 8) != 0)
                {
                    result.push_back(other + 3);
                }
            }
        }
        for (; other < num_samples_; ++other)
        {
            if (distance_(query, other) <= search_radius)
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
        const double  search_radius = clusteringSearchRadius(radius, metric_, minkowski_p_);

        int64_t count = 0;
        int64_t other = 0;
        if (distance_.canUseBlock4())
        {
            for (; other + 3 < num_samples_; other += 4)
            {
                const int mask = distance_.withinRadiusMask4(query, other, search_radius);
                const int matches = ((mask & 1) != 0) + ((mask & 2) != 0) + ((mask & 4) != 0) + ((mask & 8) != 0);
                if (count + matches < limit)
                {
                    count += matches;
                    continue;
                }
                if ((mask & 1) != 0 && ++count >= limit)
                {
                    return count;
                }
                if ((mask & 2) != 0 && ++count >= limit)
                {
                    return count;
                }
                if ((mask & 4) != 0 && ++count >= limit)
                {
                    return count;
                }
                if ((mask & 8) != 0 && ++count >= limit)
                {
                    return count;
                }
            }
        }
        for (; other < num_samples_; ++other)
        {
            if (distance_(query, other) <= search_radius && ++count >= limit)
            {
                return count;
            }
        }
        return count;
    }

private:
    const float             *samples_{nullptr};
    int64_t                  num_samples_{0};
    int64_t                  num_features_{0};
    ClusteringMetric         metric_{ClusteringMetric::Euclidean};
    double                   minkowski_p_{2.0};
    SearchDistanceCalculator distance_;
};

[[nodiscard]] std::vector<std::vector<int64_t>> bruteRadiusNeighborhoods(const float *samples, int64_t num_samples,
                                                                         int64_t num_features, double radius,
                                                                         ClusteringMetric metric, double minkowski_p)
{
    std::vector<std::vector<int64_t>> neighborhoods(static_cast<size_t>(num_samples));
    BruteRadiusIndex                  index(samples, num_samples, num_features, metric, minkowski_p);
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        index.radiusNeighbors(sample, radius, neighborhoods[static_cast<size_t>(sample)], true);
    }
    return neighborhoods;
}

[[nodiscard]] std::vector<double> bruteKthNeighborSearchDistances(const float *samples, int64_t num_samples,
                                                                   int64_t num_features, int64_t kth,
                                                                   ClusteringMetric metric, double minkowski_p)
{
    std::vector<double> inverse_norms;
    const std::vector<double> *inverse_norms_ptr = nullptr;
    if (metric == ClusteringMetric::Cosine)
    {
        inverse_norms     = cosineInverseNorms(samples, num_samples, num_features);
        inverse_norms_ptr = &inverse_norms;
    }

    SearchDistanceCalculator distance(samples, num_features, metric, minkowski_p, inverse_norms_ptr);
    std::vector<double>      result(static_cast<size_t>(num_samples), 0.0);
    std::vector<double>      row(static_cast<size_t>(num_samples), 0.0);
    const auto               nth = row.begin() + (kth - 1);
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        int64_t other = 0;
        if (distance.canUseBlock4())
        {
            for (; other + 3 < num_samples; other += 4)
            {
                const DistanceBlock4 distances = distance.block4(sample, other);
                row[static_cast<size_t>(other)]     = distances.first;
                row[static_cast<size_t>(other + 1)] = distances.second;
                row[static_cast<size_t>(other + 2)] = distances.third;
                row[static_cast<size_t>(other + 3)] = distances.fourth;
            }
        }
        for (; other < num_samples; ++other)
        {
            row[static_cast<size_t>(other)] = distance(sample, other);
        }
        std::nth_element(row.begin(), nth, row.end());
        result[static_cast<size_t>(sample)] = *nth;
    }
    return result;
}

[[nodiscard]] std::vector<double> bruteKthNeighborDistances(const float *samples, int64_t num_samples,
                                                             int64_t num_features, int64_t kth, ClusteringMetric metric,
                                                             double minkowski_p)
{
    std::vector<double> result
        = bruteKthNeighborSearchDistances(samples, num_samples, num_features, kth, metric, minkowski_p);
    for (double &distance : result)
    {
        distance = clusteringOutputDistance(distance, metric, minkowski_p);
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
        return makeKDTreeNeighborhoodIndex(samples, num_samples, num_features, leaf_size, metric, minkowski_p);
    case ClusteringAlgorithm::BallTree:
        return makeBallTreeNeighborhoodIndex(samples, num_samples, num_features, leaf_size, metric, minkowski_p);
    }

    throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering algorithm");
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
        auto                              index = makeKDTreeNeighborhoodIndex(samples, num_samples, num_features, leaf_size,
                                                                               metric, minkowski_p);
        std::vector<std::vector<int64_t>> neighborhoods(static_cast<size_t>(num_samples));
        for (int64_t sample = 0; sample < num_samples; ++sample)
        {
            index->radiusNeighbors(sample, radius, neighborhoods[static_cast<size_t>(sample)], true);
        }
        return neighborhoods;
    }
    case ClusteringAlgorithm::BallTree:
    {
        auto                              index = makeBallTreeNeighborhoodIndex(samples, num_samples, num_features,
                                                                                 leaf_size, metric, minkowski_p);
        std::vector<std::vector<int64_t>> neighborhoods(static_cast<size_t>(num_samples));
        for (int64_t sample = 0; sample < num_samples; ++sample)
        {
            index->radiusNeighbors(sample, radius, neighborhoods[static_cast<size_t>(sample)], true);
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
        distance = clusteringOutputDistance(distance, metric, minkowski_p);
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
        std::vector<double> result(static_cast<size_t>(num_samples), 0.0);
        kdTreeKthSearchDistances(samples, num_samples, num_features, leaf_size, metric, minkowski_p, kth, result);
        return result;
    }
    case ClusteringAlgorithm::BallTree:
    {
        std::vector<double> result(static_cast<size_t>(num_samples), 0.0);
        ballTreeKthSearchDistances(samples, num_samples, num_features, leaf_size, metric, minkowski_p, kth, result);
        return result;
    }
    }

    throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering algorithm");
}

} // namespace irt::ops::detail
