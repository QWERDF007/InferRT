#include "priv/ClusteringCommon.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/DBSCAN.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace irt::ops {
namespace {

class UnionFind final
{
public:
    explicit UnionFind(int64_t size)
        : parent_(static_cast<size_t>(size))
        , rank_(static_cast<size_t>(size), uint8_t{0})
    {
        for (int64_t index = 0; index < size; ++index)
        {
            parent_[static_cast<size_t>(index)] = index;
        }
    }

    [[nodiscard]] int64_t find(int64_t node)
    {
        int64_t root = node;
        while (parent_[static_cast<size_t>(root)] != root)
        {
            root = parent_[static_cast<size_t>(root)];
        }
        while (parent_[static_cast<size_t>(node)] != node)
        {
            const int64_t next                 = parent_[static_cast<size_t>(node)];
            parent_[static_cast<size_t>(node)] = root;
            node                               = next;
        }
        return root;
    }

    void unite(int64_t lhs, int64_t rhs)
    {
        int64_t lhs_root = find(lhs);
        int64_t rhs_root = find(rhs);
        if (lhs_root == rhs_root)
        {
            return;
        }
        if (rank_[static_cast<size_t>(lhs_root)] < rank_[static_cast<size_t>(rhs_root)])
        {
            std::swap(lhs_root, rhs_root);
        }
        parent_[static_cast<size_t>(rhs_root)] = lhs_root;
        if (rank_[static_cast<size_t>(lhs_root)] == rank_[static_cast<size_t>(rhs_root)])
        {
            ++rank_[static_cast<size_t>(lhs_root)];
        }
    }

private:
    std::vector<int64_t> parent_;
    std::vector<uint8_t> rank_;
};

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

[[nodiscard]] double squaredEuclideanDistance3(const float *samples, int64_t lhs, int64_t rhs)
{
    const float *lhs_ptr = samples + lhs * 3;
    const float *rhs_ptr = samples + rhs * 3;
    const double diff0   = static_cast<double>(lhs_ptr[0]) - static_cast<double>(rhs_ptr[0]);
    const double diff1   = static_cast<double>(lhs_ptr[1]) - static_cast<double>(rhs_ptr[1]);
    const double diff2   = static_cast<double>(lhs_ptr[2]) - static_cast<double>(rhs_ptr[2]);
    return diff0 * diff0 + diff1 * diff1 + diff2 * diff2;
}

[[nodiscard]] double dbscanSquaredEuclideanDistance(const float *samples, int64_t lhs, int64_t rhs,
                                                    int64_t num_features)
{
    return num_features == 3 ? squaredEuclideanDistance3(samples, lhs, rhs)
                             : squaredEuclideanDistanceUnchecked(samples, lhs, rhs, num_features);
}

[[nodiscard]] bool isMinkowskiP3(double minkowski_p)
{
    return std::abs(minkowski_p - 3.0) <= 1e-12;
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

[[nodiscard]] std::vector<double> cosineInverseNorms(const float *samples, int64_t num_samples, int64_t num_features)
{
    std::vector<double> result(static_cast<size_t>(num_samples), 0.0);
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        const float *sample_ptr = samples + sample * num_features;
        double       norm       = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const double value = static_cast<double>(sample_ptr[feature]);
            norm += value * value;
        }
        if (norm > 0.0)
        {
            result[static_cast<size_t>(sample)] = 1.0 / std::sqrt(norm);
        }
    }
    return result;
}

[[nodiscard]] double cosineDistanceUnchecked(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features,
                                             const std::vector<double> &inverse_norms)
{
    const double lhs_inv_norm = inverse_norms[static_cast<size_t>(lhs)];
    const double rhs_inv_norm = inverse_norms[static_cast<size_t>(rhs)];
    if (lhs_inv_norm == 0.0 && rhs_inv_norm == 0.0)
    {
        return 0.0;
    }
    if (lhs_inv_norm == 0.0 || rhs_inv_norm == 0.0)
    {
        return 1.0;
    }

    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;
    double       dot     = 0.0;
    if (num_features == 3)
    {
        dot = static_cast<double>(lhs_ptr[0]) * static_cast<double>(rhs_ptr[0])
              + static_cast<double>(lhs_ptr[1]) * static_cast<double>(rhs_ptr[1])
              + static_cast<double>(lhs_ptr[2]) * static_cast<double>(rhs_ptr[2]);
    }
    else
    {
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            dot += static_cast<double>(lhs_ptr[feature]) * static_cast<double>(rhs_ptr[feature]);
        }
    }

    const double similarity = std::clamp(dot * lhs_inv_norm * rhs_inv_norm, -1.0, 1.0);
    return 1.0 - similarity;
}

[[nodiscard]] DBSCANResult dbscanEuclideanBrute(const float *samples, int64_t num_samples, int64_t num_features,
                                                const DBSCANConfig &config)
{
    const double radius_sq = static_cast<double>(config.eps) * static_cast<double>(config.eps);

    std::vector<int64_t> neighbor_counts(static_cast<size_t>(num_samples), int64_t{1});
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        for (int64_t other = sample + 1; other < num_samples; ++other)
        {
            if (dbscanSquaredEuclideanDistance(samples, sample, other, num_features) <= radius_sq)
            {
                ++neighbor_counts[static_cast<size_t>(sample)];
                ++neighbor_counts[static_cast<size_t>(other)];
            }
        }
    }

    std::vector<uint8_t> is_core(static_cast<size_t>(num_samples), uint8_t{0});
    DBSCANResult         result;
    result.labels.assign(static_cast<size_t>(num_samples), int64_t{-1});
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (neighbor_counts[static_cast<size_t>(sample)] >= config.min_samples)
        {
            is_core[static_cast<size_t>(sample)] = uint8_t{1};
            result.core_sample_indices.push_back(sample);
        }
    }

    UnionFind union_find(num_samples);
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (!is_core[static_cast<size_t>(sample)])
        {
            continue;
        }

        for (int64_t other = sample + 1; other < num_samples; ++other)
        {
            if (!is_core[static_cast<size_t>(other)])
            {
                continue;
            }
            if (dbscanSquaredEuclideanDistance(samples, sample, other, num_features) <= radius_sq)
            {
                union_find.unite(sample, other);
            }
        }
    }

    std::vector<int64_t> component_labels(static_cast<size_t>(num_samples), int64_t{-1});
    int64_t              next_label = 0;
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (!is_core[static_cast<size_t>(sample)])
        {
            continue;
        }
        const int64_t root = union_find.find(sample);
        auto         &label = component_labels[static_cast<size_t>(root)];
        if (label == -1)
        {
            label = next_label++;
        }
        result.labels[static_cast<size_t>(sample)] = label;
    }

    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (is_core[static_cast<size_t>(sample)])
        {
            continue;
        }

        int64_t best_label = std::numeric_limits<int64_t>::max();
        for (int64_t other = 0; other < num_samples; ++other)
        {
            if (!is_core[static_cast<size_t>(other)])
            {
                continue;
            }
            if (dbscanSquaredEuclideanDistance(samples, sample, other, num_features) <= radius_sq)
            {
                best_label = std::min(best_label, result.labels[static_cast<size_t>(other)]);
            }
        }
        if (best_label != std::numeric_limits<int64_t>::max())
        {
            result.labels[static_cast<size_t>(sample)] = best_label;
        }
    }

    return result;
}

constexpr size_t kMaxCachedAdjacencyBytes = 256ull * 1024ull * 1024ull;

class PairAdjacencyCache final
{
public:
    explicit PairAdjacencyCache(int64_t num_samples)
        : num_samples_(num_samples)
    {
        const size_t pair_count = pairCount(num_samples_);
        const size_t word_count = (pair_count + 63ull) / 64ull;
        if (word_count == 0 || word_count > kMaxCachedAdjacencyBytes / sizeof(uint64_t))
        {
            return;
        }
        bits_.assign(word_count, uint64_t{0});
    }

    [[nodiscard]] bool enabled() const
    {
        return !bits_.empty();
    }

    void set(int64_t lhs, int64_t rhs)
    {
        const size_t index = pairIndex(lhs, rhs, num_samples_);
        bits_[index / 64ull] |= uint64_t{1} << (index % 64ull);
    }

    [[nodiscard]] bool test(int64_t lhs, int64_t rhs) const
    {
        if (lhs == rhs)
        {
            return true;
        }
        const size_t index = pairIndex(lhs, rhs, num_samples_);
        return (bits_[index / 64ull] & (uint64_t{1} << (index % 64ull))) != 0;
    }

private:
    [[nodiscard]] static size_t pairCount(int64_t num_samples)
    {
        if (num_samples <= 1)
        {
            return 0;
        }
        const auto n = static_cast<size_t>(num_samples);
        return n * (n - 1ull) / 2ull;
    }

    [[nodiscard]] static size_t pairIndex(int64_t lhs, int64_t rhs, int64_t num_samples)
    {
        if (lhs > rhs)
        {
            std::swap(lhs, rhs);
        }
        const auto n   = static_cast<size_t>(num_samples);
        const auto row = static_cast<size_t>(lhs);
        return row * (2ull * n - row - 1ull) / 2ull + static_cast<size_t>(rhs - lhs - 1);
    }

    int64_t               num_samples_{0};
    std::vector<uint64_t> bits_;
};

template <typename Distance>
[[nodiscard]] DBSCANResult dbscanBruteWithDistance(int64_t num_samples, const DBSCANConfig &config,
                                                   double search_radius, Distance distance)
{
    std::vector<int64_t> neighbor_counts(static_cast<size_t>(num_samples), int64_t{1});
    PairAdjacencyCache    adjacency(num_samples);
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        for (int64_t other = sample + 1; other < num_samples; ++other)
        {
            if (distance(sample, other) <= search_radius)
            {
                ++neighbor_counts[static_cast<size_t>(sample)];
                ++neighbor_counts[static_cast<size_t>(other)];
                if (adjacency.enabled())
                {
                    adjacency.set(sample, other);
                }
            }
        }
    }

    std::vector<uint8_t> is_core(static_cast<size_t>(num_samples), uint8_t{0});
    DBSCANResult         result;
    result.labels.assign(static_cast<size_t>(num_samples), int64_t{-1});
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (neighbor_counts[static_cast<size_t>(sample)] >= config.min_samples)
        {
            is_core[static_cast<size_t>(sample)] = uint8_t{1};
            result.core_sample_indices.push_back(sample);
        }
    }

    UnionFind union_find(num_samples);
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (!is_core[static_cast<size_t>(sample)])
        {
            continue;
        }

        for (int64_t other = sample + 1; other < num_samples; ++other)
        {
            if (!is_core[static_cast<size_t>(other)])
            {
                continue;
            }
            const bool within_radius
                = adjacency.enabled() ? adjacency.test(sample, other) : distance(sample, other) <= search_radius;
            if (within_radius)
            {
                union_find.unite(sample, other);
            }
        }
    }

    std::vector<int64_t> component_labels(static_cast<size_t>(num_samples), int64_t{-1});
    int64_t              next_label = 0;
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (!is_core[static_cast<size_t>(sample)])
        {
            continue;
        }
        const int64_t root = union_find.find(sample);
        auto         &label = component_labels[static_cast<size_t>(root)];
        if (label == -1)
        {
            label = next_label++;
        }
        result.labels[static_cast<size_t>(sample)] = label;
    }

    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (is_core[static_cast<size_t>(sample)])
        {
            continue;
        }

        int64_t best_label = std::numeric_limits<int64_t>::max();
        for (int64_t other = 0; other < num_samples; ++other)
        {
            if (!is_core[static_cast<size_t>(other)])
            {
                continue;
            }
            const bool within_radius
                = adjacency.enabled() ? adjacency.test(sample, other) : distance(sample, other) <= search_radius;
            if (within_radius)
            {
                best_label = std::min(best_label, result.labels[static_cast<size_t>(other)]);
            }
        }
        if (best_label != std::numeric_limits<int64_t>::max())
        {
            result.labels[static_cast<size_t>(sample)] = best_label;
        }
    }

    return result;
}

[[nodiscard]] DBSCANResult dbscanBrute(const float *samples, int64_t num_samples, int64_t num_features,
                                       const DBSCANConfig &config)
{
    const double search_radius = detail::clusteringSearchRadius(config.eps, config.metric, config.minkowski_p);
    switch (config.metric)
    {
    case ClusteringMetric::Manhattan:
        return dbscanBruteWithDistance(num_samples, config, search_radius,
                                       [samples, num_features](int64_t lhs, int64_t rhs)
                                       { return manhattanDistanceUnchecked(samples, lhs, rhs, num_features); });
    case ClusteringMetric::Chebyshev:
        return dbscanBruteWithDistance(num_samples, config, search_radius,
                                       [samples, num_features](int64_t lhs, int64_t rhs)
                                       { return chebyshevDistanceUnchecked(samples, lhs, rhs, num_features); });
    case ClusteringMetric::Minkowski:
    {
        const double minkowski_p = config.minkowski_p;
        return dbscanBruteWithDistance(num_samples, config, search_radius,
                                       [samples, num_features, minkowski_p](int64_t lhs, int64_t rhs)
                                       {
                                           return minkowskiPoweredDistanceUnchecked(samples, lhs, rhs, num_features,
                                                                                   minkowski_p);
                                       });
    }
    case ClusteringMetric::Cosine:
    {
        const auto inverse_norms = cosineInverseNorms(samples, num_samples, num_features);
        return dbscanBruteWithDistance(num_samples, config, search_radius,
                                       [samples, num_features, &inverse_norms](int64_t lhs, int64_t rhs)
                                       {
                                           return cosineDistanceUnchecked(samples, lhs, rhs, num_features,
                                                                          inverse_norms);
                                       });
    }
    case ClusteringMetric::Euclidean:
        break;
    }

    return dbscanBruteWithDistance(num_samples, config, search_radius,
                                   [samples, num_features, metric = config.metric,
                                    minkowski_p = config.minkowski_p](int64_t lhs, int64_t rhs)
                                   {
                                       return detail::clusteringSearchDistance(samples, lhs, rhs, num_features,
                                                                               metric, minkowski_p);
                                   });
}

[[nodiscard]] DBSCANResult dbscanIndexed(const float *samples, int64_t num_samples, int64_t num_features,
                                         const DBSCANConfig &config)
{
    auto index = detail::makeRadiusNeighborhoodIndex(samples, num_samples, num_features, config.algorithm,
                                                     config.leaf_size, config.metric, config.minkowski_p);

    std::vector<uint8_t> is_core(static_cast<size_t>(num_samples), uint8_t{0});
    DBSCANResult         result;
    result.labels.assign(static_cast<size_t>(num_samples), int64_t{-1});
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (index->radiusNeighborCount(sample, config.eps, config.min_samples) >= config.min_samples)
        {
            is_core[static_cast<size_t>(sample)] = uint8_t{1};
            result.core_sample_indices.push_back(sample);
        }
    }

    UnionFind            union_find(num_samples);
    std::vector<int64_t> neighbors;
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (!is_core[static_cast<size_t>(sample)])
        {
            continue;
        }

        index->radiusNeighbors(sample, config.eps, neighbors, false);
        for (const int64_t other : neighbors)
        {
            if (other > sample && is_core[static_cast<size_t>(other)])
            {
                union_find.unite(sample, other);
            }
        }
    }

    std::vector<int64_t> component_labels(static_cast<size_t>(num_samples), int64_t{-1});
    int64_t              next_label = 0;
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (!is_core[static_cast<size_t>(sample)])
        {
            continue;
        }
        const int64_t root = union_find.find(sample);
        auto         &label = component_labels[static_cast<size_t>(root)];
        if (label == -1)
        {
            label = next_label++;
        }
        result.labels[static_cast<size_t>(sample)] = label;
    }

    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        if (is_core[static_cast<size_t>(sample)])
        {
            continue;
        }

        int64_t best_label = std::numeric_limits<int64_t>::max();
        index->radiusNeighbors(sample, config.eps, neighbors, false);
        for (const int64_t other : neighbors)
        {
            if (is_core[static_cast<size_t>(other)])
            {
                best_label = std::min(best_label, result.labels[static_cast<size_t>(other)]);
            }
        }
        if (best_label != std::numeric_limits<int64_t>::max())
        {
            result.labels[static_cast<size_t>(sample)] = best_label;
        }
    }

    return result;
}

void validateInputs(const float *samples, int64_t num_samples, int64_t num_features, const DBSCANConfig &config)
{
    detail::validateSampleMatrix(samples, num_samples, num_features);
    detail::validateNeighborSearchConfig(config.algorithm, config.leaf_size);
    detail::validateMetricConfig(config.metric, config.minkowski_p);
    if (!std::isfinite(config.eps) || config.eps <= 0.0f)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "eps must be positive and finite");
    }
    if (config.min_samples < 1)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "min_samples must be positive, got %lld",
                        static_cast<long long>(config.min_samples));
    }
}

} // namespace

DBSCANResult dbscan(const float *samples, int64_t num_samples, int64_t num_features, const DBSCANConfig &config)
{
    validateInputs(samples, num_samples, num_features, config);
    if (num_samples == 0)
    {
        return {};
    }
    if (config.metric == ClusteringMetric::Euclidean && config.algorithm == ClusteringAlgorithm::Brute)
    {
        return dbscanEuclideanBrute(samples, num_samples, num_features, config);
    }
    if (config.algorithm == ClusteringAlgorithm::Brute)
    {
        return dbscanBrute(samples, num_samples, num_features, config);
    }
    return dbscanIndexed(samples, num_samples, num_features, config);
}

} // namespace irt::ops
