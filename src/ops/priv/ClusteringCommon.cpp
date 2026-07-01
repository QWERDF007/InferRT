#include "ClusteringCommon.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

namespace irt::ops::detail {
namespace {

constexpr int64_t kAutoTreeFeatureLimit = 15;

using NeighborHeapItem = std::pair<double, int64_t>;

[[nodiscard]] double sampleValue(const float *samples, int64_t sample, int64_t feature, int64_t num_features)
{
    return static_cast<double>(samples[sample * num_features + feature]);
}

[[nodiscard]] double squaredDistanceToCenter(const float *samples, int64_t sample, int64_t num_features,
                                             const std::vector<double> &center)
{
    double sum = 0.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        const double diff = sampleValue(samples, sample, feature, num_features) - center[static_cast<size_t>(feature)];
        sum += diff * diff;
    }
    return sum;
}

void pushNeighbor(std::priority_queue<NeighborHeapItem> &heap, int64_t max_size, double distance_sq, int64_t index)
{
    if (static_cast<int64_t>(heap.size()) < max_size)
    {
        heap.emplace(distance_sq, index);
        return;
    }

    const auto &worst = heap.top();
    if (distance_sq < worst.first || (distance_sq == worst.first && index < worst.second))
    {
        heap.pop();
        heap.emplace(distance_sq, index);
    }
}

[[nodiscard]] int64_t widestDimension(const float *samples, const std::vector<int64_t> &indices, int64_t num_features)
{
    int64_t best_dim   = 0;
    double  best_range = -1.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        double min_value = std::numeric_limits<double>::infinity();
        double max_value = -std::numeric_limits<double>::infinity();
        for (const int64_t index : indices)
        {
            const double value = sampleValue(samples, index, feature, num_features);
            min_value          = std::min(min_value, value);
            max_value          = std::max(max_value, value);
        }

        const double range = max_value - min_value;
        if (range > best_range)
        {
            best_dim   = feature;
            best_range = range;
        }
    }
    return best_dim;
}

class KDTreeIndex final
{
public:
    KDTreeIndex(const float *samples, int64_t num_samples, int64_t num_features, int64_t leaf_size)
        : samples_(samples)
        , num_features_(num_features)
        , leaf_size_(leaf_size)
    {
        std::vector<int64_t> indices(static_cast<size_t>(num_samples));
        std::iota(indices.begin(), indices.end(), int64_t{0});
        root_ = build(indices);
    }

    [[nodiscard]] std::vector<int64_t> radiusNeighbors(int64_t query, double radius_sq) const
    {
        std::vector<int64_t> result;
        radiusSearch(root_, query, radius_sq, result);
        std::sort(result.begin(), result.end());
        return result;
    }

    [[nodiscard]] double kthDistance(int64_t query, int64_t kth) const
    {
        std::priority_queue<NeighborHeapItem> heap;
        knnSearch(root_, query, kth, heap);
        return std::sqrt(heap.top().first);
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
        std::vector<int64_t> indices;
    };

    [[nodiscard]] size_t build(std::vector<int64_t> &indices)
    {
        const size_t node_index = nodes_.size();
        nodes_.push_back({});
        auto &node = nodes_.back();

        if (static_cast<int64_t>(indices.size()) <= leaf_size_)
        {
            node.leaf    = true;
            node.indices = std::move(indices);
            return node_index;
        }

        const int64_t split_dim = widestDimension(samples_, indices, num_features_);
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

    void radiusSearch(size_t node_index, int64_t query, double radius_sq, std::vector<int64_t> &result) const
    {
        const auto &node = nodes_[node_index];
        if (node.leaf)
        {
            for (const int64_t index : node.indices)
            {
                if (squaredEuclideanDistance(samples_, query, index, num_features_) <= radius_sq)
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

        radiusSearch(near_child, query, radius_sq, result);
        if (diff * diff <= radius_sq)
        {
            radiusSearch(far_child, query, radius_sq, result);
        }
    }

    void knnSearch(size_t node_index, int64_t query, int64_t kth, std::priority_queue<NeighborHeapItem> &heap) const
    {
        const auto &node = nodes_[node_index];
        if (node.leaf)
        {
            for (const int64_t index : node.indices)
            {
                pushNeighbor(heap, kth, squaredEuclideanDistance(samples_, query, index, num_features_), index);
            }
            return;
        }

        const double query_value = sampleValue(samples_, query, node.split_dim, num_features_);
        const double diff        = query_value - node.split_value;
        const size_t near_child  = diff <= 0.0 ? node.left : node.right;
        const size_t far_child   = diff <= 0.0 ? node.right : node.left;

        knnSearch(near_child, query, kth, heap);
        if (static_cast<int64_t>(heap.size()) < kth || diff * diff <= heap.top().first)
        {
            knnSearch(far_child, query, kth, heap);
        }
    }

    const float      *samples_{nullptr};
    int64_t           num_features_{0};
    int64_t           leaf_size_{0};
    std::vector<Node> nodes_;
    size_t            root_{kInvalidNode};
};

class BallTreeIndex final
{
public:
    BallTreeIndex(const float *samples, int64_t num_samples, int64_t num_features, int64_t leaf_size)
        : samples_(samples)
        , num_features_(num_features)
        , leaf_size_(leaf_size)
    {
        std::vector<int64_t> indices(static_cast<size_t>(num_samples));
        std::iota(indices.begin(), indices.end(), int64_t{0});
        root_ = build(indices);
    }

    [[nodiscard]] std::vector<int64_t> radiusNeighbors(int64_t query, double radius_sq) const
    {
        std::vector<int64_t> result;
        radiusSearch(root_, query, radius_sq, result);
        std::sort(result.begin(), result.end());
        return result;
    }

    [[nodiscard]] double kthDistance(int64_t query, int64_t kth) const
    {
        std::priority_queue<NeighborHeapItem> heap;
        knnSearch(root_, query, kth, heap);
        return std::sqrt(heap.top().first);
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
        std::vector<int64_t> indices;
    };

    [[nodiscard]] size_t build(std::vector<int64_t> &indices)
    {
        const size_t node_index = nodes_.size();
        nodes_.push_back({});
        auto &node = nodes_.back();

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

        double radius_sq = 0.0;
        for (const int64_t index : indices)
        {
            radius_sq = std::max(radius_sq, squaredDistanceToCenter(samples_, index, num_features_, node.center));
        }
        node.radius = std::sqrt(radius_sq);

        if (static_cast<int64_t>(indices.size()) <= leaf_size_ || radius_sq == 0.0)
        {
            node.leaf    = true;
            node.indices = std::move(indices);
            return node_index;
        }

        const int64_t split_dim = widestDimension(samples_, indices, num_features_);
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

    [[nodiscard]] double minDistanceToNodeSq(int64_t query, const Node &node) const
    {
        const double center_distance = std::sqrt(squaredDistanceToCenter(samples_, query, num_features_, node.center));
        if (center_distance <= node.radius)
        {
            return 0.0;
        }
        const double diff = center_distance - node.radius;
        return diff * diff;
    }

    void radiusSearch(size_t node_index, int64_t query, double radius_sq, std::vector<int64_t> &result) const
    {
        const auto &node = nodes_[node_index];
        if (minDistanceToNodeSq(query, node) > radius_sq)
        {
            return;
        }

        if (node.leaf)
        {
            for (const int64_t index : node.indices)
            {
                if (squaredEuclideanDistance(samples_, query, index, num_features_) <= radius_sq)
                {
                    result.push_back(index);
                }
            }
            return;
        }

        radiusSearch(node.left, query, radius_sq, result);
        radiusSearch(node.right, query, radius_sq, result);
    }

    void knnSearch(size_t node_index, int64_t query, int64_t kth, std::priority_queue<NeighborHeapItem> &heap) const
    {
        const auto &node        = nodes_[node_index];
        const auto  min_dist_sq = minDistanceToNodeSq(query, node);
        if (static_cast<int64_t>(heap.size()) >= kth && min_dist_sq > heap.top().first)
        {
            return;
        }

        if (node.leaf)
        {
            for (const int64_t index : node.indices)
            {
                pushNeighbor(heap, kth, squaredEuclideanDistance(samples_, query, index, num_features_), index);
            }
            return;
        }

        const auto &left_node    = nodes_[node.left];
        const auto &right_node   = nodes_[node.right];
        const bool  left_is_near = minDistanceToNodeSq(query, left_node) <= minDistanceToNodeSq(query, right_node);
        const auto  first_child  = left_is_near ? node.left : node.right;
        const auto  second_child = left_is_near ? node.right : node.left;
        knnSearch(first_child, query, kth, heap);
        knnSearch(second_child, query, kth, heap);
    }

    const float      *samples_{nullptr};
    int64_t           num_features_{0};
    int64_t           leaf_size_{0};
    std::vector<Node> nodes_;
    size_t            root_{kInvalidNode};
};

[[nodiscard]] std::vector<std::vector<int64_t>> bruteRadiusNeighborhoods(const float *samples, int64_t num_samples,
                                                                         int64_t num_features, double radius,
                                                                         ClusteringMetric metric, double minkowski_p)
{
    std::vector<std::vector<int64_t>> neighborhoods(static_cast<size_t>(num_samples));
    const double                      radius_sq = radius * radius;
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        auto &neighbors = neighborhoods[static_cast<size_t>(sample)];
        neighbors.reserve(static_cast<size_t>(num_samples));
        for (int64_t other = 0; other < num_samples; ++other)
        {
            const bool in_radius = metric == ClusteringMetric::Euclidean
                                     ? squaredEuclideanDistance(samples, sample, other, num_features) <= radius_sq
                                     : clusteringDistance(samples, sample, other, num_features, metric, minkowski_p)
                                           <= radius;
            if (in_radius)
            {
                neighbors.push_back(other);
            }
        }
    }
    return neighborhoods;
}

[[nodiscard]] std::vector<double> bruteKthNeighborDistances(const float *samples, int64_t num_samples,
                                                            int64_t num_features, int64_t kth, ClusteringMetric metric,
                                                            double minkowski_p)
{
    std::vector<double> result(static_cast<size_t>(num_samples), 0.0);
    std::vector<double> row(static_cast<size_t>(num_samples), 0.0);
    const auto          nth = row.begin() + (kth - 1);
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        for (int64_t other = 0; other < num_samples; ++other)
        {
            row[static_cast<size_t>(other)] = metric == ClusteringMetric::Euclidean
                                                ? squaredEuclideanDistance(samples, sample, other, num_features)
                                                : clusteringDistance(samples, sample, other, num_features, metric,
                                                                     minkowski_p);
        }
        std::nth_element(row.begin(), nth, row.end());
        result[static_cast<size_t>(sample)] = metric == ClusteringMetric::Euclidean ? std::sqrt(*nth) : *nth;
    }
    return result;
}

} // namespace

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
    case ClusteringAlgorithm::Auto:
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
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;

    switch (metric)
    {
    case ClusteringMetric::Euclidean:
        return euclideanDistance(samples, lhs, rhs, num_features);
    case ClusteringMetric::Manhattan:
    {
        double sum = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            sum += std::abs(static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature]));
        }
        return sum;
    }
    case ClusteringMetric::Minkowski:
    {
        if (!std::isfinite(minkowski_p) || minkowski_p <= 0.0)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "minkowski_p must be positive and finite");
        }

        double sum = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const double diff = std::abs(static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature]));
            sum += std::pow(diff, minkowski_p);
        }
        return std::pow(sum, 1.0 / minkowski_p);
    }
    case ClusteringMetric::Cosine:
    {
        double dot      = 0.0;
        double lhs_norm = 0.0;
        double rhs_norm = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const double lhs_value = static_cast<double>(lhs_ptr[feature]);
            const double rhs_value = static_cast<double>(rhs_ptr[feature]);
            dot += lhs_value * rhs_value;
            lhs_norm += lhs_value * lhs_value;
            rhs_norm += rhs_value * rhs_value;
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
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

ClusteringAlgorithm selectNeighborSearchAlgorithm(ClusteringAlgorithm requested, int64_t num_features,
                                                  ClusteringMetric metric)
{
    // KDTree/BallTree pruning in this implementation is Euclidean-only.
    if (metric != ClusteringMetric::Euclidean)
    {
        return ClusteringAlgorithm::Brute;
    }
    if (requested != ClusteringAlgorithm::Auto)
    {
        return requested;
    }
    return num_features <= kAutoTreeFeatureLimit ? ClusteringAlgorithm::KDTree : ClusteringAlgorithm::Brute;
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

    validateMetricConfig(metric, minkowski_p);
    const double radius_sq = radius * radius;
    switch (selectNeighborSearchAlgorithm(algorithm, num_features, metric))
    {
    case ClusteringAlgorithm::Brute:
        return bruteRadiusNeighborhoods(samples, num_samples, num_features, radius, metric, minkowski_p);
    case ClusteringAlgorithm::KDTree:
    {
        KDTreeIndex                       index(samples, num_samples, num_features, leaf_size);
        std::vector<std::vector<int64_t>> neighborhoods(static_cast<size_t>(num_samples));
        for (int64_t sample = 0; sample < num_samples; ++sample)
        {
            neighborhoods[static_cast<size_t>(sample)] = index.radiusNeighbors(sample, radius_sq);
        }
        return neighborhoods;
    }
    case ClusteringAlgorithm::BallTree:
    {
        BallTreeIndex                     index(samples, num_samples, num_features, leaf_size);
        std::vector<std::vector<int64_t>> neighborhoods(static_cast<size_t>(num_samples));
        for (int64_t sample = 0; sample < num_samples; ++sample)
        {
            neighborhoods[static_cast<size_t>(sample)] = index.radiusNeighbors(sample, radius_sq);
        }
        return neighborhoods;
    }
    case ClusteringAlgorithm::Auto:
        break;
    }

    throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering algorithm");
}

std::vector<double> kthNeighborDistances(const float *samples, int64_t num_samples, int64_t num_features, int64_t kth,
                                         ClusteringAlgorithm algorithm, int64_t leaf_size, ClusteringMetric metric,
                                         double minkowski_p)
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

    validateMetricConfig(metric, minkowski_p);
    switch (selectNeighborSearchAlgorithm(algorithm, num_features, metric))
    {
    case ClusteringAlgorithm::Brute:
        return bruteKthNeighborDistances(samples, num_samples, num_features, kth, metric, minkowski_p);
    case ClusteringAlgorithm::KDTree:
    {
        KDTreeIndex         index(samples, num_samples, num_features, leaf_size);
        std::vector<double> result(static_cast<size_t>(num_samples), 0.0);
        for (int64_t sample = 0; sample < num_samples; ++sample)
        {
            result[static_cast<size_t>(sample)] = index.kthDistance(sample, kth);
        }
        return result;
    }
    case ClusteringAlgorithm::BallTree:
    {
        BallTreeIndex       index(samples, num_samples, num_features, leaf_size);
        std::vector<double> result(static_cast<size_t>(num_samples), 0.0);
        for (int64_t sample = 0; sample < num_samples; ++sample)
        {
            result[static_cast<size_t>(sample)] = index.kthDistance(sample, kth);
        }
        return result;
    }
    case ClusteringAlgorithm::Auto:
        break;
    }

    throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering algorithm");
}

} // namespace irt::ops::detail
