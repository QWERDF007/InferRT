#include "priv/ClusteringCommon.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/HDBSCAN.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace irt::ops {
namespace {

constexpr int64_t kNoiseLabel = -1;

struct MSTEdge
{
    int64_t current_node{0};
    int64_t next_node{0};
    double  distance{0.0};
};

struct LinkageNode
{
    int64_t left_node{0};
    int64_t right_node{0};
    double  value{0.0};
    int64_t cluster_size{0};
};

struct CondensedNode
{
    int64_t parent{0};
    int64_t child{0};
    double  value{0.0};
    int64_t cluster_size{0};
};

class LinkageUnionFind final
{
public:
    explicit LinkageUnionFind(int64_t size)
        : parent_(static_cast<size_t>(2 * size - 1), int64_t{-1})
        , size_(static_cast<size_t>(2 * size - 1), int64_t{0})
        , next_label_(size)
    {
        std::fill(size_.begin(), size_.begin() + size, int64_t{1});
    }

    [[nodiscard]] int64_t find(int64_t node)
    {
        int64_t root = node;
        while (parent_[static_cast<size_t>(root)] != -1)
        {
            root = parent_[static_cast<size_t>(root)];
        }

        while (parent_[static_cast<size_t>(node)] != -1 && parent_[static_cast<size_t>(node)] != root)
        {
            const int64_t next                 = parent_[static_cast<size_t>(node)];
            parent_[static_cast<size_t>(node)] = root;
            node                               = next;
        }
        return root;
    }

    void unite(int64_t lhs, int64_t rhs)
    {
        parent_[static_cast<size_t>(lhs)]       = next_label_;
        parent_[static_cast<size_t>(rhs)]       = next_label_;
        size_[static_cast<size_t>(next_label_)] = size(lhs) + size(rhs);
        ++next_label_;
    }

    [[nodiscard]] int64_t size(int64_t node) const
    {
        return size_[static_cast<size_t>(node)];
    }

private:
    std::vector<int64_t> parent_;
    std::vector<int64_t> size_;
    int64_t              next_label_{0};
};

class TreeUnionFind final
{
public:
    explicit TreeUnionFind(int64_t size)
        : parent_(static_cast<size_t>(size))
        , rank_(static_cast<size_t>(size), int64_t{0})
    {
        std::iota(parent_.begin(), parent_.end(), int64_t{0});
    }

    [[nodiscard]] int64_t find(int64_t node)
    {
        if (parent_[static_cast<size_t>(node)] != node)
        {
            parent_[static_cast<size_t>(node)] = find(parent_[static_cast<size_t>(node)]);
        }
        return parent_[static_cast<size_t>(node)];
    }

    void unite(int64_t lhs, int64_t rhs)
    {
        const int64_t lhs_root = find(lhs);
        const int64_t rhs_root = find(rhs);
        if (lhs_root == rhs_root)
        {
            return;
        }

        if (rank_[static_cast<size_t>(lhs_root)] < rank_[static_cast<size_t>(rhs_root)])
        {
            parent_[static_cast<size_t>(lhs_root)] = rhs_root;
        }
        else if (rank_[static_cast<size_t>(lhs_root)] > rank_[static_cast<size_t>(rhs_root)])
        {
            parent_[static_cast<size_t>(rhs_root)] = lhs_root;
        }
        else
        {
            parent_[static_cast<size_t>(rhs_root)] = lhs_root;
            ++rank_[static_cast<size_t>(lhs_root)];
        }
    }

private:
    std::vector<int64_t> parent_;
    std::vector<int64_t> rank_;
};

void validateInputs(const float *samples, int64_t num_samples, int64_t num_features, const HDBSCANConfig &config)
{
    detail::validateSampleMatrix(samples, num_samples, num_features);
    detail::validateNeighborSearchConfig(config.algorithm, config.leaf_size);
    if (num_samples <= 1)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "HDBSCAN requires more than one sample, got %lld",
                        static_cast<long long>(num_samples));
    }
    if (config.min_cluster_size < 2)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "min_cluster_size must be at least 2, got %lld",
                        static_cast<long long>(config.min_cluster_size));
    }
    if (config.min_samples < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "min_samples must be non-negative, got %lld",
                        static_cast<long long>(config.min_samples));
    }
    const int64_t min_samples = config.min_samples == 0 ? config.min_cluster_size : config.min_samples;
    if (min_samples < 1 || min_samples > num_samples)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT,
                        "min_samples must be positive and at most num_samples, got %lld",
                        static_cast<long long>(min_samples));
    }
    if (!std::isfinite(config.cluster_selection_epsilon) || config.cluster_selection_epsilon < 0.0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "cluster_selection_epsilon must be finite and non-negative");
    }
    if (config.max_cluster_size < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "max_cluster_size must be non-negative, got %lld",
                        static_cast<long long>(config.max_cluster_size));
    }
    if (!std::isfinite(config.alpha) || config.alpha <= 0.0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "alpha must be positive and finite");
    }
}

std::vector<double> pairwiseDistances(const float *samples, int64_t num_samples, int64_t num_features, double alpha)
{
    std::vector<double> distances(static_cast<size_t>(num_samples * num_samples), 0.0);
    for (int64_t lhs = 0; lhs < num_samples; ++lhs)
    {
        for (int64_t rhs = lhs + 1; rhs < num_samples; ++rhs)
        {
            const double distance = detail::euclideanDistance(samples, lhs, rhs, num_features) / alpha;
            distances[static_cast<size_t>(lhs * num_samples + rhs)] = distance;
            distances[static_cast<size_t>(rhs * num_samples + lhs)] = distance;
        }
    }
    return distances;
}

std::vector<double> coreDistances(const float *samples, int64_t num_samples, int64_t num_features, int64_t min_samples,
                                  double alpha, ClusteringAlgorithm algorithm, int64_t leaf_size)
{
    auto result = detail::kthNeighborDistances(samples, num_samples, num_features, min_samples, algorithm, leaf_size);
    for (double &distance : result)
    {
        distance /= alpha;
    }
    return result;
}

double mutualReachability(const std::vector<double> &distances, const std::vector<double> &core_distances,
                          int64_t num_samples, int64_t lhs, int64_t rhs)
{
    return std::max({core_distances[static_cast<size_t>(lhs)], core_distances[static_cast<size_t>(rhs)],
                     distances[static_cast<size_t>(lhs * num_samples + rhs)]});
}

std::vector<MSTEdge> minimumSpanningTree(const std::vector<double> &distances,
                                         const std::vector<double> &core_distances, int64_t num_samples)
{
    std::vector<MSTEdge> mst;
    mst.reserve(static_cast<size_t>(num_samples - 1));

    std::vector<uint8_t> in_tree(static_cast<size_t>(num_samples), uint8_t{0});
    std::vector<double>  min_reachability(static_cast<size_t>(num_samples), std::numeric_limits<double>::infinity());
    std::vector<int64_t> current_sources(static_cast<size_t>(num_samples), int64_t{1});

    int64_t current_node = 0;
    for (int64_t edge_index = 0; edge_index < num_samples - 1; ++edge_index)
    {
        in_tree[static_cast<size_t>(current_node)] = uint8_t{1};

        double  new_reachability = std::numeric_limits<double>::max();
        int64_t source_node      = 0;
        int64_t new_node         = 0;

        for (int64_t next_node = 0; next_node < num_samples; ++next_node)
        {
            if (in_tree[static_cast<size_t>(next_node)])
            {
                continue;
            }

            const double  next_node_min_reach = min_reachability[static_cast<size_t>(next_node)];
            const int64_t next_node_source    = current_sources[static_cast<size_t>(next_node)];
            const double  reachability
                = mutualReachability(distances, core_distances, num_samples, current_node, next_node);

            if (reachability < next_node_min_reach)
            {
                min_reachability[static_cast<size_t>(next_node)] = reachability;
                current_sources[static_cast<size_t>(next_node)]  = current_node;
                if (reachability < new_reachability)
                {
                    new_reachability = reachability;
                    source_node      = current_node;
                    new_node         = next_node;
                }
            }
            else if (next_node_min_reach < new_reachability)
            {
                new_reachability = next_node_min_reach;
                source_node      = next_node_source;
                new_node         = next_node;
            }
        }

        mst.push_back({source_node, new_node, new_reachability});
        current_node = new_node;
    }

    std::stable_sort(mst.begin(), mst.end(),
                     [](const MSTEdge &lhs, const MSTEdge &rhs) { return lhs.distance < rhs.distance; });
    return mst;
}

std::vector<LinkageNode> makeSingleLinkage(const std::vector<MSTEdge> &mst, int64_t num_samples)
{
    std::vector<LinkageNode> linkage;
    linkage.reserve(static_cast<size_t>(num_samples - 1));
    LinkageUnionFind union_find(num_samples);

    for (const auto &edge : mst)
    {
        const int64_t current_cluster = union_find.find(edge.current_node);
        const int64_t next_cluster    = union_find.find(edge.next_node);
        linkage.push_back({current_cluster, next_cluster, edge.distance,
                           union_find.size(current_cluster) + union_find.size(next_cluster)});
        union_find.unite(current_cluster, next_cluster);
    }
    return linkage;
}

std::vector<int64_t> bfsFromHierarchy(const std::vector<LinkageNode> &hierarchy, int64_t root)
{
    const int64_t        n_samples = static_cast<int64_t>(hierarchy.size()) + 1;
    std::vector<int64_t> result;
    std::vector<int64_t> process_queue{root};

    while (!process_queue.empty())
    {
        result.insert(result.end(), process_queue.begin(), process_queue.end());

        std::vector<int64_t> hierarchy_indices;
        for (const int64_t node : process_queue)
        {
            if (node >= n_samples)
            {
                hierarchy_indices.push_back(node - n_samples);
            }
        }

        process_queue.clear();
        for (const int64_t index : hierarchy_indices)
        {
            const auto &children = hierarchy[static_cast<size_t>(index)];
            process_queue.push_back(children.left_node);
            process_queue.push_back(children.right_node);
        }
    }

    return result;
}

std::vector<CondensedNode> condenseTree(const std::vector<LinkageNode> &hierarchy, int64_t min_cluster_size)
{
    const int64_t root       = 2 * static_cast<int64_t>(hierarchy.size());
    const int64_t n_samples  = static_cast<int64_t>(hierarchy.size()) + 1;
    int64_t       next_label = n_samples + 1;

    const auto           node_list = bfsFromHierarchy(hierarchy, root);
    std::vector<int64_t> relabel(static_cast<size_t>(root + 1), int64_t{0});
    std::vector<uint8_t> ignore(static_cast<size_t>(root + 1), uint8_t{0});
    relabel[static_cast<size_t>(root)] = n_samples;

    std::vector<CondensedNode> result;
    result.reserve(static_cast<size_t>(n_samples));

    const auto appendLeaves = [&](int64_t parent, int64_t subtree, double lambda_value)
    {
        for (const int64_t sub_node : bfsFromHierarchy(hierarchy, subtree))
        {
            if (sub_node < n_samples)
            {
                result.push_back({parent, sub_node, lambda_value, int64_t{1}});
            }
            ignore[static_cast<size_t>(sub_node)] = uint8_t{1};
        }
    };

    for (const int64_t node : node_list)
    {
        if (node < n_samples || ignore[static_cast<size_t>(node)])
        {
            continue;
        }

        const auto   &children = hierarchy[static_cast<size_t>(node - n_samples)];
        const int64_t left     = children.left_node;
        const int64_t right    = children.right_node;
        const double  lambda_value
            = children.value > 0.0 ? 1.0 / children.value : std::numeric_limits<double>::infinity();
        const int64_t left_count
            = left >= n_samples ? hierarchy[static_cast<size_t>(left - n_samples)].cluster_size : int64_t{1};
        const int64_t right_count
            = right >= n_samples ? hierarchy[static_cast<size_t>(right - n_samples)].cluster_size : int64_t{1};
        const int64_t parent = relabel[static_cast<size_t>(node)];

        if (left_count >= min_cluster_size && right_count >= min_cluster_size)
        {
            relabel[static_cast<size_t>(left)] = next_label++;
            result.push_back({parent, relabel[static_cast<size_t>(left)], lambda_value, left_count});

            relabel[static_cast<size_t>(right)] = next_label++;
            result.push_back({parent, relabel[static_cast<size_t>(right)], lambda_value, right_count});
        }
        else if (left_count < min_cluster_size && right_count < min_cluster_size)
        {
            appendLeaves(parent, left, lambda_value);
            appendLeaves(parent, right, lambda_value);
        }
        else if (left_count < min_cluster_size)
        {
            relabel[static_cast<size_t>(right)] = parent;
            appendLeaves(parent, left, lambda_value);
        }
        else
        {
            relabel[static_cast<size_t>(left)] = parent;
            appendLeaves(parent, right, lambda_value);
        }
    }

    return result;
}

std::map<int64_t, double> computeStability(const std::vector<CondensedNode> &condensed_tree)
{
    int64_t largest_child    = condensed_tree.front().child;
    int64_t largest_parent   = condensed_tree.front().parent;
    int64_t smallest_cluster = condensed_tree.front().parent;
    for (const auto &node : condensed_tree)
    {
        largest_child    = std::max(largest_child, node.child);
        largest_parent   = std::max(largest_parent, node.parent);
        smallest_cluster = std::min(smallest_cluster, node.parent);
    }

    std::vector<double> births(static_cast<size_t>(std::max({largest_child, largest_parent, smallest_cluster}) + 1),
                               std::numeric_limits<double>::quiet_NaN());
    for (const auto &node : condensed_tree)
    {
        births[static_cast<size_t>(node.child)] = node.value;
    }
    births[static_cast<size_t>(smallest_cluster)] = 0.0;

    std::vector<double> result(static_cast<size_t>(largest_parent - smallest_cluster + 1), 0.0);
    for (const auto &node : condensed_tree)
    {
        result[static_cast<size_t>(node.parent - smallest_cluster)]
            += (node.value - births[static_cast<size_t>(node.parent)]) * static_cast<double>(node.cluster_size);
    }

    std::map<int64_t, double> stability;
    for (int64_t cluster = smallest_cluster; cluster <= largest_parent; ++cluster)
    {
        stability[cluster] = result[static_cast<size_t>(cluster - smallest_cluster)];
    }
    return stability;
}

std::vector<CondensedNode> clusterTree(const std::vector<CondensedNode> &condensed_tree)
{
    std::vector<CondensedNode> result;
    for (const auto &node : condensed_tree)
    {
        if (node.cluster_size > 1)
        {
            result.push_back(node);
        }
    }
    return result;
}

std::vector<int64_t> bfsFromClusterTree(const std::vector<CondensedNode> &tree, int64_t root)
{
    std::vector<int64_t> result;
    std::vector<int64_t> process_queue{root};
    while (!process_queue.empty())
    {
        result.insert(result.end(), process_queue.begin(), process_queue.end());

        const std::unordered_set<int64_t> active(process_queue.begin(), process_queue.end());
        process_queue.clear();
        for (const auto &node : tree)
        {
            if (active.find(node.parent) != active.end())
            {
                process_queue.push_back(node.child);
            }
        }
    }
    return result;
}

std::vector<int64_t> recurseLeafDfs(const std::vector<CondensedNode> &tree, int64_t current_node)
{
    std::vector<int64_t> children;
    for (const auto &node : tree)
    {
        if (node.parent == current_node)
        {
            children.push_back(node.child);
        }
    }
    if (children.empty())
    {
        return {current_node};
    }

    std::vector<int64_t> leaves;
    for (const int64_t child : children)
    {
        auto child_leaves = recurseLeafDfs(tree, child);
        leaves.insert(leaves.end(), child_leaves.begin(), child_leaves.end());
    }
    return leaves;
}

std::vector<int64_t> getClusterTreeLeaves(const std::vector<CondensedNode> &tree)
{
    if (tree.empty())
    {
        return {};
    }

    int64_t root = tree.front().parent;
    for (const auto &node : tree)
    {
        root = std::min(root, node.parent);
    }
    return recurseLeafDfs(tree, root);
}

bool findParentForChild(const std::vector<CondensedNode> &tree, int64_t child, int64_t *parent, double *value)
{
    for (const auto &node : tree)
    {
        if (node.child == child)
        {
            *parent = node.parent;
            *value  = node.value;
            return true;
        }
    }
    return false;
}

int64_t traverseUpwards(const std::vector<CondensedNode> &tree, double cluster_selection_epsilon, int64_t leaf,
                        bool allow_single_cluster)
{
    int64_t root = tree.front().parent;
    for (const auto &node : tree)
    {
        root = std::min(root, node.parent);
    }

    int64_t parent = 0;
    double  value  = 0.0;
    if (!findParentForChild(tree, leaf, &parent, &value))
    {
        return leaf;
    }
    if (parent == root)
    {
        return allow_single_cluster ? parent : leaf;
    }

    int64_t grand_parent = 0;
    double  parent_value = 0.0;
    if (!findParentForChild(tree, parent, &grand_parent, &parent_value))
    {
        return parent;
    }

    const double parent_eps = 1.0 / parent_value;
    if (parent_eps > cluster_selection_epsilon)
    {
        return parent;
    }
    return traverseUpwards(tree, cluster_selection_epsilon, parent, allow_single_cluster);
}

std::set<int64_t> epsilonSearch(const std::set<int64_t> &leaves, const std::vector<CondensedNode> &tree,
                                double cluster_selection_epsilon, bool allow_single_cluster)
{
    std::set<int64_t> selected_clusters;
    std::set<int64_t> processed;
    for (const int64_t leaf : leaves)
    {
        int64_t parent = 0;
        double  value  = 0.0;
        if (!findParentForChild(tree, leaf, &parent, &value))
        {
            selected_clusters.insert(leaf);
            continue;
        }

        const double eps = 1.0 / value;
        if (eps < cluster_selection_epsilon)
        {
            if (processed.find(leaf) == processed.end())
            {
                const int64_t epsilon_child
                    = traverseUpwards(tree, cluster_selection_epsilon, leaf, allow_single_cluster);
                selected_clusters.insert(epsilon_child);
                for (const int64_t sub_node : bfsFromClusterTree(tree, epsilon_child))
                {
                    if (sub_node != epsilon_child)
                    {
                        processed.insert(sub_node);
                    }
                }
            }
        }
        else
        {
            selected_clusters.insert(leaf);
        }
    }
    return selected_clusters;
}

int64_t maxNodeId(const std::vector<CondensedNode> &condensed_tree)
{
    int64_t max_node = 0;
    for (const auto &node : condensed_tree)
    {
        max_node = std::max({max_node, node.parent, node.child});
    }
    return max_node;
}

std::vector<int64_t> doLabelling(const std::vector<CondensedNode> &condensed_tree, const std::set<int64_t> &clusters,
                                 const std::map<int64_t, int64_t> &cluster_label_map, bool allow_single_cluster,
                                 double cluster_selection_epsilon)
{
    int64_t root_cluster = condensed_tree.front().parent;
    for (const auto &node : condensed_tree)
    {
        root_cluster = std::min(root_cluster, node.parent);
    }

    std::vector<int64_t> result(static_cast<size_t>(root_cluster), kNoiseLabel);
    TreeUnionFind        union_find(maxNodeId(condensed_tree) + 1);

    for (const auto &node : condensed_tree)
    {
        if (clusters.find(node.child) == clusters.end())
        {
            union_find.unite(node.parent, node.child);
        }
    }

    for (int64_t point = 0; point < root_cluster; ++point)
    {
        const int64_t cluster = union_find.find(point);
        int64_t       label   = kNoiseLabel;
        if (cluster != root_cluster)
        {
            const auto it = cluster_label_map.find(cluster);
            if (it != cluster_label_map.end())
            {
                label = it->second;
            }
        }
        else if (clusters.size() == 1 && allow_single_cluster)
        {
            double parent_lambda       = 0.0;
            bool   found_parent_lambda = false;
            for (const auto &node : condensed_tree)
            {
                if (node.child == point)
                {
                    parent_lambda       = node.value;
                    found_parent_lambda = true;
                    break;
                }
            }

            double threshold = 0.0;
            if (cluster_selection_epsilon != 0.0)
            {
                threshold = 1.0 / cluster_selection_epsilon;
            }
            else
            {
                for (const auto &node : condensed_tree)
                {
                    if (node.parent == cluster)
                    {
                        threshold = std::max(threshold, node.value);
                    }
                }
            }

            if (found_parent_lambda && parent_lambda >= threshold)
            {
                const auto it = cluster_label_map.find(cluster);
                if (it != cluster_label_map.end())
                {
                    label = it->second;
                }
            }
        }
        result[static_cast<size_t>(point)] = label;
    }
    return result;
}

std::vector<double> maxLambdas(const std::vector<CondensedNode> &condensed_tree)
{
    int64_t largest_parent = 0;
    for (const auto &node : condensed_tree)
    {
        largest_parent = std::max(largest_parent, node.parent);
    }

    std::vector<double> deaths(static_cast<size_t>(largest_parent + 1), 0.0);
    for (const auto &node : condensed_tree)
    {
        deaths[static_cast<size_t>(node.parent)] = std::max(deaths[static_cast<size_t>(node.parent)], node.value);
    }
    return deaths;
}

std::vector<double> getProbabilities(const std::vector<CondensedNode> &condensed_tree,
                                     const std::map<int64_t, int64_t> &reverse_cluster_map,
                                     const std::vector<int64_t>       &labels)
{
    int64_t root_cluster = condensed_tree.front().parent;
    for (const auto &node : condensed_tree)
    {
        root_cluster = std::min(root_cluster, node.parent);
    }

    std::vector<double> result(labels.size(), 0.0);
    const auto          deaths = maxLambdas(condensed_tree);
    for (const auto &node : condensed_tree)
    {
        const int64_t point = node.child;
        if (point >= root_cluster)
        {
            continue;
        }

        const int64_t cluster_num = labels[static_cast<size_t>(point)];
        if (cluster_num == kNoiseLabel)
        {
            continue;
        }

        const auto cluster_it = reverse_cluster_map.find(cluster_num);
        if (cluster_it == reverse_cluster_map.end())
        {
            continue;
        }
        const int64_t cluster    = cluster_it->second;
        const double  max_lambda = deaths[static_cast<size_t>(cluster)];
        if (max_lambda == 0.0 || std::isinf(node.value))
        {
            result[static_cast<size_t>(point)] = 1.0;
        }
        else
        {
            result[static_cast<size_t>(point)] = std::min(node.value, max_lambda) / max_lambda;
        }
    }
    return result;
}

HDBSCANResult getClusters(const std::vector<CondensedNode> &condensed_tree, const HDBSCANConfig &config)
{
    if (condensed_tree.empty())
    {
        return {};
    }

    auto stability = computeStability(condensed_tree);
    auto tree      = clusterTree(condensed_tree);

    std::vector<int64_t> node_list;
    node_list.reserve(stability.size());
    for (auto it = stability.rbegin(); it != stability.rend(); ++it)
    {
        node_list.push_back(it->first);
    }
    if (!config.allow_single_cluster && !node_list.empty())
    {
        node_list.pop_back();
    }

    std::map<int64_t, bool> is_cluster;
    for (const int64_t node : node_list)
    {
        is_cluster[node] = true;
    }

    int64_t n_samples = 0;
    for (const auto &node : condensed_tree)
    {
        if (node.cluster_size == 1)
        {
            n_samples = std::max(n_samples, node.child + 1);
        }
    }

    const int64_t max_cluster_size = config.max_cluster_size == 0 ? n_samples + 1 : config.max_cluster_size;
    std::unordered_map<int64_t, int64_t> cluster_sizes;
    for (const auto &node : tree)
    {
        cluster_sizes[node.child] = node.cluster_size;
    }
    if (config.allow_single_cluster && !node_list.empty())
    {
        const int64_t root      = node_list.back();
        int64_t       root_size = 0;
        for (const auto &node : tree)
        {
            if (node.parent == root)
            {
                root_size += node.cluster_size;
            }
        }
        cluster_sizes[root] = root_size;
    }

    if (config.cluster_selection_method == HDBSCANClusterSelectionMethod::Eom)
    {
        for (const int64_t node : node_list)
        {
            double subtree_stability = 0.0;
            for (const auto &tree_node : tree)
            {
                if (tree_node.parent == node)
                {
                    subtree_stability += stability[tree_node.child];
                }
            }

            const auto    size_it      = cluster_sizes.find(node);
            const int64_t cluster_size = size_it == cluster_sizes.end() ? int64_t{0} : size_it->second;
            if (subtree_stability > stability[node] || cluster_size > max_cluster_size)
            {
                is_cluster[node] = false;
                stability[node]  = subtree_stability;
            }
            else
            {
                for (const int64_t sub_node : bfsFromClusterTree(tree, node))
                {
                    if (sub_node != node)
                    {
                        is_cluster[sub_node] = false;
                    }
                }
            }
        }

        if (config.cluster_selection_epsilon != 0.0 && !tree.empty())
        {
            std::set<int64_t> eom_clusters;
            for (const auto &[cluster, selected] : is_cluster)
            {
                if (selected)
                {
                    eom_clusters.insert(cluster);
                }
            }

            std::set<int64_t> selected_clusters;
            int64_t           root = tree.front().parent;
            for (const auto &node : tree)
            {
                root = std::min(root, node.parent);
            }
            if (eom_clusters.size() == 1 && *eom_clusters.begin() == root)
            {
                if (config.allow_single_cluster)
                {
                    selected_clusters = eom_clusters;
                }
            }
            else
            {
                selected_clusters
                    = epsilonSearch(eom_clusters, tree, config.cluster_selection_epsilon, config.allow_single_cluster);
            }
            for (auto &[cluster, selected] : is_cluster)
            {
                selected = selected_clusters.find(cluster) != selected_clusters.end();
            }
        }
    }
    else
    {
        auto              leaf_vector = getClusterTreeLeaves(tree);
        std::set<int64_t> leaves(leaf_vector.begin(), leaf_vector.end());
        if (leaves.empty())
        {
            for (auto &[cluster, selected] : is_cluster)
            {
                selected = false;
            }
            int64_t root = condensed_tree.front().parent;
            for (const auto &node : condensed_tree)
            {
                root = std::min(root, node.parent);
            }
            is_cluster[root] = true;
        }

        const std::set<int64_t> selected_clusters
            = config.cluster_selection_epsilon == 0.0
                ? leaves
                : epsilonSearch(leaves, tree, config.cluster_selection_epsilon, config.allow_single_cluster);
        for (auto &[cluster, selected] : is_cluster)
        {
            selected = selected_clusters.find(cluster) != selected_clusters.end();
        }
    }

    std::set<int64_t> clusters;
    for (const auto &[cluster, selected] : is_cluster)
    {
        if (selected)
        {
            clusters.insert(cluster);
        }
    }

    std::map<int64_t, int64_t> cluster_map;
    std::map<int64_t, int64_t> reverse_cluster_map;
    int64_t                    label = 0;
    for (const int64_t cluster : clusters)
    {
        cluster_map[cluster]       = label;
        reverse_cluster_map[label] = cluster;
        ++label;
    }

    HDBSCANResult result;
    result.labels        = doLabelling(condensed_tree, clusters, cluster_map, config.allow_single_cluster,
                                       config.cluster_selection_epsilon);
    result.probabilities = getProbabilities(condensed_tree, reverse_cluster_map, result.labels);
    return result;
}

} // namespace

HDBSCANResult hdbscan(const float *samples, int64_t num_samples, int64_t num_features, const HDBSCANConfig &config)
{
    validateInputs(samples, num_samples, num_features, config);

    const int64_t min_samples    = config.min_samples == 0 ? config.min_cluster_size : config.min_samples;
    const auto    distances      = pairwiseDistances(samples, num_samples, num_features, config.alpha);
    const auto    core_distances = coreDistances(samples, num_samples, num_features, min_samples, config.alpha,
                                                 config.algorithm, config.leaf_size);
    const auto    mst            = minimumSpanningTree(distances, core_distances, num_samples);
    const auto    linkage        = makeSingleLinkage(mst, num_samples);
    const auto    condensed_tree = condenseTree(linkage, config.min_cluster_size);
    return getClusters(condensed_tree, config);
}

} // namespace irt::ops
