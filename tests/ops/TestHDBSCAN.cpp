#include "TestClusteringCommon.hpp"

#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/HDBSCAN.hpp>

#include <algorithm>
#include <vector>

TEST(HDBSCANTest, DefaultsToKDTreeEuclideanMetric)
{
    const irt::ops::HDBSCANConfig config;
    EXPECT_EQ(config.algorithm, irt::ops::ClusteringAlgorithm::KDTree);
    EXPECT_EQ(config.metric, irt::ops::ClusteringMetric::Euclidean);
}

TEST(HDBSCANTest, MatchesSklearnDocumentedExample)
{
    const std::vector<float> samples{
        1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 8.0f, 7.0f, 8.0f, 8.0f, 25.0f, 80.0f,
    };
    irt::ops::HDBSCANConfig config;
    config.min_cluster_size = 2;
    config.min_samples      = 2;
    config.metric           = irt::ops::ClusteringMetric::Euclidean;

    const auto result = irt::ops::hdbscan(samples.data(), 6, 2, config);

    EXPECT_EQ(result.labels, (std::vector<int64_t>{0, 0, 0, 1, 1, -1}));
    ASSERT_EQ(result.probabilities.size(), result.labels.size());
    EXPECT_DOUBLE_EQ(result.probabilities[0], 1.0);
    EXPECT_DOUBLE_EQ(result.probabilities[5], 0.0);
}

TEST(HDBSCANTest, RecoversSevenClustersFromAssetData)
{
    const auto              data = irt::test::loadClusterTestData(__FILE__);
    irt::ops::HDBSCANConfig config;
    config.min_cluster_size = 5;
    config.min_samples      = 5;
    config.metric           = irt::ops::ClusteringMetric::Euclidean;

    const auto result = irt::ops::hdbscan(data.samples.data(), data.num_samples, data.num_features, config);

    ASSERT_EQ(result.labels.size(), static_cast<size_t>(data.num_samples));
    ASSERT_EQ(result.probabilities.size(), result.labels.size());
    EXPECT_EQ(std::count(result.labels.begin(), result.labels.end(), int64_t{-1}), 0);
    EXPECT_EQ(irt::test::countClusters(result.labels), 7);
    irt::test::expectSamePartition(result.labels, data.expected_labels);
    for (const double probability : result.probabilities)
    {
        EXPECT_GE(probability, 0.0);
        EXPECT_LE(probability, 1.0);
    }
}

TEST(HDBSCANTest, SupportsNeighborSearchAlgorithms)
{
    const auto data = irt::test::loadClusterTestData(__FILE__);

    for (const auto algorithm : {irt::ops::ClusteringAlgorithm::Brute, irt::ops::ClusteringAlgorithm::KDTree,
                                 irt::ops::ClusteringAlgorithm::BallTree})
    {
        irt::ops::HDBSCANConfig config;
        config.min_cluster_size = 5;
        config.min_samples      = 5;
        config.algorithm        = algorithm;
        config.leaf_size        = 8;
        config.metric           = irt::ops::ClusteringMetric::Euclidean;

        const auto result = irt::ops::hdbscan(data.samples.data(), data.num_samples, data.num_features, config);

        EXPECT_EQ(std::count(result.labels.begin(), result.labels.end(), int64_t{-1}), 0);
        EXPECT_EQ(irt::test::countClusters(result.labels), 7);
        irt::test::expectSamePartition(result.labels, data.expected_labels);
        ASSERT_EQ(result.probabilities.size(), result.labels.size());
        for (const double probability : result.probabilities)
        {
            EXPECT_GE(probability, 0.0);
            EXPECT_LE(probability, 1.0);
        }
    }
}

TEST(HDBSCANTest, TreeAlgorithmsMatchBruteForSupportedNonEuclideanMetrics)
{
    const auto data = irt::test::loadClusterTestData(__FILE__);

    struct MetricCase
    {
        irt::ops::ClusteringMetric metric;
        double                     minkowski_p;
    };

    const std::vector<MetricCase> metric_cases{
        {irt::ops::ClusteringMetric::Manhattan, 2.0},
        {irt::ops::ClusteringMetric::Chebyshev, 2.0},
        {irt::ops::ClusteringMetric::Minkowski, 3.0},
    };

    for (const auto &metric_case : metric_cases)
    {
        irt::ops::HDBSCANConfig brute_config;
        brute_config.min_cluster_size = 5;
        brute_config.min_samples      = 5;
        brute_config.algorithm        = irt::ops::ClusteringAlgorithm::Brute;
        brute_config.leaf_size        = 8;
        brute_config.metric           = metric_case.metric;
        brute_config.minkowski_p      = metric_case.minkowski_p;

        const auto expected
            = irt::ops::hdbscan(data.samples.data(), data.num_samples, data.num_features, brute_config);

        for (const auto algorithm : {irt::ops::ClusteringAlgorithm::KDTree, irt::ops::ClusteringAlgorithm::BallTree})
        {
            auto tree_config      = brute_config;
            tree_config.algorithm = algorithm;

            const auto actual
                = irt::ops::hdbscan(data.samples.data(), data.num_samples, data.num_features, tree_config);

            EXPECT_EQ(actual.labels, expected.labels);
            EXPECT_EQ(actual.probabilities, expected.probabilities);
        }
    }
}

TEST(HDBSCANTest, RejectsCosineMetricForTreeAlgorithms)
{
    const std::vector<float> samples{
        1.0f, 0.0f, 0.9f, 0.1f, 0.0f, 1.0f, 0.1f, 0.9f,
    };

    for (const auto algorithm : {irt::ops::ClusteringAlgorithm::KDTree, irt::ops::ClusteringAlgorithm::BallTree})
    {
        irt::ops::HDBSCANConfig config;
        config.min_cluster_size = 2;
        config.min_samples      = 2;
        config.algorithm        = algorithm;
        config.metric           = irt::ops::ClusteringMetric::Cosine;

        EXPECT_THROW((void)irt::ops::hdbscan(samples.data(), 4, 2, config), irt::Exception);
    }
}

TEST(HDBSCANTest, RejectsInvalidArguments)
{
    const std::vector<float> samples{0.0f, 0.0f, 1.0f, 1.0f};
    irt::ops::HDBSCANConfig  config;
    config.metric = irt::ops::ClusteringMetric::Euclidean;

    EXPECT_THROW((void)irt::ops::hdbscan(nullptr, 2, 2, config), irt::Exception);
    EXPECT_THROW((void)irt::ops::hdbscan(samples.data(), 1, 2, config), irt::Exception);
    EXPECT_THROW((void)irt::ops::hdbscan(samples.data(), 2, 0, config), irt::Exception);

    config.min_cluster_size = 1;
    EXPECT_THROW((void)irt::ops::hdbscan(samples.data(), 2, 2, config), irt::Exception);

    config.min_cluster_size = 2;
    config.min_samples      = 3;
    EXPECT_THROW((void)irt::ops::hdbscan(samples.data(), 2, 2, config), irt::Exception);

    config.min_samples = 0;
    config.alpha       = 0.0;
    EXPECT_THROW((void)irt::ops::hdbscan(samples.data(), 2, 2, config), irt::Exception);

    config.alpha     = 1.0;
    config.leaf_size = 0;
    EXPECT_THROW((void)irt::ops::hdbscan(samples.data(), 2, 2, config), irt::Exception);
}
