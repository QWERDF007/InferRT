#include "TestClusteringCommon.hpp"

#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/DBSCAN.hpp>

#include <algorithm>
#include <vector>

TEST(DBSCANTest, DefaultsToKDTreeEuclideanMetric)
{
    const irt::ops::DBSCANConfig config;
    EXPECT_EQ(config.algorithm, irt::ops::ClusteringAlgorithm::KDTree);
    EXPECT_EQ(config.metric, irt::ops::ClusteringMetric::Euclidean);
}

TEST(DBSCANTest, MatchesSklearnDocumentedExample)
{
    const std::vector<float> samples{
        1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 8.0f, 7.0f, 8.0f, 8.0f, 25.0f, 80.0f,
    };

    irt::ops::DBSCANConfig config;
    config.eps         = 3.0f;
    config.min_samples = 2;
    config.metric      = irt::ops::ClusteringMetric::Euclidean;

    const auto result = irt::ops::dbscan(samples.data(), 6, 2, config);

    EXPECT_EQ(result.core_sample_indices, (std::vector<int64_t>{0, 1, 2, 3, 4}));
    EXPECT_EQ(result.labels, (std::vector<int64_t>{0, 0, 0, 1, 1, -1}));
}

TEST(DBSCANTest, RecoversSevenClustersFromAssetData)
{
    const auto data = irt::test::loadClusterTestData(__FILE__);

    irt::ops::DBSCANConfig config;
    config.eps         = 0.8f;
    config.min_samples = 4;
    config.metric      = irt::ops::ClusteringMetric::Euclidean;

    const auto result = irt::ops::dbscan(data.samples.data(), data.num_samples, data.num_features, config);

    ASSERT_EQ(result.labels.size(), static_cast<size_t>(data.num_samples));
    EXPECT_EQ(std::count(result.labels.begin(), result.labels.end(), int64_t{-1}), 0);
    EXPECT_EQ(irt::test::countClusters(result.labels), 7);
    irt::test::expectSamePartition(result.labels, data.expected_labels);
}

TEST(DBSCANTest, SupportsNeighborSearchAlgorithms)
{
    const auto data = irt::test::loadClusterTestData(__FILE__);

    for (const auto algorithm : {irt::ops::ClusteringAlgorithm::Brute, irt::ops::ClusteringAlgorithm::KDTree,
                                 irt::ops::ClusteringAlgorithm::BallTree})
    {
        irt::ops::DBSCANConfig config;
        config.eps         = 0.8f;
        config.min_samples = 4;
        config.algorithm   = algorithm;
        config.leaf_size   = 8;
        config.metric      = irt::ops::ClusteringMetric::Euclidean;

        const auto result = irt::ops::dbscan(data.samples.data(), data.num_samples, data.num_features, config);

        EXPECT_EQ(std::count(result.labels.begin(), result.labels.end(), int64_t{-1}), 0);
        EXPECT_EQ(irt::test::countClusters(result.labels), 7);
        irt::test::expectSamePartition(result.labels, data.expected_labels);
    }
}

TEST(DBSCANTest, TreeAlgorithmsMatchBruteForSupportedNonEuclideanMetrics)
{
    const auto data = irt::test::loadClusterTestData(__FILE__);

    struct MetricCase
    {
        irt::ops::ClusteringMetric metric;
        double                     minkowski_p;
        float                      eps;
    };

    const std::vector<MetricCase> metric_cases{
        {irt::ops::ClusteringMetric::Manhattan, 2.0, 1.0f},
        {irt::ops::ClusteringMetric::Chebyshev, 2.0, 0.8f},
        {irt::ops::ClusteringMetric::Minkowski, 3.0, 1.0f},
    };

    for (const auto &metric_case : metric_cases)
    {
        irt::ops::DBSCANConfig brute_config;
        brute_config.eps         = metric_case.eps;
        brute_config.min_samples = 4;
        brute_config.algorithm   = irt::ops::ClusteringAlgorithm::Brute;
        brute_config.leaf_size   = 8;
        brute_config.metric      = metric_case.metric;
        brute_config.minkowski_p = metric_case.minkowski_p;

        const auto expected = irt::ops::dbscan(data.samples.data(), data.num_samples, data.num_features, brute_config);

        for (const auto algorithm : {irt::ops::ClusteringAlgorithm::KDTree, irt::ops::ClusteringAlgorithm::BallTree})
        {
            auto tree_config      = brute_config;
            tree_config.algorithm = algorithm;

            const auto actual
                = irt::ops::dbscan(data.samples.data(), data.num_samples, data.num_features, tree_config);

            EXPECT_EQ(actual.core_sample_indices, expected.core_sample_indices);
            EXPECT_EQ(actual.labels, expected.labels);
        }
    }
}

TEST(DBSCANTest, RejectsCosineMetricForTreeAlgorithms)
{
    const std::vector<float> samples{
        1.0f, 0.0f, 0.9f, 0.1f, 0.0f, 1.0f, 0.1f, 0.9f,
    };

    for (const auto algorithm : {irt::ops::ClusteringAlgorithm::KDTree, irt::ops::ClusteringAlgorithm::BallTree})
    {
        irt::ops::DBSCANConfig config;
        config.eps         = 0.2f;
        config.min_samples = 2;
        config.algorithm   = algorithm;
        config.metric      = irt::ops::ClusteringMetric::Cosine;

        EXPECT_THROW((void)irt::ops::dbscan(samples.data(), 4, 2, config), irt::Exception);
    }
}

TEST(DBSCANTest, RejectsInvalidArguments)
{
    const std::vector<float> samples{0.0f, 0.0f};
    irt::ops::DBSCANConfig   config;

    EXPECT_THROW((void)irt::ops::dbscan(nullptr, 1, 2, config), irt::Exception);
    EXPECT_THROW((void)irt::ops::dbscan(samples.data(), -1, 2, config), irt::Exception);
    EXPECT_THROW((void)irt::ops::dbscan(samples.data(), 1, 0, config), irt::Exception);

    config.eps = 0.0f;
    EXPECT_THROW((void)irt::ops::dbscan(samples.data(), 1, 2, config), irt::Exception);

    config.eps         = 0.5f;
    config.min_samples = 0;
    EXPECT_THROW((void)irt::ops::dbscan(samples.data(), 1, 2, config), irt::Exception);

    config.min_samples = 1;
    config.leaf_size   = 0;
    EXPECT_THROW((void)irt::ops::dbscan(samples.data(), 1, 2, config), irt::Exception);
}
