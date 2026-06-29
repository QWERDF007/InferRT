#include "TestClusteringCommon.hpp"

#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/HDBSCAN.hpp>

#include <algorithm>
#include <vector>

TEST(HDBSCANTest, MatchesSklearnDocumentedExample)
{
    const std::vector<float> samples{
        1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 8.0f, 7.0f, 8.0f, 8.0f, 25.0f, 80.0f,
    };
    irt::ops::HDBSCANConfig config;
    config.min_cluster_size = 2;
    config.min_samples      = 2;

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

TEST(HDBSCANTest, RejectsInvalidArguments)
{
    const std::vector<float> samples{0.0f, 0.0f, 1.0f, 1.0f};
    irt::ops::HDBSCANConfig  config;

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
}
