#include "TestClusteringCommon.hpp"

#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/DBSCAN.hpp>

#include <algorithm>
#include <vector>

TEST(DBSCANTest, MatchesSklearnDocumentedExample)
{
    const std::vector<float> samples{
        1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 8.0f, 7.0f, 8.0f, 8.0f, 25.0f, 80.0f,
    };

    irt::ops::DBSCANConfig config;
    config.eps         = 3.0f;
    config.min_samples = 2;

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

    const auto result = irt::ops::dbscan(data.samples.data(), data.num_samples, data.num_features, config);

    ASSERT_EQ(result.labels.size(), static_cast<size_t>(data.num_samples));
    EXPECT_EQ(std::count(result.labels.begin(), result.labels.end(), int64_t{-1}), 0);
    EXPECT_EQ(irt::test::countClusters(result.labels), 7);
    irt::test::expectSamePartition(result.labels, data.expected_labels);
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
}
