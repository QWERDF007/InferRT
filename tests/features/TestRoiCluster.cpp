/**
 * @file TestRoiCluster.cpp
 * @brief ``RoiCluster`` 公共配置与输入校验测试。
 */

#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/features/RoiCluster.hpp>

TEST(RoiClusterTest, DefaultConstructsWithRoiFeatureDefaults)
{
    const irt::features::RoiCluster cluster;

    EXPECT_EQ(cluster.config().model_name, irt::features::RoiCluster::kDefaultModelName);
    EXPECT_EQ(cluster.config().feature_name, irt::features::RoiCluster::kDefaultFeatureName);
    EXPECT_EQ(cluster.config().pooled_height, irt::features::kDefaultRoiFeaturePooledHeight);
    EXPECT_EQ(cluster.config().pooled_width, irt::features::kDefaultRoiFeaturePooledWidth);
    EXPECT_EQ(cluster.config().sampling_ratio, -1);
    EXPECT_FALSE(cluster.config().use_pca);
    EXPECT_EQ(cluster.featureDim(), 0);
}

TEST(RoiClusterTest, ConstructorRejectsInvalidRoiFeatureConfig)
{
    irt::features::RoiClusterConfig config;
    config.pooled_width = 0;

    EXPECT_THROW(irt::features::RoiCluster cluster(config), irt::Exception);
}
