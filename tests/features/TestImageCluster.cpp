/**
 * @file TestImageCluster.cpp
 * @brief ``ImageCluster`` API 与输入校验单元测试。
 */

#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/features/ImageCluster.hpp>

#include <utility>

namespace {

template<typename Fn>
void expectIrtExceptionCode(Fn &&fn, irt::Status expected_code)
{
    try
    {
        std::forward<Fn>(fn)();
        FAIL() << "Expected irt::Exception";
    }
    catch (const irt::Exception &e)
    {
        EXPECT_EQ(e.code(), expected_code);
    }
}

} // namespace

TEST(ImageClusterTest, DefaultConstructsWithImageFeatureDefaults)
{
    const irt::features::ImageCluster cluster;

    EXPECT_EQ(cluster.config().model_name, irt::features::ImageCluster::kDefaultModelName);
    EXPECT_EQ(cluster.config().feature_name, irt::features::ImageCluster::kDefaultFeatureName);
    EXPECT_EQ(cluster.config().model_runtime, irt::model::ModelRuntime{});
    EXPECT_EQ(cluster.config().model_precision, irt::model::ModelPrecision::FP32);
    EXPECT_FALSE(cluster.config().use_pca);
    EXPECT_EQ(cluster.config().pca_dim, 0);
    EXPECT_EQ(cluster.config().hdbscan.min_cluster_size, 5);
    EXPECT_EQ(cluster.featureDim(), 0);
}

TEST(ImageClusterTest, ConstructorStoresConfig)
{
    irt::features::ImageClusterConfig config;
    config.model_name                        = "dinov2_vits14";
    config.feature_name                      = "x_norm_patchtokens";
        config.model_runtime                     = irt::model::ModelRuntime::parse("onnxruntime:1");
    config.model_precision                   = irt::model::ModelPrecision::FP16;
    config.norm                              = irt::features::ImageSearchFeatureNorm::L1;
    config.model_batch_size                  = 2;
    config.use_pca                           = true;
    config.pca_dim                           = 8;
    config.hdbscan.min_cluster_size          = 3;
    config.hdbscan.min_samples               = 2;
    config.hdbscan.cluster_selection_epsilon = 0.1;
    config.hdbscan.allow_single_cluster      = true;

    const irt::features::ImageCluster cluster(config);

    EXPECT_EQ(cluster.config().model_name, "dinov2_vits14");
    EXPECT_EQ(cluster.config().feature_name, "x_norm_patchtokens");
        EXPECT_EQ(cluster.config().model_runtime.toString(), "onnxruntime:1");
    EXPECT_EQ(cluster.config().model_precision, irt::model::ModelPrecision::FP16);
    EXPECT_EQ(cluster.config().norm, irt::features::ImageSearchFeatureNorm::L1);
    EXPECT_EQ(cluster.config().model_batch_size, 2U);
    EXPECT_TRUE(cluster.config().use_pca);
    EXPECT_EQ(cluster.config().pca_dim, 8);
    EXPECT_EQ(cluster.config().hdbscan.min_cluster_size, 3);
    EXPECT_EQ(cluster.config().hdbscan.min_samples, 2);
    EXPECT_DOUBLE_EQ(cluster.config().hdbscan.cluster_selection_epsilon, 0.1);
    EXPECT_TRUE(cluster.config().hdbscan.allow_single_cluster);
}

TEST(ImageClusterTest, ConstructorRejectsInvalidConfig)
{
    irt::features::ImageClusterConfig unsupported_model;
    unsupported_model.model_name = "not_a_model";
    expectIrtExceptionCode([&] { irt::features::ImageCluster cluster(unsupported_model); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    irt::features::ImageClusterConfig empty_feature;
    empty_feature.feature_name = "";
    expectIrtExceptionCode([&] { irt::features::ImageCluster cluster(empty_feature); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    irt::features::ImageClusterConfig bad_pca_dim;
    bad_pca_dim.pca_dim = -1;
    expectIrtExceptionCode([&] { irt::features::ImageCluster cluster(bad_pca_dim); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    irt::features::ImageClusterConfig missing_pca_dim;
    missing_pca_dim.use_pca = true;
    missing_pca_dim.pca_dim = 0;
    expectIrtExceptionCode([&] { irt::features::ImageCluster cluster(missing_pca_dim); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    expectIrtExceptionCode(
        [&] { irt::model::ModelRuntime::parse("gpu:-1"); }, irt::Status::ERROR_INVALID_ARGUMENT);
}
