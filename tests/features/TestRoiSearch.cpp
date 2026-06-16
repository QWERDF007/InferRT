/**
 * @file TestRoiSearch.cpp
 * @brief ``RoiSearch`` API 与输入校验单元测试。
 */

#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/features/RoiSearch.hpp>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

/**
 * @brief 自动清理的临时目录。
 */
class TempDir
{
public:
    TempDir()
    {
        static std::atomic<int> counter{0};
        path_ = fs::temp_directory_path()
              / fs::path("inferrt_roi_search_test_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
        fs::create_directories(path_);
    }

    ~TempDir()
    {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }

    const fs::path &path() const noexcept
    {
        return path_;
    }

    TempDir(const TempDir &)            = delete;
    TempDir &operator=(const TempDir &) = delete;

private:
    fs::path path_; ///< 临时目录路径。
};

/**
 * @brief 写入一个占位文件。
 */
void writeFile(const fs::path &path)
{
    fs::create_directories(path.parent_path());
    std::ofstream(path) << "x";
}

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

TEST(RoiSearchTest, DefaultConstructsNotReadySearcher)
{
    const irt::features::RoiSearch search;

    EXPECT_EQ(search.config().model_name, irt::features::RoiSearch::kDefaultModelName);
    EXPECT_EQ(search.config().feature_name, irt::features::RoiSearch::kDefaultFeatureName);
    EXPECT_EQ(search.config().pooled_height, irt::features::kDefaultRoiSearchPooledHeight);
    EXPECT_EQ(search.config().pooled_width, irt::features::kDefaultRoiSearchPooledWidth);
    EXPECT_EQ(search.config().sampling_ratio, -1);
    EXPECT_FALSE(search.config().aligned);
    EXPECT_FALSE(search.config().use_pca);
    EXPECT_EQ(search.config().pca_dim, 0);
    EXPECT_FALSE(search.isReady());
    EXPECT_TRUE(search.indexPath().empty());
    EXPECT_TRUE(search.galleryItems().empty());
    EXPECT_EQ(search.featureDim(), 0);
}

TEST(RoiSearchTest, ConstructorStoresConfig)
{
    irt::features::RoiSearchConfig config;
    config.model_name            = "resnet18";
    config.feature_name          = "layer3";
    config.model_backend         = irt::model::ModelBackend::ONNXRuntime;
    config.model_device          = irt::model::ModelDevice::CPU;
    config.norm                  = irt::features::ImageSearchFeatureNorm::L1;
    config.index_storage         = irt::features::ImageSearchIndexStorage::Disk;
    config.disk_build_batch_size = 4;
    config.model_batch_size      = 2;
    config.pooled_height         = 3;
    config.pooled_width          = 5;
    config.sampling_ratio        = 2;
    config.aligned               = true;
    config.use_pca               = true;
    config.pca_dim               = 32;

    const irt::features::RoiSearch search(config);

    EXPECT_EQ(search.config().model_name, "resnet18");
    EXPECT_EQ(search.config().feature_name, "layer3");
    EXPECT_EQ(search.config().model_backend, irt::model::ModelBackend::ONNXRuntime);
    EXPECT_EQ(search.config().model_device, irt::model::ModelDevice::CPU);
    EXPECT_EQ(search.config().norm, irt::features::ImageSearchFeatureNorm::L1);
    EXPECT_EQ(search.config().index_storage, irt::features::ImageSearchIndexStorage::Disk);
    EXPECT_EQ(search.config().disk_build_batch_size, 4U);
    EXPECT_EQ(search.config().model_batch_size, 2U);
    EXPECT_EQ(search.config().pooled_height, 3);
    EXPECT_EQ(search.config().pooled_width, 5);
    EXPECT_EQ(search.config().sampling_ratio, 2);
    EXPECT_TRUE(search.config().aligned);
    EXPECT_TRUE(search.config().use_pca);
    EXPECT_EQ(search.config().pca_dim, 32);
}

TEST(RoiSearchTest, ConstructorNormalizesGpuFaissStorage)
{
    irt::features::RoiSearchConfig config;
    config.model_name    = "resnet18";
    config.feature_name  = "layer4";
    config.faiss_backend = irt::features::ImageSearchFaissBackend::GPU;
    config.index_storage = irt::features::ImageSearchIndexStorage::Disk;

    const irt::features::RoiSearch search(config);

    EXPECT_EQ(search.config().faiss_backend, irt::features::ImageSearchFaissBackend::GPU);
    EXPECT_EQ(search.config().index_storage, irt::features::ImageSearchIndexStorage::RAM);
}

TEST(RoiSearchTest, ConstructorRejectsInvalidConfig)
{
    irt::features::RoiSearchConfig unsupported_model;
    unsupported_model.model_name = "not_a_model";
    expectIrtExceptionCode([&] { irt::features::RoiSearch search(unsupported_model); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    irt::features::RoiSearchConfig empty_feature;
    empty_feature.feature_name = "";
    expectIrtExceptionCode([&] { irt::features::RoiSearch search(empty_feature); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    irt::features::RoiSearchConfig bad_pool;
    bad_pool.pooled_width = 0;
    expectIrtExceptionCode([&] { irt::features::RoiSearch search(bad_pool); }, irt::Status::ERROR_INVALID_ARGUMENT);

    irt::features::RoiSearchConfig bad_sampling;
    bad_sampling.sampling_ratio = -2;
    expectIrtExceptionCode([&] { irt::features::RoiSearch search(bad_sampling); }, irt::Status::ERROR_INVALID_ARGUMENT);

    irt::features::RoiSearchConfig bad_pca_dim;
    bad_pca_dim.pca_dim = -1;
    expectIrtExceptionCode([&] { irt::features::RoiSearch search(bad_pca_dim); }, irt::Status::ERROR_INVALID_ARGUMENT);

    irt::features::RoiSearchConfig missing_pca_dim;
    missing_pca_dim.use_pca = true;
    missing_pca_dim.pca_dim = 0;
    expectIrtExceptionCode([&] { irt::features::RoiSearch search(missing_pca_dim); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

TEST(RoiSearchTest, DefaultIndexPathUsesRoiExtension)
{
    const auto path = irt::features::RoiSearch::defaultIndexPath("gallery", "wide_resnet50_2", "layer/4.out");

    EXPECT_EQ(path.generic_string(), "gallery/wide_resnet50_2_layer_4_out.roi.faiss");
}

TEST(RoiSearchTest, SearchBeforeBuildThrowsInvalidOperation)
{
    irt::features::RoiSearchConfig config;
    config.model_name   = "resnet18";
    config.feature_name = "layer4";
    irt::features::RoiSearch search(config);

    expectIrtExceptionCode([&] { search.search("query.jpg", {0.0f, 0.0f, 10.0f, 10.0f}); },
                           irt::Status::ERROR_INVALID_OPERATION);
}

TEST(RoiSearchTest, BuildRequiresExplicitIndexFile)
{
    TempDir temp;
    writeFile(temp.path() / "a.jpg");

    irt::features::RoiSearch                        search;
    const std::vector<irt::features::RoiSearchItem> items{
        {temp.path() / "a.jpg", {0.0f, 0.0f, 10.0f, 10.0f}},
    };

    expectIrtExceptionCode([&] { search.build("weights.wts", items, {}); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

TEST(RoiSearchTest, BuildRejectsBadItemsBeforeLoadingModel)
{
    TempDir temp;
    writeFile(temp.path() / "a.jpg");

    irt::features::RoiSearch search;
    const auto               index_path = temp.path() / "roi.faiss";

    expectIrtExceptionCode([&] { search.build("weights.wts", {}, index_path); }, irt::Status::ERROR_INVALID_ARGUMENT);

    const std::vector<irt::features::RoiSearchItem> bad_roi{
        {temp.path() / "a.jpg", {10.0f, 0.0f, 5.0f, 10.0f}},
    };
    expectIrtExceptionCode([&] { search.build("weights.wts", bad_roi, index_path); },
                           irt::Status::ERROR_INVALID_ARGUMENT);

    const std::vector<irt::features::RoiSearchItem> missing_image{
        {temp.path() / "missing.jpg", {0.0f, 0.0f, 5.0f, 10.0f}},
    };
    expectIrtExceptionCode([&] { search.build("weights.wts", missing_image, index_path); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}
