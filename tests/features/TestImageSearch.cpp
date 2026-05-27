/**
 * @file TestImageSearch.cpp
 * @brief ``ImageSearch`` 与 CPU 磁盘 Faiss 索引的单元测试。
 */

#include <gtest/gtest.h>

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/features/ImageSearch.hpp>

#include "ImageSearchFaissIndex.hpp"

#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

/**
 * @brief 自动清理的临时目录，避免测试文件污染仓库。
 */
class TempDir
{
public:
    /**
     * @brief 创建唯一临时目录。
     */
    TempDir()
    {
        static std::atomic<int> counter{0};
        path_ = fs::temp_directory_path()
              / fs::path("inferrt_image_search_test_"
                         + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
        fs::create_directories(path_);
    }

    /**
     * @brief 析构时递归删除临时目录。
     */
    ~TempDir()
    {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }

    /**
     * @brief 获取临时目录路径。
     * @return 临时目录路径。
     */
    const fs::path &path() const noexcept
    {
        return path_;
    }

    /** @brief 禁止拷贝构造。 */
    TempDir(const TempDir &)            = delete;
    /** @brief 禁止拷贝赋值。 */
    TempDir &operator=(const TempDir &) = delete;

private:
    fs::path path_; ///< 临时目录绝对路径。
};

/**
 * @brief 写入一个简单测试文件。
 * @param path 文件路径。
 */
void writeFile(const fs::path &path)
{
    fs::create_directories(path.parent_path());
    std::ofstream(path) << "x";
}

/**
 * @brief 断言调用抛出 InferRT 异常且错误码符合预期。
 * @tparam Fn 可调用对象类型。
 * @param fn 待执行调用。
 * @param expected_code 期望错误码。
 */
template <typename Fn>
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

/**
 * @brief 默认构造应使用约定模型和特征名，且初始状态不可搜索。
 */
TEST(ImageSearchTest, DefaultConstructsNotReadySearcher)
{
    const irt::features::ImageSearch search;

    EXPECT_EQ(search.config().model_name, irt::features::ImageSearch::kDefaultModelName);
    EXPECT_EQ(search.config().feature_name, irt::features::ImageSearch::kDefaultFeatureName);
    EXPECT_FALSE(search.isReady());
    EXPECT_TRUE(search.indexPath().empty());
    EXPECT_TRUE(search.galleryImages().empty());
    EXPECT_EQ(search.featureDim(), 0);
    EXPECT_EQ(search.config().preprocess_backend, irt::features::ImageSearchPreprocessBackend::CPU);
    EXPECT_EQ(search.config().norm, irt::features::ImageSearchFeatureNorm::L2);
    EXPECT_EQ(search.config().faiss_backend, irt::features::ImageSearchFaissBackend::CPU);
    EXPECT_EQ(search.config().index_storage, irt::features::ImageSearchIndexStorage::RAM);
    EXPECT_EQ(search.config().disk_build_batch_size, irt::features::kDefaultImageSearchDiskBuildBatchSize);
}

/**
 * @brief ImageSearch config 应允许选择 CPU 后端和特征归一化策略。
 */
TEST(ImageSearchTest, ConstructorStoresConfig)
{
    irt::features::ImageSearchConfig config;
    config.model_name = "resnet18";
    config.feature_name = "layer4";
    config.norm = irt::features::ImageSearchFeatureNorm::L1;
    config.index_storage = irt::features::ImageSearchIndexStorage::Disk;
    config.disk_build_batch_size = 3;

    const irt::features::ImageSearch search(config);

    EXPECT_EQ(search.config().model_name, "resnet18");
    EXPECT_EQ(search.config().feature_name, "layer4");
    EXPECT_EQ(search.config().preprocess_backend, irt::features::ImageSearchPreprocessBackend::CPU);
    EXPECT_EQ(search.config().norm, irt::features::ImageSearchFeatureNorm::L1);
    EXPECT_EQ(search.config().faiss_backend, irt::features::ImageSearchFaissBackend::CPU);
    EXPECT_EQ(search.config().index_storage, irt::features::ImageSearchIndexStorage::Disk);
    EXPECT_EQ(search.config().disk_build_batch_size, 3U);

    config.norm = irt::features::ImageSearchFeatureNorm::None;
    const irt::features::ImageSearch default_search(config);

    EXPECT_EQ(default_search.config().model_name, irt::features::ImageSearch::kDefaultModelName);
    EXPECT_EQ(default_search.config().feature_name, irt::features::ImageSearch::kDefaultFeatureName);
    EXPECT_EQ(default_search.config().norm, irt::features::ImageSearchFeatureNorm::None);
    EXPECT_EQ(default_search.config().index_storage, irt::features::ImageSearchIndexStorage::Disk);
    EXPECT_EQ(default_search.config().disk_build_batch_size, 3U);
}

/**
 * @brief GPU 预处理当前只是预留配置项，应在构造时明确返回未实现。
 */
TEST(ImageSearchTest, ConstructorRejectsGpuPreprocessPlaceholder)
{
    irt::features::ImageSearchConfig config;
    config.model_name = "resnet18";
    config.feature_name = "layer4";
    config.preprocess_backend = irt::features::ImageSearchPreprocessBackend::GPU;

    expectIrtExceptionCode([&] { irt::features::ImageSearch search(config); },
                           irt::Status::ERROR_NOT_IMPLEMENTED);
}

/**
 * @brief GPU Faiss 后端应可通过 config 启用，实际资源在 buildOrLoad 时创建。
 */
TEST(ImageSearchTest, ConstructorAcceptsGpuFaissBackend)
{
    irt::features::ImageSearchConfig config;
    config.model_name = "resnet18";
    config.feature_name = "layer4";
    config.faiss_backend = irt::features::ImageSearchFaissBackend::GPU;
    config.index_storage = irt::features::ImageSearchIndexStorage::Disk;

    const irt::features::ImageSearch search(config);

    EXPECT_EQ(search.config().faiss_backend, irt::features::ImageSearchFaissBackend::GPU);
    EXPECT_EQ(search.config().index_storage, irt::features::ImageSearchIndexStorage::RAM);
    EXPECT_FALSE(search.isReady());
}

/**
 * @brief CPU disk 构建批量必须为正数，避免批处理循环无法推进。
 */
TEST(ImageSearchTest, ConstructorRejectsZeroDiskBuildBatchSize)
{
    irt::features::ImageSearchConfig config;
    config.model_name = "resnet18";
    config.feature_name = "layer4";
    config.disk_build_batch_size = 0;

    expectIrtExceptionCode([&] { irt::features::ImageSearch search(config); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief DINOv2/DINOv3 可直接使用归一化 CLS token 作为图像检索向量。
 */
TEST(ImageSearchTest, ConstructorAcceptsDinoBackbonesForClsTokenSearch)
{
    irt::features::ImageSearchConfig dinov2_config;
    dinov2_config.model_name = "dinov2_vits14";
    dinov2_config.feature_name = "x_norm_clstoken";
    const irt::features::ImageSearch dinov2(dinov2_config);

    irt::features::ImageSearchConfig dinov3_config;
    dinov3_config.model_name = "dinov3_vitb16";
    dinov3_config.feature_name = "x_norm_clstoken";
    const irt::features::ImageSearch dinov3(dinov3_config);

    EXPECT_EQ(dinov2.config().model_name, "dinov2_vits14");
    EXPECT_EQ(dinov2.config().feature_name, "x_norm_clstoken");
    EXPECT_FALSE(dinov2.isReady());

    EXPECT_EQ(dinov3.config().model_name, "dinov3_vitb16");
    EXPECT_EQ(dinov3.config().feature_name, "x_norm_clstoken");
    EXPECT_FALSE(dinov3.isReady());
}

/**
 * @brief ImageSearch 应识别常见图片扩展名，并忽略大小写。
 */
TEST(ImageSearchTest, IsImageFileAcceptsKnownExtensions)
{
    EXPECT_TRUE(irt::features::ImageSearch::isImageFile("a.jpg"));
    EXPECT_TRUE(irt::features::ImageSearch::isImageFile("a.JPEG"));
    EXPECT_TRUE(irt::features::ImageSearch::isImageFile("a.png"));
    EXPECT_TRUE(irt::features::ImageSearch::isImageFile("a.bmp"));
    EXPECT_TRUE(irt::features::ImageSearch::isImageFile("a.webp"));
    EXPECT_TRUE(irt::features::ImageSearch::isImageFile("dir.with.dot/a.jpg"));
    EXPECT_FALSE(irt::features::ImageSearch::isImageFile("a.txt"));
    EXPECT_FALSE(irt::features::ImageSearch::isImageFile("no_extension"));
    EXPECT_FALSE(irt::features::ImageSearch::isImageFile(".jpg"));
}

/**
 * @brief 默认索引路径应将模型名和特征名消毒为安全文件名。
 */
TEST(ImageSearchTest, DefaultIndexPathSanitizesModelAndFeatureNames)
{
    const auto path = irt::features::ImageSearch::defaultIndexPath("gallery", "wide_resnet50_2", "layer/4.out");
    EXPECT_EQ(path.generic_string(), "gallery/wide_resnet50_2_layer_4_out.faiss");
}

/**
 * @brief DINO 特征名包含下划线或点号时，也应生成稳定的索引文件名。
 */
TEST(ImageSearchTest, DefaultIndexPathHandlesDinoFeatureNames)
{
    const auto cls_path
        = irt::features::ImageSearch::defaultIndexPath("gallery", "dinov3_vitb16", "x_norm_clstoken");
    const auto block_path = irt::features::ImageSearch::defaultIndexPath("gallery", "dinov2_vits14", "blocks.11");

    EXPECT_EQ(cls_path.generic_string(), "gallery/dinov3_vitb16_x_norm_clstoken.faiss");
    EXPECT_EQ(block_path.generic_string(), "gallery/dinov2_vits14_blocks_11.faiss");
}

/**
 * @brief RAM 模式应使用内存 IVF-PQ 压缩索引，而不是 Flat 全量浮点索引。
 */
TEST(ImageSearchTest, RamIndexUsesInMemoryIvfPqCompression)
{
    constexpr int    feature_dim  = 16;
    constexpr size_t vector_count = 64;

    std::vector<float> features(vector_count * feature_dim, 0.0f);
    for (size_t row = 0; row < vector_count; ++row)
    {
        features[row * feature_dim + (row % feature_dim)] = 1.0f;
        features[row * feature_dim + ((row * 3 + 1) % feature_dim)] += 0.25f;
    }

    auto load_feature = [&](size_t row) {
        const auto begin = features.begin() + static_cast<std::ptrdiff_t>(row * feature_dim);
        return std::vector<float>(begin, begin + feature_dim);
    };

    auto index = irt::features::priv::buildRamIvfPqIndex(vector_count, feature_dim, 16, load_feature);
    ASSERT_TRUE(index);
    EXPECT_EQ(index->ntotal, static_cast<faiss::idx_t>(vector_count));
    EXPECT_EQ(index->metric_type, faiss::METRIC_INNER_PRODUCT);

    auto *ivfpq = dynamic_cast<faiss::IndexIVFPQ *>(index.get());
    ASSERT_NE(ivfpq, nullptr);
    EXPECT_LT(ivfpq->code_size, static_cast<size_t>(feature_dim) * sizeof(float));
    EXPECT_GE(ivfpq->nprobe, 1U);

    auto query = load_feature(0);
    std::vector<float> distances(3);
    std::vector<faiss::idx_t> labels(3);
    index->search(1, query.data(), static_cast<faiss::idx_t>(labels.size()), distances.data(), labels.data());
    EXPECT_GE(labels[0], 0);
}

/**
 * @brief GPU 兼容的 RAM IVF-PQ 索引应固定使用 8 位子码（``require_gpu_compatible=true``）。
 */
TEST(ImageSearchTest, RamIvfPqGpuCompatibleIndexUsesEightBitCodes)
{
    constexpr int    feature_dim  = 16;
    constexpr size_t vector_count = 16;

    std::vector<float> features(vector_count * feature_dim, 0.0f);
    for (size_t row = 0; row < vector_count; ++row)
    {
        features[row * feature_dim + (row % feature_dim)] = 1.0f;
        features[row * feature_dim + ((row * 3 + 1) % feature_dim)] += 0.25f;
    }

    auto load_feature = [&](size_t row) {
        const auto begin = features.begin() + static_cast<std::ptrdiff_t>(row * feature_dim);
        return std::vector<float>(begin, begin + feature_dim);
    };

    auto index = irt::features::priv::buildRamIvfPqIndex(vector_count, feature_dim, 16, load_feature, true);
    ASSERT_TRUE(index);
    EXPECT_EQ(index->ntotal, static_cast<faiss::idx_t>(vector_count));

    auto *ivfpq = dynamic_cast<faiss::IndexIVFPQ *>(index.get());
    ASSERT_NE(ivfpq, nullptr);
    EXPECT_EQ(ivfpq->pq.nbits, 8U);
}

/**
 * @brief CPU disk 模式应使用 IVF + mmap/OnDiskInvertedLists，而不是搜索时整索引读入内存。
 */
TEST(ImageSearchTest, CpuDiskIndexUsesOnDiskIvfInvertedLists)
{
    constexpr int feature_dim = 4;
    constexpr size_t vector_count = 16;

    std::vector<float> features(vector_count * feature_dim, 0.0f);
    for (size_t row = 0; row < vector_count; ++row)
    {
        features[row * feature_dim + (row % feature_dim)] = 1.0f;
        features[row * feature_dim + ((row + 1) % feature_dim)] = 0.01f * static_cast<float>(row + 1);
    }

    auto load_feature = [&](size_t row) {
        const auto begin = features.begin() + static_cast<std::ptrdiff_t>(row * feature_dim);
        return std::vector<float>(begin, begin + feature_dim);
    };

    TempDir temp;
    const auto index_path = temp.path() / "synthetic_disk.faiss";
    auto index = irt::features::priv::buildCpuOnDiskIvfFlatIndex(vector_count, feature_dim, index_path, 3,
                                                                 load_feature);
    const auto data_path = irt::features::priv::cpuOnDiskIvfDataPath(index_path);

    ASSERT_TRUE(index);
    EXPECT_TRUE(fs::exists(index_path));
    EXPECT_TRUE(fs::exists(data_path));
    EXPECT_GT(fs::file_size(data_path), 0);
    EXPECT_EQ(index->ntotal, static_cast<faiss::idx_t>(vector_count));

    auto *ivf_index = dynamic_cast<faiss::IndexIVF *>(index.get());
    ASSERT_NE(ivf_index, nullptr);
    ASSERT_NE(ivf_index->invlists, nullptr);
    EXPECT_EQ(index->metric_type, faiss::METRIC_INNER_PRODUCT);
    EXPECT_GE(ivf_index->nprobe, 1U);

    const std::string invlists_type = typeid(*ivf_index->invlists).name();
    EXPECT_NE(invlists_type.find("OnDisk"), std::string::npos) << invlists_type;

    auto query = load_feature(0);
    std::vector<float> distances(3);
    std::vector<faiss::idx_t> labels(3);
    index->search(1, query.data(), static_cast<faiss::idx_t>(labels.size()), distances.data(), labels.data());
    EXPECT_GE(labels[0], 0);

    auto reloaded = irt::features::priv::loadCpuOnDiskIvfFlatIndex(index_path);
    EXPECT_EQ(reloaded->ntotal, static_cast<faiss::idx_t>(vector_count));
    auto *reloaded_ivf = dynamic_cast<faiss::IndexIVF *>(reloaded.get());
    ASSERT_NE(reloaded_ivf, nullptr);
    ASSERT_NE(reloaded_ivf->invlists, nullptr);
    const std::string reloaded_invlists_type = typeid(*reloaded_ivf->invlists).name();
    EXPECT_NE(reloaded_invlists_type.find("OnDisk"), std::string::npos) << reloaded_invlists_type;
}

/**
 * @brief 高维 DINO 特征下，IVF 训练/构建批大小应受内存上限约束。
 *
 * 验证 ``chooseCpuOnDiskIvf*`` 在约 1369×384 维、12500 条向量规模下不会超出
 * ``kCpuOnDiskIvfMaxTrainingBytes`` 与 ``kCpuOnDiskIvfMaxBatchBytes``。
 */
TEST(ImageSearchTest, CpuDiskIndexBoundsWideFeatureBuffers)
{
    constexpr size_t vector_count = 12500;
    constexpr int    feature_dim  = 1369 * 384;

    const auto nlist = irt::features::priv::chooseCpuOnDiskIvfListCount(vector_count, feature_dim);
    const auto training_count =
        irt::features::priv::chooseCpuOnDiskIvfTrainingCount(vector_count, feature_dim, nlist);
    const auto batch_size = irt::features::priv::chooseCpuOnDiskIvfBuildBatchSize(256, vector_count, feature_dim);

    const auto bytes_per_feature = static_cast<size_t>(feature_dim) * sizeof(float);

    EXPECT_GE(nlist, 1U);
    EXPECT_LE(nlist, training_count);
    EXPECT_LT(training_count, irt::features::priv::kCpuOnDiskIvfMaxTrainingVectors);
    EXPECT_LE(training_count * bytes_per_feature, irt::features::priv::kCpuOnDiskIvfMaxTrainingBytes);
    EXPECT_LT(batch_size, 256U);
    EXPECT_LE(batch_size * bytes_per_feature, irt::features::priv::kCpuOnDiskIvfMaxBatchBytes);
}

/**
 * @brief 384 维 CLS 特征、万级图库应能成功构建 CPU 磁盘 IVF 索引并落盘。
 */
TEST(ImageSearchTest, CpuDiskIndexBuildsDinoClsSizedGallery)
{
    constexpr size_t vector_count = 12500;
    constexpr int    feature_dim  = 384;

    std::vector<float> features(vector_count * feature_dim, 0.0f);
    for (size_t row = 0; row < vector_count; ++row)
    {
        const size_t first  = row % feature_dim;
        const size_t second = (row * 37 + 11) % feature_dim;
        features[row * feature_dim + first] = 1.0f;
        features[row * feature_dim + second] += 0.25f;
    }

    auto load_feature = [&](size_t row) {
        const auto begin = features.begin() + static_cast<std::ptrdiff_t>(row * feature_dim);
        return std::vector<float>(begin, begin + feature_dim);
    };

    TempDir temp;
    const auto index_path = temp.path() / "dino_cls_sized_disk.faiss";
    auto index = irt::features::priv::buildCpuOnDiskIvfFlatIndex(vector_count, feature_dim, index_path, 256,
                                                                 load_feature);

    ASSERT_TRUE(index);
    EXPECT_EQ(index->ntotal, static_cast<faiss::idx_t>(vector_count));
    EXPECT_TRUE(fs::exists(index_path));
    EXPECT_TRUE(fs::exists(irt::features::priv::cpuOnDiskIvfDataPath(index_path)));
}

/**
 * @brief collectGalleryImages 应递归收集图片文件，忽略非图片，并返回排序后的绝对路径。
 */
TEST(ImageSearchTest, CollectGalleryImagesReturnsSortedAbsoluteImages)
{
    TempDir temp;
    writeFile(temp.path() / "b.png");
    writeFile(temp.path() / "nested" / "c.webp");
    writeFile(temp.path() / "a.JPG");
    writeFile(temp.path() / "notes.txt");

    auto images = irt::features::ImageSearch::collectGalleryImages(temp.path());

    std::vector<fs::path> expected{
        fs::absolute(temp.path() / "a.JPG"),
        fs::absolute(temp.path() / "b.png"),
        fs::absolute(temp.path() / "nested" / "c.webp"),
    };
    std::sort(expected.begin(), expected.end());

    EXPECT_EQ(images, expected);
}

/**
 * @brief collectGalleryImages 应拒绝不存在的图库目录。
 */
TEST(ImageSearchTest, CollectGalleryImagesRejectsMissingDirectory)
{
    TempDir temp;
    expectIrtExceptionCode([&] { irt::features::ImageSearch::collectGalleryImages(temp.path() / "missing"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief collectGalleryImages 应拒绝没有任何图片文件的图库目录。
 */
TEST(ImageSearchTest, CollectGalleryImagesRejectsEmptyGallery)
{
    TempDir temp;
    writeFile(temp.path() / "notes.txt");

    expectIrtExceptionCode([&] { irt::features::ImageSearch::collectGalleryImages(temp.path()); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 显式图片路径列表构建时必须指定 Faiss 索引路径。
 */
TEST(ImageSearchTest, BuildFromExplicitImagePathsRequiresIndexFile)
{
    TempDir temp;
    writeFile(temp.path() / "a.jpg");

    irt::features::ImageSearch search;
    const std::vector<fs::path> images{temp.path() / "a.jpg"};

    expectIrtExceptionCode([&] { search.build("weights.wts", images, {}); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 显式图片路径列表构建时，空列表应在加载权重前被拒绝。
 */
TEST(ImageSearchTest, BuildFromExplicitImagePathsRejectsEmptyList)
{
    TempDir temp;

    irt::features::ImageSearch search;
    const std::vector<fs::path> images;

    expectIrtExceptionCode([&] { search.build("weights.wts", images, temp.path() / "index.faiss"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 未注册的模型名应在 ImageSearch 构造时被拒绝。
 */
TEST(ImageSearchTest, ConstructorRejectsUnsupportedModel)
{
    irt::features::ImageSearchConfig config;
    config.model_name = "not_a_model";
    config.feature_name = "layer4";

    expectIrtExceptionCode([&] { irt::features::ImageSearch search(config); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 空特征名应在构造 ImageSearch 时被拒绝。
 */
TEST(ImageSearchTest, ConstructorRejectsEmptyFeatureName)
{
    irt::features::ImageSearchConfig config;
    config.model_name = "resnet18";
    config.feature_name = "";

    expectIrtExceptionCode([&] { irt::features::ImageSearch search(config); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 未 buildOrLoad 的检索器调用 search 时应返回未就绪错误。
 */
TEST(ImageSearchTest, SearchBeforeBuildOrLoadThrowsInvalidOperation)
{
    irt::features::ImageSearchConfig config;
    config.model_name = "resnet18";
    config.feature_name = "layer4";
    irt::features::ImageSearch search(config);

    expectIrtExceptionCode([&] { search.search("query.jpg"); }, irt::Status::ERROR_INVALID_OPERATION);

    config.index_storage = irt::features::ImageSearchIndexStorage::Disk;
    irt::features::ImageSearch disk_search(config);

    expectIrtExceptionCode([&] { disk_search.search("query.jpg"); }, irt::Status::ERROR_INVALID_OPERATION);
}

/**
 * @brief 移动构造后的检索器应保留配置字段，便于作为可移动资源返回。
 */
TEST(ImageSearchTest, MoveConstructedSearcherKeepsConfiguration)
{
    irt::features::ImageSearchConfig config;
    config.model_name = "resnet18";
    config.feature_name = "layer4";
    irt::features::ImageSearch source(config);
    irt::features::ImageSearch moved(std::move(source));

    EXPECT_EQ(moved.config().model_name, "resnet18");
    EXPECT_EQ(moved.config().feature_name, "layer4");
    EXPECT_FALSE(moved.isReady());
}
