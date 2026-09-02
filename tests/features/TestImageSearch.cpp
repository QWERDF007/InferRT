/**
 * @file TestImageSearch.cpp
 * @brief ``ImageSearch`` 与 CPU 磁盘 Faiss 索引的单元测试。
 */

#include "ImageSearchFaissIndex.hpp"

#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>
#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.h>
#include <inferrt/features/ImageSearch.hpp>

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
        path_
            = fs::temp_directory_path()
            / fs::path("inferrt_image_search_test_" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
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
    TempDir(const TempDir &) = delete;
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
    EXPECT_TRUE(search.galleryIds().empty());
    EXPECT_EQ(search.featureDim(), 0);
    EXPECT_EQ(search.config().preprocess_backend, irt::features::ImageSearchPreprocessBackend::CPU);
    EXPECT_EQ(search.config().model_runtime, irt::model::ModelRuntime{});
    EXPECT_EQ(search.config().model_precision, irt::model::ModelPrecision::FP32);
    EXPECT_EQ(search.config().norm, irt::features::ImageSearchFeatureNorm::L2);
    EXPECT_EQ(search.config().faiss_backend, irt::features::ImageSearchFaissBackend::CPU);
    EXPECT_EQ(search.config().index_storage, irt::features::ImageSearchIndexStorage::RAM);
    EXPECT_EQ(search.config().model_batch_size, irt::features::kDefaultImageSearchModelBatchSize);
    EXPECT_EQ(search.config().preprocess.input_width, 0);
    EXPECT_EQ(search.config().preprocess.input_height, 0);
    EXPECT_EQ(search.config().preprocess.input_channels, 3);
}

TEST(ImageSearchTest, PreprocessSpecIsValidatedAsPartOfFeatureConfig)
{
    irt::features::ImageSearchConfig config;
    EXPECT_NO_THROW((void)irt::features::ImageSearch(config));

    config.preprocess.input_width = 224;
    config.preprocess.input_height = 224;
    config.preprocess.input_channels = 1;
    config.preprocess.src_color = irt::ColorFormat::GRAY;
    config.preprocess.dst_color = irt::ColorFormat::GRAY;
    config.preprocess.mean = {0.0F};
    config.preprocess.stddev = {1.0F};
    EXPECT_NO_THROW((void)irt::features::ImageSearch(config));

    config.preprocess.stddev = {0.0F};
    EXPECT_THROW((void)irt::features::ImageSearch(config), irt::Exception);
}

/**
 * @brief ImageSearch config 应允许选择 CPU 后端和特征归一化策略。
 */
TEST(ImageSearchTest, ConstructorStoresConfig)
{
    irt::features::ImageSearchConfig config;
    config.model_name            = "resnet18";
    config.feature_name          = "layer4";
    config.model_runtime         = irt::model::ModelRuntime::parse("openvino:2");
    config.model_precision       = irt::model::ModelPrecision::FP16;
    config.norm                  = irt::features::ImageSearchFeatureNorm::L1;
    config.index_storage         = irt::features::ImageSearchIndexStorage::Disk;
    config.model_batch_size      = 3;

    const irt::features::ImageSearch search(config);

    EXPECT_EQ(search.config().model_name, "resnet18");
    EXPECT_EQ(search.config().feature_name, "layer4");
    EXPECT_EQ(search.config().model_runtime.toString(), "openvino:2");
    EXPECT_EQ(search.config().model_precision, irt::model::ModelPrecision::FP16);
    EXPECT_EQ(search.config().preprocess_backend, irt::features::ImageSearchPreprocessBackend::CPU);
    EXPECT_EQ(search.config().norm, irt::features::ImageSearchFeatureNorm::L1);
    EXPECT_EQ(search.config().faiss_backend, irt::features::ImageSearchFaissBackend::CPU);
    EXPECT_EQ(search.config().index_storage, irt::features::ImageSearchIndexStorage::Disk);
    EXPECT_EQ(search.config().model_batch_size, 3U);

    config.norm = irt::features::ImageSearchFeatureNorm::None;
    const irt::features::ImageSearch default_search(config);

    EXPECT_EQ(default_search.config().model_name, irt::features::ImageSearch::kDefaultModelName);
    EXPECT_EQ(default_search.config().feature_name, irt::features::ImageSearch::kDefaultFeatureName);
    EXPECT_EQ(default_search.config().model_runtime.toString(), "openvino:2");
    EXPECT_EQ(default_search.config().model_precision, irt::model::ModelPrecision::FP16);
    EXPECT_EQ(default_search.config().norm, irt::features::ImageSearchFeatureNorm::None);
    EXPECT_EQ(default_search.config().index_storage, irt::features::ImageSearchIndexStorage::Disk);
    EXPECT_EQ(default_search.config().model_batch_size, 3U);
}

/**
 * @brief TensorRT 图像检索配置应允许设置特征提取模型 batch。
 */
TEST(ImageSearchTest, ConstructorStoresTensorRtModelBatchSize)
{
    irt::features::ImageSearchConfig config;
    config.model_name       = "resnet18";
    config.feature_name     = "layer4";
    config.model_batch_size = 4;

    const irt::features::ImageSearch search(config);

    EXPECT_EQ(search.config().model_batch_size, 4U);
}

/**
 * @brief ImageSearch config 应允许选择 ONNX Runtime 图后端。
 */
TEST(ImageSearchTest, ConstructorAcceptsOnnxRuntimeBackend)
{
    irt::features::ImageSearchConfig config;
    config.model_name    = "resnet18";
    config.feature_name  = "layer4";
    config.model_runtime = irt::model::ModelRuntime::parse("onnxruntime:cpu");

    const irt::features::ImageSearch search(config);

    EXPECT_EQ(search.config().model_runtime.toString(), "onnxruntime:cpu");
    EXPECT_FALSE(search.isReady());
}

/**
 * @brief TensorRT 特征提取后端需要 GPU 设备。
 */
TEST(ImageSearchTest, ConstructorRejectsTensorRtCpuDevice)
{
    irt::features::ImageSearchConfig config;
    config.model_name    = "resnet18";
    config.feature_name  = "layer4";
    expectIrtExceptionCode(
        [&]
        {
            config.model_runtime = irt::model::ModelRuntime::parse("tensorrt:cpu");
            irt::features::ImageSearch search(config);
        },
        irt::Status::ERROR_NOT_IMPLEMENTED);
}

/**
 * @brief GPU 预处理当前只是预留配置项，应在构造时明确返回未实现。
 */
TEST(ImageSearchTest, ConstructorRejectsGpuPreprocessPlaceholder)
{
    irt::features::ImageSearchConfig config;
    config.model_name         = "resnet18";
    config.feature_name       = "layer4";
    config.preprocess_backend = irt::features::ImageSearchPreprocessBackend::GPU;

    expectIrtExceptionCode([&] { irt::features::ImageSearch search(config); }, irt::Status::ERROR_NOT_IMPLEMENTED);
}

/**
 * @brief GPU Faiss 后端应可通过 config 启用，实际资源在 buildOrLoad 时创建。
 */
TEST(ImageSearchTest, ConstructorAcceptsGpuFaissBackend)
{
    irt::features::ImageSearchConfig config;
    config.model_name    = "resnet18";
    config.feature_name  = "layer4";
    config.faiss_backend = irt::features::ImageSearchFaissBackend::GPU;
    config.index_storage = irt::features::ImageSearchIndexStorage::Disk;

    const irt::features::ImageSearch search(config);

    EXPECT_EQ(search.config().faiss_backend, irt::features::ImageSearchFaissBackend::GPU);
    EXPECT_EQ(search.config().index_storage, irt::features::ImageSearchIndexStorage::RAM);
    EXPECT_FALSE(search.isReady());
}

/**
 * @brief 模型推理 batch 必须为正数。
 */
TEST(ImageSearchTest, ConstructorRejectsZeroModelBatchSize)
{
    irt::features::ImageSearchConfig config;
    config.model_name       = "resnet18";
    config.feature_name     = "layer4";
    config.model_batch_size = 0;

    expectIrtExceptionCode([&] { irt::features::ImageSearch search(config); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 图后端也应保留用户配置的模型 batch，实际动态性在加载图模型时校验。
 */
TEST(ImageSearchTest, ConstructorStoresGraphBackendModelBatchSize)
{
    const std::vector<irt::model::ModelRuntime::Backend> graph_backends{
        irt::model::ModelRuntime::Backend::ONNXRuntime,
        irt::model::ModelRuntime::Backend::OpenVINO,
    };

    for (const auto backend : graph_backends)
    {
        irt::features::ImageSearchConfig config;
        config.model_name       = "resnet18";
        config.feature_name     = "layer4";
        config.model_runtime    = {backend, irt::model::ModelRuntime::Device::CPU};
        config.model_batch_size = 2;

        const irt::features::ImageSearch search(config);

        EXPECT_EQ(search.config().model_runtime.backend(), backend);
        EXPECT_EQ(search.config().model_batch_size, 2U);
        EXPECT_FALSE(search.isReady());
    }
}

/**
 * @brief DINOv2/DINOv3 可直接使用归一化 CLS token 作为图像检索向量。
 */
TEST(ImageSearchTest, ConstructorAcceptsDinoBackbonesForClsTokenSearch)
{
    irt::features::ImageSearchConfig dinov2_config;
    dinov2_config.model_name   = "dinov2_vits14";
    dinov2_config.feature_name = "x_norm_clstoken";
    const irt::features::ImageSearch dinov2(dinov2_config);

    irt::features::ImageSearchConfig dinov3_config;
    dinov3_config.model_name   = "dinov3_vitb16";
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
 * @brief 默认索引路径使用图库目录下的时间戳文件名。
 */
TEST(ImageSearchTest, DefaultIndexPathUsesTimestampFileName)
{
    const auto path = irt::features::ImageSearch::defaultIndexPath("gallery", "wide_resnet50_2", "layer/4.out");

    EXPECT_EQ(path.parent_path().generic_string(), "gallery");
    EXPECT_EQ(path.extension().string(), ".faiss");
    EXPECT_FALSE(path.stem().empty());
    EXPECT_EQ(path.filename().string().find("wide_resnet50_2"), std::string::npos);
    EXPECT_EQ(path.filename().string().find("layer"), std::string::npos);
}

/**
 * @brief 图像检索配置应拒绝未知模型精度。
 */
TEST(ImageSearchTest, ConstructorRejectsInvalidModelPrecision)
{
    irt::features::ImageSearchConfig config;
    config.model_precision = static_cast<irt::model::ModelPrecision>(99);

    expectIrtExceptionCode([&] { irt::features::ImageSearch search(config); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

TEST(ImageSearchTest, ConstructorRejectsInvalidModelRuntime)
{
    irt::features::ImageSearchConfig config;

    expectIrtExceptionCode(
        [&]
        {
            config.model_runtime = irt::model::ModelRuntime::parse("cuda:-1");
            irt::features::ImageSearch search(config);
        },
        irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief DINO 模型和特征名不再写入默认索引文件名。
 */
TEST(ImageSearchTest, DefaultIndexPathOmitsDinoFeatureNames)
{
    const auto cls_path   = irt::features::ImageSearch::defaultIndexPath("gallery", "dinov3_vitb16", "x_norm_clstoken");
    const auto block_path = irt::features::ImageSearch::defaultIndexPath("gallery", "dinov2_vits14", "blocks.11");

    EXPECT_EQ(cls_path.parent_path().generic_string(), "gallery");
    EXPECT_EQ(block_path.parent_path().generic_string(), "gallery");
    EXPECT_EQ(cls_path.extension().string(), ".faiss");
    EXPECT_EQ(block_path.extension().string(), ".faiss");
    EXPECT_EQ(cls_path.filename().string().find("dinov3"), std::string::npos);
    EXPECT_EQ(block_path.filename().string().find("blocks"), std::string::npos);
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

    auto load_feature = [&](size_t row)
    {
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

    auto                      query = load_feature(0);
    std::vector<float>        distances(3);
    std::vector<faiss::idx_t> labels(3);
    index->search(1, query.data(), static_cast<faiss::idx_t>(labels.size()), distances.data(), labels.data());
    EXPECT_GE(labels[0], 0);
}

/**
 * @brief RAM IVF-PQ build reports vector-add batches through the unified index-build stage.
 */
TEST(ImageSearchTest, RamIndexBuildReportsBatchProgress)
{
    constexpr int    feature_dim  = 16;
    constexpr size_t vector_count = 17;

    std::vector<float> features(vector_count * feature_dim, 0.0f);
    for (size_t row = 0; row < vector_count; ++row)
    {
        features[row * feature_dim + (row % feature_dim)] = 1.0f;
        features[row * feature_dim + ((row * 5 + 3) % feature_dim)] += 0.125f;
    }

    size_t single_loads = 0;
    auto load_feature = [&](size_t row)
    {
        ++single_loads;
        const auto begin = features.begin() + static_cast<std::ptrdiff_t>(row * feature_dim);
        return std::vector<float>(begin, begin + feature_dim);
    };

    std::vector<irt::features::ImageSearchBuildStage> stages;
    std::vector<std::pair<size_t, size_t>>            add_batches;
    size_t                                            build_progress = 0;
    auto progress_callback = [&](const irt::features::ImageSearchBuildProgress &progress)
    {
        stages.push_back(progress.stage);
        if (progress.stage == irt::features::ImageSearchBuildStage::BuildingIndex)
        {
            build_progress = std::max(build_progress, progress.processed_count);
            if (progress.batch_count > 0)
            {
                add_batches.emplace_back(progress.batch_begin, progress.batch_count);
            }
        }
    };

    auto index = irt::features::priv::buildRamIvfPqIndex(vector_count, feature_dim, 5, load_feature, progress_callback);

    ASSERT_TRUE(index);
    EXPECT_NE(std::find(stages.begin(), stages.end(), irt::features::ImageSearchBuildStage::BuildingIndex),
              stages.end());
    const std::vector<std::pair<size_t, size_t>> expected_add_batches{
        { 0, 5},
        { 5, 5},
        {10, 5},
        {15, 2}
    };
    EXPECT_EQ(add_batches, expected_add_batches);
    EXPECT_EQ(build_progress, vector_count);
    EXPECT_EQ(single_loads, vector_count * 2);
}

TEST(ImageSearchTest, FeatureSamplingUsesContiguousBatchCallback)
{
    constexpr int    feature_dim     = 2;
    constexpr size_t vector_count    = 10;
    constexpr size_t training_count  = 10;
    constexpr size_t stride          = 1;
    constexpr size_t training_batch  = 4;
    size_t           single_loads    = 0;
    size_t           indexed_loads   = 0;

    auto load_feature = [&](size_t row)
    {
        ++single_loads;
        return std::vector<float>{static_cast<float>(row), static_cast<float>(row + 1)};
    };

    std::vector<std::pair<size_t, size_t>> loaded_batches;
    auto load_feature_batch = [&](size_t begin, size_t count)
    {
        loaded_batches.emplace_back(begin, count);
        std::vector<float> features;
        features.reserve(count * feature_dim);
        for (size_t row = begin; row < begin + count; ++row)
        {
            features.push_back(static_cast<float>(row));
            features.push_back(static_cast<float>(row + 1));
        }
        return features;
    };

    auto load_feature_index_batch = [&](const std::vector<size_t> &indices)
    {
        ++indexed_loads;
        std::vector<float> features;
        features.reserve(indices.size() * feature_dim);
        for (const auto row : indices)
        {
            features.push_back(static_cast<float>(row));
            features.push_back(static_cast<float>(row + 1));
        }
        return features;
    };


    const auto sample = irt::features::priv::loadTrainingFeatures(
        vector_count, feature_dim, training_count, stride, training_batch, load_feature, load_feature_batch,
        load_feature_index_batch);

    const std::vector<std::pair<size_t, size_t>> expected_batches{
        {0, 4},
        {4, 4},
        {8, 2}
    };
    EXPECT_EQ(loaded_batches, expected_batches);
    EXPECT_EQ(single_loads, 0U);
    EXPECT_EQ(indexed_loads, 0U);
    EXPECT_EQ(sample.count, training_count);
    EXPECT_EQ(sample.indices, (std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST(ImageSearchTest, FeatureSamplingUsesIndexedBatchCallbackForStridedSamples)
{
    constexpr int    feature_dim     = 2;
    constexpr size_t vector_count    = 10;
    constexpr size_t training_count  = 5;
    constexpr size_t stride          = 2;
    constexpr size_t training_batch  = 3;
    size_t           single_loads    = 0;
    size_t           contiguous_loads = 0;

    auto load_feature = [&](size_t row)
    {
        ++single_loads;
        return std::vector<float>{static_cast<float>(row), static_cast<float>(row + 1)};
    };

    auto load_feature_batch = [&](size_t begin, size_t count)
    {
        ++contiguous_loads;
        std::vector<float> features;
        features.reserve(count * feature_dim);
        for (size_t row = begin; row < begin + count; ++row)
        {
            features.push_back(static_cast<float>(row));
            features.push_back(static_cast<float>(row + 1));
        }
        return features;
    };

    std::vector<std::vector<size_t>> loaded_index_batches;
    auto load_feature_index_batch = [&](const std::vector<size_t> &indices)
    {
        loaded_index_batches.push_back(indices);
        std::vector<float> features;
        features.reserve(indices.size() * feature_dim);
        for (const auto row : indices)
        {
            features.push_back(static_cast<float>(row));
            features.push_back(static_cast<float>(row + 1));
        }
        return features;
    };


    const auto sample = irt::features::priv::loadTrainingFeatures(
        vector_count, feature_dim, training_count, stride, training_batch, load_feature, load_feature_batch,
        load_feature_index_batch);

    const std::vector<std::vector<size_t>> expected_index_batches{
        {0, 2, 4},
        {6, 8}
    };
    EXPECT_EQ(loaded_index_batches, expected_index_batches);
    EXPECT_EQ(single_loads, 0U);
    EXPECT_EQ(contiguous_loads, 0U);
    EXPECT_EQ(sample.count, training_count);
    EXPECT_EQ(sample.indices, (std::vector<size_t>{0, 2, 4, 6, 8}));
}

/**
 * @brief GPU-compatible RAM IVF-PQ indexes use fixed 8-bit PQ codes.
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

    auto load_feature = [&](size_t row)
    {
        const auto begin = features.begin() + static_cast<std::ptrdiff_t>(row * feature_dim);
        return std::vector<float>(begin, begin + feature_dim);
    };

    auto index = irt::features::priv::buildRamIvfPqIndex(vector_count, feature_dim, 16, load_feature, {}, true);
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
    constexpr int    feature_dim  = 4;
    constexpr size_t vector_count = 16;

    std::vector<float> features(vector_count * feature_dim, 0.0f);
    for (size_t row = 0; row < vector_count; ++row)
    {
        features[row * feature_dim + (row % feature_dim)]       = 1.0f;
        features[row * feature_dim + ((row + 1) % feature_dim)] = 0.01f * static_cast<float>(row + 1);
    }

    auto load_feature = [&](size_t row)
    {
        const auto begin = features.begin() + static_cast<std::ptrdiff_t>(row * feature_dim);
        return std::vector<float>(begin, begin + feature_dim);
    };

    TempDir                                temp;
    const auto                             index_path = temp.path() / "synthetic_disk.faiss";
    std::vector<std::pair<size_t, size_t>> batches;
    auto                                   index = irt::features::priv::buildCpuOnDiskIvfFlatIndex(
        vector_count, feature_dim, index_path, 3, load_feature,
        [&](const irt::features::ImageSearchBuildProgress &progress)
        {
            if (progress.stage == irt::features::ImageSearchBuildStage::BuildingIndex
                && progress.batch_count > 0 && progress.processed_count > vector_count)
            {
                batches.emplace_back(progress.batch_begin, progress.batch_count);
            }
        });
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
    const std::vector<std::pair<size_t, size_t>> expected_batches{
        { 0, 3},
        { 3, 3},
        { 6, 3},
        { 9, 3},
        {12, 3},
        {15, 1}
    };
    EXPECT_EQ(batches, expected_batches);

    const std::string invlists_type = typeid(*ivf_index->invlists).name();
    EXPECT_NE(invlists_type.find("OnDisk"), std::string::npos) << invlists_type;

    auto                      query = load_feature(0);
    std::vector<float>        distances(3);
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
 * @brief 高维 DINO 特征下，IVF 训练/Faiss 建库批大小应受内存上限约束。
 *
 * 验证 ``chooseCpuOnDiskIvf*`` 在约 1369×384 维、12500 条向量规模下不会超出
 * ``kCpuOnDiskIvfMaxTrainingBytes`` 与 ``kFaissIndexBuildMaxBatchBytes``。
 */
TEST(ImageSearchTest, CpuDiskIndexBoundsWideFeatureBuffers)
{
    constexpr size_t vector_count = 12500;
    constexpr int    feature_dim  = 1369 * 384;

    const auto nlist          = irt::features::priv::chooseCpuOnDiskIvfListCount(vector_count, feature_dim);
    const auto training_count = irt::features::priv::chooseCpuOnDiskIvfTrainingCount(vector_count, feature_dim, nlist);
    const auto batch_size     = irt::features::priv::chooseFaissIndexBuildBatchSize(256, vector_count, feature_dim);

    const auto bytes_per_feature = static_cast<size_t>(feature_dim) * sizeof(float);

    EXPECT_GE(nlist, 1U);
    EXPECT_LE(nlist, training_count);
    EXPECT_LT(training_count, irt::features::priv::kCpuOnDiskIvfMaxTrainingVectors);
    EXPECT_LE(training_count * bytes_per_feature, irt::features::priv::kCpuOnDiskIvfMaxTrainingBytes);
    EXPECT_LT(batch_size, 256U);
    EXPECT_LE(batch_size * bytes_per_feature, irt::features::priv::kFaissIndexBuildMaxBatchBytes);
}

TEST(ImageSearchTest, FeatureBatchSizeOverflowIsRejectedBeforeCallback)
{
    bool callback_called = false;
    const auto load = [&](size_t, size_t)
    {
        callback_called = true;
        return std::vector<float>{};
    };

    EXPECT_THROW(irt::features::priv::loadFeatureBatch(0, std::numeric_limits<size_t>::max(), 2, load),
                 irt::Exception);
    EXPECT_FALSE(callback_called);
}

TEST(ImageSearchTest, CpuDiskLayoutOverflowIsRejected)
{
    const std::vector<uint64_t> list_sizes{std::numeric_limits<uint64_t>::max()};
    EXPECT_THROW(irt::features::priv::makeCpuOnDiskIvfListMeta(list_sizes, sizeof(float)), irt::Exception);
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
        const size_t first                  = row % feature_dim;
        const size_t second                 = (row * 37 + 11) % feature_dim;
        features[row * feature_dim + first] = 1.0f;
        features[row * feature_dim + second] += 0.25f;
    }

    auto load_feature = [&](size_t row)
    {
        const auto begin = features.begin() + static_cast<std::ptrdiff_t>(row * feature_dim);
        return std::vector<float>(begin, begin + feature_dim);
    };

    TempDir    temp;
    const auto index_path = temp.path() / "dino_cls_sized_disk.faiss";
    auto       index
        = irt::features::priv::buildCpuOnDiskIvfFlatIndex(vector_count, feature_dim, index_path, 256, load_feature);

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
 * @brief 加载已有索引时必须显式指定 Faiss 索引路径。
 */
TEST(ImageSearchTest, LoadRequiresExplicitIndexFile)
{
    TempDir temp;

    irt::features::ImageSearch  search;

    expectIrtExceptionCode([&] { search.load("weights.wts", {}); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 显式图片路径列表构建时，空列表应在加载权重前被拒绝。
 */
TEST(ImageSearchTest, BuildFromExplicitImagePathsRejectsEmptyList)
{
    TempDir temp;

    irt::features::ImageSearch  search;
    const std::vector<irt::features::ImageSearchItem> images;

    expectIrtExceptionCode([&] { search.build("weights.wts", images, temp.path() / "index.faiss"); },
                           irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 未注册的模型名应在 ImageSearch 构造时被拒绝。
 */
TEST(ImageSearchTest, ConstructorRejectsUnsupportedModel)
{
    irt::features::ImageSearchConfig config;
    config.model_name   = "not_a_model";
    config.feature_name = "layer4";

    expectIrtExceptionCode([&] { irt::features::ImageSearch search(config); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 空特征名应在构造 ImageSearch 时被拒绝。
 */
TEST(ImageSearchTest, ConstructorRejectsEmptyFeatureName)
{
    irt::features::ImageSearchConfig config;
    config.model_name   = "resnet18";
    config.feature_name = "";

    expectIrtExceptionCode([&] { irt::features::ImageSearch search(config); }, irt::Status::ERROR_INVALID_ARGUMENT);
}

/**
 * @brief 未 buildOrLoad 的检索器调用 search 时应返回未就绪错误。
 */
TEST(ImageSearchTest, SearchBeforeBuildOrLoadThrowsInvalidOperation)
{
    irt::features::ImageSearchConfig config;
    config.model_name   = "resnet18";
    config.feature_name = "layer4";
    irt::features::ImageSearch search(config);

    expectIrtExceptionCode([&] { search.search("query.jpg"); }, irt::Status::INVALID_OPERATION);

    config.index_storage = irt::features::ImageSearchIndexStorage::Disk;
    irt::features::ImageSearch disk_search(config);

    expectIrtExceptionCode([&] { disk_search.search("query.jpg"); }, irt::Status::INVALID_OPERATION);
}

/**
 * @brief 移动构造后的检索器应保留配置字段，便于作为可移动资源返回。
 */
TEST(ImageSearchTest, MoveConstructedSearcherKeepsConfiguration)
{
    irt::features::ImageSearchConfig config;
    config.model_name   = "resnet18";
    config.feature_name = "layer4";
    irt::features::ImageSearch source(config);
    irt::features::ImageSearch moved(std::move(source));

    EXPECT_EQ(moved.config().model_name, "resnet18");
    EXPECT_EQ(moved.config().feature_name, "layer4");
    EXPECT_FALSE(moved.isReady());
}
