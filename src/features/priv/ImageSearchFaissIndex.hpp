#pragma once

/**
 * @file ImageSearchFaissIndex.hpp
 * @brief ImageSearch 的 Faiss 索引构建接口。
 *
 * 索引格式、磁盘映射和训练实现位于同名私有 `.cpp`；调用方只依赖
 * 构建回调、进度回调和索引生命周期，不承担 Faiss 文件布局细节。
 */

#include <inferrt/features/ImageSearch.hpp>

#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/Index.h>
#pragma warning(pop)

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace irt::features::priv {

using BuildProgressCallback = ImageSearchBuildProgressCallback;
using LoadFeatureCallback = std::function<std::vector<float>(size_t)>;
using LoadFeatureBatchCallback = std::function<std::vector<float>(size_t, size_t)>;
using LoadFeatureIndexedBatchCallback = std::function<std::vector<float>(const std::vector<size_t> &)>;

inline constexpr size_t kCpuOnDiskIvfMaxTrainingVectors = 8192;
inline constexpr size_t kCpuOnDiskIvfMaxTrainingBytes = 512ULL * 1024ULL * 1024ULL;
inline constexpr size_t kFaissIndexBuildMaxBatchBytes = 256ULL * 1024ULL * 1024ULL;

struct TrainingFeatureSample
{
    std::vector<float> features;
    std::vector<size_t> indices;
    size_t count{0};
};

struct CpuOnDiskIvfListMeta
{
    uint64_t size{0};
    uint64_t ids_offset{0};
    uint64_t codes_offset{0};
};

/** Report one batch or stage transition during index construction. */
INFERRT_FEATURES_API void reportBuildProgress(const BuildProgressCallback &progress_callback, ImageSearchBuildStage stage,
                                              size_t batch_index = 0, size_t batch_begin = 0, size_t batch_count = 0,
                                              size_t processed_count = 0, size_t total_count = 0);

INFERRT_FEATURES_API std::vector<float> loadFeatureBatch(size_t begin, size_t count, int feature_dim,
                                                         const LoadFeatureCallback &load_feature);
INFERRT_FEATURES_API std::vector<float> loadFeatureBatch(size_t begin, size_t count, int feature_dim,
                                                         const LoadFeatureBatchCallback &load_feature_batch);

INFERRT_FEATURES_API TrainingFeatureSample loadTrainingFeatures(
    size_t vector_count, int feature_dim, size_t training_count, size_t stride, size_t batch_size,
    const LoadFeatureCallback &load_feature, const LoadFeatureBatchCallback &load_feature_batch,
    const LoadFeatureIndexedBatchCallback &load_feature_index_batch);

INFERRT_FEATURES_API std::vector<CpuOnDiskIvfListMeta> makeCpuOnDiskIvfListMeta(
    const std::vector<uint64_t> &list_sizes, size_t code_size);

/** Return the sidecar path used by a CPU on-disk IVF index. */
INFERRT_FEATURES_API std::filesystem::path cpuOnDiskIvfDataPath(const std::filesystem::path &index_path);

/** Select a bounded IVF list count from gallery size. */
INFERRT_FEATURES_API size_t chooseCpuOnDiskIvfListCount(size_t vector_count);

/** Select a bounded IVF list count from gallery size and feature memory budget. */
INFERRT_FEATURES_API size_t chooseCpuOnDiskIvfListCount(size_t vector_count, int feature_dim);

/** Select the number of vectors used to train a CPU on-disk IVF index. */
INFERRT_FEATURES_API size_t chooseCpuOnDiskIvfTrainingCount(size_t vector_count, int feature_dim, size_t nlist);

/** Limit an index build batch to the configured memory budget. */
INFERRT_FEATURES_API size_t chooseFaissIndexBuildBatchSize(size_t requested_batch_size, size_t vector_count,
                                                          int feature_dim);

/** Load an on-disk IVF+Flat index and attach its memory-mapped sidecar. */
INFERRT_FEATURES_API std::unique_ptr<faiss::Index> loadCpuOnDiskIvfFlatIndex(
    const std::filesystem::path &index_path);

/** Build an exact RAM index for galleries too small to benefit from IVF-PQ. */
INFERRT_FEATURES_API std::unique_ptr<faiss::Index> buildRamFlatIndex(
    size_t vector_count, int feature_dim, size_t batch_size, const LoadFeatureCallback &load_feature,
    const LoadFeatureBatchCallback &load_feature_batch, const BuildProgressCallback &progress_callback = {});

INFERRT_FEATURES_API std::unique_ptr<faiss::Index> buildCpuOnDiskIvfFlatIndex(
    size_t vector_count, int feature_dim, const std::filesystem::path &index_path, size_t batch_size,
    const LoadFeatureCallback &load_feature, const BuildProgressCallback &progress_callback = {});

INFERRT_FEATURES_API std::unique_ptr<faiss::Index> buildCpuOnDiskIvfFlatIndex(
    size_t vector_count, int feature_dim, const std::filesystem::path &index_path, size_t batch_size,
    const LoadFeatureCallback &load_feature, const LoadFeatureBatchCallback &load_feature_batch,
    const BuildProgressCallback &progress_callback = {});

INFERRT_FEATURES_API std::unique_ptr<faiss::Index> buildCpuOnDiskIvfFlatIndex(
    size_t vector_count, int feature_dim, const std::filesystem::path &index_path, size_t batch_size,
    const LoadFeatureCallback &load_feature, const LoadFeatureBatchCallback &load_feature_batch,
    const LoadFeatureIndexedBatchCallback &load_feature_index_batch,
    const BuildProgressCallback &progress_callback);

INFERRT_FEATURES_API std::unique_ptr<faiss::Index> buildRamIvfPqIndex(
    size_t vector_count, int feature_dim, size_t batch_size, const LoadFeatureCallback &load_feature,
    const BuildProgressCallback &progress_callback = {}, bool require_gpu_compatible = false);

INFERRT_FEATURES_API std::unique_ptr<faiss::Index> buildRamIvfPqIndex(
    size_t vector_count, int feature_dim, size_t batch_size, const LoadFeatureCallback &load_feature,
    const LoadFeatureBatchCallback &load_feature_batch, const BuildProgressCallback &progress_callback,
    bool require_gpu_compatible);

INFERRT_FEATURES_API std::unique_ptr<faiss::Index> buildRamIvfPqIndex(
    size_t vector_count, int feature_dim, size_t batch_size, const LoadFeatureCallback &load_feature,
    const LoadFeatureBatchCallback &load_feature_batch,
    const LoadFeatureIndexedBatchCallback &load_feature_index_batch,
    const BuildProgressCallback &progress_callback = {}, bool require_gpu_compatible = false);

} // namespace irt::features::priv
