/**
 * @file FeatureSearchCommon.cpp
 * @brief 图像级检索与 ROI 检索共用工具实现。
 */

#include "FeatureSearchCommon.hpp"

#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/gpu/GpuCloner.h>
#include <faiss/index_io.h>
#pragma warning(pop)
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <system_error>

namespace irt::features::priv {
namespace fs = std::filesystem;
namespace {

/**
 * @brief 对特征向量做 L2 归一化。
 */
void l2Normalize(float *values, size_t count)
{
    float sum_sq = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        sum_sq += values[i] * values[i];
    }
    if (sum_sq <= 0.0f)
    {
        return;
    }

    const float inv_norm = 1.0f / std::sqrt(sum_sq);
    for (size_t i = 0; i < count; ++i)
    {
        values[i] *= inv_norm;
    }
}

/**
 * @brief 对特征向量做 L1 归一化。
 */
void l1Normalize(float *values, size_t count)
{
    float sum_abs = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        sum_abs += std::abs(values[i]);
    }
    if (sum_abs <= 0.0f)
    {
        return;
    }

    const float inv_norm = 1.0f / sum_abs;
    for (size_t i = 0; i < count; ++i)
    {
        values[i] *= inv_norm;
    }
}

/**
 * @brief 为小图库构建精确的内积索引。
 *
 * IVF-PQ 需要足够多的训练样本；在只有少量图片时，聚类既没有收益，
 * 还会触发 Faiss 的小样本训练边界。IndexFlatIP 对小图库更快且结果精确。
 */
std::unique_ptr<faiss::Index> buildRamFlatIndex(
    size_t vector_count, int feature_dim, size_t requested_batch_size,
    const LoadFeatureCallback &load_feature, const LoadFeatureBatchCallback &load_feature_batch,
    const ImageSearchBuildProgressCallback &progress_callback)
{
    if (vector_count == 0 || feature_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Flat Faiss index requires non-empty features");
    }

    auto index = std::make_unique<faiss::IndexFlatIP>(feature_dim);
    const auto batch_size = chooseFaissIndexBuildBatchSize(requested_batch_size, vector_count, feature_dim);
    size_t     batch_index = 0;
    for (size_t begin = 0; begin < vector_count; begin += batch_size)
    {
        const size_t count = std::min(batch_size, vector_count - begin);
        const auto features
            = load_feature_batch ? loadFeatureBatch(begin, count, feature_dim, load_feature_batch)
                                  : loadFeatureBatch(begin, count, feature_dim, load_feature);
        index->add(static_cast<faiss::idx_t>(count), features.data());
        reportBuildProgress(progress_callback, ImageSearchBuildStage::BuildingIndex, batch_index++, begin, count,
                            begin + count, vector_count);
    }
    return index;
}

} // namespace

void processFeatureBatches(size_t item_count, size_t batch_size, int feature_dim,
                           const FeatureBatchLoader &loader, const FeatureBatchConsumer &consumer,
                           const FeatureBatchProgressCallback &progress_callback)
{
    if (batch_size == 0 || feature_dim <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature batch configuration is invalid");
    }

    size_t batch_index{0};
    for (size_t begin = 0; begin < item_count; begin += batch_size, ++batch_index)
    {
        const size_t count    = std::min(batch_size, item_count - begin);
        const auto   features = loader(begin, count);
        if (features.size() != count * static_cast<size_t>(feature_dim))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature batch size mismatch");
        }

        if (consumer)
        {
            consumer(begin, count, features);
        }
        if (progress_callback)
        {
            progress_callback(FeatureBatchProgress{batch_index, begin, count, begin + count, item_count});
        }
    }
}

std::filesystem::path normalizeImageFilePath(const std::filesystem::path &image_path, const char *owner_name)
{
    const std::string owner = owner_name == nullptr ? "FeatureSearch" : owner_name;
    if (image_path.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s image path must not be empty", owner.c_str());
    }

    std::error_code ec;
    if (!fs::is_regular_file(image_path, ec))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s image path does not exist: %s", owner.c_str(),
                             image_path.string().c_str());
    }
    if (!ImageSearch::isImageFile(image_path))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported %s image file: %s", owner.c_str(),
                             image_path.string().c_str());
    }
    return fs::absolute(image_path);
}

std::filesystem::path featureStorePath(const std::filesystem::path &index_path)
{
    return index_path.string() + ".features.tmp";
}

FeatureStore::FeatureStore(std::filesystem::path path, size_t item_count, int feature_dim)
    : path_(std::move(path))
    , item_count_(item_count)
    , feature_dim_(feature_dim)
{
    if (path_.empty() || item_count_ == 0 || feature_dim_ <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature store configuration is invalid");
    }
    if (!path_.parent_path().empty())
    {
        fs::create_directories(path_.parent_path());
    }

    std::error_code ec;
    fs::remove(path_, ec);
    if (ec)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to prepare feature store: %s",
                             path_.string().c_str());
    }

    output_.open(path_, std::ios::binary | std::ios::trunc);
    if (!output_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create feature store: %s",
                             path_.string().c_str());
    }
    const auto byte_count = item_count_ * static_cast<size_t>(feature_dim_) * sizeof(float);
    if (byte_count > 0)
    {
        output_.seekp(static_cast<std::streamoff>(byte_count - 1), std::ios::beg);
        output_.put('\0');
        output_.seekp(0, std::ios::beg);
    }
    written_.assign(item_count_, 0);
}

FeatureStore::~FeatureStore()
{
    if (output_.is_open())
    {
        output_.close();
    }
    std::error_code ec;
    fs::remove(path_, ec);
}

void FeatureStore::validateRange(size_t begin, size_t count) const
{
    if (begin > item_count_ || count > item_count_ - begin)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature store range is invalid");
    }
}

void FeatureStore::writeBatch(size_t begin, size_t count, const std::vector<float> &features)
{
    if (writing_finished_ || !output_ || begin != next_write_index_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Feature store is not writable");
    }
    writeBatchAt(begin, count, features);
    next_write_index_ += count;
}

void FeatureStore::writeBatchAt(size_t begin, size_t count, const std::vector<float> &features)
{
    if (writing_finished_ || !output_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Feature store is not writable");
    }
    validateRange(begin, count);
    if (features.size() != count * static_cast<size_t>(feature_dim_))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature store batch size mismatch");
    }

    for (size_t index = begin; index < begin + count; ++index)
    {
        if (written_[index] != 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Feature store range was already written");
        }
    }

    const auto offset = static_cast<std::streamoff>(begin * static_cast<size_t>(feature_dim_) * sizeof(float));
    output_.seekp(offset, std::ios::beg);
    output_.write(reinterpret_cast<const char *>(features.data()),
                  static_cast<std::streamsize>(features.size() * sizeof(float)));
    if (!output_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write feature store: %s",
                             path_.string().c_str());
    }
    for (size_t index = begin; index < begin + count; ++index)
    {
        written_[index] = 1;
    }
    written_count_ += count;
}

void FeatureStore::finishWriting()
{
    if (writing_finished_ || written_count_ != item_count_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Feature store is incomplete");
    }
    output_.close();
    writing_finished_ = true;
}

std::vector<float> FeatureStore::read(size_t index) const
{
    return readBatch(index, 1);
}

std::vector<float> FeatureStore::readBatch(size_t begin, size_t count) const
{
    validateRange(begin, count);
    if (!writing_finished_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Feature store is not ready for reading");
    }

    std::ifstream input(path_, std::ios::binary);
    if (!input)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open feature store: %s",
                             path_.string().c_str());
    }

    const auto offset = static_cast<std::streamoff>(begin * static_cast<size_t>(feature_dim_) * sizeof(float));
    input.seekg(offset, std::ios::beg);
    std::vector<float> features(count * static_cast<size_t>(feature_dim_));
    input.read(reinterpret_cast<char *>(features.data()), static_cast<std::streamsize>(features.size() * sizeof(float)));
    if (!input)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to read feature store: %s",
                             path_.string().c_str());
    }
    return features;
}

std::vector<float> FeatureStore::readBatch(const std::vector<size_t> &indices) const
{
    if (!writing_finished_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Feature store is not ready for reading");
    }

    std::ifstream input(path_, std::ios::binary);
    if (!input)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open feature store: %s",
                             path_.string().c_str());
    }

    const auto feature_size = static_cast<size_t>(feature_dim_);
    std::vector<float> features;
    features.reserve(indices.size() * feature_size);
    std::vector<float> batch(feature_size);
    for (const size_t index : indices)
    {
        validateRange(index, 1);
        const auto offset = static_cast<std::streamoff>(index * feature_size * sizeof(float));
        input.seekg(offset, std::ios::beg);
        input.read(reinterpret_cast<char *>(batch.data()), static_cast<std::streamsize>(batch.size() * sizeof(float)));
        if (!input)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to read feature store: %s",
                                 path_.string().c_str());
        }
        features.insert(features.end(), batch.begin(), batch.end());
    }
    return features;
}

bool usesTensorRtModelBackend(const ImageSearchConfig &config) noexcept
{
    return config.model_runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT;
}

bool useCpuDiskIndex(const ImageSearchConfig &config) noexcept
{
    return config.faiss_backend == ImageSearchFaissBackend::CPU
        && config.index_storage == ImageSearchIndexStorage::Disk;
}

void validateFeatureSearchConfig(const ImageSearchConfig &config, const char *owner_name)
{
    const char *owner = owner_name == nullptr ? "FeatureSearch" : owner_name;

    config.model_runtime.validate();

    switch (config.model_precision)
    {
    case irt::model::ModelPrecision::FP32:
    case irt::model::ModelPrecision::FP16:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported %s model precision", owner);
    }

    switch (config.preprocess_backend)
    {
    case ImageSearchPreprocessBackend::CPU:
        break;
    case ImageSearchPreprocessBackend::GPU:
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "%s GPU preprocessing is not implemented", owner);
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported %s preprocessing backend", owner);
    }

    switch (config.norm)
    {
    case ImageSearchFeatureNorm::None:
    case ImageSearchFeatureNorm::L1:
    case ImageSearchFeatureNorm::L2:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported %s feature norm", owner);
    }

    switch (config.faiss_backend)
    {
    case ImageSearchFaissBackend::CPU:
    case ImageSearchFaissBackend::GPU:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported %s Faiss backend", owner);
    }

    switch (config.index_storage)
    {
    case ImageSearchIndexStorage::RAM:
    case ImageSearchIndexStorage::Disk:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported %s index storage", owner);
    }

    if (config.model_batch_size == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s model batch size must be positive", owner);
    }
    if (config.model_batch_size > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s model batch size is too large", owner);
    }
}

const char *preprocessBackendName(ImageSearchPreprocessBackend backend) noexcept
{
    switch (backend)
    {
    case ImageSearchPreprocessBackend::CPU:
        return "cpu";
    case ImageSearchPreprocessBackend::GPU:
        return "gpu";
    }
    return "unknown";
}

const char *featureNormName(ImageSearchFeatureNorm norm) noexcept
{
    switch (norm)
    {
    case ImageSearchFeatureNorm::None:
        return "none";
    case ImageSearchFeatureNorm::L1:
        return "l1";
    case ImageSearchFeatureNorm::L2:
        return "l2";
    }
    return "unknown";
}

const char *faissBackendName(ImageSearchFaissBackend backend) noexcept
{
    switch (backend)
    {
    case ImageSearchFaissBackend::CPU:
        return "cpu";
    case ImageSearchFaissBackend::GPU:
        return "gpu";
    }
    return "unknown";
}

const char *indexStorageName(ImageSearchIndexStorage storage) noexcept
{
    switch (storage)
    {
    case ImageSearchIndexStorage::RAM:
        return "ram";
    case ImageSearchIndexStorage::Disk:
        return "disk";
    }
    return "unknown";
}

const char *indexKindName(const ImageSearchConfig &config) noexcept
{
    return useCpuDiskIndex(config) ? "ivf_flat_ondisk" : "ivf_pq_ram";
}

void normalizeFeature(float *values, size_t count, ImageSearchFeatureNorm norm)
{
    switch (norm)
    {
    case ImageSearchFeatureNorm::None:
        return;
    case ImageSearchFeatureNorm::L1:
        l1Normalize(values, count);
        return;
    case ImageSearchFeatureNorm::L2:
        l2Normalize(values, count);
        return;
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported feature norm");
}

void normalizeFeature(std::vector<float> &values, ImageSearchFeatureNorm norm)
{
    normalizeFeature(values.data(), values.size(), norm);
}

FaissIndexBundle moveCpuIndexToConfiguredBackend(std::unique_ptr<faiss::Index> cpu_index,
                                                 ImageSearchFaissBackend       backend, int device_id)
{
    if (!cpu_index)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Cannot move empty Faiss index");
    }

    FaissIndexBundle bundle;
    switch (backend)
    {
    case ImageSearchFaissBackend::CPU:
        bundle.index = std::move(cpu_index);
        break;
    case ImageSearchFaissBackend::GPU:
    {
        irt::model::setCudaDevice(device_id);
        bundle.gpu_resources = std::make_unique<faiss::gpu::StandardGpuResources>();
        faiss::gpu::GpuClonerOptions options;
        options.useFloat16 = true;
        bundle.index.reset(
            faiss::gpu::index_cpu_to_gpu(bundle.gpu_resources.get(), device_id, cpu_index.get(), &options));
        if (!bundle.index)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to clone Faiss index to GPU");
        }
        break;
    }
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported Faiss backend");
    }

    return bundle;
}

FaissIndexBundle buildConfiguredFaissIndex(size_t vector_count, int feature_dim,
                                           const std::filesystem::path &index_path, const ImageSearchConfig &config,
                                           const LoadFeatureCallback              &load_feature,
                                           const LoadFeatureBatchCallback         &load_feature_batch,
                                           const LoadFeatureIndexedBatchCallback  &load_feature_index_batch,
                                           const ImageSearchBuildProgressCallback &progress_callback)
{
    if (useCpuDiskIndex(config))
    {
        FaissIndexBundle bundle;
        bundle.index
            = buildCpuOnDiskIvfFlatIndex(vector_count, feature_dim, index_path, config.model_batch_size, load_feature,
                                         load_feature_batch, load_feature_index_batch, progress_callback);
        return bundle;
    }

    // IVF-PQ is not meaningful for a tiny gallery and its training code requires
    // substantially more samples than are available here. Keep the exact flat
    // index in that case; it is also a safer default for sample-sized galleries.
    constexpr size_t kExactFlatIndexMaxVectors = 256;
    auto             cpu_index
        = vector_count <= kExactFlatIndexMaxVectors
              ? buildRamFlatIndex(vector_count, feature_dim, config.model_batch_size, load_feature,
                                  load_feature_batch, progress_callback)
              : buildRamIvfPqIndex(vector_count, feature_dim, config.model_batch_size, load_feature,
                                   load_feature_batch, load_feature_index_batch, progress_callback,
                                   config.faiss_backend == ImageSearchFaissBackend::GPU);

    faiss::write_index(cpu_index.get(), index_path.string().c_str());
    return moveCpuIndexToConfiguredBackend(std::move(cpu_index), config.faiss_backend, config.model_runtime.deviceId());
}

FaissIndexBundle loadConfiguredFaissIndex(const std::filesystem::path &index_path, const ImageSearchConfig &config)
{
    auto cpu_index = useCpuDiskIndex(config)
                       ? loadCpuOnDiskIvfFlatIndex(index_path)
                       : std::unique_ptr<faiss::Index>(faiss::read_index(index_path.string().c_str()));
    if (!cpu_index)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load Faiss index: %s",
                             index_path.string().c_str());
    }

    if (useCpuDiskIndex(config))
    {
        FaissIndexBundle bundle;
        bundle.index = std::move(cpu_index);
        return bundle;
    }

    return moveCpuIndexToConfiguredBackend(std::move(cpu_index), config.faiss_backend,
                                           config.model_runtime.deviceId());
}

} // namespace irt::features::priv
