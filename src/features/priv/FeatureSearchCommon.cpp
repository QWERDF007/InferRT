/**
 * @file FeatureSearchCommon.cpp
 * @brief 图像级检索与 ROI 检索共用工具实现。
 */

#include "FeatureSearchCommon.hpp"

#include <cuda_runtime_api.h>

#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/gpu/GpuCloner.h>
#include <faiss/index_io.h>
#pragma warning(pop)
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace irt::features::priv {
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
 * @brief 获取当前 CUDA 设备编号，供 GPU Faiss 使用。
 */
int currentCudaDevice()
{
    int device{0};
    irt::model::checkCuda(cudaGetDevice(&device), "cudaGetDevice(Faiss GPU backend)");
    return device;
}

} // namespace

bool usesTensorRtModelBackend(const ImageSearchConfig &config) noexcept
{
    return config.model_backend == irt::model::ModelBackend::TensorRT;
}

bool useCpuDiskIndex(const ImageSearchConfig &config) noexcept
{
    return config.faiss_backend == ImageSearchFaissBackend::CPU
        && config.index_storage == ImageSearchIndexStorage::Disk;
}

void validateFeatureSearchConfig(const ImageSearchConfig &config, const char *owner_name)
{
    const char *owner = owner_name == nullptr ? "FeatureSearch" : owner_name;

    switch (config.model_backend)
    {
    case irt::model::ModelBackend::TensorRT:
    case irt::model::ModelBackend::OpenVINO:
    case irt::model::ModelBackend::ONNXRuntime:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported %s model backend", owner);
    }

    switch (config.model_device)
    {
    case irt::model::ModelDevice::CPU:
    case irt::model::ModelDevice::GPU:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported %s model device", owner);
    }

    if (usesTensorRtModelBackend(config) && config.model_device == irt::model::ModelDevice::CPU)
    {
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "%s TensorRT backend requires GPU device", owner);
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

    if (config.disk_build_batch_size == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s disk build batch size must be positive", owner);
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

std::filesystem::path mappingPathFromIndex(const std::filesystem::path &index_path)
{
    return index_path.string() + ".paths.txt";
}

std::filesystem::path metadataPathFromIndex(const std::filesystem::path &index_path)
{
    return index_path.string() + ".meta.txt";
}

FaissIndexBundle moveCpuIndexToConfiguredBackend(std::unique_ptr<faiss::Index> cpu_index,
                                                 ImageSearchFaissBackend       backend)
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
        bundle.gpu_resources = std::make_unique<faiss::gpu::StandardGpuResources>();
        faiss::gpu::GpuClonerOptions options;
        options.useFloat16 = true;
        bundle.index.reset(
            faiss::gpu::index_cpu_to_gpu(bundle.gpu_resources.get(), currentCudaDevice(), cpu_index.get(), &options));
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
        bundle.index = buildCpuOnDiskIvfFlatIndex(vector_count, feature_dim, index_path, config.disk_build_batch_size,
                                                  config.model_batch_size, load_feature, load_feature_batch,
                                                  load_feature_index_batch, progress_callback);
        return bundle;
    }

    auto cpu_index
        = buildRamIvfPqIndex(vector_count, feature_dim, config.disk_build_batch_size, config.model_batch_size,
                             load_feature, load_feature_batch, load_feature_index_batch, progress_callback,
                             config.faiss_backend == ImageSearchFaissBackend::GPU);

    reportBuildProgress(progress_callback, ImageSearchBuildStage::WritingIndex, 0, 0, 0, 0, 1);
    faiss::write_index(cpu_index.get(), index_path.string().c_str());
    reportBuildProgress(progress_callback, ImageSearchBuildStage::WritingIndex, 0, 0, 0, 1, 1);

    reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 0, 1);
    auto bundle = moveCpuIndexToConfiguredBackend(std::move(cpu_index), config.faiss_backend);
    reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 1, 1);
    return bundle;
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

    return moveCpuIndexToConfiguredBackend(std::move(cpu_index), config.faiss_backend);
}

} // namespace irt::features::priv
