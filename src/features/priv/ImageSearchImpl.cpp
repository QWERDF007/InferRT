/**
 * @file ImageSearchImpl.cpp
 * @brief 图像检索 PIMPL：特征提取、Faiss 索引构建/加载与 Top-K 搜索。
 */

#include "ImageSearchImpl.hpp"

#include "FeatureSearchCommon.hpp"
#include "ImageFeatureExtractor.hpp"
#include "ImageSearchFaissIndex.hpp"

#include <cuda_runtime_api.h>

#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>
#pragma warning(pop)
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <inferrt/util/FileManifest.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace irt::features {

namespace {

using DeviceBuffer = irt::model::DeviceBuffer;
using irt::model::checkCuda;
using irt::model::dimsToCsv;
using irt::model::elementCount;

/**
 * @brief Faiss 索引及其关联资源的打包结构。
 *
 * 构建或加载索引时的中间返回值，便于一并转移 GPU 资源、索引实例与 ID 映射。
 */
struct FaissIndexBundle
{
    std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_resources; ///< GPU Faiss 资源（可选）。
    std::unique_ptr<faiss::Index>                     index;         ///< Faiss 内积索引。
    std::vector<int64_t>                              image_ids;     ///< 与索引向量一一对应的图像 ID。
};

/**
 * @brief 对特征向量做 L2 归一化（原地修改）。
 * @param values 特征分量数组。
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
 * @brief 对特征向量做 L1 归一化（原地修改）。
 * @param values 特征分量数组。
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
 * @brief 按配置对特征向量做归一化。
 * @param values 特征分量数组（原地修改）。
 * @param norm 归一化方式。
 */
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

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ImageSearch feature norm");
}

/**
 * @brief 对整条特征向量做原地归一化。
 * @param values 待归一化的特征向量。
 * @param norm 归一化策略。
 */
void normalizeFeature(std::vector<float> &values, ImageSearchFeatureNorm norm)
{
    normalizeFeature(values.data(), values.size(), norm);
}

/** 
 * @brief 将预处理后端枚举序列化为元数据文件中的字符串。 
 */
const char *preprocessBackendName(ImageSearchPreprocessBackend backend)
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

/**
 * @brief 判断当前特征提取模型是否走 TensorRT 后端。
 * @param config 图像检索配置。
 * @return 使用 TensorRT 时返回 true。
 */
bool usesTensorRtModelBackend(const ImageSearchConfig &config)
{
    return config.model_backend == irt::model::ModelBackend::TensorRT;
}

/**
 * @brief 将特征归一化枚举序列化为元数据文件中的字符串。
 **/
const char *featureNormName(ImageSearchFeatureNorm norm)
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

/**
 * @brief 将 Faiss 后端枚举序列化为元数据文件中的字符串。 
 **/
const char *faissBackendName(ImageSearchFaissBackend backend)
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

/** 
 * @brief 将索引存储位置枚举序列化为元数据文件中的字符串。 
 **/
const char *indexStorageName(ImageSearchIndexStorage storage)
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

/**
 * @brief 判断是否使用 CPU 磁盘 IVF 索引模式。
 * @param config 检索配置。
 * @return CPU 后端且 ``index_storage`` 为 Disk 时返回 true。
 */
bool useCpuDiskIndex(const ImageSearchConfig &config)
{
    return config.faiss_backend == ImageSearchFaissBackend::CPU
        && config.index_storage == ImageSearchIndexStorage::Disk;
}

/** @brief 返回写入元数据的索引类型标识。 */
const char *indexKindName(const ImageSearchConfig &config)
{
    return useCpuDiskIndex(config) ? "ivf_flat_ondisk" : "ivf_pq_ram";
}

/**
 * @brief 校验检索配置中的枚举与批大小是否合法。
 * @param config 待校验配置。
 * @throws irt::Exception 含未实现选项或非法值时抛出。
 */
void validateConfig(const ImageSearchConfig &config)
{
    switch (config.model_backend)
    {
    case irt::model::ModelBackend::TensorRT:
    case irt::model::ModelBackend::OpenVINO:
    case irt::model::ModelBackend::ONNXRuntime:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ImageSearch model backend");
    }

    switch (config.model_device)
    {
    case irt::model::ModelDevice::CPU:
    case irt::model::ModelDevice::GPU:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ImageSearch model device");
    }

    if (usesTensorRtModelBackend(config) && config.model_device == irt::model::ModelDevice::CPU)
    {
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "ImageSearch TensorRT backend requires GPU device");
    }

    switch (config.preprocess_backend)
    {
    case ImageSearchPreprocessBackend::CPU:
        break;
    case ImageSearchPreprocessBackend::GPU:
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "ImageSearch GPU preprocessing is not implemented");
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ImageSearch preprocessing backend");
    }

    switch (config.norm)
    {
    case ImageSearchFeatureNorm::None:
    case ImageSearchFeatureNorm::L1:
    case ImageSearchFeatureNorm::L2:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ImageSearch feature norm");
    }

    switch (config.faiss_backend)
    {
    case ImageSearchFaissBackend::CPU:
        break;
    case ImageSearchFaissBackend::GPU:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ImageSearch Faiss backend");
    }

    switch (config.index_storage)
    {
    case ImageSearchIndexStorage::RAM:
    case ImageSearchIndexStorage::Disk:
        break;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ImageSearch index storage");
    }

    if (config.model_batch_size == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch model batch size must be positive");
    }
    if (config.model_batch_size > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch model batch size is too large");
    }
}

/**
 * @brief 获取当前 CUDA 设备编号，供 GPU Faiss 索引使用。
 * @return 设备 ID。
 */
int currentCudaDevice()
{
    int device{0};
    checkCuda(cudaGetDevice(&device), "cudaGetDevice(Faiss GPU backend)");
    return device;
}

/**
 * @brief 将磁盘读入的 CPU 索引迁移到配置指定的后端。
 * @param cpu_index 从文件加载的 CPU 索引。
 * @param backend 目标 Faiss 后端。
 * @return 可在配置后端上搜索的索引包。
 */
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
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ImageSearch Faiss backend");
    }

    return bundle;
}

/**
 * @brief 解析 Faiss 索引文件路径。
 *
 * 若 ``index_file`` 为空，则在图库目录下生成时间戳文件名。
 *
 * @param gallery_dir 图库目录。
 * @param index_file 用户指定的索引路径；可为空。
 * @return 最终使用的索引文件路径。
 */
fs::path resolveIndexPath(const fs::path &gallery_dir, const fs::path &index_file)
{
    return irt::util::resolveOutputFilePath(index_file, gallery_dir, ".faiss");
}

/**
 * @brief 解析显式图片列表构建时的 Faiss 输出路径。
 */
fs::path resolveExplicitIndexPath(const fs::path &index_file)
{
    return irt::util::resolveOutputFilePath(index_file, {}, ".faiss");
}

/**
 * @brief 生成绝对路径形式的 manifest 值。
 */
std::string absolutePathManifestValue(const fs::path &path)
{
    return path.empty() ? std::string{} : fs::absolute(path).generic_string();
}

/**
 * @brief 生成图库目录在元数据文件中的 canonical 值。
 * @param gallery_dir 图库目录。
 * @return 绝对路径的 generic 字符串，用于 ``gallery_dir`` manifest 字段校验。
 */
std::string galleryDirectoryMetadataValue(const fs::path &gallery_dir)
{
    return absolutePathManifestValue(gallery_dir);
}

/**
 * @brief 显式路径列表构建时写入元数据的 ``gallery_dir`` 占位值。
 * @return 固定哨兵字符串 ``"<explicit_path_list>"``。
 */
std::string explicitPathListMetadataValue()
{
    return "<explicit_path_list>";
}

/**
 * @brief 校验并规范化显式传入的图库图片条目列表。
 *
 * 要求列表非空、路径存在且为支持的图片文件，并统一转为绝对路径。
 */
std::vector<ImageSearchItem> normalizeImageItems(const std::vector<ImageSearchItem> &gallery_items)
{
    if (gallery_items.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Image item list must not be empty");
    }

    std::vector<ImageSearchItem> normalized;
    normalized.reserve(gallery_items.size());
    for (const auto &item : gallery_items)
    {
        if (item.image_path.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Image path must not be empty");
        }

        std::error_code ec;
        const bool      exists          = fs::exists(item.image_path, ec);
        const bool      is_regular_file = exists && fs::is_regular_file(item.image_path, ec);
        if (!is_regular_file)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Image path does not exist: %s",
                                 item.image_path.string().c_str());
        }
        if (!ImageSearch::isImageFile(item.image_path))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported image file: %s",
                                 item.image_path.string().c_str());
        }
        normalized.push_back(ImageSearchItem{item.image_id, fs::absolute(item.image_path)});
    }
    return normalized;
}

std::vector<ImageSearchItem> makeImageItemsFromPaths(const std::vector<fs::path> &image_paths)
{
    std::vector<ImageSearchItem> items;
    items.reserve(image_paths.size());
    for (size_t i = 0; i < image_paths.size(); ++i)
    {
        items.push_back(ImageSearchItem{static_cast<int64_t>(i), image_paths[i]});
    }
    return items;
}

std::vector<fs::path> imageItemPaths(const std::vector<ImageSearchItem> &items)
{
    std::vector<fs::path> paths;
    paths.reserve(items.size());
    for (const auto &item : items)
    {
        paths.push_back(item.image_path);
    }
    return paths;
}

std::vector<int64_t> imageItemIds(const std::vector<ImageSearchItem> &items)
{
    std::vector<int64_t> ids;
    ids.reserve(items.size());
    for (const auto &item : items)
    {
        ids.push_back(item.image_id);
    }
    return ids;
}

bool imageIdsMatch(const std::vector<ImageSearchItem> &items, const std::vector<int64_t> &ids)
{
    if (items.size() != ids.size())
    {
        return false;
    }
    for (size_t i = 0; i < items.size(); ++i)
    {
        if (items[i].image_id != ids[i])
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief 构造图像检索 manifest 条目。
 */
irt::util::ManifestEntries imageSearchManifestEntries(const fs::path &index_path, const std::string &gallery_value,
                                                      const ImageSearchConfig     &config,
                                                      const std::vector<int64_t>  &image_ids)
{
    irt::util::ManifestEntries entries{
        {"version", "1"},
        {"kind", "image_search"},
        {"index_file", absolutePathManifestValue(index_path)},
        {"model", config.model_name},
        {"feature", config.feature_name},
        {"gallery_dir", gallery_value},
        {"model_backend", modelBackendName(config.model_backend)},
        {"model_device", modelDeviceName(config.model_device)},
        {"preprocess_backend", preprocessBackendName(config.preprocess_backend)},
        {"norm", featureNormName(config.norm)},
        {"faiss_backend", faissBackendName(config.faiss_backend)},
        {"index_storage", indexStorageName(config.index_storage)},
        {"model_batch_size", std::to_string(config.model_batch_size)},
        {"index_kind", indexKindName(config)},
    };
    if (useCpuDiskIndex(config))
    {
        entries.emplace_back("ivf_data_file", absolutePathManifestValue(priv::cpuOnDiskIvfDataPath(index_path)));
    }
    entries.emplace_back("image_count", std::to_string(image_ids.size()));
    for (size_t i = 0; i < image_ids.size(); ++i)
    {
        entries.emplace_back("image." + std::to_string(i) + ".id", std::to_string(image_ids[i]));
    }
    return entries;
}

/**
 * @brief 将图像检索配置和 ID 映射写入 ``xxx.manifest.yaml``。
 */
void saveImageSearchManifest(const fs::path &index_path, const std::string &gallery_value,
                             const ImageSearchConfig &config, const std::vector<int64_t> &image_ids)
{
    irt::util::writeYamlManifest(irt::util::manifestPathForDataFile(index_path),
                                 imageSearchManifestEntries(index_path, gallery_value, config, image_ids));
}

/**
 * @brief 从 manifest 解析图库图像 ID。
 */
std::vector<int64_t> loadImageIdsFromManifest(const fs::path &index_path)
{
    const auto manifest = irt::util::loadYamlManifest(irt::util::manifestPathForDataFile(index_path));
    if (manifest.empty() || !irt::util::manifestValueEquals(manifest, "kind", "image_search"))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch manifest is missing or invalid: %s",
                             irt::util::manifestPathForDataFile(index_path).string().c_str());
    }

    const auto count_text = irt::util::manifestValue(manifest, "image_count");
    if (count_text.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch manifest has no image_count: %s",
                             irt::util::manifestPathForDataFile(index_path).string().c_str());
    }

    size_t image_count{0};
    try
    {
        image_count = static_cast<size_t>(std::stoull(count_text));
    }
    catch (const std::exception &)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid image_count in ImageSearch manifest: %s",
                             count_text.c_str());
    }

    std::vector<int64_t> image_ids;
    image_ids.reserve(image_count);
    for (size_t i = 0; i < image_count; ++i)
    {
        const auto key   = "image." + std::to_string(i) + ".id";
        const auto value = irt::util::manifestValue(manifest, key);
        if (value.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch manifest is missing %s",
                                 key.c_str());
        }
        try
        {
            image_ids.push_back(std::stoll(value));
        }
        catch (const std::exception &)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Invalid image ID in ImageSearch manifest: %s",
                                 value.c_str());
        }
    }
    return image_ids;
}

/**
 * @brief 判断磁盘上已有索引是否与当前图库及配置兼容，可直接加载。
 */
bool existingIndexMatchesConfig(const fs::path &index_path, const fs::path &gallery_dir,
                                const ImageSearchConfig &config)
{
    const auto manifest_path = irt::util::manifestPathForDataFile(index_path);
    if (!fs::exists(manifest_path))
    {
        return false;
    }

    const auto manifest = irt::util::loadYamlManifest(manifest_path);
    if (manifest.empty())
    {
        return false;
    }

    const bool gallery_matches
        = gallery_dir.empty()
       || irt::util::manifestValueEquals(manifest, "gallery_dir", galleryDirectoryMetadataValue(gallery_dir))
       || irt::util::manifestValueEquals(manifest, "gallery_dir", explicitPathListMetadataValue());

    return irt::util::manifestValueEquals(manifest, "kind", "image_search")
        && irt::util::manifestValueEquals(manifest, "index_file", absolutePathManifestValue(index_path))
        && irt::util::manifestValueEquals(manifest, "model", config.model_name)
        && irt::util::manifestValueEquals(manifest, "feature", config.feature_name)
        && gallery_matches
        && irt::util::manifestValueEquals(manifest, "norm", featureNormName(config.norm))
        && irt::util::manifestValueEquals(manifest, "index_storage", indexStorageName(config.index_storage))
        && irt::util::manifestValueEquals(manifest, "index_kind", indexKindName(config));
}

} // namespace

namespace priv {

/**
 * @brief 基于 InferRT 模型的单图特征提取器。
 *
 * 加载 TensorRT 引擎，对查询/图库图片做 ImageNet 预处理，经 ``forwardFeatures``
 * 导出指定中间层 float 特征并按配置归一化。
 */
class ImageSearchFeatureExtractor
{
public:
    /**
     * @brief 创建特征提取器并构建或加载 TensorRT 引擎。
     * @param model_name 内置模型名称。
     * @param feature_name 中间特征张量名。
     * @param weights_file ``.wts`` 权重路径。
     * @param config 预处理与归一化配置。
     */
    ImageSearchFeatureExtractor(std::string model_name, std::string feature_name, const fs::path &weights_file,
                                ImageSearchConfig config)
        : model_name_(std::move(model_name))
        , feature_name_(std::move(feature_name))
        , config_(config)
    {
        validateConfig(config_);

        auto model_config = std::make_unique<irt::model::IModelConfig>();
        model_config->setFeatureTensorNames({feature_name_});
        model_config->setOutputTensorNames({feature_name_});
        model_config->setFeatureOnly(true);
        model_config->setBackend(config_.model_backend);
        model_config->setDevice(config_.model_device);
        if (usesTensorRtModelBackend(config_) && config_.model_batch_size > 1)
        {
            const int batch = static_cast<int>(config_.model_batch_size);
            model_config->setDynamicBatchRange(1, batch, batch);
        }

        const std::string runtime_model_name = usesTensorRtModelBackend(config_) ? model_name_ : "onnx";
        model_                               = irt::model::CreateModel(runtime_model_name, std::move(model_config));
        if (!model_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create model: %s",
                                 runtime_model_name.c_str());
        }

        model_->setLogLevel(nvinfer1::ILogger::Severity::kINFO);
        model_->buildOrLoad(weights_file.string());

        const auto input_tensor_names = model_->ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
        if (input_tensor_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageSearch model must expose at least one input tensor");
        }

        input_name_            = input_tensor_names.front();
        const auto input_shape = model_->tensorShape(input_name_);
        const auto input_type  = model_->tensorDataType(input_tensor_names.front());
        if (input_type != nvinfer1::DataType::kFLOAT)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch expects float32 input tensor");
        }
        input_shape_ = resolveInputShape(input_shape);

        const auto output_tensor_names = model_->ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
        if (output_tensor_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageSearch model must expose at least one output tensor");
        }
        output_name_ = output_tensor_names.front();
        output_dims_ = model_->tensorShape(output_name_);
        output_type_ = model_->tensorDataType(output_name_);
        if (output_type_ != nvinfer1::DataType::kFLOAT)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Expected float feature tensor for %s, got unsupported data type",
                                 output_name_.c_str());
        }
        if (output_dims_.nbDims <= 0 || output_dims_.d[0] != input_shape_.d[0])
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageSearch feature output must preserve model batch dimension");
        }

        max_batch_size_            = static_cast<size_t>(input_shape_.d[0]);
        input_height_              = static_cast<int>(input_shape_.d[2]);
        input_width_               = static_cast<int>(input_shape_.d[3]);
        input_elements_per_sample_ = elementCount(input_shape_) / max_batch_size_;
        feature_dim_               = elementCount(output_dims_) / max_batch_size_;
        if (usesTensorRtModelBackend(config_))
        {
            device_input_.resize(max_batch_size_ * input_elements_per_sample_, nvinfer1::DataType::kFLOAT);
            device_output_.resize(max_batch_size_ * feature_dim_, nvinfer1::DataType::kFLOAT);
        }
    }

    /** @brief 析构；先释放 TensorRT 模型再销毁 CUDA 缓冲区。 */
    ~ImageSearchFeatureExtractor()
    {
        // Release TensorRT context/engine before CUDA buffers.
        model_.reset();
    }

    /**
     * @brief 获取特征向量维度。
     * @return 输出张量元素个数。
     */
    int featureDim() const noexcept
    {
        return static_cast<int>(feature_dim_);
    }

    /**
     * @brief 从单张图片提取归一化后的检索特征。
     * @param image_path 图片路径。
     * @return 长度为 ``featureDim()`` 的 float 特征向量。
     */
    std::vector<float> extract(const fs::path &image_path)
    {
        const std::vector<fs::path> image_paths{image_path};
        return extractBatch(image_paths, 0, 1);
    }

    /**
     * @brief 从连续图像区间批量提取归一化特征。
     * @param image_paths 图像路径数组。
     * @param begin 起始图像下标。
     * @param count 图像数量；大于模型 batch 时会自动拆分为多次前向。
     * @return 扁平化特征数组，布局为 ``count x featureDim()``。
     */
    std::vector<float> extractBatch(const std::vector<fs::path> &image_paths, size_t begin, size_t count)
    {
        if (count == 0)
        {
            return {};
        }
        if (begin > image_paths.size() || count > image_paths.size() - begin)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch feature batch range is invalid");
        }
        if (count > max_batch_size_)
        {
            std::vector<float> features;
            features.reserve(count * feature_dim_);
            for (size_t offset = 0; offset < count; offset += max_batch_size_)
            {
                const size_t chunk_count = std::min(max_batch_size_, count - offset);
                auto         chunk       = extractBatch(image_paths, begin + offset, chunk_count);
                features.insert(features.end(), chunk.begin(), chunk.end());
            }
            return features;
        }

        auto       input_data      = preprocessBatch(image_paths, begin, count);
        const auto output_dims     = setRuntimeBatchSize(count);
        const auto output_elements = elementCount(output_dims);
        if (output_elements != count * feature_dim_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageSearch feature output size does not match runtime batch");
        }

        std::vector<float> features(output_elements);
        if (usesTensorRtModelBackend(config_))
        {
            std::vector<void *> buffers{device_input_.data(), device_output_.data()};
            const auto          stream = model_->resolveExecutionStream();

            checkCuda(cudaMemcpyAsync(device_input_.data(), input_data.data(), input_data.size() * sizeof(float),
                                      cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync(H2D input)");
            model_->forwardFeatures(buffers, stream, true);
            checkCuda(cudaMemcpyAsync(features.data(), device_output_.data(), features.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync(D2H feature)");
            checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(feature extraction)");
        }
        else
        {
            std::vector<void *> buffers{input_data.data(), features.data()};
            model_->forwardFeatures(buffers, nullptr, false);
        }

        for (size_t i = 0; i < count; ++i)
        {
            normalizeFeature(features.data() + i * feature_dim_, feature_dim_, config_.norm);
        }
        return features;
    }

    /**
     * @brief 按任意下标列表批量提取图库特征。
     *
     * IVF 训练采样可能按 stride 选择非连续图片；该重载把这些图片重排为临时连续批次，
     * 复用常规 ``extractBatch`` 的预处理、前向和归一化逻辑。
     *
     * @param image_paths 完整图库图片路径列表。
     * @param indices 待提取图片在 ``image_paths`` 中的下标列表。
     * @return 扁平化 ``indices.size() x featureDim()`` 特征。
     */
    std::vector<float> extractBatch(const std::vector<fs::path> &image_paths, const std::vector<size_t> &indices)
    {
        if (indices.empty())
        {
            return {};
        }

        std::vector<fs::path> selected_images;
        selected_images.reserve(indices.size());
        for (const auto index : indices)
        {
            if (index >= image_paths.size())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch feature batch index is invalid");
            }
            selected_images.push_back(image_paths[index]);
        }
        return extractBatch(selected_images, 0, selected_images.size());
    }

private:
    /**
     * @brief 解析模型输入 batch，并把动态 graph 后端设置到配置的最大 batch。
     *
     * TensorRT 通过 dynamic profile 支持 ``1..model_batch_size``；ONNX Runtime/OpenVINO
     * 只有在图输入第 0 维为动态维时才能同时服务图库 batch 和单图查询。固定 batch 的
     * graph 模型只能安全使用 batch=1。
     *
     * @param input_shape 后端暴露的原始输入形状。
     * @return 已解析到可分配缓冲区的输入形状。
     */
    nvinfer1::Dims resolveInputShape(nvinfer1::Dims input_shape)
    {
        if (input_shape.nbDims != 4 || input_shape.d[1] != 3 || input_shape.d[2] <= 0 || input_shape.d[3] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch expects input shape Nx3xHxW, got %s",
                                 dimsToCsv(input_shape).c_str());
        }

        if (input_shape.d[0] < 0)
        {
            input_shape.d[0] = static_cast<int32_t>(config_.model_batch_size);
            model_->setTensorShape(input_name_, input_shape);
            return model_->tensorShape(input_name_);
        }

        if (input_shape.d[0] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch expects input shape Nx3xHxW, got %s",
                                 dimsToCsv(input_shape).c_str());
        }

        const auto runtime_batch = static_cast<size_t>(input_shape.d[0]);
        if (usesTensorRtModelBackend(config_))
        {
            if (runtime_batch != config_.model_batch_size)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "ImageSearch TensorRT model batch does not match configured model batch size");
            }
            return input_shape;
        }

        if (runtime_batch != 1 || config_.model_batch_size != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageSearch graph backends require dynamic batch to use model batch size %zu; "
                                 "export the ONNX/OpenVINO model with dynamic batch",
                                 config_.model_batch_size);
        }
        model_->setTensorShape(input_name_, input_shape);
        return model_->tensorShape(input_name_);
    }

    /**
     * @brief 读取并预处理一个连续图像区间，拼接为 NCHW batch。
     */
    std::vector<float> preprocessBatch(const std::vector<fs::path> &image_paths, size_t begin, size_t count) const
    {
        std::vector<float> input_data;
        input_data.reserve(count * input_elements_per_sample_);
        for (size_t i = 0; i < count; ++i)
        {
            const auto &image_path = image_paths[begin + i];
            cv::Mat     image      = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            if (image.empty())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                     image_path.string().c_str());
            }

            switch (config_.preprocess_backend)
            {
            case ImageSearchPreprocessBackend::CPU:
            {
                const auto preprocessed
                    = irt::model::ImageNetUtil::preprocess(image, cv::Size(input_width_, input_height_));
                auto single = irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);
                input_data.insert(input_data.end(), single.begin(), single.end());
                break;
            }
            case ImageSearchPreprocessBackend::GPU:
                throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                                     "ImageSearch GPU preprocessing is not implemented");
            }
        }
        return input_data;
    }

    /**
     * @brief 设置运行时 batch，并返回当前输出形状。
     */
    nvinfer1::Dims setRuntimeBatchSize(size_t batch_size)
    {
        auto dims = input_shape_;
        dims.d[0] = static_cast<int32_t>(batch_size);
        model_->setTensorShape(input_name_, dims);

        auto output_dims = model_->tensorShape(output_name_);
        if (output_dims.nbDims <= 0 || output_dims.d[0] != dims.d[0])
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageSearch feature output must preserve runtime batch dimension");
        }
        return output_dims;
    }

    ///< 模型名称。
    std::string model_name_;

    ///< 导出的中间特征名。
    std::string feature_name_;

    ///< 预处理与归一化配置。
    ImageSearchConfig config_{};

    ///< TensorRT 推理模型。
    std::unique_ptr<irt::model::IModel> model_;

    ///< 输入张量名称。
    std::string input_name_;

    ///< 特征输出张量名。
    std::string output_name_;

    ///< 最大 batch 对应的输入形状；推理前只修改第 0 维。
    nvinfer1::Dims input_shape_{};

    ///< 输出张量形状。
    nvinfer1::Dims output_dims_{};

    ///< 输出张量数据类型。
    nvinfer1::DataType output_type_{};

    ///< 模型输入高度。
    int input_height_{224};

    ///< 模型输入宽度。
    int input_width_{224};

    ///< 特征向量维度。
    size_t feature_dim_{0};

    ///< 特征提取模型一次前向允许的最大 batch。
    size_t max_batch_size_{1};

    ///< 单张图像输入元素数量。
    size_t input_elements_per_sample_{0};

    ///< 设备侧输入缓冲区。
    DeviceBuffer device_input_;

    ///< 设备侧特征输出缓冲区。
    DeviceBuffer device_output_;
};

} // namespace priv

namespace {

/**
 * @brief 构建 CPU 磁盘 IVF 索引。
 * @param gallery_images 图库图片路径列表，顺序对应 Faiss 向量 ID。
 * @param extractor 已加载的特征提取器。
 * @param index_path ``.faiss`` 输出路径。
 * @param config 图像检索配置。
 * @param progress_callback 构建进度回调。
 * @return 索引及相关资源包。
 */
FaissIndexBundle buildCpuOnDiskIndex(const std::vector<fs::path> &gallery_images,
                                     priv::ImageFeatureExtractor &extractor, const fs::path &index_path,
                                     const ImageSearchConfig                &config,
                                     const ImageSearchBuildProgressCallback &progress_callback)
{
    FaissIndexBundle bundle;
    bundle.index = priv::buildCpuOnDiskIvfFlatIndex(
        gallery_images.size(), extractor.featureDim(), index_path, config.model_batch_size,
        [&](size_t index) { return extractor.extract(gallery_images[index]); },
        [&](size_t begin, size_t count) { return extractor.extractBatch(gallery_images, begin, count); },
        [&](const std::vector<size_t> &indices) { return extractor.extractBatch(gallery_images, indices); },
        progress_callback);
    return bundle;
}

/**
 * @brief 构建内存 IVF-PQ 压缩索引，并按配置保留在 CPU 或迁移到 GPU。
 * @param gallery_images 图库图片路径列表，顺序对应 Faiss 向量 ID。
 * @param extractor 已加载的特征提取器。
 * @param index_path ``.faiss`` 输出路径。
 * @param config 图像检索配置。
 * @param progress_callback 构建进度回调。
 * @return 索引及相关资源包；GPU 后端会额外持有 ``StandardGpuResources``。
 */
FaissIndexBundle buildRamIvfPqIndex(const std::vector<fs::path> &gallery_images, priv::ImageFeatureExtractor &extractor,
                                    const fs::path &index_path, const ImageSearchConfig &config,
                                    const ImageSearchBuildProgressCallback &progress_callback)
{
    auto cpu_index = priv::buildRamIvfPqIndex(
        gallery_images.size(), extractor.featureDim(), config.model_batch_size,
        [&](size_t index) { return extractor.extract(gallery_images[index]); },
        [&](size_t begin, size_t count) { return extractor.extractBatch(gallery_images, begin, count); },
        [&](const std::vector<size_t> &indices) { return extractor.extractBatch(gallery_images, indices); },
        progress_callback, config.faiss_backend == ImageSearchFaissBackend::GPU);

    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::WritingIndex, 0, 0, 0, 0, 1);
    faiss::write_index(cpu_index.get(), index_path.string().c_str());
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::WritingIndex, 0, 0, 0, 1, 1);

    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 0, 1);
    auto bundle = moveCpuIndexToConfiguredBackend(std::move(cpu_index), config.faiss_backend);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 1, 1);
    return bundle;
}

/**
 * @brief 按配置构建 Faiss 索引（内存 IVF-PQ 或 CPU 磁盘 IVF）。
 * @param gallery_images 图库图片路径列表。
 * @param extractor 已加载的特征提取器。
 * @param index_path ``.faiss`` 输出路径。
 * @param config 图像检索配置。
 * @param progress_callback 构建进度回调。
 * @return 构建完成的索引资源包。
 */
FaissIndexBundle buildIndex(const std::vector<fs::path> &gallery_images, priv::ImageFeatureExtractor &extractor,
                            const fs::path &index_path, const ImageSearchConfig &config,
                            const ImageSearchBuildProgressCallback &progress_callback)
{
    if (useCpuDiskIndex(config))
    {
        return buildCpuOnDiskIndex(gallery_images, extractor, index_path, config, progress_callback);
    }

    return buildRamIvfPqIndex(gallery_images, extractor, index_path, config, progress_callback);
}

/**
 * @brief 从磁盘加载 Faiss 索引、ID 映射，并迁移到配置指定的后端。
 * @param index_path ``.faiss`` 索引路径。
 * @param config 图像检索配置。
 * @return 加载完成的索引资源包。
 */
FaissIndexBundle loadIndex(const fs::path &index_path, const ImageSearchConfig &config)
{
    auto cpu_index = useCpuDiskIndex(config)
                       ? priv::loadCpuOnDiskIvfFlatIndex(index_path)
                       : std::unique_ptr<faiss::Index>(faiss::read_index(index_path.string().c_str()));
    if (!cpu_index)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load Faiss index: %s",
                             index_path.string().c_str());
    }

    auto image_ids = loadImageIdsFromManifest(index_path);
    if (static_cast<faiss::idx_t>(image_ids.size()) != cpu_index->ntotal)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Index size (%lld) does not match ID mapping size (%zu)",
                             static_cast<long long>(cpu_index->ntotal), image_ids.size());
    }

    FaissIndexBundle bundle;
    if (useCpuDiskIndex(config))
    {
        bundle.index = std::move(cpu_index);
    }
    else
    {
        bundle = moveCpuIndexToConfiguredBackend(std::move(cpu_index), config.faiss_backend);
    }
    bundle.image_ids = std::move(image_ids);
    return bundle;
}

} // namespace

ImageSearch::Impl::Impl(ImageSearchConfig config)
    : config_(std::move(config))
{
    if (config_.faiss_backend == ImageSearchFaissBackend::GPU)
    {
        config_.index_storage = ImageSearchIndexStorage::RAM;
    }
    if (!irt::model::isSupportedModel(config_.model_name))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model: %s", config_.model_name.c_str());
    }
    if (config_.feature_name.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "feature name must not be empty");
    }
    validateConfig(config_);
}

ImageSearch::Impl::~Impl() = default;

void ImageSearch::Impl::buildOrLoad(const fs::path &weights_file, const fs::path &gallery_dir,
                                    const fs::path &index_file, bool rebuild_index,
                                    ImageSearchBuildProgressCallback progress_callback)
{
    const fs::path resolved_index_path = resolveIndexPath(gallery_dir, index_file);
    if (!rebuild_index && fs::exists(resolved_index_path)
        && fs::exists(irt::util::manifestPathForDataFile(resolved_index_path))
        && existingIndexMatchesConfig(resolved_index_path, gallery_dir, config_))
    {
        priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Started);
        priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 0, 1);
        load(weights_file, resolved_index_path);
        priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 1, 1);
        priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Finished, 0, 0, 0, gallery_ids_.size(),
                                  gallery_ids_.size());
        return;
    }

    build(weights_file, gallery_dir, resolved_index_path, std::move(progress_callback));
}

void ImageSearch::Impl::build(const fs::path &weights_file, const fs::path &gallery_dir, const fs::path &index_file,
                              ImageSearchBuildProgressCallback progress_callback)
{
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Started);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages);
    auto images = ImageSearch::collectGalleryImages(gallery_dir);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages, 0, 0, 0, images.size(),
                              images.size());
    buildWithImages(weights_file, gallery_dir, makeImageItemsFromPaths(images), resolveIndexPath(gallery_dir, index_file),
                    galleryDirectoryMetadataValue(gallery_dir), std::move(progress_callback));
}

void ImageSearch::Impl::buildOrLoad(const fs::path &weights_file, const std::vector<ImageSearchItem> &gallery_items,
                                    const fs::path &index_file, bool rebuild_index,
                                    ImageSearchBuildProgressCallback progress_callback)
{
    const fs::path resolved_index_path = resolveExplicitIndexPath(index_file);

    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Started);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages);
    auto normalized_items = normalizeImageItems(gallery_items);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages, 0, 0, 0,
                              normalized_items.size(), normalized_items.size());

    if (!rebuild_index && fs::exists(resolved_index_path)
        && fs::exists(irt::util::manifestPathForDataFile(resolved_index_path))
        && existingIndexMatchesConfig(resolved_index_path, {}, config_))
    {
        const auto mapped_ids = loadImageIdsFromManifest(resolved_index_path);
        if (imageIdsMatch(normalized_items, mapped_ids))
        {
            priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 0, 1);
            load(weights_file, resolved_index_path);
            priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingIndex, 0, 0, 0, 1, 1);
            priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Finished, 0, 0, 0,
                                      gallery_ids_.size(), gallery_ids_.size());
            return;
        }
    }

    buildWithImages(weights_file, {}, std::move(normalized_items), resolved_index_path, explicitPathListMetadataValue(),
                    std::move(progress_callback));
}

void ImageSearch::Impl::build(const fs::path &weights_file, const std::vector<ImageSearchItem> &gallery_items,
                              const fs::path &index_file, ImageSearchBuildProgressCallback progress_callback)
{
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Started);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages);
    auto normalized_items = normalizeImageItems(gallery_items);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::CollectingImages, 0, 0, 0,
                              normalized_items.size(), normalized_items.size());
    buildWithImages(weights_file, {}, std::move(normalized_items), resolveExplicitIndexPath(index_file),
                    explicitPathListMetadataValue(), std::move(progress_callback));
}

void ImageSearch::Impl::load(const fs::path &weights_file, const fs::path &index_file)
{
    if (index_file.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageSearch index_file must not be empty when loading");
    }
    const fs::path resolved_index_path = index_file;
    if (!fs::exists(resolved_index_path))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index file does not exist: %s",
                             resolved_index_path.string().c_str());
    }
    if (!fs::exists(irt::util::manifestPathForDataFile(resolved_index_path)))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Index manifest file does not exist: %s",
                             irt::util::manifestPathForDataFile(resolved_index_path).string().c_str());
    }
    if (!existingIndexMatchesConfig(resolved_index_path, {}, config_))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Index manifest does not match current ImageSearch config: %s",
                             irt::util::manifestPathForDataFile(resolved_index_path).string().c_str());
    }

    auto loaded = loadIndex(resolved_index_path, config_);
    index_.reset();
    faiss_gpu_resources_.reset();
    weights_file_        = weights_file;
    gallery_dir_.clear();
    index_path_          = resolved_index_path;
    faiss_gpu_resources_ = std::move(loaded.gpu_resources);
    index_               = std::move(loaded.index);
    gallery_ids_         = std::move(loaded.image_ids);
    extractor_.reset();
    feature_dim_ = static_cast<int>(index_->d);
}

void ImageSearch::Impl::buildWithImages(const fs::path &weights_file, const fs::path &gallery_dir,
                                        std::vector<ImageSearchItem> gallery_items, const fs::path &index_path,
                                        const std::string               &metadata_gallery_value,
                                        ImageSearchBuildProgressCallback progress_callback)
{
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingModel, 0, 0, 0, 0, 1);
    auto extractor = std::make_unique<priv::ImageFeatureExtractor>(config_.model_name, config_.feature_name,
                                                                   weights_file, config_);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::LoadingModel, 0, 0, 0, 1, 1);
    if (!index_path.parent_path().empty())
    {
        fs::create_directories(index_path.parent_path());
    }

    auto gallery_paths = imageItemPaths(gallery_items);
    auto gallery_ids   = imageItemIds(gallery_items);
    auto built         = buildIndex(gallery_paths, *extractor, index_path, config_, progress_callback);
    index_.reset();
    faiss_gpu_resources_.reset();
    weights_file_        = weights_file;
    gallery_dir_         = gallery_dir;
    index_path_          = index_path;
    faiss_gpu_resources_ = std::move(built.gpu_resources);
    index_               = std::move(built.index);
    gallery_ids_         = std::move(gallery_ids);
    feature_dim_         = extractor->featureDim();
    extractor_           = std::move(extractor);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::SavingMetadata, 0, 0, 0, 0, 1);
    saveImageSearchManifest(index_path_, metadata_gallery_value, config_, gallery_ids_);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::SavingMetadata, 0, 0, 0, 1, 1);
    priv::reportBuildProgress(progress_callback, ImageSearchBuildStage::Finished, 0, 0, 0, gallery_ids_.size(),
                              gallery_ids_.size());
}

std::vector<ImageSearchResult> ImageSearch::Impl::search(const fs::path &query_image, int top_k)
{
    if (!index_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "ImageSearch index is not ready");
    }
    if (top_k <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "top_k must be positive");
    }
    if (index_->ntotal <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "ImageSearch index is empty");
    }

    ensureExtractor();
    const auto query_feature = extractor_->extract(query_image);

    const int                 result_count = std::min(top_k, static_cast<int>(index_->ntotal));
    std::vector<faiss::idx_t> indices(result_count);
    std::vector<float>        distances(result_count);
    index_->search(1, query_feature.data(), result_count, distances.data(), indices.data());

    std::vector<ImageSearchResult> results;
    results.reserve(static_cast<size_t>(result_count));
    for (int i = 0; i < result_count; ++i)
    {
        if (indices[i] < 0 || static_cast<size_t>(indices[i]) >= gallery_ids_.size())
        {
            continue;
        }
        results.push_back({distances[i], gallery_ids_[static_cast<size_t>(indices[i])]});
    }
    return results;
}

bool ImageSearch::Impl::isReady() const noexcept
{
    return index_ != nullptr;
}

const ImageSearchConfig &ImageSearch::Impl::config() const noexcept
{
    return config_;
}

const fs::path &ImageSearch::Impl::indexPath() const noexcept
{
    return index_path_;
}

std::vector<int64_t> ImageSearch::Impl::galleryIds() const
{
    return gallery_ids_;
}

int ImageSearch::Impl::featureDim() const noexcept
{
    return feature_dim_;
}

void ImageSearch::Impl::ensureExtractor()
{
    if (!extractor_)
    {
        extractor_   = std::make_unique<priv::ImageFeatureExtractor>(config_.model_name, config_.feature_name,
                                                                     weights_file_, config_);
        feature_dim_ = extractor_->featureDim();
    }
}

} // namespace irt::features
