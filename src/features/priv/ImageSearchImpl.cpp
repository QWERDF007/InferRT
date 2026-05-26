#include "ImageSearchImpl.hpp"
#include "ImageSearchFaissIndex.hpp"

#include <cuda_runtime_api.h>
#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>
#pragma warning(pop)
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace irt::features {

namespace {

using DeviceBuffer = irt::model::DeviceBuffer;
using irt::model::checkCuda;
using irt::model::dimsToCsv;
using irt::model::elementCount;

struct FaissIndexBundle
{
    std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_resources;
    std::unique_ptr<faiss::Index>                     index;
    std::vector<fs::path>                             image_paths;
};

void l2Normalize(std::vector<float> &values)
{
    const float sum_sq = std::inner_product(values.begin(), values.end(), values.begin(), 0.0f);
    if (sum_sq <= 0.0f)
    {
        return;
    }

    const float inv_norm = 1.0f / std::sqrt(sum_sq);
    for (float &value : values)
    {
        value *= inv_norm;
    }
}

void l1Normalize(std::vector<float> &values)
{
    float sum_abs = 0.0f;
    for (const float value : values)
    {
        sum_abs += std::abs(value);
    }
    if (sum_abs <= 0.0f)
    {
        return;
    }

    const float inv_norm = 1.0f / sum_abs;
    for (float &value : values)
    {
        value *= inv_norm;
    }
}

void normalizeFeature(std::vector<float> &values, ImageSearchFeatureNorm norm)
{
    switch (norm)
    {
    case ImageSearchFeatureNorm::None:
        return;
    case ImageSearchFeatureNorm::L1:
        l1Normalize(values);
        return;
    case ImageSearchFeatureNorm::L2:
        l2Normalize(values);
        return;
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ImageSearch feature norm");
}

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

bool useCpuDiskIndex(const ImageSearchConfig &config)
{
    return config.faiss_backend == ImageSearchFaissBackend::CPU
        && config.index_storage == ImageSearchIndexStorage::Disk;
}

const char *indexKindName(const ImageSearchConfig &config)
{
    return useCpuDiskIndex(config) ? "ivf_flat_ondisk" : "flat_ip";
}

bool isDefaultConfig(const ImageSearchConfig &config)
{
    return config.model_name == ImageSearch::kDefaultModelName
        && config.feature_name == ImageSearch::kDefaultFeatureName
        && config.preprocess_backend == ImageSearchPreprocessBackend::CPU && config.norm == ImageSearchFeatureNorm::L2
        && config.faiss_backend == ImageSearchFaissBackend::CPU && config.index_storage == ImageSearchIndexStorage::RAM;
}

void validateConfig(const ImageSearchConfig &config)
{
    switch (config.preprocess_backend)
    {
    case ImageSearchPreprocessBackend::CPU:
        break;
    case ImageSearchPreprocessBackend::GPU:
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                             "ImageSearch GPU preprocessing is not implemented");
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

    if (config.disk_build_batch_size == 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageSearch disk build batch size must be positive");
    }
}

int currentCudaDevice()
{
    int device{0};
    checkCuda(cudaGetDevice(&device), "cudaGetDevice(Faiss GPU backend)");
    return device;
}

FaissIndexBundle createEmptyFaissIndex(int feature_dim, ImageSearchFaissBackend backend)
{
    FaissIndexBundle bundle;
    switch (backend)
    {
    case ImageSearchFaissBackend::CPU:
        bundle.index = std::make_unique<faiss::IndexFlatIP>(feature_dim);
        break;
    case ImageSearchFaissBackend::GPU:
    {
        bundle.gpu_resources = std::make_unique<faiss::gpu::StandardGpuResources>();
        faiss::gpu::GpuIndexFlatConfig gpu_config;
        gpu_config.device = currentCudaDevice();
        bundle.index = std::make_unique<faiss::gpu::GpuIndexFlatIP>(bundle.gpu_resources.get(), feature_dim,
                                                                    gpu_config);
        break;
    }
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ImageSearch Faiss backend");
    }

    return bundle;
}

FaissIndexBundle moveCpuIndexToConfiguredBackend(std::unique_ptr<faiss::Index> cpu_index,
                                                 ImageSearchFaissBackend backend)
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

void writeIndex(const faiss::Index &index, ImageSearchFaissBackend backend, const fs::path &index_path)
{
    switch (backend)
    {
    case ImageSearchFaissBackend::CPU:
        faiss::write_index(&index, index_path.string().c_str());
        break;
    case ImageSearchFaissBackend::GPU:
    {
        auto cpu_index = std::unique_ptr<faiss::Index>(faiss::gpu::index_gpu_to_cpu(&index));
        if (!cpu_index)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to clone Faiss GPU index to CPU");
        }
        faiss::write_index(cpu_index.get(), index_path.string().c_str());
        break;
    }
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported ImageSearch Faiss backend");
    }
}

fs::path mappingPathFromIndex(const fs::path &index_path)
{
    return index_path.string() + ".paths.txt";
}

fs::path metadataPathFromIndex(const fs::path &index_path)
{
    return index_path.string() + ".meta.txt";
}

void savePathMapping(const fs::path &mapping_path, const std::vector<fs::path> &image_paths)
{
    std::ofstream output(mapping_path);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open mapping file: %s",
                             mapping_path.string().c_str());
    }

    for (const auto &image_path : image_paths)
    {
        output << image_path.generic_string() << "\n";
    }
}

std::vector<fs::path> loadPathMapping(const fs::path &mapping_path)
{
    std::ifstream input(mapping_path);
    if (!input)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open mapping file: %s",
                             mapping_path.string().c_str());
    }

    std::vector<fs::path> image_paths;
    std::string           line;
    while (std::getline(input, line))
    {
        if (!line.empty())
        {
            image_paths.emplace_back(line);
        }
    }
    return image_paths;
}

void saveMetadata(const fs::path &metadata_path, const fs::path &gallery_dir, const std::string &model_name,
                  const std::string &feature_name, const ImageSearchConfig &config)
{
    std::ofstream output(metadata_path);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open metadata file: %s",
                             metadata_path.string().c_str());
    }

    output << "model=" << model_name << "\n";
    output << "feature=" << feature_name << "\n";
    output << "gallery_dir=" << fs::absolute(gallery_dir).generic_string() << "\n";
    output << "preprocess_backend=" << preprocessBackendName(config.preprocess_backend) << "\n";
    output << "norm=" << featureNormName(config.norm) << "\n";
    output << "faiss_backend=" << faissBackendName(config.faiss_backend) << "\n";
    output << "index_storage=" << indexStorageName(config.index_storage) << "\n";
    output << "disk_build_batch_size=" << config.disk_build_batch_size << "\n";
    output << "index_kind=" << indexKindName(config) << "\n";
}

std::unordered_map<std::string, std::string> loadMetadata(const fs::path &metadata_path)
{
    std::ifstream input(metadata_path);
    if (!input)
    {
        return {};
    }

    std::unordered_map<std::string, std::string> metadata;
    std::string                                 line;
    while (std::getline(input, line))
    {
        const auto separator = line.find('=');
        if (separator == std::string::npos)
        {
            continue;
        }
        metadata.emplace(line.substr(0, separator), line.substr(separator + 1));
    }
    return metadata;
}

bool metadataValueEquals(const std::unordered_map<std::string, std::string> &metadata, const std::string &key,
                         const std::string &expected)
{
    const auto it = metadata.find(key);
    return it == metadata.end() || it->second == expected;
}

bool metadataConfigValueEquals(const std::unordered_map<std::string, std::string> &metadata, const std::string &key,
                               const std::string &expected, const std::string &legacy_default)
{
    const auto it = metadata.find(key);
    return it == metadata.end() ? expected == legacy_default : it->second == expected;
}

bool metadataIndexKindEquals(const std::unordered_map<std::string, std::string> &metadata,
                             const ImageSearchConfig &config)
{
    const auto it = metadata.find("index_kind");
    if (useCpuDiskIndex(config))
    {
        return it != metadata.end() && it->second == indexKindName(config);
    }
    return it == metadata.end() || it->second == indexKindName(config);
}

bool existingIndexMatchesConfig(const fs::path &index_path, const fs::path &gallery_dir,
                                const std::string &model_name, const std::string &feature_name,
                                const ImageSearchConfig &config)
{
    const fs::path metadata_path = metadataPathFromIndex(index_path);
    if (!fs::exists(metadata_path))
    {
        return isDefaultConfig(config);
    }

    const auto metadata = loadMetadata(metadata_path);
    if (metadata.empty())
    {
        return isDefaultConfig(config);
    }

    return metadataValueEquals(metadata, "model", model_name)
        && metadataValueEquals(metadata, "feature", feature_name)
        && metadataValueEquals(metadata, "gallery_dir", fs::absolute(gallery_dir).generic_string())
        && metadataConfigValueEquals(metadata, "preprocess_backend",
                                     preprocessBackendName(config.preprocess_backend), "cpu")
        && metadataConfigValueEquals(metadata, "norm", featureNormName(config.norm), "l2")
        && metadataConfigValueEquals(metadata, "faiss_backend", faissBackendName(config.faiss_backend), "cpu")
        && metadataConfigValueEquals(metadata, "index_storage", indexStorageName(config.index_storage), "ram")
        && metadataIndexKindEquals(metadata, config);
}

} // namespace

namespace priv {

class ImageSearchFeatureExtractor
{
public:
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

        model_ = irt::model::CreateModel(model_name_, std::move(model_config));
        if (!model_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create model: %s",
                                 model_name_.c_str());
        }

        model_->setLogLevel(nvinfer1::ILogger::Severity::kINFO);
        model_->buildOrLoad(weights_file.string());

        output_name_ = model_->modelConfig().outputTensorNames().front();
        output_dims_ = model_->tensorShape(output_name_);
        output_type_ = model_->tensorDataType(output_name_);
        if (output_type_ != nvinfer1::DataType::kFLOAT)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Expected float feature tensor for %s, got unsupported data type",
                                 output_name_.c_str());
        }

        const auto &input_tensor_names = model_->modelConfig().inputTensorNames();
        if (input_tensor_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageSearch model must expose at least one input tensor");
        }

        const auto input_shape = model_->tensorShape(input_tensor_names.front());
        if (input_shape.nbDims != 4 || input_shape.d[0] != 1 || input_shape.d[1] != 3 || input_shape.d[2] <= 0
            || input_shape.d[3] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "ImageSearch expects input shape 1x3xHxW, got %s",
                                 dimsToCsv(input_shape).c_str());
        }

        input_height_ = static_cast<int>(input_shape.d[2]);
        input_width_  = static_cast<int>(input_shape.d[3]);
        feature_dim_  = elementCount(output_dims_);
        device_input_.resize(elementCount(input_shape), nvinfer1::DataType::kFLOAT);
        device_output_.resize(feature_dim_, nvinfer1::DataType::kFLOAT);
    }

    ~ImageSearchFeatureExtractor()
    {
        // Release TensorRT context/engine before CUDA buffers.
        model_.reset();
    }

    int featureDim() const noexcept
    {
        return static_cast<int>(feature_dim_);
    }

    std::vector<float> extract(const fs::path &image_path)
    {
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                 image_path.string().c_str());
        }

        std::vector<float> input_data;
        switch (config_.preprocess_backend)
        {
        case ImageSearchPreprocessBackend::CPU:
        {
            const auto preprocessed
                = irt::model::ImageNetUtil::preprocess(image, cv::Size(input_width_, input_height_));
            input_data = irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);
            break;
        }
        case ImageSearchPreprocessBackend::GPU:
            throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                                 "ImageSearch GPU preprocessing is not implemented");
        }

        std::vector<float>  feature(feature_dim_);
        std::vector<void *> buffers{device_input_.data(), device_output_.data()};
        const auto          stream = model_->resolveExecutionStream();

        checkCuda(cudaMemcpyAsync(device_input_.data(), input_data.data(), input_data.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(H2D input)");
        model_->forwardFeatures(buffers, stream, true);
        checkCuda(cudaMemcpyAsync(feature.data(), device_output_.data(), feature.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync(D2H feature)");
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(feature extraction)");
        normalizeFeature(feature, config_.norm);
        return feature;
    }

private:
    std::string                         model_name_;
    std::string                         feature_name_;
    ImageSearchConfig                   config_{};
    std::unique_ptr<irt::model::IModel> model_;
    std::string                         output_name_;
    nvinfer1::Dims                      output_dims_{};
    nvinfer1::DataType                  output_type_{};
    int                                 input_height_{224};
    int                                 input_width_{224};
    size_t                              feature_dim_{0};
    DeviceBuffer                        device_input_;
    DeviceBuffer                        device_output_;
};

} // namespace priv

namespace {

FaissIndexBundle buildCpuOnDiskIndex(const std::vector<fs::path> &gallery_images,
                                     priv::ImageSearchFeatureExtractor &extractor,
                                     const fs::path &index_path,
                                     const ImageSearchConfig &config)
{
    savePathMapping(mappingPathFromIndex(index_path), gallery_images);

    FaissIndexBundle bundle;
    bundle.index = priv::buildCpuOnDiskIvfFlatIndex(gallery_images.size(), extractor.featureDim(), index_path,
                                                    config.disk_build_batch_size,
                                                    [&](size_t index) {
                                                        return extractor.extract(gallery_images[index]);
                                                    });
    return bundle;
}

FaissIndexBundle buildIndex(const std::vector<fs::path> &gallery_images,
                            priv::ImageSearchFeatureExtractor &extractor,
                            const fs::path &index_path,
                            const ImageSearchConfig &config)
{
    if (useCpuDiskIndex(config))
    {
        return buildCpuOnDiskIndex(gallery_images, extractor, index_path, config);
    }

    auto bundle = createEmptyFaissIndex(extractor.featureDim(), config.faiss_backend);

    for (const auto &image_path : gallery_images)
    {
        const auto feature = extractor.extract(image_path);
        bundle.index->add(1, feature.data());
    }

    writeIndex(*bundle.index, config.faiss_backend, index_path);
    savePathMapping(mappingPathFromIndex(index_path), gallery_images);
    return bundle;
}

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

    auto image_paths = loadPathMapping(mappingPathFromIndex(index_path));
    if (static_cast<faiss::idx_t>(image_paths.size()) != cpu_index->ntotal)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Index size (%lld) does not match path mapping size (%zu)",
                             static_cast<long long>(cpu_index->ntotal), image_paths.size());
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
    bundle.image_paths = std::move(image_paths);
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
                                    const fs::path &index_file, bool rebuild_index)
{
    weights_file_ = weights_file;
    gallery_dir_  = gallery_dir;
    index_path_ = index_file.empty()
                    ? ImageSearch::defaultIndexPath(gallery_dir_, config_.model_name, config_.feature_name)
                    : index_file;

    if (!rebuild_index && fs::exists(index_path_) && fs::exists(mappingPathFromIndex(index_path_))
        && existingIndexMatchesConfig(index_path_, gallery_dir_, config_.model_name, config_.feature_name, config_))
    {
        auto loaded = loadIndex(index_path_, config_);
        index_.reset();
        faiss_gpu_resources_.reset();
        faiss_gpu_resources_ = std::move(loaded.gpu_resources);
        index_               = std::move(loaded.index);
        gallery_images_      = std::move(loaded.image_paths);
        feature_dim_         = static_cast<int>(index_->d);
        return;
    }

    auto extractor = std::make_unique<priv::ImageSearchFeatureExtractor>(config_.model_name, config_.feature_name,
                                                                         weights_file_, config_);
    gallery_images_ = ImageSearch::collectGalleryImages(gallery_dir_);
    if (!index_path_.parent_path().empty())
    {
        fs::create_directories(index_path_.parent_path());
    }
    auto built = buildIndex(gallery_images_, *extractor, index_path_, config_);
    index_.reset();
    faiss_gpu_resources_.reset();
    faiss_gpu_resources_ = std::move(built.gpu_resources);
    index_               = std::move(built.index);
    feature_dim_         = extractor->featureDim();
    extractor_           = std::move(extractor);
    saveMetadata(metadataPathFromIndex(index_path_), gallery_dir_, config_.model_name, config_.feature_name,
                 config_);
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
        if (indices[i] < 0 || static_cast<size_t>(indices[i]) >= gallery_images_.size())
        {
            continue;
        }
        results.push_back({distances[i], gallery_images_[static_cast<size_t>(indices[i])]});
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

std::vector<fs::path> ImageSearch::Impl::galleryImages() const
{
    return gallery_images_;
}

int ImageSearch::Impl::featureDim() const noexcept
{
    return feature_dim_;
}

void ImageSearch::Impl::ensureExtractor()
{
    if (!extractor_)
    {
        extractor_ = std::make_unique<priv::ImageSearchFeatureExtractor>(config_.model_name, config_.feature_name,
                                                                         weights_file_, config_);
        feature_dim_ = extractor_->featureDim();
    }
}

} // namespace irt::features
