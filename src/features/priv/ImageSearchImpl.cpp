#include "ImageSearchImpl.hpp"

#include <cuda_runtime_api.h>
#pragma warning(push)
#pragma warning(disable : 4244)
#include <faiss/IndexFlat.h>
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
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace irt::features {

namespace {

using DeviceBuffer = irt::model::DeviceBuffer;
using irt::model::checkCuda;
using irt::model::dimsToCsv;
using irt::model::elementCount;

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
                  const std::string &feature_name)
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
}

} // namespace

namespace priv {

class ImageSearchFeatureExtractor
{
public:
    ImageSearchFeatureExtractor(std::string model_name, std::string feature_name, const fs::path &weights_file)
        : model_name_(std::move(model_name))
        , feature_name_(std::move(feature_name))
    {
        auto config = std::make_unique<irt::model::IModelConfig>();
        config->setFeatureTensorNames({feature_name_});
        config->setOutputTensorNames({feature_name_});
        config->setFeatureOnly(true);

        model_ = irt::model::CreateModel(model_name_, std::move(config));
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

        const auto preprocessed = irt::model::ImageNetUtil::preprocess(image, cv::Size(input_width_, input_height_));
        const auto input_data   = irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);

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
        l2Normalize(feature);
        return feature;
    }

private:
    std::string                         model_name_;
    std::string                         feature_name_;
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

std::unique_ptr<faiss::Index> buildIndex(const std::vector<fs::path> &gallery_images,
                                         priv::ImageSearchFeatureExtractor &extractor,
                                         const fs::path &index_path)
{
    auto index = std::make_unique<faiss::IndexFlatIP>(extractor.featureDim());

    for (const auto &image_path : gallery_images)
    {
        const auto feature = extractor.extract(image_path);
        index->add(1, feature.data());
    }

    faiss::write_index(index.get(), index_path.string().c_str());
    savePathMapping(mappingPathFromIndex(index_path), gallery_images);
    return index;
}

std::pair<std::unique_ptr<faiss::Index>, std::vector<fs::path>> loadIndex(const fs::path &index_path)
{
    auto index = std::unique_ptr<faiss::Index>(faiss::read_index(index_path.string().c_str()));
    if (!index)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load Faiss index: %s",
                             index_path.string().c_str());
    }

    auto image_paths = loadPathMapping(mappingPathFromIndex(index_path));
    if (static_cast<faiss::idx_t>(image_paths.size()) != index->ntotal)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Index size (%lld) does not match path mapping size (%zu)",
                             static_cast<long long>(index->ntotal), image_paths.size());
    }

    return {std::move(index), std::move(image_paths)};
}

} // namespace

ImageSearch::Impl::Impl(std::string model_name, std::string feature_name)
    : model_name_(std::move(model_name))
    , feature_name_(std::move(feature_name))
{
    if (!irt::model::isSupportedModel(model_name_))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported model: %s", model_name_.c_str());
    }
    if (feature_name_.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "feature name must not be empty");
    }
}

ImageSearch::Impl::~Impl() = default;

void ImageSearch::Impl::buildOrLoad(const fs::path &weights_file, const fs::path &gallery_dir,
                                    const fs::path &index_file, bool rebuild_index)
{
    weights_file_ = weights_file;
    gallery_dir_  = gallery_dir;
    index_path_   = index_file.empty() ? ImageSearch::defaultIndexPath(gallery_dir_, model_name_, feature_name_)
                                       : index_file;

    if (!rebuild_index && fs::exists(index_path_) && fs::exists(mappingPathFromIndex(index_path_)))
    {
        auto loaded     = loadIndex(index_path_);
        index_          = std::move(loaded.first);
        gallery_images_ = std::move(loaded.second);
        feature_dim_    = static_cast<int>(index_->d);
        return;
    }

    auto extractor = std::make_unique<priv::ImageSearchFeatureExtractor>(model_name_, feature_name_, weights_file_);
    gallery_images_ = ImageSearch::collectGalleryImages(gallery_dir_);
    if (!index_path_.parent_path().empty())
    {
        fs::create_directories(index_path_.parent_path());
    }
    index_       = buildIndex(gallery_images_, *extractor, index_path_);
    feature_dim_ = extractor->featureDim();
    extractor_   = std::move(extractor);
    saveMetadata(metadataPathFromIndex(index_path_), gallery_dir_, model_name_, feature_name_);
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

const std::string &ImageSearch::Impl::modelName() const noexcept
{
    return model_name_;
}

const std::string &ImageSearch::Impl::featureName() const noexcept
{
    return feature_name_;
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
        extractor_   = std::make_unique<priv::ImageSearchFeatureExtractor>(model_name_, feature_name_, weights_file_);
        feature_dim_ = extractor_->featureDim();
    }
}

} // namespace irt::features
