/**
 * @file ImageFeatureExtractor.cpp
 * @brief 图像特征抽取器实现。
 */

#include "ImageFeatureExtractor.hpp"

#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <utility>

namespace fs = std::filesystem;

namespace irt::features::priv {

using irt::model::checkCuda;
using irt::model::dimsToCsv;
using irt::model::elementCount;

ImageFeatureExtractor::ImageFeatureExtractor(std::string model_name, std::string feature_name,
                                             const fs::path &weights_file, ImageSearchConfig config)
    : model_name_(std::move(model_name))
    , feature_name_(std::move(feature_name))
    , config_(config)
{
    validateFeatureSearchConfig(config_, "ImageFeatureExtractor");

    auto model_config = std::make_unique<irt::model::IModelConfig>();
    model_config->setFeatureTensorNames({feature_name_});
    model_config->setOutputTensorNames({feature_name_});
    model_config->setFeatureOnly(true);
    model_config->setBackend(config_.model_backend);
    model_config->setDevice(config_.model_device);
    model_config->setDeviceId(config_.model_device_id);
    model_config->setPrecision(config_.model_precision);
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
    if (usesTensorRtModelBackend(config_))
    {
        irt::model::setCudaDevice(config_.model_device_id);
    }

    const auto input_tensor_names = model_->ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
    if (input_tensor_names.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor model must expose at least one input tensor");
    }

    input_name_            = input_tensor_names.front();
    const auto input_shape = model_->tensorShape(input_name_);
    const auto input_type  = model_->tensorDataType(input_tensor_names.front());
    if (input_type != nvinfer1::DataType::kFLOAT)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageFeatureExtractor expects float32 input tensor");
    }
    input_shape_ = resolveInputShape(input_shape);

    const auto output_tensor_names = model_->ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
    if (output_tensor_names.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor model must expose at least one output tensor");
    }
    output_name_ = output_tensor_names.front();
    output_dims_ = model_->tensorShape(output_name_);
    output_type_ = model_->tensorDataType(output_name_);
    if (output_type_ != nvinfer1::DataType::kFLOAT)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Expected float feature tensor for %s, got unsupported data type", output_name_.c_str());
    }
    if (output_dims_.nbDims <= 0 || output_dims_.d[0] != input_shape_.d[0])
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor output must preserve model batch dimension");
    }

    max_batch_size_            = static_cast<size_t>(input_shape_.d[0]);
    input_height_              = static_cast<int>(input_shape_.d[2]);
    input_width_               = static_cast<int>(input_shape_.d[3]);
    input_elements_per_sample_ = elementCount(input_shape_) / max_batch_size_;
    feature_dim_               = elementCount(output_dims_) / max_batch_size_;
    if (usesTensorRtModelBackend(config_))
    {
        irt::model::setCudaDevice(config_.model_device_id);
        device_input_.resize(max_batch_size_ * input_elements_per_sample_, nvinfer1::DataType::kFLOAT);
        device_output_.resize(max_batch_size_ * feature_dim_, nvinfer1::DataType::kFLOAT);
    }
}

ImageFeatureExtractor::~ImageFeatureExtractor()
{
    // 先释放模型上下文，再释放 CUDA 缓冲区，避免后端仍持有执行资源。
    model_.reset();
}

int ImageFeatureExtractor::featureDim() const noexcept
{
    return static_cast<int>(feature_dim_);
}

int ImageFeatureExtractor::inputWidth() const noexcept
{
    return input_width_;
}

int ImageFeatureExtractor::inputHeight() const noexcept
{
    return input_height_;
}

size_t ImageFeatureExtractor::maxBatchSize() const noexcept
{
    return max_batch_size_;
}

nvinfer1::Dims ImageFeatureExtractor::featureTensorShape() const noexcept
{
    return output_dims_;
}

std::vector<float> ImageFeatureExtractor::extract(const fs::path &image_path)
{
    const std::vector<fs::path> image_paths{image_path};
    return extractBatch(image_paths, 0, 1);
}

std::vector<float> ImageFeatureExtractor::extractBatch(const std::vector<fs::path> &image_paths, size_t begin,
                                                       size_t count)
{
    if (count == 0)
    {
        return {};
    }
    if (begin > image_paths.size() || count > image_paths.size() - begin)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature batch range is invalid");
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

    auto tensor = extractFeatureTensorBatch(image_paths, begin, count);
    if (tensor.data.size() != count * feature_dim_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature output size does not match flattened batch");
    }

    for (size_t i = 0; i < count; ++i)
    {
        normalizeFeature(tensor.data.data() + i * feature_dim_, feature_dim_, config_.norm);
    }
    return std::move(tensor.data);
}

std::vector<float> ImageFeatureExtractor::extractBatch(const std::vector<fs::path> &image_paths,
                                                       const std::vector<size_t>   &indices)
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
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature batch index is invalid");
        }
        selected_images.push_back(image_paths[index]);
    }
    return extractBatch(selected_images, 0, selected_images.size());
}

FeatureTensorBatch ImageFeatureExtractor::extractFeatureTensorBatch(const std::vector<fs::path> &image_paths,
                                                                    size_t begin, size_t count)
{
    if (count == 0)
    {
        return {};
    }
    if (begin > image_paths.size() || count > image_paths.size() - begin)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature tensor batch range is invalid");
    }
    if (count > max_batch_size_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Feature tensor batch size exceeds model max batch size");
    }

    auto       input_batch     = preprocessBatch(image_paths, begin, count);
    if (usesTensorRtModelBackend(config_))
    {
        irt::model::setCudaDevice(config_.model_device_id);
    }
    const auto output_dims     = setRuntimeBatchSize(count);
    const auto output_elements = elementCount(output_dims);
    if (output_elements != count * feature_dim_)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature output size does not match runtime batch");
    }

    std::vector<float> features(output_elements);
    if (usesTensorRtModelBackend(config_))
    {
        std::vector<void *> buffers{device_input_.data(), device_output_.data()};
        const auto          stream = model_->resolveExecutionStream();

        checkCuda(cudaMemcpyAsync(device_input_.data(), input_batch.input_data.data(),
                                  input_batch.input_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(H2D input)");
        model_->forwardFeatures(buffers, stream, true);
        checkCuda(cudaMemcpyAsync(features.data(), device_output_.data(), features.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync(D2H feature)");
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(feature extraction)");
    }
    else
    {
        std::vector<void *> buffers{input_batch.input_data.data(), features.data()};
        model_->forwardFeatures(buffers, nullptr, false);
    }

    FeatureTensorBatch output;
    output.data           = std::move(features);
    output.dims           = output_dims;
    output.original_sizes = std::move(input_batch.image_sizes);
    return output;
}

nvinfer1::Dims ImageFeatureExtractor::resolveInputShape(nvinfer1::Dims input_shape)
{
    if (input_shape.nbDims != 4 || input_shape.d[1] != 3 || input_shape.d[2] <= 0 || input_shape.d[3] <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor expects input shape Nx3xHxW, got %s",
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
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor expects input shape Nx3xHxW, got %s",
                             dimsToCsv(input_shape).c_str());
    }

    const auto runtime_batch = static_cast<size_t>(input_shape.d[0]);
    if (usesTensorRtModelBackend(config_))
    {
        if (runtime_batch != config_.model_batch_size)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "TensorRT model batch does not match configured model batch size");
        }
        return input_shape;
    }

    if (runtime_batch != 1 || config_.model_batch_size != 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Graph backends require dynamic batch to use model batch size %zu; "
                             "export the ONNX/OpenVINO model with dynamic batch",
                             config_.model_batch_size);
    }
    model_->setTensorShape(input_name_, input_shape);
    return model_->tensorShape(input_name_);
}

ImageFeatureExtractor::PreprocessedBatch ImageFeatureExtractor::preprocessBatch(
    const std::vector<fs::path> &image_paths, size_t begin, size_t count) const
{
    PreprocessedBatch batch;
    batch.input_data.reserve(count * input_elements_per_sample_);
    batch.image_sizes.reserve(count);

    for (size_t i = 0; i < count; ++i)
    {
        const auto &image_path = image_paths[begin + i];
        cv::Mat     image      = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                 image_path.string().c_str());
        }
        batch.image_sizes.push_back(ImageSize{image.cols, image.rows});

        switch (config_.preprocess_backend)
        {
        case ImageSearchPreprocessBackend::CPU:
        {
            const auto preprocessed
                = irt::model::ImageNetUtil::preprocess(image, cv::Size(input_width_, input_height_));
            auto single = irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);
            batch.input_data.insert(batch.input_data.end(), single.begin(), single.end());
            break;
        }
        case ImageSearchPreprocessBackend::GPU:
            throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                                 "ImageFeatureExtractor GPU preprocessing is not implemented");
        }
    }

    return batch;
}

nvinfer1::Dims ImageFeatureExtractor::setRuntimeBatchSize(size_t batch_size)
{
    auto dims = input_shape_;
    dims.d[0] = static_cast<int32_t>(batch_size);
    model_->setTensorShape(input_name_, dims);

    auto output_dims = model_->tensorShape(output_name_);
    if (output_dims.nbDims <= 0 || output_dims.d[0] != dims.d[0])
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Feature output must preserve runtime batch dimension");
    }
    return output_dims;
}

} // namespace irt::features::priv
