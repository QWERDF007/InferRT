/**
 * @file ImageFeatureExtractor.cpp
 * @brief 图像特征抽取器实现。
 */

#include "ImageFeatureExtractor.hpp"

#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <utility>

namespace fs = std::filesystem;

namespace irt::features::priv {

using irt::model::checkCuda;

namespace {

cv::Mat loadImageForPreprocess(const fs::path &path, const irt::PreprocessSpec &spec)
{
    int flags = cv::IMREAD_COLOR;
    switch (spec.src_color)
    {
    case irt::ColorFormat::GRAY:
        flags = cv::IMREAD_GRAYSCALE;
        break;
    case irt::ColorFormat::BGRA:
    case irt::ColorFormat::RGBA:
        flags = cv::IMREAD_UNCHANGED;
        break;
    case irt::ColorFormat::BGR:
    case irt::ColorFormat::RGB:
        flags = cv::IMREAD_COLOR;
        break;
    }

    cv::Mat image = cv::imread(path.string(), flags);
    if (image.empty())
    {
        return image;
    }

    // OpenCV decodes colour images as BGR/BGRA.  Convert the decoded storage
    // to the declared source order before the shared preprocessing routine.
    if (spec.src_color == irt::ColorFormat::RGB && image.channels() == 3)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }
    else if (spec.src_color == irt::ColorFormat::RGBA && image.channels() == 4)
    {
        cv::cvtColor(image, image, cv::COLOR_BGRA2RGBA);
    }
    return image;
}

} // namespace

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
    model_config->setRuntime(config_.model_runtime);
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

    model_->setLogLevel(irt::model::LogLevel::Info);
    model_->buildOrLoad(weights_file.string());
    if (usesTensorRtModelBackend(config_))
    {
        irt::model::setCudaDevice(config_.model_runtime.deviceId());
    }

    const auto input_tensor_names = model_->ioTensorNames(irt::TensorIOMode::Input);
    if (input_tensor_names.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor model must expose at least one input tensor");
    }

    input_name_            = input_tensor_names.front();
    const auto input_shape = model_->tensorShape(input_name_);
    const auto input_type  = model_->tensorDataType(input_tensor_names.front());
    if (input_type != irt::TensorDataType::F32)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "ImageFeatureExtractor expects float32 input tensor");
    }
    input_shape_ = resolveInputShape(toTensorRtDims(input_shape));
    resolvePreprocessSpec();

    const auto output_tensor_names = model_->ioTensorNames(irt::TensorIOMode::Output);
    if (output_tensor_names.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor model must expose at least one output tensor");
    }
    output_name_ = output_tensor_names.front();
    output_dims_ = toTensorRtDims(model_->tensorShape(output_name_));
    output_type_ = model_->tensorDataType(output_name_);
    if (output_type_ != irt::TensorDataType::F32)
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
    input_elements_per_sample_ = tensorElementCount(input_shape_) / max_batch_size_;
    feature_dim_               = tensorElementCount(output_dims_) / max_batch_size_;
    if (usesTensorRtModelBackend(config_))
    {
        irt::model::setCudaDevice(config_.model_runtime.deviceId());
        device_input_.resize(irt::checkedSizeMul(max_batch_size_, input_elements_per_sample_,
                                                 "Feature device input elements"),
                             irt::TensorDataType::F32);
        device_output_.resize(irt::checkedSizeMul(max_batch_size_, feature_dim_,
                                                  "Feature device output elements"),
                              irt::TensorDataType::F32);
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

const irt::PreprocessSpec &ImageFeatureExtractor::preprocessSpec() const noexcept
{
    return preprocess_spec_;
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
        features.reserve(irt::checkedSizeMul(count, feature_dim_, "Feature batch output elements"));
        for (size_t offset = 0; offset < count; offset += max_batch_size_)
        {
            const size_t chunk_count = std::min(max_batch_size_, count - offset);
            auto         chunk       = extractBatch(image_paths, begin + offset, chunk_count);
            features.insert(features.end(), chunk.begin(), chunk.end());
        }
        return features;
    }

    auto tensor = extractFeatureTensorBatch(image_paths, begin, count);
    const auto expected_elements = irt::checkedSizeMul(count, feature_dim_, "Feature output elements");
    if (tensor.data.size() != expected_elements)
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
        irt::model::setCudaDevice(config_.model_runtime.deviceId());
    }
    const auto output_dims     = setRuntimeBatchSize(count);
    const auto output_elements = tensorElementCount(output_dims);
    const auto expected_elements = irt::checkedSizeMul(count, feature_dim_, "Feature runtime output elements");
    if (output_elements != expected_elements)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature output size does not match runtime batch");
    }

    std::vector<float> features(output_elements);
    const auto input_names  = model_->ioTensorNames(irt::TensorIOMode::Input);
    const auto output_names = model_->ioTensorNames(irt::TensorIOMode::Output);
    if (input_names.size() != 1 || output_names.size() != 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor requires one model input and one feature output");
    }
    if (usesTensorRtModelBackend(config_))
    {
        std::vector<irt::BufferView> buffers{
            irt::BufferView::fromBytes(device_input_.data(), device_input_.sizeBytes(), irt::MemoryKind::DEVICE,
                                       input_names.front()),
            irt::BufferView::fromBytes(device_output_.data(), device_output_.sizeBytes(), irt::MemoryKind::DEVICE,
                                       output_names.front())};
        const auto stream_handle = model_->resolveExecutionStream();
        const auto stream        = reinterpret_cast<cudaStream_t>(stream_handle);
        if (stream == nullptr)
        {
            throw irt::Exception(irt::Status::INVALID_OPERATION,
                                 "TensorRT feature extraction requires a valid CUDA stream");
        }

        checkCuda(cudaMemcpyAsync(device_input_.data(), input_batch.input_data.data(),
                                  irt::checkedSizeMul(input_batch.input_data.size(), sizeof(float),
                                                      "Feature H2D bytes"),
                                  cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync(H2D input)");
        model_->forwardFeatures(buffers, stream_handle, true);
        checkCuda(cudaMemcpyAsync(features.data(), device_output_.data(),
                                  irt::checkedSizeMul(features.size(), sizeof(float), "Feature D2H bytes"),
                                  cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync(D2H feature)");
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(feature extraction)");
    }
    else
    {
        std::vector<irt::BufferView> buffers{
            irt::BufferView::fromBytes(input_batch.input_data.data(),
                                       irt::checkedSizeMul(input_batch.input_data.size(), sizeof(float),
                                                           "Feature host input bytes"),
                                       irt::MemoryKind::HOST, input_names.front()),
            irt::BufferView::fromBytes(features.data(),
                                       irt::checkedSizeMul(features.size(), sizeof(float),
                                                           "Feature host output bytes"),
                                       irt::MemoryKind::HOST, output_names.front())};
        model_->forwardFeatures(buffers, 0, false);
    }

    FeatureTensorBatch output;
    output.data           = std::move(features);
    output.dims           = output_dims;
    output.original_sizes = std::move(input_batch.image_sizes);
    return output;
}

nvinfer1::Dims ImageFeatureExtractor::resolveInputShape(nvinfer1::Dims input_shape)
{
    if (input_shape.nbDims != 4 || input_shape.d[1] <= 0 || input_shape.d[2] <= 0 || input_shape.d[3] <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor expects input shape NxCxHxW, got %s",
                             tensorDimsToCsv(input_shape).c_str());
    }

    if (config_.preprocess.input_channels > 0 && input_shape.d[1] != config_.preprocess.input_channels)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor input channels (%d) do not match PreprocessSpec (%d)",
                             input_shape.d[1], config_.preprocess.input_channels);
    }

    if (input_shape.d[0] < 0)
    {
        input_shape.d[0] = static_cast<int32_t>(config_.model_batch_size);
        model_->setTensorShape(input_name_,
                               toCoreShape(input_shape));
        return toTensorRtDims(model_->tensorShape(input_name_));
    }

    if (input_shape.d[0] <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor expects input shape Nx3xHxW, got %s",
                              tensorDimsToCsv(input_shape).c_str());
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
    model_->setTensorShape(input_name_, toCoreShape(input_shape));
    return toTensorRtDims(model_->tensorShape(input_name_));
}

void ImageFeatureExtractor::resolvePreprocessSpec()
{
    preprocess_spec_ = config_.preprocess;
    if (preprocess_spec_.input_width == 0)
    {
        preprocess_spec_.input_width = input_shape_.d[3];
    }
    else if (preprocess_spec_.input_width != input_shape_.d[3])
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor input width (%d) does not match PreprocessSpec (%d)",
                             input_shape_.d[3], preprocess_spec_.input_width);
    }
    if (preprocess_spec_.input_height == 0)
    {
        preprocess_spec_.input_height = input_shape_.d[2];
    }
    else if (preprocess_spec_.input_height != input_shape_.d[2])
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "ImageFeatureExtractor input height (%d) does not match PreprocessSpec (%d)",
                             input_shape_.d[2], preprocess_spec_.input_height);
    }
    preprocess_spec_.validate();
    config_.preprocess = preprocess_spec_;
}

ImageFeatureExtractor::PreprocessedBatch ImageFeatureExtractor::preprocessBatch(
    const std::vector<fs::path> &image_paths, size_t begin, size_t count) const
{
    PreprocessedBatch batch;
    batch.input_data.resize(irt::checkedSizeMul(count, input_elements_per_sample_, "Feature preprocess elements"));
    batch.image_sizes.resize(count);

    if (config_.preprocess_backend == ImageSearchPreprocessBackend::GPU)
    {
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                             "ImageFeatureExtractor GPU preprocessing is not implemented");
    }

    // 图像解码、颜色转换、缩放和归一化互不依赖。按样本并行处理，写入各自固定的
    // 输入切片，保持原始顺序和数值路径不变；这条路径同时被 ImageSearch、ImageCluster
    // 和 RoiSearch 使用。
    std::atomic<size_t> failed_index{count};
    std::atomic<bool>   invalid_tensor{false};
    cv::parallel_for_(cv::Range(0, irt::checkedSizeToInt(count, "Feature preprocess batch")),
                      [&](const cv::Range &range)
    {
        for (int offset = range.start; offset < range.end; ++offset)
        {
            const auto index      = static_cast<size_t>(offset);
            const auto &image_path = image_paths[begin + index];
            cv::Mat     image       = loadImageForPreprocess(image_path, preprocess_spec_);
            if (image.empty())
            {
                size_t expected = failed_index.load(std::memory_order_relaxed);
                while (index < expected
                       && !failed_index.compare_exchange_weak(expected, index, std::memory_order_relaxed))
                {
                }
                continue;
            }

            batch.image_sizes[index] = ImageSize{image.cols, image.rows};
            const auto preprocessed = irt::model::ImageNetUtil::preprocess(image, preprocess_spec_);
            const auto single = irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);
            if (single.size() != input_elements_per_sample_)
            {
                invalid_tensor.store(true, std::memory_order_relaxed);
                continue;
            }
            std::copy(single.begin(), single.end(),
                      batch.input_data.begin()
                          + static_cast<std::ptrdiff_t>(irt::checkedSizeMul(index, input_elements_per_sample_,
                                                                            "Feature preprocess offset")));
        }
    });

    const auto failed = failed_index.load(std::memory_order_relaxed);
    if (failed < count)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                             image_paths[begin + failed].string().c_str());
    }
    if (invalid_tensor.load(std::memory_order_relaxed))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Preprocessed image tensor size does not match model input");
    }

    return batch;
}

nvinfer1::Dims ImageFeatureExtractor::setRuntimeBatchSize(size_t batch_size)
{
    auto dims = input_shape_;
    dims.d[0] = static_cast<int32_t>(batch_size);
    model_->setTensorShape(input_name_, toCoreShape(dims));

    auto output_dims = toTensorRtDims(model_->tensorShape(output_name_));
    if (output_dims.nbDims <= 0 || output_dims.d[0] != dims.d[0])
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Feature output must preserve runtime batch dimension");
    }
    return output_dims;
}

} // namespace irt::features::priv
