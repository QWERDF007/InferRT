#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpCvtColor.h>
#include <inferrt/cvcuda/OpLetterBox.h>
#include <inferrt/cvcuda/OpNormalize.h>
#include <inferrt/cvcuda/OpResize.h>
#include <inferrt/engine/BuiltinOperators.hpp>
#include <inferrt/model/Utils.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace irt::engine {
namespace {

const TensorView &input(const OperatorContext &context, const std::string &name)
{
    const auto found = context.tensors.find(name);
    if (found == context.tensors.end())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline input tensor is unavailable: %s",
                             name.c_str());
    }
    return found->second;
}

TensorView &output(OperatorContext &context, const std::string &name)
{
    const auto found = context.tensors.find(name);
    if (found == context.tensors.end())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Pipeline output tensor is unavailable: %s",
                             name.c_str());
    }
    return found->second;
}

void requireStatus(const IRTStatus status, const char *name)
{
    if (status != IRT_SUCCESS)
    {
        throw irt::Exception(static_cast<irt::Status>(status), "%s failed", name);
    }
}

int preprocessInterpolation(const irt::Interpolation interpolation)
{
    switch (interpolation)
    {
    case irt::Interpolation::Nearest:
        return cv::INTER_NEAREST;
    case irt::Interpolation::Linear:
        return cv::INTER_LINEAR;
    case irt::Interpolation::Cubic:
        return cv::INTER_CUBIC;
    case irt::Interpolation::Area:
        return cv::INTER_AREA;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported preprocessing interpolation");
}

int preprocessColorConversion(const irt::ColorFormat source, const irt::ColorFormat destination)
{
    if (source == destination)
    {
        return -1;
    }
    using irt::ColorFormat;
    switch (source)
    {
    case ColorFormat::BGR:
        switch (destination)
        {
        case ColorFormat::RGB:  return cv::COLOR_BGR2RGB;
        case ColorFormat::GRAY: return cv::COLOR_BGR2GRAY;
        case ColorFormat::BGRA: return cv::COLOR_BGR2BGRA;
        case ColorFormat::RGBA: return cv::COLOR_BGR2RGBA;
        default:                break;
        }
        break;
    case ColorFormat::RGB:
        switch (destination)
        {
        case ColorFormat::BGR:  return cv::COLOR_RGB2BGR;
        case ColorFormat::GRAY: return cv::COLOR_RGB2GRAY;
        case ColorFormat::BGRA: return cv::COLOR_RGB2BGRA;
        case ColorFormat::RGBA: return cv::COLOR_RGB2RGBA;
        default:                break;
        }
        break;
    case ColorFormat::GRAY:
        switch (destination)
        {
        case ColorFormat::BGR:  return cv::COLOR_GRAY2BGR;
        case ColorFormat::RGB:  return cv::COLOR_GRAY2RGB;
        case ColorFormat::BGRA: return cv::COLOR_GRAY2BGRA;
        case ColorFormat::RGBA: return cv::COLOR_GRAY2RGBA;
        default:                break;
        }
        break;
    case ColorFormat::BGRA:
        switch (destination)
        {
        case ColorFormat::BGR:  return cv::COLOR_BGRA2BGR;
        case ColorFormat::RGB:  return cv::COLOR_BGRA2RGB;
        case ColorFormat::GRAY: return cv::COLOR_BGRA2GRAY;
        case ColorFormat::RGBA: return cv::COLOR_BGRA2RGBA;
        default:                break;
        }
        break;
    case ColorFormat::RGBA:
        switch (destination)
        {
        case ColorFormat::BGR:  return cv::COLOR_RGBA2BGR;
        case ColorFormat::RGB:  return cv::COLOR_RGBA2RGB;
        case ColorFormat::GRAY: return cv::COLOR_RGBA2GRAY;
        case ColorFormat::BGRA: return cv::COLOR_RGBA2BGRA;
        default:                break;
        }
        break;
    }
    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported preprocessing color conversion");
}

template<typename T>
const T &parameters(const NodeConfig &config, const char *operator_type)
{
    try
    {
        return std::any_cast<const T &>(config.parameters);
    }
    catch (const std::bad_any_cast &)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "%s received invalid C++ parameters", operator_type);
    }
}

class CpuImageCopyOperator final : public IOperator
{
public:
    explicit CpuImageCopyOperator(const NodeConfig &config)
    {
        if (!config.inputs.empty() || config.outputs.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.copy_image requires 0 inputs and exactly 1 output");
        }
        output_ = config.outputs.front();
    }

    OperatorContract contract() const override
    {
        return {ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 0, 1, false, false, 0};
    }

    void execute(OperatorContext &context) override
    {
        if (context.image == nullptr || context.image->empty() || context.image->depth() != CV_8U)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.copy_image requires an 8-bit input image");
        }
        auto &destination = output(context, output_);
        if (destination.desc.memory_kind != MemoryKind::HOST || destination.desc.data_type != TensorDataType::U8
            || destination.desc.layout != TensorLayout::HWC || context.image->cols != destination.desc.width()
            || context.image->rows != destination.desc.height() || context.image->channels() != destination.desc.channels())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.copy_image input does not match its declared host HWC tensor");
        }
        const size_t row_bytes = irt::checkedSizeMul(static_cast<size_t>(context.image->cols),
                                                     static_cast<size_t>(context.image->channels()),
                                                     "cpu.copy_image row bytes");
        const size_t expected_bytes = irt::checkedSizeMul(row_bytes, static_cast<size_t>(context.image->rows),
                                                          "cpu.copy_image bytes");
        if (destination.bytes_per_request != expected_bytes)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.copy_image destination expects %zu bytes, got %zu", expected_bytes,
                                 destination.bytes_per_request);
        }
        auto *destination_data = destination.dataForRequest(context.request_index);
        if (destination_data == nullptr && expected_bytes != 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.copy_image destination buffer is unavailable");
        }
        for (int row = 0; row < context.image->rows; ++row)
        {
            std::memcpy(static_cast<uint8_t *>(destination_data) + static_cast<size_t>(row) * row_bytes,
                        context.image->ptr(row), row_bytes);
        }
    }

private:
    std::string output_;
};

class CpuCopyTensorOperator final : public IOperator
{
public:
    explicit CpuCopyTensorOperator(const NodeConfig &config)
        : options_(parameters<CpuCopyTensorOptions>(config, "cpu.copy_tensor"))
    {
        if (!config.inputs.empty() || config.outputs.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.copy_tensor requires 0 inputs and exactly 1 output");
        }
        if (options_.input_name.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.copy_tensor requires a non-empty input_name");
        }
        output_ = config.outputs.front();
    }

    OperatorContract contract() const override
    {
        return {ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 0, 1, false, false, 0};
    }

    void execute(OperatorContext &context) override
    {
        if (context.inputs == nullptr)
        {
            throw irt::Exception(irt::Status::INVALID_OPERATION, "cpu.copy_tensor requires named submit inputs");
        }
        const auto source = context.inputs->find(options_.input_name);
        if (source == context.inputs->end())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Named TensorRT input is missing: %s",
                                 options_.input_name.c_str());
        }
        auto &destination = output(context, output_);
        if (destination.desc.memory_kind != MemoryKind::HOST || destination.desc.data_type != TensorDataType::F32)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.copy_tensor requires a host float32 destination tensor");
        }
        const size_t element_count = destination.bytes_per_request / sizeof(float);
        if (source->second.size() != element_count)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Named input %s has %zu elements, expected %zu",
                                 options_.input_name.c_str(), source->second.size(), element_count);
        }
        std::memcpy(destination.dataForRequest(context.request_index), source->second.data(),
                    destination.bytes_per_request);
    }

private:
    std::string          output_;
    CpuCopyTensorOptions options_;
};

class CpuImageToTensorOperator final : public IOperator
{
public:
    explicit CpuImageToTensorOperator(const NodeConfig &config)
        : options_(parameters<CpuImageToTensorOptions>(config, "cpu.image_to_tensor"))
    {
        if (!config.inputs.empty() || config.outputs.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.image_to_tensor requires 0 inputs and exactly 1 output");
        }
        options_.preprocess.validate();
        if (options_.preprocess.output_layout != TensorLayout::NCHW
            || options_.preprocess.output_dtype != TensorDataType::F32)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.image_to_tensor requires float32 NCHW output");
        }
        output_ = config.outputs.front();
    }

    OperatorContract contract() const override
    {
        return {ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 0, 1, false, false, 0};
    }

    void execute(OperatorContext &context) override
    {
        if (context.image == nullptr || context.image->empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Input image is empty");
        }
        auto &destination = output(context, output_);
        if (destination.desc.memory_kind != MemoryKind::HOST || destination.desc.data_type != TensorDataType::F32
            || destination.desc.layout != options_.preprocess.output_layout
            || destination.desc.width() != options_.preprocess.input_width
            || destination.desc.height() != options_.preprocess.input_height
            || destination.desc.channels() != options_.preprocess.input_channels)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.image_to_tensor output does not match PreprocessSpec");
        }

        const auto processed = irt::model::ImageNetUtil::preprocessWithGeometry(*context.image, options_.preprocess);
        const auto expected_type = CV_MAKETYPE(CV_32F, options_.preprocess.input_channels);
        if (processed.image.type() != expected_type)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "PreprocessSpec produced unexpected image type: %d", processed.image.type());
        }
        if (destination.bytes_per_request % sizeof(float) != 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.image_to_tensor destination size is not float-aligned");
        }
        const auto tensor = irt::model::ImageNetUtil::imageToTensorCHW(processed.image);
        const size_t expected_elements = destination.bytes_per_request / sizeof(float);
        if (tensor.size() != expected_elements)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "PreprocessSpec produced %zu elements, destination expects %zu", tensor.size(),
                                 expected_elements);
        }
        auto *destination_data = static_cast<float *>(destination.dataForRequest(context.request_index));
        if (destination_data == nullptr && !tensor.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.image_to_tensor destination buffer is unavailable");
        }
        std::memcpy(destination_data, tensor.data(), destination.bytes_per_request);
    }

private:
    std::string             output_;
    CpuImageToTensorOptions options_;
};

class UploadOperator final : public IOperator
{
public:
    explicit UploadOperator(const NodeConfig &config)
    {
        if (config.inputs.size() != 1 || config.outputs.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cuda.upload requires exactly 1 input and 1 output");
        }
        input_  = config.inputs.front();
        output_ = config.outputs.front();
    }

    OperatorContract contract() const override
    {
        return {ExecutionKind::CUDA, PipelineStage::H2D, 1, 1, false, false, 0};
    }

    void execute(OperatorContext &context) override
    {
        const auto &source      = input(context, input_);
        auto       &destination = output(context, output_);
        if (source.desc.memory_kind != MemoryKind::HOST || destination.desc.memory_kind != MemoryKind::DEVICE
            || source.bytes_per_request != destination.bytes_per_request)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cuda.upload requires equally sized host and device tensors");
        }
        irt::model::checkCuda(cudaMemcpyAsync(destination.dataForRequest(context.request_index),
                                              source.dataForRequest(context.request_index), source.bytes_per_request,
                                              cudaMemcpyHostToDevice, context.stream),
                              "cudaMemcpyAsync(pipeline H2D)");
    }

private:
    std::string input_;
    std::string output_;
};

class CenterCropOperator final : public IOperator
{
public:
    explicit CenterCropOperator(const NodeConfig &config)
        : options_(parameters<CenterCropOptions>(config, "cuda.center_crop"))
    {
        if (config.inputs.size() != 1 || config.outputs.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cuda.center_crop requires exactly 1 input and 1 output");
        }
        options_.preprocess.validate();
        if (options_.preprocess.padding_mode != irt::PaddingMode::CenterCrop)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cuda.center_crop requires CenterCrop padding mode");
        }
        input_  = config.inputs.front();
        output_ = config.outputs.front();
    }

    OperatorContract contract() const override
    {
        return {ExecutionKind::CUDA, PipelineStage::CUDA_PREPROCESS, 1, 1, false, false, 0};
    }

    void execute(OperatorContext &context) override
    {
        const auto &source      = input(context, input_);
        auto       &destination = output(context, output_);
        if (source.desc.memory_kind != MemoryKind::DEVICE || destination.desc.memory_kind != MemoryKind::DEVICE
            || source.desc.data_type != TensorDataType::U8 || destination.desc.data_type != TensorDataType::U8
            || source.desc.layout != TensorLayout::HWC || destination.desc.layout != TensorLayout::HWC
            || source.desc.channels() != destination.desc.channels() || source.desc.width() < destination.desc.width()
            || source.desc.height() < destination.desc.height())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cuda.center_crop requires a device uint8 HWC source not smaller than the output");
        }

        const auto channels = static_cast<size_t>(source.desc.channels());
        const auto source_row_bytes
            = irt::checkedSizeMul(static_cast<size_t>(source.desc.width()), channels, "center crop source row");
        const auto destination_row_bytes
            = irt::checkedSizeMul(static_cast<size_t>(destination.desc.width()), channels,
                                  "center crop destination row");
        const auto source_bytes = irt::checkedSizeMul(source_row_bytes, static_cast<size_t>(source.desc.height()),
                                                      "center crop source bytes");
        const auto destination_bytes
            = irt::checkedSizeMul(destination_row_bytes, static_cast<size_t>(destination.desc.height()),
                                  "center crop destination bytes");
        if (source.bytes_per_request != source_bytes || destination.bytes_per_request != destination_bytes)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cuda.center_crop tensor capacity does not match its descriptor");
        }

        const auto geometry = irt::resolvePreprocessGeometry(options_.preprocess, source.desc.width(),
                                                             source.desc.height());
        if (geometry.resized_width != source.desc.width() || geometry.resized_height != source.desc.height()
            || geometry.crop_left + destination.desc.width() > source.desc.width()
            || geometry.crop_top + destination.desc.height() > source.desc.height()
            || destination.desc.width() != options_.preprocess.input_width
            || destination.desc.height() != options_.preprocess.input_height)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cuda.center_crop tensors do not match resolved PreprocessSpec geometry");
        }

        const auto *source_data = static_cast<const uint8_t *>(source.dataForRequest(context.request_index));
        auto       *destination_data = static_cast<uint8_t *>(destination.dataForRequest(context.request_index));
        if (source_data == nullptr || destination_data == nullptr)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cuda.center_crop tensor buffer is unavailable");
        }

        const auto left = static_cast<size_t>(geometry.crop_left);
        const auto top  = static_cast<size_t>(geometry.crop_top);
        const auto source_offset = irt::checkedSizeAdd(
            irt::checkedSizeMul(top, source_row_bytes, "center crop source row offset"),
            irt::checkedSizeMul(left, channels, "center crop source column offset"), "center crop source offset");
        irt::model::checkCuda(cudaMemcpy2DAsync(destination_data, destination_row_bytes, source_data + source_offset,
                                                source_row_bytes, destination_row_bytes,
                                                static_cast<size_t>(destination.desc.height()),
                                                cudaMemcpyDeviceToDevice, context.stream),
                              "cudaMemcpy2DAsync(center crop)");
    }

private:
    std::string        input_;
    std::string        output_;
    CenterCropOptions options_;
};

class ResizeOperator final : public IOperator
{
public:
    explicit ResizeOperator(const NodeConfig &config)
        : options_(parameters<ResizeOptions>(config, "cvcuda.resize"))
    {
        if (config.inputs.size() != 1 || config.outputs.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.resize requires exactly 1 input and 1 output");
        }
        options_.preprocess.validate();
        input_  = config.inputs.front();
        output_ = config.outputs.front();
    }

    OperatorContract contract() const override
    {
        return {ExecutionKind::CUDA, PipelineStage::CUDA_PREPROCESS, 1, 1, false, false, 0};
    }

    void execute(OperatorContext &context) override
    {
        const auto &source      = input(context, input_);
        auto       &destination = output(context, output_);
        if (source.desc.memory_kind != MemoryKind::DEVICE || destination.desc.memory_kind != MemoryKind::DEVICE
            || source.desc.data_type != TensorDataType::U8 || destination.desc.data_type != TensorDataType::U8
            || source.desc.layout != TensorLayout::HWC || destination.desc.layout != TensorLayout::HWC
            || source.desc.channels() != destination.desc.channels())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.resize requires device uint8 HWC tensors with matching channels");
        }
        requireStatus(
            irt::cvcuda::resize<uint8_t>(static_cast<const uint8_t *>(source.dataForRequest(context.request_index)),
                                         static_cast<uint8_t *>(destination.dataForRequest(context.request_index)),
                                         cv::Size(source.desc.width(), source.desc.height()),
                                         cv::Size(destination.desc.width(), destination.desc.height()),
                                          source.desc.channels(), preprocessInterpolation(options_.preprocess.interpolation),
                                          context.stream),
            "cvcuda.resize");
    }

private:
    std::string   input_;
    std::string   output_;
    ResizeOptions options_;
};

class CvtColorOperator final : public IOperator
{
public:
    explicit CvtColorOperator(const NodeConfig &config)
        : options_(parameters<CvtColorOptions>(config, "cvcuda.cvt_color"))
    {
        if (config.inputs.size() != 1 || config.outputs.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.cvt_color requires exactly 1 input and 1 output");
        }
        options_.preprocess.validate();
        input_  = config.inputs.front();
        output_ = config.outputs.front();
    }

    OperatorContract contract() const override
    {
        return {ExecutionKind::CUDA, PipelineStage::CUDA_PREPROCESS, 1, 1, false, false, 0};
    }

    void execute(OperatorContext &context) override
    {
        const auto &source      = input(context, input_);
        auto       &destination = output(context, output_);
        if (source.desc.memory_kind != MemoryKind::DEVICE || destination.desc.memory_kind != MemoryKind::DEVICE
            || source.desc.data_type != TensorDataType::U8 || destination.desc.data_type != TensorDataType::U8
            || source.desc.layout != TensorLayout::HWC || destination.desc.layout != TensorLayout::HWC
            || source.desc.width() != destination.desc.width() || source.desc.height() != destination.desc.height()
            || source.desc.channels() != irt::colorChannels(options_.preprocess.src_color)
            || destination.desc.channels() != irt::colorChannels(options_.preprocess.dst_color))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.cvt_color requires equal-sized device uint8 HWC tensors");
        }
        const int conversion = preprocessColorConversion(options_.preprocess.src_color, options_.preprocess.dst_color);
        if (source.desc.channels() != irt::colorChannels(options_.preprocess.src_color)
            || destination.desc.channels() != irt::colorChannels(options_.preprocess.dst_color))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.cvt_color tensor channels do not match PreprocessSpec");
        }
        if (conversion < 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.cvt_color requires distinct source and destination color formats");
        }
        requireStatus(irt::cvcuda::cvtColor<uint8_t>(
                          static_cast<const uint8_t *>(source.dataForRequest(context.request_index)),
                          static_cast<uint8_t *>(destination.dataForRequest(context.request_index)),
                          cv::Size(source.desc.width(), source.desc.height()), conversion, context.stream),
                      "cvcuda.cvtColor");
    }

private:
    std::string     input_;
    std::string     output_;
    CvtColorOptions options_;
};

class NormalizeOperator final : public IOperator
{
public:
    explicit NormalizeOperator(const NodeConfig &config)
        : options_(parameters<NormalizeOptions>(config, "cvcuda.normalize"))
    {
        if (config.inputs.size() != 1 || config.outputs.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.normalize requires exactly 1 input and 1 output");
        }
        options_.preprocess.validate();
        input_  = config.inputs.front();
        output_ = config.outputs.front();
    }

    OperatorContract contract() const override
    {
        return {ExecutionKind::CUDA, PipelineStage::CUDA_PREPROCESS, 1, 1, false, false, 0};
    }

    void execute(OperatorContext &context) override
    {
        const auto &source      = input(context, input_);
        auto       &destination = output(context, output_);
        if (source.desc.memory_kind != MemoryKind::DEVICE || destination.desc.memory_kind != MemoryKind::DEVICE
            || source.desc.data_type != TensorDataType::U8 || destination.desc.data_type != TensorDataType::F32
            || source.desc.layout != TensorLayout::HWC || destination.desc.layout != TensorLayout::NCHW
            || source.desc.width() != destination.desc.width() || source.desc.height() != destination.desc.height()
            || source.desc.width() != options_.preprocess.input_width
            || source.desc.height() != options_.preprocess.input_height
            || source.desc.channels() != destination.desc.channels()
            || source.desc.channels() != options_.preprocess.input_channels
            || source.desc.channels() != irt::colorChannels(options_.preprocess.dst_color))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.normalize requires matching device uint8 HWC and float32 NCHW tensors");
        }
        requireStatus(irt::cvcuda::normalize(static_cast<const uint8_t *>(source.dataForRequest(context.request_index)),
                                             static_cast<float *>(destination.dataForRequest(context.request_index)),
                                             cv::Size(source.desc.width(), source.desc.height()), source.desc.channels(),
                                             options_.preprocess.mean.data(), options_.preprocess.stddev.data(),
                                             options_.preprocess.scale, context.stream),
                      "cvcuda.normalize");
    }

private:
    std::string      input_;
    std::string      output_;
    NormalizeOptions options_;
};

class LetterBoxOperator final : public IOperator
{
public:
    explicit LetterBoxOperator(const NodeConfig &config)
    {
        if (config.inputs.size() != 1 || config.outputs.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.letterbox requires exactly 1 input and 1 output");
        }
        if (config.parameters.has_value())
        {
            options_ = parameters<LetterBoxOptions>(config, "cvcuda.letterbox");
        }
        input_  = config.inputs.front();
        output_ = config.outputs.front();
    }

    OperatorContract contract() const override
    {
        return {ExecutionKind::CUDA, PipelineStage::CUDA_PREPROCESS, 1, 1, false, false, 0};
    }

    void execute(OperatorContext &context) override
    {
        const auto &source      = input(context, input_);
        auto       &destination = output(context, output_);
        if (source.desc.memory_kind != MemoryKind::DEVICE || destination.desc.memory_kind != MemoryKind::DEVICE
            || source.desc.data_type != TensorDataType::U8 || destination.desc.data_type != TensorDataType::F32
            || source.desc.layout != TensorLayout::HWC || destination.desc.layout != TensorLayout::NCHW
            || source.desc.channels() != destination.desc.channels())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.letterbox requires device uint8 HWC and float32 NCHW tensors");
        }
        if (source.desc.channels() != irt::colorChannels(options_.preprocess.src_color)
            || destination.desc.channels() != irt::colorChannels(options_.preprocess.dst_color))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.letterbox tensor channels do not match PreprocessSpec");
        }
        requireStatus(irt::cvcuda::letterBox(
                          static_cast<const uint8_t *>(source.dataForRequest(context.request_index)),
                          static_cast<float *>(destination.dataForRequest(context.request_index)),
                          cv::Size(source.desc.width(), source.desc.height()),
                          cv::Size(destination.desc.width(), destination.desc.height()), source.desc.channels(),
                          options_.preprocess, context.stream),
                      "cvcuda.letterBox");
    }

private:
    std::string input_;
    std::string output_;
    LetterBoxOptions options_{};
};

void registerOrThrow(OperatorRegistry &registry, const char *type, OperatorContract contract,
                     OperatorCreator creator, OperatorValidator validator = nullptr)
{
    if (!registry.registerCreator(type, contract, std::move(creator), std::move(validator)))
    {
        throw irt::Exception(irt::Status::INVALID_OPERATION, "Failed to register pipeline operator: %s", type);
    }
}

} // namespace

void registerBuiltinOperators(OperatorRegistry &registry)
{
    registerOrThrow(
        registry, "cpu.copy_image",
        OperatorContract{ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 0, 1, false, false, 0},
        [](const NodeConfig &config) { return std::make_unique<CpuImageCopyOperator>(config); },
        [](const NodeConfig &config)
        {
        if (!config.inputs.empty() || config.outputs.size() != 1)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "cpu.copy_image requires 0 inputs and exactly 1 output");
            }
        });

    registerOrThrow(
        registry, "cpu.image_to_tensor",
        OperatorContract{ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 0, 1, false, false, 0},
        [](const NodeConfig &config) { return std::make_unique<CpuImageToTensorOperator>(config); },
        [](const NodeConfig &config)
        {
            if (!config.inputs.empty() || config.outputs.size() != 1)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "cpu.image_to_tensor requires 0 inputs and exactly 1 output");
            }
            if (config.parameters.has_value())
            {
                parameters<CpuImageToTensorOptions>(config, "cpu.image_to_tensor");
            }
        });

    registerOrThrow(
        registry, "cpu.copy_tensor",
        OperatorContract{ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 0, 1, false, false, 0},
        [](const NodeConfig &config) { return std::make_unique<CpuCopyTensorOperator>(config); },
        [](const NodeConfig &config)
        {
            if (!config.inputs.empty() || config.outputs.size() != 1)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "cpu.copy_tensor requires 0 inputs and exactly 1 output");
            }
            const auto &options = parameters<CpuCopyTensorOptions>(config, "cpu.copy_tensor");
            if (options.input_name.empty())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "cpu.copy_tensor requires a non-empty input_name");
            }
        });

    registerOrThrow(
        registry, "cuda.upload",
        OperatorContract{ExecutionKind::CUDA, PipelineStage::H2D, 1, 1, false, false, 0},
        [](const NodeConfig &config) { return std::make_unique<UploadOperator>(config); },
        [](const NodeConfig &config)
        {
            if (config.inputs.size() != 1 || config.outputs.size() != 1)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "cuda.upload requires exactly 1 input and 1 output");
            }
        });

    registerOrThrow(
        registry, "cuda.center_crop",
        OperatorContract{ExecutionKind::CUDA, PipelineStage::CUDA_PREPROCESS, 1, 1, false, false, 0},
        [](const NodeConfig &config) { return std::make_unique<CenterCropOperator>(config); },
        [](const NodeConfig &config)
        {
            if (config.inputs.size() != 1 || config.outputs.size() != 1)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "cuda.center_crop requires exactly 1 input and 1 output");
            }
            const auto &options = parameters<CenterCropOptions>(config, "cuda.center_crop");
            options.preprocess.validate();
            if (options.preprocess.padding_mode != irt::PaddingMode::CenterCrop)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "cuda.center_crop requires CenterCrop padding mode");
            }
        });

    registerOrThrow(
        registry, "cvcuda.resize",
        OperatorContract{ExecutionKind::CUDA, PipelineStage::CUDA_PREPROCESS, 1, 1, false, false, 0},
        [](const NodeConfig &config) { return std::make_unique<ResizeOperator>(config); },
        [](const NodeConfig &config)
        {
            if (config.inputs.size() != 1 || config.outputs.size() != 1)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "cvcuda.resize requires exactly 1 input and 1 output");
            }
            if (config.parameters.has_value())
            {
                parameters<ResizeOptions>(config, "cvcuda.resize");
            }
        });

    registerOrThrow(
        registry, "cvcuda.cvt_color",
        OperatorContract{ExecutionKind::CUDA, PipelineStage::CUDA_PREPROCESS, 1, 1, false, false, 0},
        [](const NodeConfig &config) { return std::make_unique<CvtColorOperator>(config); },
        [](const NodeConfig &config)
        {
            if (config.inputs.size() != 1 || config.outputs.size() != 1)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "cvcuda.cvt_color requires exactly 1 input and 1 output");
            }
            if (config.parameters.has_value())
            {
                parameters<CvtColorOptions>(config, "cvcuda.cvt_color");
            }
        });

    registerOrThrow(
        registry, "cvcuda.normalize",
        OperatorContract{ExecutionKind::CUDA, PipelineStage::CUDA_PREPROCESS, 1, 1, false, false, 0},
        [](const NodeConfig &config) { return std::make_unique<NormalizeOperator>(config); },
        [](const NodeConfig &config)
        {
            if (config.inputs.size() != 1 || config.outputs.size() != 1)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "cvcuda.normalize requires exactly 1 input and 1 output");
            }
            parameters<NormalizeOptions>(config, "cvcuda.normalize");
        });

    registerOrThrow(
        registry, "cvcuda.letterbox",
        OperatorContract{ExecutionKind::CUDA, PipelineStage::CUDA_PREPROCESS, 1, 1, false, false, 0},
        [](const NodeConfig &config) { return std::make_unique<LetterBoxOperator>(config); },
         [](const NodeConfig &config)
         {
             if (config.inputs.size() != 1 || config.outputs.size() != 1)
             {
                 throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                      "cvcuda.letterbox requires exactly 1 input and 1 output");
             }
             const auto &options = parameters<LetterBoxOptions>(config, "cvcuda.letterbox");
             options.preprocess.validate();
         });
}

} // namespace irt::engine
