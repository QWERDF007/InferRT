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
        : output_(config.outputs.front())
    {
    }

    OperatorContract contract() const override
    {
        return {ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 0, 1, false, false, 0};
    }

    void execute(OperatorContext &context) override
    {
        if (context.image == nullptr || context.image->type() != CV_8UC3 || !context.image->isContinuous())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.copy_image requires a continuous CV_8UC3 input image");
        }
        auto &destination = output(context, output_);
        if (destination.desc.memory_kind != MemoryKind::HOST || destination.desc.data_type != TensorDataType::U8
            || destination.desc.layout != TensorLayout::HWC || context.image->cols != destination.desc.width
            || context.image->rows != destination.desc.height || context.image->channels() != destination.desc.channels)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.copy_image input does not match its declared host HWC tensor");
        }
        std::memcpy(destination.dataForRequest(context.request_index), context.image->data,
                    destination.bytes_per_request);
    }

private:
    std::string output_;
};

class CpuCopyTensorOperator final : public IOperator
{
public:
    explicit CpuCopyTensorOperator(const NodeConfig &config)
        : output_(config.outputs.front())
        , options_(parameters<CpuCopyTensorOptions>(config, "cpu.copy_tensor"))
    {
        if (options_.input_name.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.copy_tensor requires a non-empty input_name");
        }
    }

    OperatorContract contract() const override
    {
        return {ExecutionKind::CPU, PipelineStage::CPU_PREPROCESS, 0, 1, false, false, 0};
    }

    void execute(OperatorContext &context) override
    {
        if (context.inputs == nullptr)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "cpu.copy_tensor requires named submit inputs");
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
        : output_(config.outputs.front())
        , options_(parameters<CpuImageToTensorOptions>(config, "cpu.image_to_tensor"))
    {
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
            || destination.desc.layout != TensorLayout::NCHW || destination.desc.channels != 3)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cpu.image_to_tensor requires a host float32 NCHW output with three channels");
        }

        cv::Mat bgr;
        if (context.image->channels() == 3)
        {
            bgr = *context.image;
        }
        else if (context.image->channels() == 1)
        {
            cv::cvtColor(*context.image, bgr, cv::COLOR_GRAY2BGR);
        }
        else if (context.image->channels() == 4)
        {
            cv::cvtColor(*context.image, bgr, cv::COLOR_BGRA2BGR);
        }
        else
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported image channel count: %d",
                                 context.image->channels());
        }

        cv::Mat rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

        cv::Mat resized;
        if (options_.letterbox)
        {
            const double   scale = std::min(static_cast<double>(destination.desc.width) / rgb.cols,
                                            static_cast<double>(destination.desc.height) / rgb.rows);
            const cv::Size resized_size(
                std::min(destination.desc.width, std::max(1, static_cast<int>(std::round(rgb.cols * scale)))),
                std::min(destination.desc.height, std::max(1, static_cast<int>(std::round(rgb.rows * scale)))));
            cv::resize(rgb, resized, resized_size, 0.0, 0.0, cv::INTER_LINEAR);

            cv::Mat letterboxed(destination.desc.height, destination.desc.width, CV_8UC3, cv::Scalar(114, 114, 114));
            const cv::Rect content((destination.desc.width - resized.cols) / 2,
                                   (destination.desc.height - resized.rows) / 2, resized.cols, resized.rows);
            resized.copyTo(letterboxed(content));
            resized = std::move(letterboxed);
        }
        else
        {
            cv::resize(rgb, resized, cv::Size(destination.desc.width, destination.desc.height), 0.0, 0.0,
                       cv::INTER_LINEAR);
        }
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        auto        *tensor = static_cast<float *>(destination.dataForRequest(context.request_index));
        const size_t plane  = static_cast<size_t>(destination.desc.width) * destination.desc.height;
        for (int y = 0; y < destination.desc.height; ++y)
        {
            const auto *row = resized.ptr<cv::Vec3f>(y);
            for (int x = 0; x < destination.desc.width; ++x)
            {
                const size_t offset = static_cast<size_t>(y) * destination.desc.width + x;
                for (int channel = 0; channel < 3; ++channel)
                {
                    tensor[static_cast<size_t>(channel) * plane + offset]
                        = (row[x][channel] - options_.mean[static_cast<size_t>(channel)])
                        / options_.stddev[static_cast<size_t>(channel)];
                }
            }
        }
    }

private:
    std::string             output_;
    CpuImageToTensorOptions options_;
};

class UploadOperator final : public IOperator
{
public:
    explicit UploadOperator(const NodeConfig &config)
        : input_(config.inputs.front())
        , output_(config.outputs.front())
    {
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

class ResizeOperator final : public IOperator
{
public:
    explicit ResizeOperator(const NodeConfig &config)
        : input_(config.inputs.front())
        , output_(config.outputs.front())
        , options_(parameters<ResizeOptions>(config, "cvcuda.resize"))
    {
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
            || source.desc.channels != destination.desc.channels)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.resize requires device uint8 HWC tensors with matching channels");
        }
        requireStatus(
            irt::cvcuda::resize<uint8_t>(static_cast<const uint8_t *>(source.dataForRequest(context.request_index)),
                                         static_cast<uint8_t *>(destination.dataForRequest(context.request_index)),
                                         cv::Size(source.desc.width, source.desc.height),
                                         cv::Size(destination.desc.width, destination.desc.height),
                                         source.desc.channels, options_.interpolation, context.stream),
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
        : input_(config.inputs.front())
        , output_(config.outputs.front())
        , options_(parameters<CvtColorOptions>(config, "cvcuda.cvt_color"))
    {
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
            || source.desc.width != destination.desc.width || source.desc.height != destination.desc.height
            || source.desc.channels != destination.desc.channels)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.cvt_color requires equal-sized device uint8 HWC tensors");
        }
        requireStatus(irt::cvcuda::cvtColor<uint8_t>(
                          static_cast<const uint8_t *>(source.dataForRequest(context.request_index)),
                          static_cast<uint8_t *>(destination.dataForRequest(context.request_index)),
                          cv::Size(source.desc.width, source.desc.height), options_.code, context.stream),
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
        : input_(config.inputs.front())
        , output_(config.outputs.front())
        , options_(parameters<NormalizeOptions>(config, "cvcuda.normalize"))
    {
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
            || source.desc.width != destination.desc.width || source.desc.height != destination.desc.height
            || source.desc.channels != destination.desc.channels)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.normalize requires matching device uint8 HWC and float32 NCHW tensors");
        }
        requireStatus(irt::cvcuda::normalize(static_cast<const uint8_t *>(source.dataForRequest(context.request_index)),
                                             static_cast<float *>(destination.dataForRequest(context.request_index)),
                                             cv::Size(source.desc.width, source.desc.height), source.desc.channels,
                                             options_.mean.data(), options_.stddev.data(), context.stream),
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
        : input_(config.inputs.front())
        , output_(config.outputs.front())
    {
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
            || source.desc.channels != destination.desc.channels)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "cvcuda.letterbox requires device uint8 HWC and float32 NCHW tensors");
        }
        requireStatus(irt::cvcuda::letterBox(static_cast<const uint8_t *>(source.dataForRequest(context.request_index)),
                                             static_cast<float *>(destination.dataForRequest(context.request_index)),
                                             cv::Size(source.desc.width, source.desc.height),
                                             cv::Size(destination.desc.width, destination.desc.height),
                                             source.desc.channels, context.stream),
                      "cvcuda.letterBox");
    }

private:
    std::string input_;
    std::string output_;
};

void registerOrThrow(OperatorRegistry &registry, const char *type, OperatorCreator creator)
{
    if (!registry.registerCreator(type, std::move(creator)))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION, "Failed to register pipeline operator: %s", type);
    }
}

} // namespace

void registerBuiltinOperators(OperatorRegistry &registry)
{
    registerOrThrow(registry, "cpu.copy_image",
                    [](const NodeConfig &config) { return std::make_unique<CpuImageCopyOperator>(config); });
    registerOrThrow(registry, "cpu.image_to_tensor",
                    [](const NodeConfig &config) { return std::make_unique<CpuImageToTensorOperator>(config); });
    registerOrThrow(registry, "cpu.copy_tensor",
                    [](const NodeConfig &config) { return std::make_unique<CpuCopyTensorOperator>(config); });
    registerOrThrow(registry, "cuda.upload",
                    [](const NodeConfig &config) { return std::make_unique<UploadOperator>(config); });
    registerOrThrow(registry, "cvcuda.resize",
                    [](const NodeConfig &config) { return std::make_unique<ResizeOperator>(config); });
    registerOrThrow(registry, "cvcuda.cvt_color",
                    [](const NodeConfig &config) { return std::make_unique<CvtColorOperator>(config); });
    registerOrThrow(registry, "cvcuda.normalize",
                    [](const NodeConfig &config) { return std::make_unique<NormalizeOperator>(config); });
    registerOrThrow(registry, "cvcuda.letterbox",
                    [](const NodeConfig &config) { return std::make_unique<LetterBoxOperator>(config); });
}

} // namespace irt::engine
