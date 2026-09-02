#include "EngineLegacyPipeline.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/BuiltinOperators.hpp>

#include <algorithm>
#include <utility>

namespace irt::engine::priv {
namespace {

TensorDesc tensor(const TensorDataType data_type, const TensorLayout layout, const MemoryKind memory_kind,
                  const int width, const int height, const int channels)
{
    return {data_type, layout, memory_kind, width, height, channels};
}

} // namespace

std::shared_ptr<const PipelinePlan> makeLegacyPipeline(const EngineConfig &config)
{
    const auto &preprocess = config.preprocessSpec();
    preprocess.validate();

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);

    PipelineBuilder builder;
    const auto model_input = tensor(TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE,
                                    preprocess.input_width, preprocess.input_height, preprocess.input_channels);
    builder.addTensor("model_input", model_input).setModelInput("model_input");

    if (preprocess.backend == irt::PreprocessBackend::CPU)
    {
        builder.addTensor("host_input", tensor(TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST,
                                               preprocess.input_width, preprocess.input_height,
                                               preprocess.input_channels));
        builder.addNode("cpu.image_to_tensor",
                        {
                            {},
                            {"host_input"},
                            CpuImageToTensorOptions{preprocess}
        });
        builder.addNode("cuda.upload", {{"host_input"}, {"model_input"}, {}});
        return builder.build(std::move(registry));
    }

    if (preprocess.source_width <= 0 || preprocess.source_height <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Legacy CUDA preprocessing requires source_width and source_height");
    }
    const auto geometry = irt::resolvePreprocessGeometry(preprocess, preprocess.source_width, preprocess.source_height);
    const int source_channels      = irt::colorChannels(preprocess.src_color);
    const int destination_channels = irt::colorChannels(preprocess.dst_color);
    auto source = tensor(TensorDataType::U8, TensorLayout::HWC, MemoryKind::HOST, preprocess.source_width,
                         preprocess.source_height, source_channels);
    builder.addTensor("host_image", source);
    source.memory_kind = MemoryKind::DEVICE;
    builder.addTensor("device_image", source);
    builder.addNode("cpu.copy_image", {{}, {"host_image"}, {}});
    builder.addNode("cuda.upload", {{"host_image"}, {"device_image"}, {}});

    if (preprocess.padding_mode == irt::PaddingMode::Letterbox)
    {
        std::string letterbox_input = "device_image";
        auto        letterbox_spec  = preprocess;
        if (source_channels != destination_channels)
        {
            builder.addTensor("converted_source",
                              tensor(TensorDataType::U8, TensorLayout::HWC, MemoryKind::DEVICE,
                                     preprocess.source_width, preprocess.source_height, destination_channels));
            builder.addNode("cvcuda.cvt_color", {{"device_image"}, {"converted_source"},
                                                  CvtColorOptions{preprocess}});
            letterbox_input              = "converted_source";
            letterbox_spec.src_color     = preprocess.dst_color;
            letterbox_spec.source_channels = destination_channels;
        }
        builder.addNode("cvcuda.letterbox", {{letterbox_input}, {"model_input"},
                                              LetterBoxOptions{letterbox_spec}});
        return builder.build(std::move(registry));
    }
    else
    {
        const auto resized = tensor(TensorDataType::U8, TensorLayout::HWC, MemoryKind::DEVICE,
                                    geometry.resized_width, geometry.resized_height, source_channels);
        const auto converted = tensor(TensorDataType::U8, TensorLayout::HWC, MemoryKind::DEVICE,
                                      preprocess.input_width, preprocess.input_height, destination_channels);
        builder.addTensor("resized_source", resized);
        if (preprocess.padding_mode == irt::PaddingMode::CenterCrop)
        {
            builder.addTensor("cropped_source",
                              tensor(TensorDataType::U8, TensorLayout::HWC, MemoryKind::DEVICE,
                                     preprocess.input_width, preprocess.input_height, source_channels));
        }
        builder.addTensor("converted", converted);
        builder.addNode("cvcuda.resize", {{"device_image"}, {"resized_source"}, ResizeOptions{preprocess}});
        std::string color_input = "resized_source";
        if (preprocess.padding_mode == irt::PaddingMode::CenterCrop)
        {
            builder.addNode("cuda.center_crop", {{"resized_source"}, {"cropped_source"},
                                                  CenterCropOptions{preprocess}});
            color_input = "cropped_source";
        }
        auto color_spec      = preprocess;
        color_spec.input_channels = destination_channels;
        builder.addNode("cvcuda.cvt_color", {{color_input}, {"converted"}, CvtColorOptions{color_spec}});
        builder.addNode("cvcuda.normalize", {{"converted"}, {"model_input"}, NormalizeOptions{preprocess}});
    }
    return builder.build(std::move(registry));
}

} // namespace irt::engine::priv
