/**
 * @file EngineExample.hpp
 * @brief Engine sample and benchmark 的代码定义模型配置与 DAG。
 *
 * 这里刻意不读取 YAML：示例的模型输入、动态 batch 和预处理图都由 C++ API
 * 显式声明。应用接入时可直接以这些函数中的 PipelineBuilder 调用为起点，按需
 * 重定义自己的不可变 PipelinePlan。
 */

#pragma once

#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/BuiltinOperators.hpp>
#include <inferrt/engine/EngineConfig.hpp>
#include <inferrt/engine/Pipeline.hpp>

#include <opencv2/core.hpp>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string_view>
#include <utility>

namespace irt::sample::engine {

inline bool usesLetterbox(const std::string_view model_name)
{
    return model_name == "yolov8n";
}

inline irt::engine::EngineConfig makeConfig(const std::string_view example, std::filesystem::path engine_file,
                                             const int device_id)
{
    irt::engine::EngineConfig config;
    config.engine_file = std::move(engine_file);
    config.device_id   = device_id;

    if (example == "resnet18")
    {
        config.model_name     = "resnet18";
        config.input_width    = 224;
        config.input_height   = 224;
        config.min_batch_size = 1;
        config.opt_batch_size = 4;
        config.max_batch_size = 8;
        config.max_wait       = std::chrono::microseconds(2000);
        config.execution_slots = 3;
        config.queue_capacity  = 64;
    }
    else if (example == "yolov8n")
    {
        config.model_name     = "yolov8n";
        config.input_width    = 640;
        config.input_height   = 640;
        config.min_batch_size = 1;
        config.opt_batch_size = 4;
        config.max_batch_size = 8;
        config.max_wait       = std::chrono::microseconds(2000);
        config.execution_slots = 3;
        config.queue_capacity  = 64;
        config.mean            = {0.0F, 0.0F, 0.0F};
        config.stddev          = {1.0F, 1.0F, 1.0F};
    }
    else if (example == "dinov2_vits14")
    {
        config.model_name          = "dinov2_vits14";
        config.input_width         = 518;
        config.input_height        = 518;
        config.output_tensor_names = {"x_norm_clstoken"};
        config.feature_tensor_names = {"x_norm_clstoken"};
        config.feature_only         = true;
        config.execution_slots      = 2;
        config.queue_capacity       = 32;
    }
    else
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Unknown --example: %.*s (expected resnet18, yolov8n, or dinov2_vits14)",
                             static_cast<int>(example.size()), example.data());
    }

    config.validate();
    return config;
}

inline std::shared_ptr<const irt::engine::PipelinePlan>
makePipeline(const irt::engine::EngineConfig &config, const cv::Size source_size, const bool cuda_preprocess)
{
    using namespace irt::engine;

    if (source_size.width <= 0 || source_size.height <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Input image size must be positive");
    }

    const auto makeTensor = [](const TensorDataType data_type, const TensorLayout layout, const MemoryKind memory_kind,
                               const int width, const int height, const int channels)
    {
        return TensorDesc{data_type, layout, memory_kind, width, height, channels};
    };

    auto registry = std::make_shared<OperatorRegistry>();
    registerBuiltinOperators(*registry);

    PipelineBuilder builder;
    builder
        .addTensor("model_input", makeTensor(TensorDataType::F32, TensorLayout::NCHW, MemoryKind::DEVICE,
                                               config.input_width, config.input_height, config.input_channels))
        .setModelInput("model_input");

    if (!cuda_preprocess)
    {
        builder.addTensor("host_input", makeTensor(TensorDataType::F32, TensorLayout::NCHW, MemoryKind::HOST,
                                                     config.input_width, config.input_height, config.input_channels));
        builder.addNode("cpu.image_to_tensor",
                        {{}, {"host_input"}, CpuImageToTensorOptions{config.mean, config.stddev,
                                                                        usesLetterbox(config.model_name)}});
        builder.addNode("cuda.upload", {{"host_input"}, {"model_input"}, {}});
        return builder.build(std::move(registry));
    }

    auto source = makeTensor(TensorDataType::U8, TensorLayout::HWC, MemoryKind::HOST, source_size.width,
                             source_size.height, config.input_channels);
    builder.addTensor("host_image", source);
    source.memory_kind = MemoryKind::DEVICE;
    builder.addTensor("device_image", source);
    builder.addNode("cpu.copy_image", {{}, {"host_image"}, {}});
    builder.addNode("cuda.upload", {{"host_image"}, {"device_image"}, {}});

    if (usesLetterbox(config.model_name))
    {
        builder.addNode("cvcuda.letterbox", {{"device_image"}, {"model_input"}, {}});
    }
    else
    {
        const auto resized = makeTensor(TensorDataType::U8, TensorLayout::HWC, MemoryKind::DEVICE,
                                        config.input_width, config.input_height, config.input_channels);
        builder.addTensor("resized_bgr", resized).addTensor("rgb", resized);
        builder.addNode("cvcuda.resize", {{"device_image"}, {"resized_bgr"}, ResizeOptions{}});
        builder.addNode("cvcuda.cvt_color", {{"resized_bgr"}, {"rgb"}, CvtColorOptions{}});
        builder.addNode("cvcuda.normalize", {{"rgb"}, {"model_input"}, NormalizeOptions{config.mean, config.stddev}});
    }
    return builder.build(std::move(registry));
}

} // namespace irt::sample::engine
