#include <cuda_runtime_api.h>
#include <cxxopts.hpp>
#include <inferrt/engine/EngineConfig.hpp>
#include <inferrt/engine/InferenceEngine.hpp>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/IModelConfig.hpp>
#include <inferrt/model/ModelFactory.hpp>
#include <inferrt/model/ModelRuntime.hpp>
#include <inferrt/model/Utils.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "EngineExample.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

irt::engine::InferenceResult runModel(const cv::Mat &image, const irt::engine::EngineConfig &config)
{
    auto model_config = std::make_unique<irt::model::IModelConfig>();
    model_config->setRuntime(
        {irt::model::ModelRuntime::Backend::TensorRT, irt::model::ModelRuntime::Device::GPU, config.device_id});
        model_config->setInputShape(irt::Shape{1, config.preprocess.input_channels, config.preprocess.input_height,
                                              config.preprocess.input_width});
    if (!config.output_tensor_names.empty())
    {
        model_config->setOutputTensorNames(config.output_tensor_names);
    }
    if (config.feature_only)
    {
        model_config->setFeatureTensorNames(config.feature_tensor_names);
        model_config->setFeatureOnly(true);
    }
    if (config.min_batch_size != config.max_batch_size)
    {
        model_config->setDynamicBatchRange(config.min_batch_size, config.opt_batch_size, config.max_batch_size);
    }

    auto model = irt::model::CreateModel(config.model_name, std::move(model_config));
    if (!model)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unknown model: %s", config.model_name.c_str());
    }
    model->load(config.engine_file.string());

    const auto inputs  = model->ioTensorNames(irt::TensorIOMode::Input);
    const auto outputs = model->ioTensorNames(irt::TensorIOMode::Output);
    if (inputs.size() != 1)
    {
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Sample supports exactly one input tensor");
    }

        model->setTensorShape(inputs.front(), irt::Shape{1, config.preprocess.input_channels,
                                                         config.preprocess.input_height,
                                                         config.preprocess.input_width});
    const auto input = irt::sample::engine::preprocessForDirect(config, image);

    irt::model::DeviceBuffer device_input(input.size(), irt::TensorDataType::F32);
    irt::model::checkCuda(
        cudaMemcpy(device_input.data(), input.data(), device_input.sizeBytes(), cudaMemcpyHostToDevice),
        "cudaMemcpy(input)");

    std::vector<irt::model::DeviceBuffer> device_outputs;
    std::vector<std::vector<float>>       host_outputs;
    std::vector<irt::BufferView>          buffers{
        irt::BufferView::fromBytes(device_input.data(), device_input.sizeBytes(), irt::MemoryKind::DEVICE,
                                   inputs.front())};
    device_outputs.reserve(outputs.size());
    host_outputs.reserve(outputs.size());
    for (const auto &name : outputs)
    {
        if (model->tensorDataType(name) != irt::TensorDataType::F32)
        {
            throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Sample supports float32 outputs only");
        }
        const size_t elements = model->tensorShape(name).elementCount();
        device_outputs.emplace_back(elements, irt::TensorDataType::F32);
        host_outputs.emplace_back(elements);
        buffers.push_back(irt::BufferView::fromBytes(device_outputs.back().data(), device_outputs.back().sizeBytes(),
                                                     irt::MemoryKind::DEVICE, name));
    }

    model->infer(buffers);

    irt::engine::InferenceResult result;
    for (size_t index = 0; index < outputs.size(); ++index)
    {
        irt::model::checkCuda(cudaMemcpy(host_outputs[index].data(), device_outputs[index].data(),
                                         device_outputs[index].sizeBytes(), cudaMemcpyDeviceToHost),
                              "cudaMemcpy(output)");
        result.outputs.emplace(outputs[index], std::move(host_outputs[index]));
    }
    return result;
}

float maxDifference(const irt::engine::InferenceResult &lhs, const irt::engine::InferenceResult &rhs)
{
    float difference = 0.0F;
    for (const auto &[name, values] : lhs.outputs)
    {
        const auto found = rhs.outputs.find(name);
        if (found == rhs.outputs.end() || found->second.size() != values.size())
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Output mismatch for tensor: %s", name.c_str());
        }
        for (size_t index = 0; index < values.size(); ++index)
        {
            difference = std::max(difference, std::abs(values[index] - found->second[index]));
        }
    }
    return difference;
}

void printSummary(const irt::engine::InferenceResult &result, const std::string &feature_name = "")
{
    std::cout << "request=" << result.request_id << ", outputs=" << result.outputs.size() << '\n';
    for (const auto &[name, values] : result.outputs)
    {
        std::cout << "  " << name << ": " << values.size() << " float values" << '\n';
    }
    if (!feature_name.empty())
    {
        const auto found = result.outputs.find(feature_name);
        if (found == result.outputs.end())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature output not found: %s",
                                 feature_name.c_str());
        }
        double squared_norm = 0.0;
        for (const float value : found->second)
        {
            squared_norm += static_cast<double>(value) * value;
        }
        std::cout << "  feature " << feature_name << ": l2_norm=" << std::sqrt(squared_norm) << '\n';
    }
}

} // namespace

int main(int argc, char **argv)
{
    cxxopts::Options options("inferrt_sample_engine", "Compare direct IModel and asynchronous engine execution");
    options.add_options()("example", "resnet18 | yolov8n | dinov2_vits14",
                          cxxopts::value<std::string>()->default_value("resnet18"))(
        "engine", "TensorRT engine file", cxxopts::value<std::string>())("image", "Input image",
                                                                             cxxopts::value<std::string>())(
        "device", "CUDA device id", cxxopts::value<int>()->default_value("0"))(
        "preprocess", "Engine preprocessing backend: cpu or cuda",
        cxxopts::value<std::string>()->default_value("cuda"))("mode", "model | engine | compare",
                                                               cxxopts::value<std::string>()->default_value("compare"))(
        "feature", "Optional feature output name to report", cxxopts::value<std::string>()->default_value(""))(
        "input-size", "Optional square input size override (e.g. 1024 for yolov8n@1024)",
        cxxopts::value<int>()->default_value("0"))("batch-min", "Optional dynamic batch profile minimum",
                                                   cxxopts::value<int>()->default_value("0"))(
        "batch-opt", "Optional dynamic batch profile optimum", cxxopts::value<int>()->default_value("0"))(
        "batch-max", "Optional dynamic batch profile maximum", cxxopts::value<int>()->default_value("0"))(
        "timeout-ms", "Synchronous engine timeout", cxxopts::value<int>()->default_value("10000"))("help", "Show help");

    const auto args = options.parse(argc, argv);
    if (args.count("help") || !args.count("engine") || !args.count("image"))
    {
        std::cout << options.help() << std::endl;
        return args.count("help") ? 0 : 1;
    }

    try
    {
        const auto image = cv::imread(args["image"].as<std::string>(), cv::IMREAD_UNCHANGED);
        if (image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load input image");
        }
        cv::Mat display_image = image;
        if (image.channels() == 1)
        {
            cv::cvtColor(image, display_image, cv::COLOR_GRAY2BGR);
        }
        else if (image.channels() == 4)
        {
            cv::cvtColor(image, display_image, cv::COLOR_BGRA2BGR);
        }
        if (!display_image.isContinuous())
        {
            display_image = display_image.clone();
        }

        const auto mode         = args["mode"].as<std::string>();
        const auto feature_name = args["feature"].as<std::string>();
        const auto preprocess_backend = args["preprocess"].as<std::string>();
        if (preprocess_backend != "cpu" && preprocess_backend != "cuda")
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--preprocess must be cpu or cuda");
        }
        const auto config = irt::sample::engine::makeConfig(args["example"].as<std::string>(),
                                                              args["engine"].as<std::string>(),
                                                              args["device"].as<int>(),
                                                              args["input-size"].as<int>(),
                                                              args["batch-min"].as<int>(),
                                                              args["batch-opt"].as<int>(),
                                                              args["batch-max"].as<int>());
        if (mode == "model")
        {
            printSummary(runModel(display_image, config), feature_name);
            return 0;
        }

        auto engine = irt::engine::InferenceEngine::create(
            config, irt::sample::engine::makePipeline(config, display_image.size(), preprocess_backend == "cuda"));
        engine->start();
        const auto engine_result = engine->infer(display_image, std::chrono::milliseconds(args["timeout-ms"].as<int>()));
        engine->shutdown();

        if (mode == "engine")
        {
            printSummary(engine_result, feature_name);
            return 0;
        }
        if (mode != "compare")
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unknown mode: %s", mode.c_str());
        }

        const auto model_result = runModel(display_image, config);
        std::cout << "max_abs_difference=" << maxDifference(model_result, engine_result) << '\n';
        printSummary(engine_result, feature_name);
    }
    catch (const std::exception &error)
    {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
