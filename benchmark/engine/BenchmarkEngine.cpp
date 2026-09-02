#include "EngineExample.hpp"

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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <future>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class DirectRunner
{
public:
    explicit DirectRunner(const irt::engine::EngineConfig &config)
        : config_(config)
    {
        auto model_config = std::make_unique<irt::model::IModelConfig>();
        model_config->setRuntime(
            {irt::model::ModelRuntime::Backend::TensorRT, irt::model::ModelRuntime::Device::GPU, config_.device_id});
        model_config->setInputShape(
            irt::Shape{1, config_.preprocess.input_channels, config_.preprocess.input_height,
                       config_.preprocess.input_width});
        if (!config_.output_tensor_names.empty())
        {
            model_config->setOutputTensorNames(config_.output_tensor_names);
        }
        if (config_.feature_only)
        {
            model_config->setFeatureTensorNames(config_.feature_tensor_names);
            model_config->setFeatureOnly(true);
        }
        if (config_.min_batch_size != config_.max_batch_size)
        {
            model_config->setDynamicBatchRange(config_.min_batch_size, config_.opt_batch_size, config_.max_batch_size);
        }

        model_ = irt::model::CreateModel(config_.model_name, std::move(model_config));
        if (!model_)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unknown model: %s", config_.model_name.c_str());
        }
        model_->load(config_.engine_file.string());

        const auto inputs = model_->ioTensorNames(irt::TensorIOMode::Input);
        outputs_          = model_->ioTensorNames(irt::TensorIOMode::Output);
        if (inputs.size() != 1 || outputs_.empty())
        {
            throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                                 "Benchmark supports one input and at least one output");
        }
        input_name_ = inputs.front();
        model_->setTensorShape(input_name_,
                               irt::Shape{1, config_.preprocess.input_channels, config_.preprocess.input_height,
                                          config_.preprocess.input_width});

        const size_t input_elements = irt::checkedSizeProduct(
            {static_cast<size_t>(config_.preprocess.input_channels),
             static_cast<size_t>(config_.preprocess.input_height), static_cast<size_t>(config_.preprocess.input_width)},
            "Engine benchmark input elements");
        device_input_.resize(input_elements, irt::TensorDataType::F32);
        buffers_.push_back(irt::BufferView::fromBytes(device_input_.data(), device_input_.sizeBytes(),
                                                      irt::MemoryKind::DEVICE, input_name_));

        device_outputs_.reserve(outputs_.size());
        host_outputs_.reserve(outputs_.size());
        for (const auto &name : outputs_)
        {
            if (model_->tensorDataType(name) != irt::TensorDataType::F32)
            {
                throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Benchmark supports float32 outputs only");
            }
            const size_t elements = irt::model::elementCount(model_->tensorShape(name));
            device_outputs_.emplace_back(elements, irt::TensorDataType::F32);
            host_outputs_.emplace_back(elements);
            buffers_.push_back(irt::BufferView::fromBytes(device_outputs_.back().data(),
                                                          device_outputs_.back().sizeBytes(), irt::MemoryKind::DEVICE,
                                                          name));
        }
    }

    void run(const cv::Mat &image)
    {
        const auto input = irt::sample::engine::preprocessForDirect(config_, image);
        irt::model::checkCuda(
            cudaMemcpy(device_input_.data(), input.data(), device_input_.sizeBytes(), cudaMemcpyHostToDevice),
            "cudaMemcpy(H2D)");
        model_->infer(buffers_);
        for (size_t index = 0; index < device_outputs_.size(); ++index)
        {
            irt::model::checkCuda(cudaMemcpy(host_outputs_[index].data(), device_outputs_[index].data(),
                                             device_outputs_[index].sizeBytes(), cudaMemcpyDeviceToHost),
                                  "cudaMemcpy(D2H)");
        }
    }

private:
    irt::engine::EngineConfig             config_;
    std::unique_ptr<irt::model::IModel>   model_;
    std::string                           input_name_;
    std::vector<std::string>              outputs_;
    irt::model::DeviceBuffer              device_input_;
    std::vector<irt::model::DeviceBuffer> device_outputs_;
    std::vector<std::vector<float>>       host_outputs_;
    std::vector<irt::BufferView>          buffers_;
};

template<typename Run>
double measure(int repeat, Run &&run)
{
    const auto start = std::chrono::steady_clock::now();
    for (int index = 0; index < repeat; ++index)
    {
        run();
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void printSyncResult(std::string_view name, int repeat, double elapsed_ms)
{
    const double latency = elapsed_ms / static_cast<double>(repeat);
    const double fps     = 1000.0 / latency;
    std::cout << std::left << std::setw(22) << name << " average_latency=" << std::fixed << std::setprecision(3)
              << latency << " ms, throughput=" << std::setprecision(2) << fps << " images/s" << '\n';
}

} // namespace

int main(int argc, char **argv)
{
    cxxopts::Options options("inferrt_benchmark_engine", "End-to-end IModel and Engine comparison");
    options.add_options()("example", "resnet18 | yolov8n | dinov2_vits14",
                          cxxopts::value<std::string>()->default_value("resnet18"))("engine", "TensorRT engine file",
                                                                                    cxxopts::value<std::string>())(
        "image", "Input image", cxxopts::value<std::string>())("device", "CUDA device id",
                                                               cxxopts::value<int>()->default_value("0"))(
        "warmup", "Warmup iterations", cxxopts::value<int>()->default_value("10"))(
        "repeat", "Measured iterations", cxxopts::value<int>()->default_value("100"))(
        "inflight", "Maximum asynchronous engine requests", cxxopts::value<int>()->default_value("8"))(
        "preprocess", "Engine preprocessing backend: cpu, cuda, or both",
        cxxopts::value<std::string>()->default_value("cpu"))("input-size", "Optional square input size override",
                                                             cxxopts::value<int>()->default_value("0"))(
        "batch-min", "Optional dynamic batch profile minimum", cxxopts::value<int>()->default_value("0"))(
        "batch-opt", "Optional dynamic batch profile optimum", cxxopts::value<int>()->default_value("0"))(
        "batch-max", "Optional dynamic batch profile maximum", cxxopts::value<int>()->default_value("0"))("help",
                                                                                                         "Show help");

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
        const int  warmup     = args["warmup"].as<int>();
        const int  repeat     = args["repeat"].as<int>();
        const int  inflight   = args["inflight"].as<int>();
        const auto preprocess = args["preprocess"].as<std::string>();
        const auto config     = irt::sample::engine::makeConfig(args["example"].as<std::string>(),
                                                                args["engine"].as<std::string>(), args["device"].as<int>(),
                                                                args["input-size"].as<int>(), args["batch-min"].as<int>(),
                                                                args["batch-opt"].as<int>(), args["batch-max"].as<int>());
        if (warmup < 0 || repeat <= 0 || inflight <= 0 || static_cast<size_t>(inflight) > config.queue_capacity)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "warmup/repeat/inflight are invalid for the engine queue capacity");
        }
        if (preprocess != "cpu" && preprocess != "cuda" && preprocess != "both")
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--preprocess must be cpu, cuda, or both");
        }

        std::cout << "Timed scope excludes image decode, model/engine loading, allocation, and warmup. "
                     "It includes preprocessing, transfers, inference, output materialization, and Engine queueing."
                  << '\n';

        DirectRunner direct(config);
        for (int index = 0; index < warmup; ++index)
        {
            direct.run(display_image);
        }
        printSyncResult("IModel direct (CPU)", repeat, measure(repeat, [&] { direct.run(display_image); }));

        const auto runEngine = [&](const bool cuda_preprocess, const std::string_view backend_name)
        {
            auto engine = irt::engine::InferenceEngine::create(
                config, irt::sample::engine::makePipeline(config, display_image.size(), cuda_preprocess));
            engine->start();
            for (int index = 0; index < warmup; ++index)
            {
                (void)engine->infer(display_image, std::chrono::seconds(30));
            }
            printSyncResult(std::string("Engine ") + std::string(backend_name) + " sync", repeat,
                            measure(repeat, [&] { (void)engine->infer(display_image, std::chrono::seconds(30)); }));

            const auto async_start = std::chrono::steady_clock::now();
            std::deque<std::pair<std::future<irt::engine::InferenceResult>, std::chrono::steady_clock::time_point>>
                                pending;
            std::vector<double> completion_latencies_ms;
            completion_latencies_ms.reserve(static_cast<size_t>(repeat));
            for (int index = 0; index < repeat; ++index)
            {
                const auto submitted = std::chrono::steady_clock::now();
                pending.emplace_back(engine->submit(display_image), submitted);
                if (pending.size() >= static_cast<size_t>(inflight))
                {
                    pending.front().first.get();
                    const auto completed = std::chrono::steady_clock::now();
                    completion_latencies_ms.push_back(
                        std::chrono::duration<double, std::milli>(completed - pending.front().second).count());
                    pending.pop_front();
                }
            }
            while (!pending.empty())
            {
                pending.front().first.get();
                const auto completed = std::chrono::steady_clock::now();
                completion_latencies_ms.push_back(
                    std::chrono::duration<double, std::milli>(completed - pending.front().second).count());
                pending.pop_front();
            }
            const auto async_end = std::chrono::steady_clock::now();
            std::sort(completion_latencies_ms.begin(), completion_latencies_ms.end());
            const auto percentile = [&completion_latencies_ms](double quantile)
            {
                const size_t index
                    = static_cast<size_t>(std::ceil(quantile * static_cast<double>(completion_latencies_ms.size())))
                    - 1;
                return completion_latencies_ms[index];
            };
            const double elapsed_ms = std::chrono::duration<double, std::milli>(async_end - async_start).count();
            std::cout << std::left << std::setw(22) << std::string("Engine ") + std::string(backend_name) + " async"
                      << " p50=" << std::fixed << std::setprecision(3) << percentile(0.50)
                      << " ms, p95=" << percentile(0.95) << " ms, p99=" << percentile(0.99)
                      << " ms, wall_throughput=" << std::setprecision(2)
                      << 1000.0 * static_cast<double>(repeat) / elapsed_ms << " images/s" << '\n';
            const auto metrics = engine->metrics();
            std::cout << "  queue_high_watermark=" << metrics.queued_high_watermark
                      << ", prepare/gpu/post_queue_high_watermark=" << metrics.prepare_queue_high_watermark << '/'
                      << metrics.gpu_queue_high_watermark << '/' << metrics.postprocess_queue_high_watermark
                      << ", pinned_input/output_high_watermark=" << metrics.pinned_input_high_watermark << '/'
                      << metrics.pinned_output_high_watermark << ", completed=" << metrics.completed_requests
                      << ", failed=" << metrics.failed_requests << ", cancelled=" << metrics.cancelled_requests
                      << ", dropped=" << metrics.dropped_requests << '\n';
            const auto average = [](const irt::engine::StageLatencySnapshot &stage)
            {
                return stage.samples == 0 ? 0.0
                                          : static_cast<double>(stage.total_microseconds) / stage.samples / 1000.0;
            };
            std::cout << "  stage_avg_ms(queue/cpu_pre/gpu/cpu_post)=" << std::fixed << std::setprecision(3)
                      << average(metrics.queue_latency) << '/' << average(metrics.cpu_preprocess_latency) << '/'
                      << average(metrics.gpu_latency) << '/' << average(metrics.cpu_postprocess_latency)
                      << ", device_arena_bytes=" << metrics.device_arena_bytes << '\n';
            engine->shutdown();
        };

        if (preprocess == "cpu" || preprocess == "both")
        {
            runEngine(false, "CPU");
        }
        if (preprocess == "cuda" || preprocess == "both")
        {
            runEngine(true, "CUDA");
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
