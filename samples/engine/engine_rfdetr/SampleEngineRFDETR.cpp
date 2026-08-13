/**
 * @file SampleEngineRFDETR.cpp
 * @brief RF-DETR 高吞吐推理 sample：普通 IModel 与 InferenceEngine 异步流水线对比。
 *
 * 依赖一个已构建的动态 batch TensorRT engine（由 detection sample 或 IModel::save 生成），
 * 例如 1x3x1024x1024、profile 1/8/8。Engine 路径按 inflight 并发提交，输出 p50/p95/p99
 * 延迟、墙钟吞吐与队列/阶段指标，并解码 RF-DETR dets/labels 输出打印检测框。
 */

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
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using irt::model::checkCuda;
using irt::model::DeviceBuffer;
using irt::model::elementCount;

/** @brief 单个检测框，坐标为原图像素。 */
struct Detection
{
    float x1{0.0F};
    float y1{0.0F};
    float x2{0.0F};
    float y2{0.0F};
    float confidence{0.0F};
    int   class_id{0};
};

float sigmoid(float value)
{
    if (value >= 0.0F)
    {
        const float z = std::exp(-value);
        return 1.0F / (1.0F + z);
    }
    const float z = std::exp(value);
    return z / (1.0F + z);
}

float clampFloat(float value, float lower, float upper)
{
    return std::max(lower, std::min(value, upper));
}

float intersectionOverUnion(const Detection &a, const Detection &b)
{
    const float x1         = std::max(a.x1, b.x1);
    const float y1         = std::max(a.y1, b.y1);
    const float x2         = std::min(a.x2, b.x2);
    const float y2         = std::min(a.y2, b.y2);
    const float inter_area = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
    const float area_a     = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
    const float area_b     = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
    const float denom      = area_a + area_b - inter_area;
    return denom <= 0.0F ? 0.0F : inter_area / denom;
}

std::vector<Detection> nonMaximumSuppression(std::vector<Detection> detections, float threshold, int max_detections)
{
    std::sort(detections.begin(), detections.end(),
              [](const Detection &a, const Detection &b) { return a.confidence > b.confidence; });

    std::vector<Detection> kept;
    std::vector<bool>      removed(detections.size(), false);
    kept.reserve(std::min<int>(max_detections, static_cast<int>(detections.size())));

    for (size_t i = 0; i < detections.size() && static_cast<int>(kept.size()) < max_detections; ++i)
    {
        if (removed[i])
        {
            continue;
        }
        kept.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (!removed[j] && detections[i].class_id == detections[j].class_id
                && intersectionOverUnion(detections[i], detections[j]) > threshold)
            {
                removed[j] = true;
            }
        }
    }
    return kept;
}

/** @brief 从 Engine/IModel 结果解码 RF-DETR 检测框（boxes 归一化 cxcywh，labels 为 logits）。 */
std::vector<Detection> decodeRFDETR(const irt::engine::InferenceResult &result, const cv::Size &image_size,
                                    float conf_threshold, float nms_threshold, int max_detections)
{
    const auto dets_it   = result.outputs.find("dets");
    const auto labels_it = result.outputs.find("labels");
    if (dets_it == result.outputs.end() || labels_it == result.outputs.end())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Engine result is missing dets/labels outputs");
    }
    const auto &dets    = dets_it->second;
    const auto &labels  = labels_it->second;
    const int   queries = static_cast<int>(dets.size()) / 4;
    const int   classes = queries > 0 ? static_cast<int>(labels.size()) / queries : 0;
    if (queries <= 0 || classes <= 0 || static_cast<size_t>(queries) * 4 != dets.size()
        || static_cast<size_t>(queries) * classes != labels.size())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Unexpected RF-DETR output sizes: dets=%zu labels=%zu", dets.size(), labels.size());
    }

    std::vector<Detection> detections;
    for (int q = 0; q < queries; ++q)
    {
        const float cx = dets[static_cast<size_t>(q) * 4 + 0] * static_cast<float>(image_size.width);
        const float cy = dets[static_cast<size_t>(q) * 4 + 1] * static_cast<float>(image_size.height);
        const float w  = dets[static_cast<size_t>(q) * 4 + 2] * static_cast<float>(image_size.width);
        const float h  = dets[static_cast<size_t>(q) * 4 + 3] * static_cast<float>(image_size.height);

        Detection base;
        base.x1 = clampFloat(cx - 0.5F * w, 0.0F, static_cast<float>(image_size.width - 1));
        base.y1 = clampFloat(cy - 0.5F * h, 0.0F, static_cast<float>(image_size.height - 1));
        base.x2 = clampFloat(cx + 0.5F * w, 0.0F, static_cast<float>(image_size.width - 1));
        base.y2 = clampFloat(cy + 0.5F * h, 0.0F, static_cast<float>(image_size.height - 1));

        for (int cls = 0; cls < classes; ++cls)
        {
            const float confidence = sigmoid(labels[static_cast<size_t>(q) * classes + cls]);
            if (confidence < conf_threshold)
            {
                continue;
            }
            auto det       = base;
            det.confidence = confidence;
            det.class_id   = cls;
            detections.push_back(det);
        }
    }
    return nonMaximumSuppression(std::move(detections), nms_threshold, max_detections);
}

std::string className(int class_id, const std::vector<std::string> &labels)
{
    if (class_id >= 0 && static_cast<size_t>(class_id) < labels.size())
    {
        return labels[static_cast<size_t>(class_id)];
    }
    return "class_" + std::to_string(class_id);
}

void printDetections(const std::vector<Detection> &detections, const std::vector<std::string> &labels)
{
    std::cout << "Detections: " << detections.size() << '\n';
    for (size_t i = 0; i < detections.size(); ++i)
    {
        const auto &det = detections[i];
        std::cout << std::fixed << std::setprecision(4) << "  #" << i << " confidence=" << det.confidence
                  << " class=" << det.class_id << " (" << className(det.class_id, labels) << ") box=[" << det.x1 << ", "
                  << det.y1 << ", " << det.x2 << ", " << det.y2 << "]" << '\n';
    }
}

void drawDetections(const cv::Mat &image, const std::vector<Detection> &detections,
                    const std::vector<std::string> &labels, const std::filesystem::path &output_path)
{
    if (output_path.empty())
    {
        return;
    }
    cv::Mat visual = image.clone();
    for (const auto &det : detections)
    {
        const int  color_seed = det.class_id * 37;
        cv::Scalar color((color_seed * 3) % 255, (color_seed * 7 + 80) % 255, (color_seed * 11 + 160) % 255);
        cv::rectangle(visual, cv::Point(static_cast<int>(std::round(det.x1)), static_cast<int>(std::round(det.y1))),
                      cv::Point(static_cast<int>(std::round(det.x2)), static_cast<int>(std::round(det.y2))), color, 2);
        std::ostringstream text;
        text << className(det.class_id, labels) << ' ' << std::fixed << std::setprecision(2) << det.confidence;
        int            baseline  = 0;
        const cv::Size text_size = cv::getTextSize(text.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int      text_x    = static_cast<int>(std::round(det.x1));
        const int      text_y    = std::max(text_size.height + 4, static_cast<int>(std::round(det.y1)));
        cv::rectangle(
            visual,
            cv::Rect(text_x, text_y - text_size.height - 4, text_size.width + 4, text_size.height + baseline + 4),
            color, cv::FILLED);
        cv::putText(visual, text.str(), cv::Point(text_x + 2, text_y - 3), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    std::filesystem::create_directories(output_path.parent_path().empty() ? std::filesystem::path{"."}
                                                                          : output_path.parent_path());
    if (!cv::imwrite(output_path.string(), visual))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to write output image: %s",
                             output_path.string().c_str());
    }
    std::cout << "Saved detection image to: " << std::filesystem::absolute(output_path).string() << '\n';
}

std::vector<std::string> readLabelNames(const std::filesystem::path &path)
{
    if (path.empty())
    {
        return {};
    }
    std::ifstream      file(path);
    std::vector<std::string> names;
    std::string              line;
    while (std::getline(file, line))
    {
        if (!line.empty())
        {
            names.push_back(line);
        }
    }
    return names;
}

/** @brief 普通 IModel 同步路径（单图），作为端到端基线与数值参考。 */
class DirectRunner
{
public:
    DirectRunner(const irt::engine::EngineConfig &config)
        : config_(config)
    {
        auto model_config = std::make_unique<irt::model::IModelConfig>();
        model_config->setRuntime(
            {irt::model::ModelRuntime::Backend::TensorRT, irt::model::ModelRuntime::Device::GPU, config_.device_id});
        model_config->setInputShape(
            nvinfer1::Dims4{1, config_.input_channels, config_.input_height, config_.input_width});
        if (!config_.output_tensor_names.empty())
        {
            model_config->setOutputTensorNames(config_.output_tensor_names);
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

        const auto inputs = model_->ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
        outputs_          = model_->ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
        if (inputs.size() != 1 || outputs_.empty())
        {
            throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                                 "Sample supports one input and at least one output");
        }
        model_->setTensorShape(inputs.front(),
                               nvinfer1::Dims4{1, config_.input_channels, config_.input_height, config_.input_width});

        const size_t input_elements = static_cast<size_t>(config_.input_channels)
                                    * static_cast<size_t>(config_.input_height)
                                    * static_cast<size_t>(config_.input_width);
        device_input_.resize(input_elements, nvinfer1::DataType::kFLOAT);
        buffers_.push_back(device_input_.data());

        device_outputs_.reserve(outputs_.size());
        host_outputs_.reserve(outputs_.size());
        for (const auto &name : outputs_)
        {
            if (model_->tensorDataType(name) != nvinfer1::DataType::kFLOAT)
            {
                throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Sample supports float32 outputs only");
            }
            const size_t elements = elementCount(model_->tensorShape(name));
            device_outputs_.emplace_back(elements, nvinfer1::DataType::kFLOAT);
            host_outputs_.emplace_back(elements);
            buffers_.push_back(device_outputs_.back().data());
        }
    }

    irt::engine::InferenceResult run(const cv::Mat &image)
    {
        const auto input = irt::sample::engine::preprocessForDirect(config_, image);
        checkCuda(cudaMemcpy(device_input_.data(), input.data(), device_input_.sizeBytes(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(H2D)");
        model_->infer(buffers_);
        irt::engine::InferenceResult result;
        for (size_t index = 0; index < outputs_.size(); ++index)
        {
            checkCuda(cudaMemcpy(host_outputs_[index].data(), device_outputs_[index].data(),
                                 device_outputs_[index].sizeBytes(), cudaMemcpyDeviceToHost),
                      "cudaMemcpy(D2H)");
            result.outputs.emplace(outputs_[index], host_outputs_[index]);
        }
        return result;
    }

private:
    irt::engine::EngineConfig             config_;
    std::unique_ptr<irt::model::IModel>   model_;
    std::vector<std::string>              outputs_;
    irt::model::DeviceBuffer              device_input_;
    std::vector<irt::model::DeviceBuffer> device_outputs_;
    std::vector<std::vector<float>>       host_outputs_;
    std::vector<void *>                   buffers_;
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
    std::cout << std::left << std::setw(26) << name << " average_latency=" << std::fixed << std::setprecision(3)
              << latency << " ms, throughput=" << std::setprecision(2) << fps << " images/s" << '\n';
}

float maxOutputDifference(const irt::engine::InferenceResult &lhs, const irt::engine::InferenceResult &rhs)
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

} // namespace

int main(int argc, char **argv)
{
    cxxopts::Options options("inferrt_sample_engine_rfdetr",
                             "RF-DETR high-throughput InferenceEngine benchmark with detection decode");
    options.add_options()("engine", "TensorRT engine file (dynamic batch, e.g. profile 1/8/8)",
                          cxxopts::value<std::string>())("image", "Input image",
                                                         cxxopts::value<std::string>())(
        "labels", "Class label file", cxxopts::value<std::string>()->default_value(""))(
        "output", "Optional path to save image with boxes", cxxopts::value<std::string>()->default_value(""))(
        "device", "CUDA device id", cxxopts::value<int>()->default_value("0"))(
        "input-size", "Square network input size, must match the engine", cxxopts::value<int>()->default_value("1024"))(
        "batch-min", "Engine dynamic batch profile minimum", cxxopts::value<int>()->default_value("1"))(
        "batch-opt", "Engine dynamic batch profile optimum", cxxopts::value<int>()->default_value("8"))(
        "batch-max", "Engine dynamic batch profile maximum", cxxopts::value<int>()->default_value("8"))(
        "max-wait-us", "Dynamic scheduler batch wait window", cxxopts::value<int>()->default_value("2000"))(
        "slots", "Execution slots", cxxopts::value<size_t>()->default_value("1"))(
        "inflight", "Maximum asynchronous engine requests", cxxopts::value<int>()->default_value("8"))(
        "warmup", "Warmup iterations", cxxopts::value<int>()->default_value("10"))(
        "repeat", "Measured iterations", cxxopts::value<int>()->default_value("100"))(
        "preprocess", "Engine preprocessing backend: cpu, cuda, or both",
        cxxopts::value<std::string>()->default_value("both"))("conf-threshold", "Confidence threshold",
                                                              cxxopts::value<float>()->default_value("0.35"))(
        "nms-threshold", "Class-wise NMS IoU threshold", cxxopts::value<float>()->default_value("0.50"))(
        "max-detections", "Maximum detections printed after NMS", cxxopts::value<int>()->default_value("100"))(
        "help", "Show help");

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
        const std::vector<std::string> labels = readLabelNames(args["labels"].as<std::string>());
        const auto                     labels_path = args["output"].as<std::string>();
        const float                    conf_threshold = args["conf-threshold"].as<float>();
        const float                    nms_threshold  = args["nms-threshold"].as<float>();
        const int                      max_detections = args["max-detections"].as<int>();
        const int                      warmup         = args["warmup"].as<int>();
        const int                      repeat         = args["repeat"].as<int>();
        const int                      inflight       = args["inflight"].as<int>();
        const auto                     preprocess     = args["preprocess"].as<std::string>();
        if (conf_threshold < 0.0F || conf_threshold > 1.0F || nms_threshold < 0.0F || nms_threshold > 1.0F)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "thresholds must be in [0, 1]");
        }
        if (max_detections <= 0 || warmup < 0 || repeat <= 0 || inflight <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "max-detections/warmup/repeat/inflight must be positive");
        }
        if (preprocess != "cpu" && preprocess != "cuda" && preprocess != "both")
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--preprocess must be cpu, cuda, or both");
        }

        auto config = irt::sample::engine::makeConfig("rfdetr_nano", args["engine"].as<std::string>(),
                                                      args["device"].as<int>());
        const int input_size = args["input-size"].as<int>();
        if (input_size <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--input-size must be positive");
        }
        config.input_width    = input_size;
        config.input_height   = input_size;
        config.min_batch_size = args["batch-min"].as<int>();
        config.opt_batch_size = args["batch-opt"].as<int>();
        config.max_batch_size = args["batch-max"].as<int>();
        config.max_wait       = std::chrono::microseconds(args["max-wait-us"].as<int>());
        config.execution_slots = args["slots"].as<size_t>();
        if (config.min_batch_size <= 0 || config.min_batch_size > config.opt_batch_size
            || config.opt_batch_size > config.max_batch_size)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "batch profile must satisfy 1 <= min <= opt <= max");
        }
        if (static_cast<size_t>(inflight) > config.queue_capacity)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "--inflight must not exceed queue capacity");
        }
        config.validate();

        std::cout << "Timed scope excludes image decode, engine loading, and warmup; it includes preprocessing, "
                     "transfers, inference, output materialization, and Engine queueing."
                  << '\n';

        const auto direct_result = [&]
        {
            DirectRunner direct(config);
            for (int index = 0; index < warmup; ++index)
            {
                (void)direct.run(display_image);
            }
            printSyncResult("IModel direct (CPU)", repeat, measure(repeat, [&] { (void)direct.run(display_image); }));
            const auto result = direct.run(display_image);
            std::cout << "IModel direct detections:" << '\n';
            printDetections(decodeRFDETR(result, display_image.size(), conf_threshold, nms_threshold, max_detections), labels);
            return result;
        }();

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
            std::vector<double>             completion_latencies_ms;
            completion_latencies_ms.reserve(static_cast<size_t>(repeat));
            for (int index = 0; index < repeat; ++index)
            {
                const auto submitted = std::chrono::steady_clock::now();
                pending.emplace_back(engine->submit(display_image), submitted);
                if (pending.size() >= static_cast<size_t>(inflight))
                {
                    (void)pending.front().first.get();
                    const auto completed = std::chrono::steady_clock::now();
                    completion_latencies_ms.push_back(
                        std::chrono::duration<double, std::milli>(completed - pending.front().second).count());
                    pending.pop_front();
                }
            }
            while (!pending.empty())
            {
                (void)pending.front().first.get();
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
            std::cout << std::left << std::setw(26) << std::string("Engine ") + std::string(backend_name) + " async"
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

            const auto engine_result = engine->infer(display_image, std::chrono::seconds(30));
            std::cout << "Engine " << backend_name
                      << " detections (max_abs_output_diff=" << maxOutputDifference(direct_result, engine_result)
                      << "):" << '\n';
            const auto detections
                = decodeRFDETR(engine_result, display_image.size(), conf_threshold, nms_threshold, max_detections);
            printDetections(detections, labels);
            drawDetections(display_image, detections, labels, labels_path);
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
