/**
 * @file BenchmarkVisionModels.cpp
 * @brief DINOv2/DINOv3 与 LingBot-Vision TensorRT 性能基准。
 *
 * benchmark 只统计已经完成 H2D 的设备端推理时间。权重对应的 engine
 * 由 InferRT 的 buildOrLoad() 按模型、动态 batch 和精度自动缓存。
 */

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/IModelConfig.hpp>
#include <inferrt/model/ModelFactory.hpp>
#include <inferrt/model/Utils.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using irt::model::DeviceBuffer;
using irt::model::IModel;
using irt::model::IModelConfig;
using irt::model::ModelPrecision;

struct ModelSpec
{
    const char *name;
    const char *display_name;
    std::vector<const char *> weight_names;
};

const std::vector<ModelSpec> &modelSpecs()
{
    static const std::vector<ModelSpec> specs{
        {"dinov2_vits14", "DINOv2 ViT-S/14", {"dinov2_vits14.wts"}},
        {"dinov2_vitb14", "DINOv2 ViT-B/14", {"dinov2_vitb14.wts"}},
        {"dinov3_vits16", "DINOv3 ViT-S/16", {"dinov3_vits16.wts"}},
        {"dinov3_vitb16", "DINOv3 ViT-B/16", {"dinov3_vitb16.wts"}},
        {"lingbot_vision_vits16", "LingBot-Vision ViT-S/16", {"lingbot_vision_vits16.wts",
                                                                  "lingbot-vision-vit-small.wts"}},
        {"lingbot_vision_vitb16", "LingBot-Vision ViT-B/16", {"lingbot_vision_vitb16.wts",
                                                                  "lingbot-vision-vit-base.wts"}},
    };
    return specs;
}

struct Options
{
    fs::path              weights_root;
    fs::path              artifact_root;
    fs::path              explicit_weights;
    std::vector<std::string> model_names;
    std::vector<int>       batches;
    std::vector<ModelPrecision> precisions;
};

std::string lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string trim(std::string value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> splitCsv(const std::string &text)
{
    std::vector<std::string> values;
    std::stringstream        stream(text);
    std::string              item;
    while (std::getline(stream, item, ','))
    {
        item = trim(item);
        if (!item.empty())
        {
            values.push_back(item);
        }
    }
    return values;
}

std::string environmentOr(const char *name, std::string fallback)
{
#ifdef _WIN32
    char  *value = nullptr;
    size_t value_size = 0;
    if (_dupenv_s(&value, &value_size, name) == 0 && value != nullptr)
    {
        std::string result(value);
        std::free(value);
        return result.empty() ? std::move(fallback) : result;
    }
    std::free(value);
    return fallback;
#else
    if (const char *value = std::getenv(name); value != nullptr && *value != '\0')
    {
        return value;
    }
    return fallback;
#endif
}

int parsePositiveInt(const std::string &text, const char *option)
{
    try
    {
        size_t consumed = 0;
        const int value  = std::stoi(text, &consumed);
        if (consumed != text.size() || value <= 0)
        {
            throw std::invalid_argument("not positive");
        }
        return value;
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument(std::string(option) + " expects positive integers, got '" + text + "'");
    }
}

std::vector<int> parseBatches(const std::string &text)
{
    std::vector<int> batches;
    for (const auto &item : splitCsv(text))
    {
        batches.push_back(parsePositiveInt(item, "--batches"));
    }
    if (batches.empty())
    {
        throw std::invalid_argument("--batches must not be empty");
    }
    std::sort(batches.begin(), batches.end());
    batches.erase(std::unique(batches.begin(), batches.end()), batches.end());
    return batches;
}

ModelPrecision parsePrecision(const std::string &text)
{
    const auto value = lower(trim(text));
    if (value == "fp32" || value == "float32")
    {
        return ModelPrecision::FP32;
    }
    if (value == "fp16" || value == "float16")
    {
        return ModelPrecision::FP16;
    }
    throw std::invalid_argument("--precisions accepts fp32 or fp16, got '" + text + "'");
}

std::vector<ModelPrecision> parsePrecisions(const std::string &text)
{
    std::vector<ModelPrecision> precisions;
    for (const auto &item : splitCsv(text))
    {
        const auto precision = parsePrecision(item);
        if (std::find(precisions.begin(), precisions.end(), precision) == precisions.end())
        {
            precisions.push_back(precision);
        }
    }
    if (precisions.empty())
    {
        throw std::invalid_argument("--precisions must not be empty");
    }
    return precisions;
}

const ModelSpec &findModelSpec(const std::string &name)
{
    const auto normalized = lower(name);
    const auto &specs     = modelSpecs();
    const auto  it        = std::find_if(specs.begin(), specs.end(), [&](const ModelSpec &spec)
    { return normalized == spec.name; });
    if (it == specs.end())
    {
        throw std::invalid_argument("Unsupported vision benchmark model: " + name);
    }
    return *it;
}

std::vector<std::string> expandModels(const std::string &text)
{
    std::vector<std::string> models;
    const auto add = [&](const char *name)
    {
        if (std::find(models.begin(), models.end(), name) == models.end())
        {
            models.emplace_back(name);
        }
    };

    for (const auto &item : splitCsv(text))
    {
        const auto value = lower(item);
        if (value == "all" || value == "dino")
        {
            for (const auto &spec : modelSpecs())
            {
                if (value == "all" || std::string_view(spec.name).find("dino") != std::string_view::npos)
                {
                    add(spec.name);
                }
            }
        }
        else if (value == "lingbot" || value == "lingbot-vision")
        {
            add("lingbot_vision_vits16");
            add("lingbot_vision_vitb16");
        }
        else
        {
            add(findModelSpec(item).name);
        }
    }
    if (models.empty())
    {
        throw std::invalid_argument("--models must not be empty");
    }
    return models;
}

bool startsWith(std::string_view value, std::string_view prefix)
{
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

std::string optionValue(const std::string &argument, const char *name, int &index, int argc, char **argv)
{
    const std::string prefix = std::string(name) + "=";
    if (argument == name)
    {
        if (index + 1 >= argc)
        {
            throw std::invalid_argument(std::string(name) + " requires a value");
        }
        return argv[++index];
    }
    if (startsWith(argument, prefix))
    {
        return argument.substr(prefix.size());
    }
    return {};
}

void printUsage(const char *program)
{
    std::cout << "Usage: " << program << " [vision options] [google-benchmark options]\n"
              << "  --models <csv>       Models, or all/dino/lingbot (default: all)\n"
              << "  --weights-root <dir> Search root for .wts files (default: $INFERRT_MODEL_ROOT or F:/models)\n"
              << "  --artifact-root <dir> Search generated benchmark .wts files\n"
              << "  --weights-file <file> Use one .wts file when benchmarking one model\n"
              << "  --batches <csv>      Dynamic batches (default: 1,2,4,8)\n"
              << "  --precisions <csv>   fp32,fp16 (default: fp32,fp16)\n"
              << "\nLingBot-Vision .pt checkpoints are converted to .wts by benchmark/python/benchmark_vision.py.\n";
}

Options parseOptions(int argc, char **argv, std::vector<std::string> &benchmarkArguments)
{
    const fs::path executable_build_root = fs::absolute(argv[0]).parent_path().parent_path();
    Options options{
        fs::path(environmentOr("INFERRT_MODEL_ROOT", "F:/models")),
        fs::path(environmentOr("INFERRT_BENCHMARK_ARTIFACT_ROOT",
                               (executable_build_root / "benchmark_vision_artifacts").string())),
        {},
        expandModels("all"),
        parseBatches("1,2,4,8"),
        parsePrecisions("fp32,fp16"),
    };

    benchmarkArguments.emplace_back(argv[0]);
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h")
        {
            printUsage(argv[0]);
            benchmarkArguments.clear();
            return options;
        }

        auto value = optionValue(argument, "--models", index, argc, argv);
        if (!value.empty())
        {
            options.model_names = expandModels(value);
            continue;
        }
        value = optionValue(argument, "--weights-root", index, argc, argv);
        if (!value.empty())
        {
            options.weights_root = fs::path(value);
            continue;
        }
        value = optionValue(argument, "--artifact-root", index, argc, argv);
        if (!value.empty())
        {
            options.artifact_root = fs::path(value);
            continue;
        }
        value = optionValue(argument, "--weights-file", index, argc, argv);
        if (!value.empty())
        {
            options.explicit_weights = fs::path(value);
            continue;
        }
        value = optionValue(argument, "--batches", index, argc, argv);
        if (!value.empty())
        {
            options.batches = parseBatches(value);
            continue;
        }
        value = optionValue(argument, "--precisions", index, argc, argv);
        if (!value.empty())
        {
            options.precisions = parsePrecisions(value);
            continue;
        }
        benchmarkArguments.push_back(argument);
    }
    return options;
}

bool sameFilename(const fs::path &path, const std::vector<const char *> &names)
{
    const auto filename = lower(path.filename().string());
    return std::any_of(names.begin(), names.end(), [&](const char *name) { return filename == lower(name); });
}

fs::path findWeights(const Options &options, const ModelSpec &spec)
{
    if (!options.explicit_weights.empty())
    {
        if (options.model_names.size() != 1)
        {
            throw std::invalid_argument("--weights-file requires exactly one --models entry");
        }
        if (!fs::is_regular_file(options.explicit_weights))
        {
            throw std::invalid_argument("Weights file not found: " + options.explicit_weights.string());
        }
        return options.explicit_weights;
    }

    std::vector<fs::path> roots;
    for (const auto &root : {options.weights_root, options.artifact_root})
    {
        if (!root.empty() && fs::is_directory(root)
            && std::find(roots.begin(), roots.end(), root) == roots.end())
        {
            roots.push_back(root);
        }
    }

    for (const auto &root : roots)
    {
        std::error_code error;
        fs::recursive_directory_iterator iterator(root, fs::directory_options::skip_permission_denied, error);
        const fs::recursive_directory_iterator end;
        for (; iterator != end; iterator.increment(error))
        {
            if (error)
            {
                error.clear();
                continue;
            }
            if (iterator->is_regular_file(error) && !error && sameFilename(iterator->path(), spec.weight_names))
            {
                return iterator->path();
            }
        }
    }

    std::ostringstream message;
    message << "No .wts file found for " << spec.name << ". Searched roots: " << options.weights_root << ", "
            << options.artifact_root << ". Convert LingBot-Vision .pt files first or pass --weights-file.";
    throw std::invalid_argument(message.str());
}

class CudaEventPair
{
public:
    CudaEventPair()
    {
        irt::model::checkCuda(cudaEventCreate(&start_), "cudaEventCreate(start)");
        try
        {
            irt::model::checkCuda(cudaEventCreate(&stop_), "cudaEventCreate(stop)");
        }
        catch (...)
        {
            cudaEventDestroy(start_);
            start_ = nullptr;
            throw;
        }
    }

    ~CudaEventPair()
    {
        if (start_ != nullptr)
        {
            cudaEventDestroy(start_);
        }
        if (stop_ != nullptr)
        {
            cudaEventDestroy(stop_);
        }
    }

    float measure(IModel &model, const std::vector<void *> &buffers, cudaStream_t stream)
    {
        irt::model::checkCuda(cudaEventRecord(start_, stream), "cudaEventRecord(start)");
        model.infer(buffers, stream, true);
        irt::model::checkCuda(cudaEventRecord(stop_, stream), "cudaEventRecord(stop)");
        irt::model::checkCuda(cudaEventSynchronize(stop_), "cudaEventSynchronize(stop)");

        float elapsed_ms = 0.0F;
        irt::model::checkCuda(cudaEventElapsedTime(&elapsed_ms, start_, stop_), "cudaEventElapsedTime");
        return elapsed_ms;
    }

private:
    cudaEvent_t start_{nullptr};
    cudaEvent_t stop_{nullptr};
};

class VisionRunner
{
public:
    VisionRunner(const ModelSpec &spec, ModelPrecision precision, int max_batch, fs::path weights_file)
        : spec_(spec)
        , precision_(precision)
        , max_batch_(max_batch)
        , weights_file_(std::move(weights_file))
    {
        auto config = std::make_unique<IModelConfig>();
        config->setDynamicBatchRange(1, max_batch_, max_batch_);
        config->setPrecision(precision_);

        model_ = irt::model::CreateModel(spec_.name, std::move(config));
        if (!model_)
        {
            throw std::runtime_error("Failed to create model: " + std::string(spec_.name));
        }
        model_->setLogLevel(nvinfer1::ILogger::Severity::kERROR);
        model_->buildOrLoad(weights_file_.string());

        const auto input_names = model_->ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
        if (input_names.size() != 1)
        {
            throw std::runtime_error("Vision benchmark expects one input tensor for " + std::string(spec_.name));
        }
        input_name_ = input_names.front();
        output_names_ = model_->ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
        if (output_names_.empty())
        {
            throw std::runtime_error("Model has no output tensors: " + std::string(spec_.name));
        }
        stream_ = model_->resolveExecutionStream();
        if (stream_ == nullptr)
        {
            throw std::runtime_error("Model did not provide a CUDA stream: " + std::string(spec_.name));
        }
    }

    void prepare(int batch)
    {
        if (batch <= 0 || batch > max_batch_)
        {
            throw std::invalid_argument("Requested batch is outside the benchmark profile");
        }

        auto input_dims = model_->tensorShape(input_name_);
        if (input_dims.nbDims != 4)
        {
            throw std::runtime_error("Expected NCHW input for " + std::string(spec_.name));
        }
        input_dims.d[0] = batch;
        model_->setTensorShape(input_name_, input_dims);
        input_dims = model_->tensorShape(input_name_);

        input_.resize(irt::model::elementCount(input_dims), model_->tensorDataType(input_name_));
        output_buffers_.clear();
        output_buffers_.reserve(output_names_.size());
        buffers_.clear();
        buffers_.reserve(1 + output_names_.size());
        buffers_.push_back(input_.data());

        for (const auto &output_name : output_names_)
        {
            const auto dims      = model_->tensorShape(output_name);
            const auto data_type = model_->tensorDataType(output_name);
            output_buffers_.emplace_back(irt::model::elementCount(dims), data_type);
            buffers_.push_back(output_buffers_.back().data());
        }

        irt::model::checkCuda(cudaMemsetAsync(input_.data(), 0, input_.sizeBytes(), stream_),
                              "cudaMemsetAsync(input)");
        irt::model::checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(input init)");

        for (int index = 0; index < 2; ++index)
        {
            model_->infer(buffers_, stream_, true);
        }
        irt::model::checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize(warmup)");
        batch_ = batch;
    }

    void run(benchmark::State &state)
    {
        prepare(static_cast<int>(state.range(0)));
        CudaEventPair timer;

        for (auto _ : state)
        {
            const float elapsed_ms = timer.measure(*model_, buffers_, stream_);
            state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
            benchmark::DoNotOptimize(buffers_.back());
        }

        state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * batch_);
        state.counters["batch"]      = static_cast<double>(batch_);
        state.counters["input_bytes"] = static_cast<double>(input_.sizeBytes());
        state.SetLabel(std::string(spec_.display_name) + ", " + irt::model::modelPrecisionName(precision_)
                       + ", input=" + irt::model::dimsToString(model_->tensorShape(input_name_))
                       + ", weights=" + weights_file_.filename().string());
    }

private:
    const ModelSpec &spec_;
    ModelPrecision  precision_;
    int              max_batch_;
    int              batch_{0};
    fs::path         weights_file_;
    std::unique_ptr<IModel> model_;
    std::string              input_name_;
    std::vector<std::string> output_names_;
    cudaStream_t              stream_{nullptr};
    DeviceBuffer              input_;
    std::vector<DeviceBuffer> output_buffers_;
    std::vector<void *>       buffers_;
};

const Options *g_options = nullptr;
std::unique_ptr<VisionRunner> g_active_runner;
std::string                  g_active_key;

VisionRunner &activeRunner(const ModelSpec &spec, ModelPrecision precision)
{
    const std::string key = std::string(spec.name) + "/" + irt::model::modelPrecisionName(precision);
    if (g_active_runner == nullptr || g_active_key != key)
    {
        g_active_runner.reset();
        const auto weights = findWeights(*g_options, spec);
        g_active_runner   = std::make_unique<VisionRunner>(spec, precision,
                                                            *std::max_element(g_options->batches.begin(),
                                                                              g_options->batches.end()),
                                                            weights);
        g_active_key      = key;
    }
    return *g_active_runner;
}

void runBenchmark(benchmark::State &state, const ModelSpec &spec, ModelPrecision precision)
{
    try
    {
        activeRunner(spec, precision).run(state);
    }
    catch (const std::exception &error)
    {
        state.SkipWithError(error.what());
    }
}

void registerBenchmarks(const Options &options)
{
    g_options = &options;
    for (const auto &model_name : options.model_names)
    {
        const auto &spec = findModelSpec(model_name);
        for (const auto precision : options.precisions)
        {
            for (const int batch : options.batches)
            {
                const std::string name = "InferRT/" + std::string(spec.name) + "/"
                                        + irt::model::modelPrecisionName(precision) + "/batch_" + std::to_string(batch);
                benchmark::RegisterBenchmark(name.c_str(), [spec_ptr = &spec, precision](benchmark::State &state)
                { runBenchmark(state, *spec_ptr, precision); })
                    ->Args({batch})
                    ->UseManualTime()
                    ->Unit(benchmark::kMillisecond);
            }
        }
    }
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        std::vector<std::string> benchmark_arguments;
        const auto               options = parseOptions(argc, argv, benchmark_arguments);
        if (benchmark_arguments.empty())
        {
            return 0;
        }

        registerBenchmarks(options);

        std::vector<char *> benchmark_argv;
        benchmark_argv.reserve(benchmark_arguments.size());
        for (auto &argument : benchmark_arguments)
        {
            benchmark_argv.push_back(argument.data());
        }
        int benchmark_argc = static_cast<int>(benchmark_argv.size());
        benchmark::Initialize(&benchmark_argc, benchmark_argv.data());
        if (benchmark::ReportUnrecognizedArguments(benchmark_argc, benchmark_argv.data()))
        {
            return 1;
        }
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Benchmark setup failed: " << error.what() << '\n';
        return 1;
    }
}
