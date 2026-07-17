#include <cuda_runtime_api.h>
#include <cxxopts.hpp>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <inferrt/util/Path.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;
using irt::model::checkCuda;
using irt::model::dataTypeToString;
using irt::model::dimsToCsv;
using irt::model::elementCount;
using irt::model::elementSize;

double elapsedMs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * @brief 用于在显示帮助后中断主流程。
 */
struct HelpRequested
{
};

/**
 * @brief 分类 sample 的命令行参数集合。
 */
struct Arguments
{
    std::string              model_name;
    fs::path                 weights_file;
    std::vector<fs::path>    image_paths;
    fs::path                 label_file;
    fs::path                 dump_dir;
    irt::model::ModelRuntime runtime{};
    int                      warmup{0};
    int                      repeat{1};
};

struct IterationTiming
{
    double h2d_ms{0.0};
    double inference_ms{0.0};
    double d2h_ms{0.0};

    double totalMs() const noexcept
    {
        return h2d_ms + inference_ms + d2h_ms;
    }
};

struct TimingStats
{
    double total_ms{0.0};
    double avg_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};
};

std::string sanitizeFileStem(std::string_view value)
{
    std::string stem;
    stem.reserve(value.size());
    for (unsigned char ch : value)
    {
        if (std::isalnum(ch))
        {
            stem.push_back(static_cast<char>(ch));
        }
        else
        {
            stem.push_back('_');
        }
    }
    return stem.empty() ? "tensor" : stem;
}

void writeBinaryFile(const fs::path &file_path, const void *data, size_t num_bytes)
{
    std::ofstream output(file_path, std::ios::binary);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open output file: %s",
                             file_path.string().c_str());
    }
    output.write(static_cast<const char *>(data), static_cast<std::streamsize>(num_bytes));
}

std::string trim(std::string value)
{
    auto not_space = [](unsigned char ch)
    {
        return !std::isspace(ch);
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::vector<fs::path> splitPathList(const std::string &value)
{
    std::vector<fs::path> paths;
    size_t                start = 0;
    while (start <= value.size())
    {
        const size_t end   = value.find_first_of(",;", start);
        auto         token = trim(value.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (!token.empty())
        {
            paths.emplace_back(std::move(token));
        }
        if (end == std::string::npos)
        {
            break;
        }
        start = end + 1;
    }
    return paths;
}

std::vector<fs::path> resolveImagePaths(const fs::path &project_root, const std::vector<fs::path> &configured_paths)
{
    if (configured_paths.empty())
    {
        return {project_root / irt::model::ImageNetUtil::kDefaultImagePath};
    }

    std::vector<fs::path> resolved;
    resolved.reserve(configured_paths.size());
    for (const auto &path : configured_paths)
    {
        resolved.push_back(path.is_absolute() ? path : project_root / path);
    }
    return resolved;
}

std::vector<float> preprocessBatch(const std::vector<cv::Mat> &images, const nvinfer1::Dims &input_dims)
{
    if (input_dims.nbDims != 4 || input_dims.d[1] != 3 || input_dims.d[2] <= 0 || input_dims.d[3] <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Classification sample expects input shape Nx3xHxW, got %s",
                             dimsToCsv(input_dims).c_str());
    }

    const size_t image_elements = static_cast<size_t>(input_dims.d[1]) * static_cast<size_t>(input_dims.d[2])
                                * static_cast<size_t>(input_dims.d[3]);
    std::vector<float> batch_data(images.size() * image_elements);
    for (size_t i = 0; i < images.size(); ++i)
    {
        const cv::Mat preprocessed = irt::model::ImageNetUtil::preprocess(
            images[i], cv::Size(static_cast<int>(input_dims.d[3]), static_cast<int>(input_dims.d[2])));
        const auto single = irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);
        if (single.size() != image_elements)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "Unexpected preprocessed image tensor size");
        }
        std::memcpy(batch_data.data() + i * image_elements, single.data(), image_elements * sizeof(float));
    }
    return batch_data;
}

TimingStats summarizeTimings(const std::vector<double> &values)
{
    if (values.empty())
    {
        return {};
    }

    const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    const double total          = std::accumulate(values.begin(), values.end(), 0.0);
    return TimingStats{total, total / static_cast<double>(values.size()), *min_it, *max_it};
}

void printTimingStats(const char *name, const TimingStats &stats)
{
    std::cout << ", " << name << "_total=" << stats.total_ms << " ms, " << name << "_avg=" << stats.avg_ms << " ms, "
              << name << "_min=" << stats.min_ms << " ms, " << name << "_max=" << stats.max_ms << " ms";
}

/**
 * @brief 判断主输出是否可以按 ImageNet 分类 logits 解释。
 * @param labels 已加载的标签表。
 * @param num_scores 主输出展平后的元素数量。
 * @return 标签数量与输出元素数量一致时返回 true。
 */
bool isClassificationOutput(const std::vector<std::string> &labels, size_t num_scores)
{
    return !labels.empty() && labels.size() == num_scores;
}

/**
 * @brief 打印分类 logits 或 DINO 等特征向量的 Top-K 结果。
 * @param scores 已按分数降序排好前 topk 的输出值与索引。
 * @param topk 需要打印的条数。
 * @param labels ImageNet 标签表；当数量匹配输出维度时用于打印类别名。
 */
void printTopOutputs(const std::vector<std::pair<float, size_t>> &scores, size_t topk,
                     const std::vector<std::string> &labels)
{
    const bool classification_output = isClassificationOutput(labels, scores.size());
    std::cout << "\nTop-" << topk << (classification_output ? " predictions:" : " feature values:") << std::endl;

    for (size_t i = 0; i < topk; ++i)
    {
        const size_t idx   = scores[i].second;
        const float  value = scores[i].first;
        if (classification_output)
        {
            std::cout << "top: " << (i + 1) << ", confidence: " << value << ", label[" << idx << "]: " << labels[idx]
                      << std::endl;
        }
        else
        {
            std::cout << "top: " << (i + 1) << ", value: " << value << ", feature[" << idx << "]" << std::endl;
        }
    }
}

/**
 * @brief 构造命令行选项定义。
 * @param program_name 可执行文件名。
 * @return cxxopts 选项对象。
 */
cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Run InferRT classification models on one image or an image batch");
    options.add_options()("model,m", "Built-in model name (required)", cxxopts::value<std::string>())(
        "weights-file,w", "Weights/model file (.wts, .onnx or OpenVINO IR) (required)", cxxopts::value<std::string>())(
        "image-path,i", "Input image path, or comma/semicolon-separated image paths",
        cxxopts::value<std::string>()->default_value(""))("label-file,l", "ImageNet label file",
                                                          cxxopts::value<std::string>()->default_value(""))(
        "dump-dir,o", "Optional directory to dump input/output tensors for parity tests",
        cxxopts::value<std::string>()->default_value(""))("runtime",
                                                          "Model runtime: cpu, gpu:0, cuda:0, or backend:gpu-id (e.g. tensorrt:0)",
                                                          cxxopts::value<std::string>()->default_value("tensorrt:0"))(
        "warmup", "Warmup iterations before timing", cxxopts::value<int>()->default_value("0"))(
        "repeat", "Timed inference iterations", cxxopts::value<int>()->default_value("1"))("h,help", "Show help");
    return options;
}

/**
 * @brief 解析并校验命令行参数。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组。
 * @return 解析后的参数。
 */
Arguments parseArguments(int argc, char *argv[])
{
    auto       options = makeOptions(argv[0]);
    const auto result  = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        std::cout << "Default image: " << irt::model::ImageNetUtil::kDefaultImagePath.generic_string() << std::endl;
        std::cout << "Default labels: " << irt::model::ImageNetUtil::kDefaultLabelPath.generic_string() << std::endl;
        std::cout << "Supported models:";
        for (const auto &model_name : irt::model::getRegisteredModelNames())
        {
            std::cout << ' ' << model_name;
        }
        std::cout << std::endl;
        throw HelpRequested{};
    }

    if (!result.count("model") || !result.count("weights-file"))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--model and --weights-file are required");
    }

    Arguments args;
    args.model_name   = result["model"].as<std::string>();
    args.weights_file = result["weights-file"].as<std::string>();
    args.image_paths  = splitPathList(result["image-path"].as<std::string>());
    args.label_file   = result["label-file"].as<std::string>();
    args.dump_dir     = result["dump-dir"].as<std::string>();
    args.runtime      = irt::model::ModelRuntime::parse(result["runtime"].as<std::string>());
    args.warmup       = result["warmup"].as<int>();
    args.repeat       = result["repeat"].as<int>();
    if (args.warmup < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--warmup must be >= 0");
    }
    if (args.repeat <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--repeat must be > 0");
    }
    return args;
}

} // namespace

/**
 * @brief 运行单张或批量图片推理，并打印分类 logits 或特征向量的 Top-3 结果。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组。
 * @return 成功返回 0，失败返回非 0。
 */
int main(int argc, char *argv[])
{
    try
    {
        const Arguments args = parseArguments(argc, argv);
        if (!irt::model::isSupportedModel(args.model_name))
        {
            std::cerr << "Unsupported model: " << args.model_name << std::endl;
            return -1;
        }

        fs::path project_root = irt::util::findProjectRoot(
            argv[0], {irt::model::ImageNetUtil::kDefaultImagePath, irt::model::ImageNetUtil::kDefaultLabelPath},
            __FILE__);
        std::vector<fs::path> image_paths = resolveImagePaths(project_root, args.image_paths);
        if (image_paths.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "At least one image path is required");
        }
        if (image_paths.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Batch size is too large: %zu",
                                 image_paths.size());
        }
        fs::path label_file
            = args.label_file.empty() ? project_root / irt::model::ImageNetUtil::kDefaultLabelPath : args.label_file;

        auto config = std::make_unique<irt::model::IModelConfig>();
        if (image_paths.size() > 1)
        {
            const int batch_size = static_cast<int>(image_paths.size());
            config->setDynamicBatchRange(1, batch_size, batch_size);
        }
        config->setRuntime(args.runtime);

        const std::string runtime_model_name
            = args.runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT ? args.model_name : "onnx";
        auto model = irt::model::CreateModel(runtime_model_name, std::move(config));
        if (!model)
        {
            std::cerr << "Failed to create model: " << runtime_model_name << std::endl;
            return -1;
        }

        model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);

        std::cout << "Building or Loading model..." << std::endl;
        const auto build_start = Clock::now();
        model->buildOrLoad(args.weights_file.string());
        const auto build_end = Clock::now();
        std::cout << "Model loaded successfully." << std::endl;

        std::vector<cv::Mat> images;
        images.reserve(image_paths.size());
        for (const auto &image_path : image_paths)
        {
            std::cout << "Loading image: " << image_path.generic_string() << std::endl;
            cv::Mat img = cv::imread(image_path.generic_string());
            if (img.empty())
            {
                std::cerr << "Failed to load image: " << image_path.generic_string() << std::endl;
                return -1;
            }
            images.push_back(std::move(img));
        }

        const auto input_tensor_names  = model->ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
        const auto output_tensor_names = model->ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
        if (input_tensor_names.empty() || output_tensor_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Model must expose at least one input and one output tensor");
        }
        if (input_tensor_names.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Classification sample expects exactly one input tensor, got %zu",
                                 input_tensor_names.size());
        }

        const std::string &input_tensor_name = input_tensor_names.front();
        nvinfer1::Dims     input_dims        = model->tensorShape(input_tensor_name);
        const auto         input_type        = model->tensorDataType(input_tensor_name);
        if (input_type != nvinfer1::DataType::kFLOAT)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Classification sample expects float32 input tensor");
        }
        if (input_dims.nbDims != 4 || input_dims.d[1] != 3 || input_dims.d[2] <= 0 || input_dims.d[3] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Classification sample expects input shape Nx3xHxW, got %s",
                                 dimsToCsv(input_dims).c_str());
        }
        if (input_dims.d[0] != static_cast<int64_t>(images.size()))
        {
            input_dims.d[0] = static_cast<int64_t>(images.size());
            model->setTensorShape(input_tensor_name, input_dims);
        }

        const auto               preprocess_start = Clock::now();
        const std::vector<float> input_data       = preprocessBatch(images, input_dims);
        const auto               preprocess_end   = Clock::now();

        const size_t input_num_bytes = elementCount(input_dims) * elementSize(nvinfer1::DataType::kFLOAT);
        if (input_num_bytes != input_data.size() * sizeof(float))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Preprocessed input byte size does not match model input tensor");
        }

        const bool uses_tensorrt = args.runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT;
        const auto stream        = uses_tensorrt ? model->resolveExecutionStream() : nullptr;

        void *d_input = nullptr;
        if (uses_tensorrt)
        {
            checkCuda(cudaMalloc(&d_input, input_num_bytes), "cudaMalloc(input)");
        }

        std::vector<std::string>                   output_names;
        std::vector<nvinfer1::Dims>                output_dims;
        std::vector<nvinfer1::DataType>            output_types;
        std::vector<size_t>                        output_num_bytes;
        std::vector<void *>                        device_outputs;
        std::vector<std::vector<char>>             host_outputs;
        std::vector<std::vector<std::max_align_t>> graph_output_storage;

        output_names.reserve(output_tensor_names.size());
        output_dims.reserve(output_tensor_names.size());
        output_types.reserve(output_tensor_names.size());
        output_num_bytes.reserve(output_tensor_names.size());
        device_outputs.reserve(output_tensor_names.size());
        host_outputs.reserve(output_tensor_names.size());
        graph_output_storage.reserve(output_tensor_names.size());

        for (const auto &output_name : output_tensor_names)
        {
            const nvinfer1::Dims     dims      = model->tensorShape(output_name);
            const nvinfer1::DataType data_type = model->tensorDataType(output_name);
            const size_t             num_bytes = elementCount(dims) * elementSize(data_type);

            void *device_ptr = nullptr;
            if (uses_tensorrt)
            {
                checkCuda(cudaMalloc(&device_ptr, num_bytes), "cudaMalloc(output)");
            }

            output_names.push_back(output_name);
            output_dims.push_back(dims);
            output_types.push_back(data_type);
            output_num_bytes.push_back(num_bytes);
            device_outputs.push_back(device_ptr);
            host_outputs.emplace_back(num_bytes);
            if (!uses_tensorrt)
            {
                const size_t aligned_words = (num_bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
                graph_output_storage.emplace_back(aligned_words);
            }
        }

        std::vector<void *> buffers;
        buffers.reserve(1 + device_outputs.size());
        buffers.push_back(uses_tensorrt ? d_input : const_cast<float *>(input_data.data()));
        for (size_t i = 0; i < output_names.size(); ++i)
        {
            buffers.push_back(uses_tensorrt ? device_outputs[i] : graph_output_storage[i].data());
        }

        auto run_inference_once = [&]() -> IterationTiming
        {
            IterationTiming timing;
            const auto      h2d_start = Clock::now();
            if (uses_tensorrt)
            {
                checkCuda(cudaMemcpyAsync(d_input, input_data.data(), input_num_bytes, cudaMemcpyHostToDevice, stream),
                          "cudaMemcpyAsync(H2D input)");
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(H2D input)");
            }
            const auto h2d_end = Clock::now();
            timing.h2d_ms      = elapsedMs(h2d_start, h2d_end);

            const auto inference_start = Clock::now();
            model->infer(buffers, stream, true);
            if (uses_tensorrt)
            {
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(inference)");
            }
            const auto inference_end = Clock::now();
            timing.inference_ms      = elapsedMs(inference_start, inference_end);

            const auto d2h_start = Clock::now();
            if (uses_tensorrt)
            {
                for (size_t i = 0; i < host_outputs.size(); ++i)
                {
                    checkCuda(cudaMemcpyAsync(host_outputs[i].data(), device_outputs[i], output_num_bytes[i],
                                              cudaMemcpyDeviceToHost, stream),
                              "cudaMemcpyAsync(D2H output)");
                }
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(D2H output)");
            }
            else
            {
                for (size_t i = 0; i < host_outputs.size(); ++i)
                {
                    std::memcpy(host_outputs[i].data(), graph_output_storage[i].data(), output_num_bytes[i]);
                }
            }
            const auto d2h_end = Clock::now();
            timing.d2h_ms      = elapsedMs(d2h_start, d2h_end);
            return timing;
        };

        std::cout << "Running inference warmup=" << args.warmup << ", repeat=" << args.repeat << "..." << std::endl;
        for (int i = 0; i < args.warmup; ++i)
        {
            (void)run_inference_once();
        }

        std::vector<double> h2d_times_ms;
        std::vector<double> inference_times_ms;
        std::vector<double> d2h_times_ms;
        std::vector<double> end_to_end_times_ms;
        h2d_times_ms.reserve(static_cast<size_t>(args.repeat));
        inference_times_ms.reserve(static_cast<size_t>(args.repeat));
        d2h_times_ms.reserve(static_cast<size_t>(args.repeat));
        end_to_end_times_ms.reserve(static_cast<size_t>(args.repeat));
        const auto infer_start = Clock::now();
        for (int i = 0; i < args.repeat; ++i)
        {
            const auto timing = run_inference_once();
            h2d_times_ms.push_back(timing.h2d_ms);
            inference_times_ms.push_back(timing.inference_ms);
            d2h_times_ms.push_back(timing.d2h_ms);
            end_to_end_times_ms.push_back(timing.totalMs());
        }
        const auto infer_end = Clock::now();

        const auto               postprocess_start = Clock::now();
        std::vector<std::string> labels;
        if (!label_file.empty() && fs::exists(label_file))
        {
            labels = irt::model::readImagenetLabels(label_file.string());
        }

        const std::string &primary_output_name = output_names.front();
        const size_t       primary_index       = 0;
        if (output_types[primary_index] != nvinfer1::DataType::kFLOAT)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Classification sample expects float32 primary output");
        }

        const size_t num_scores   = elementCount(output_dims[primary_index]);
        const auto  *output_data  = reinterpret_cast<const float *>(host_outputs[primary_index].data());
        const size_t batch_size   = images.size();
        const bool   split_output = output_dims[primary_index].nbDims >= 1 && output_dims[primary_index].d[0] > 0
                               && static_cast<size_t>(output_dims[primary_index].d[0]) == batch_size
                               && num_scores % batch_size == 0;
        const size_t per_sample_scores = split_output ? num_scores / batch_size : num_scores;
        const auto   postprocess_end   = Clock::now();

        if (!args.dump_dir.empty())
        {
            fs::create_directories(args.dump_dir);
            writeBinaryFile(args.dump_dir / "input.bin", input_data.data(), input_num_bytes);

            std::ofstream manifest(args.dump_dir / "manifest.txt");
            if (!manifest)
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open manifest file");
            }

            manifest << "version=1\n";
            manifest << "runtime=" << args.runtime.toString() << "\n";
            manifest << "warmup=" << args.warmup << "\n";
            manifest << "repeat=" << args.repeat << "\n";
            manifest << "model_name=" << args.model_name << "\n";
            manifest << "weights_file=" << fs::absolute(args.weights_file).generic_string() << "\n";
            manifest << "image_count=" << image_paths.size() << "\n";
            manifest << "image_path=" << fs::absolute(image_paths.front()).generic_string() << "\n";
            for (size_t i = 0; i < image_paths.size(); ++i)
            {
                manifest << "image_path_" << i << "=" << fs::absolute(image_paths[i]).generic_string() << "\n";
            }
            manifest << "tensor|input|float32|" << dimsToCsv(input_dims) << "|input.bin\n";

            for (size_t i = 0; i < output_names.size(); ++i)
            {
                const std::string file_name = sanitizeFileStem(output_names[i]) + ".bin";
                writeBinaryFile(args.dump_dir / file_name, host_outputs[i].data(), output_num_bytes[i]);
                manifest << "tensor|" << output_names[i] << "|" << dataTypeToString(output_types[i]) << "|"
                         << dimsToCsv(output_dims[i]) << "|" << file_name << "\n";
            }

            std::cout << "Saved classification dump to: " << fs::absolute(args.dump_dir).string() << std::endl;
            std::cout << "Primary output tensor: " << primary_output_name << std::endl;
        }

        std::cout << "Runtime: " << args.runtime.toString() << std::endl;
        const auto h2d_stats        = summarizeTimings(h2d_times_ms);
        const auto inference_stats  = summarizeTimings(inference_times_ms);
        const auto d2h_stats        = summarizeTimings(d2h_times_ms);
        const auto end_to_end_stats = summarizeTimings(end_to_end_times_ms);
        std::cout << "Timing: build_or_load=" << elapsedMs(build_start, build_end)
                  << " ms, preprocess=" << elapsedMs(preprocess_start, preprocess_end) << " ms";
        printTimingStats("h2d", h2d_stats);
        printTimingStats("inference", inference_stats);
        printTimingStats("d2h", d2h_stats);
        printTimingStats("end_to_end", end_to_end_stats);
        std::cout << ", timed_loop_wall=" << elapsedMs(infer_start, infer_end)
                  << " ms, postprocess=" << elapsedMs(postprocess_start, postprocess_end) << " ms" << std::endl;

        const size_t samples_to_print = split_output ? batch_size : 1;
        for (size_t batch_index = 0; batch_index < samples_to_print; ++batch_index)
        {
            const auto                           *sample_data = output_data + batch_index * per_sample_scores;
            std::vector<std::pair<float, size_t>> scores;
            scores.reserve(per_sample_scores);
            for (size_t i = 0; i < per_sample_scores; ++i)
            {
                scores.push_back({sample_data[i], i});
            }
            const size_t topk = std::min<size_t>(3, scores.size());
            std::partial_sort(scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(topk), scores.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });
            if (split_output)
            {
                std::cout << "\nBatch " << batch_index << " image: " << image_paths[batch_index].generic_string()
                          << std::endl;
            }
            printTopOutputs(scores, topk, labels);
        }

        if (uses_tensorrt)
        {
            checkCuda(cudaFree(d_input), "cudaFree(input)");
            for (void *device_ptr : device_outputs)
            {
                checkCuda(cudaFree(device_ptr), "cudaFree(output)");
            }
        }

        std::cout << "\nDone!" << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    catch (const HelpRequested &)
    {
        return 0;
    }
}
