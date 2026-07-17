#include <cuda_runtime_api.h>
#include <cxxopts.hpp>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Buffers.hpp>
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

using Clock        = std::chrono::steady_clock;
using DeviceBuffer = irt::model::DeviceBuffer;
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
 * @brief 特征导出 sample 的命令行参数集合。
 */
struct Arguments
{
    std::string              model_name;
    fs::path                 weights_file;
    std::vector<std::string> feature_names;
    std::vector<fs::path>    image_paths;
    fs::path                 output_dir;
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

/**
 * @brief 单个输出特征张量的主机侧描述。
 */
struct TensorDump
{
    std::string        name;
    nvinfer1::Dims     dims;
    nvinfer1::DataType data_type;
    std::vector<char>  host_bytes;
    std::string        file_name;
};

const fs::path kDefaultOutputDir = "feature_dump_cpp";

/**
 * @brief 用于在显示帮助后中断主流程。
 */
struct HelpRequested
{
};

/**
 * @brief 去除字符串首尾空白字符。
 * @param value 待处理字符串。
 * @return 去除空白后的结果。
 */
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

/**
 * @brief 按逗号分割特征名列表，并去除每项首尾空白。
 * @param csv 逗号分隔字符串。
 * @return 特征名数组。
 */
std::vector<std::string> splitFeatureNames(const std::string &csv)
{
    std::vector<std::string> feature_names;
    size_t                   start = 0;
    while (start <= csv.size())
    {
        const size_t end   = csv.find(',', start);
        auto         token = trim(csv.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (!token.empty())
        {
            feature_names.push_back(std::move(token));
        }
        if (end == std::string::npos)
        {
            break;
        }
        start = end + 1;
    }
    return feature_names;
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
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Feature sample expects input shape Nx3xHxW, got %s",
                             dimsToCsv(input_dims).c_str());
    }

    const size_t image_elements = static_cast<size_t>(input_dims.d[1]) * static_cast<size_t>(input_dims.d[2])
                                * static_cast<size_t>(input_dims.d[3]);
    std::vector<float> batch_data(images.size() * image_elements);
    for (size_t i = 0; i < images.size(); ++i)
    {
        const auto preprocessed = irt::model::ImageNetUtil::preprocess(
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
 * @brief 构造命令行选项定义。
 * @param program_name 可执行文件名。
 * @return cxxopts 选项对象。
 */
cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Dump intermediate feature tensors from InferRT models");
    options.add_options()("model,m", "Built-in model name (required)", cxxopts::value<std::string>())(
        "weights-file,w", "Weights/model file (.wts, .onnx or OpenVINO IR) (required)", cxxopts::value<std::string>())(
        "features,f", "Comma-separated feature tensor names (required)", cxxopts::value<std::string>())(
        "image-path,i", "Input image path, or comma/semicolon-separated image paths",
        cxxopts::value<std::string>()->default_value(""))("output-dir,o", "Output directory",
                                                          cxxopts::value<std::string>()->default_value(""))(
        "runtime", "Model runtime: cpu, gpu:0, cuda:0, or backend:gpu-id (e.g. tensorrt:0)",
        cxxopts::value<std::string>()->default_value("tensorrt:0"))(
        "warmup", "Warmup iterations before timing", cxxopts::value<int>()->default_value("0"))(
        "repeat", "Timed feature forward iterations", cxxopts::value<int>()->default_value("1"))("h,help", "Show help");
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
        std::cout << "Default output dir: " << kDefaultOutputDir.generic_string() << std::endl;
        std::cout << "DINO feature hint: use x_norm_clstoken for global retrieval, x_norm_patchtokens for patch tokens"
                  << std::endl;
        std::cout << "Supported models:";
        for (const auto &model_name : irt::model::getRegisteredModelNames())
        {
            std::cout << ' ' << model_name;
        }
        std::cout << std::endl;
        throw HelpRequested{};
    }

    if (!result.count("model") || !result.count("weights-file") || !result.count("features"))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "--model, --weights-file and --features are required");
    }

    Arguments args;
    args.model_name    = result["model"].as<std::string>();
    args.weights_file  = result["weights-file"].as<std::string>();
    args.feature_names = splitFeatureNames(result["features"].as<std::string>());
    args.runtime       = irt::model::ModelRuntime::parse(result["runtime"].as<std::string>());
    args.warmup        = result["warmup"].as<int>();
    args.repeat        = result["repeat"].as<int>();
    if (args.feature_names.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "At least one feature name is required in --features");
    }
    if (args.warmup < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--warmup must be >= 0");
    }
    if (args.repeat <= 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--repeat must be > 0");
    }
    args.image_paths = splitPathList(result["image-path"].as<std::string>());
    args.output_dir  = result["output-dir"].as<std::string>();
    return args;
}

/**
 * @brief 将张量名转换为适合文件名使用的安全字符串。
 * @param value 原始名称。
 * @return 处理后的文件名 stem。
 */
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
    return stem;
}

/**
 * @brief 将原始字节写入二进制文件。
 * @param file_path 输出文件路径。
 * @param bytes 原始字节数据。
 */
void writeBinaryFile(const fs::path &file_path, const std::vector<char> &bytes)
{
    std::ofstream output(file_path, std::ios::binary);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open output file: %s",
                             file_path.string().c_str());
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

/**
 * @brief 将 float 向量写入二进制文件。
 * @param file_path 输出文件路径。
 * @param values float 数据。
 */
void writeFloatBinaryFile(const fs::path &file_path, const std::vector<float> &values)
{
    std::ofstream output(file_path, std::ios::binary);
    if (!output)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open output file: %s",
                             file_path.string().c_str());
    }
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
}

/**
 * @brief 打印单个张量的基本统计信息。
 * @param tensor 张量导出结果。
 */
void printStats(const TensorDump &tensor)
{
    if (tensor.data_type != nvinfer1::DataType::kFLOAT)
    {
        std::cout << tensor.name << " dtype=" << dataTypeToString(tensor.data_type) << " dims=["
                  << dimsToCsv(tensor.dims) << "] bytes=" << tensor.host_bytes.size() << std::endl;
        return;
    }

    const auto *values          = reinterpret_cast<const float *>(tensor.host_bytes.data());
    const auto  count           = tensor.host_bytes.size() / sizeof(float);
    const auto [min_it, max_it] = std::minmax_element(values, values + count);
    const double sum            = std::accumulate(values, values + count, 0.0);
    std::cout << tensor.name << " dims=[" << dimsToCsv(tensor.dims) << "] min=" << *min_it << " max=" << *max_it
              << " mean=" << (sum / static_cast<double>(count)) << std::endl;
}

} // namespace

/**
 * @brief 运行单张或批量图片特征导出 sample，并将输入与中间张量保存到输出目录。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组。
 * @return 成功返回 0，失败返回非 0。
 */
int main(int argc, char *argv[])
{
    try
    {
        const Arguments cli = parseArguments(argc, argv);
        if (!irt::model::isSupportedModel(cli.model_name))
        {
            std::cerr << "Unsupported model: " << cli.model_name << std::endl;
            return -1;
        }

        const fs::path project_root
            = irt::util::findProjectRoot(argv[0], {irt::model::ImageNetUtil::kDefaultImagePath}, __FILE__);
        const std::vector<fs::path> image_paths = resolveImagePaths(project_root, cli.image_paths);
        if (image_paths.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "At least one image path is required");
        }
        if (image_paths.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Batch size is too large: %zu",
                                 image_paths.size());
        }
        const fs::path output_dir = cli.output_dir.empty() ? (fs::current_path() / kDefaultOutputDir) : cli.output_dir;

        auto config = std::make_unique<irt::model::IModelConfig>();
        config->setFeatureTensorNames(cli.feature_names);
        config->setOutputTensorNames(cli.feature_names);
        config->setFeatureOnly(true);
        if (image_paths.size() > 1)
        {
            const int batch_size = static_cast<int>(image_paths.size());
            config->setDynamicBatchRange(1, batch_size, batch_size);
        }
        config->setRuntime(cli.runtime);

        const std::string runtime_model_name
            = cli.runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT ? cli.model_name : "onnx";
        auto model = irt::model::CreateModel(runtime_model_name, std::move(config));
        if (!model)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create model: %s",
                                 runtime_model_name.c_str());
        }

        model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);
        std::cout << "Building or loading feature-only model..." << std::endl;
        const auto build_start = Clock::now();
        model->buildOrLoad(cli.weights_file.string());
        const auto build_end = Clock::now();

        std::vector<cv::Mat> images;
        images.reserve(image_paths.size());
        for (const auto &image_path : image_paths)
        {
            cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
            if (img.empty())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                     image_path.string().c_str());
            }
            images.push_back(std::move(img));
        }

        const auto input_tensor_names = model->ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
        if (input_tensor_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Model must expose at least one input tensor");
        }
        if (input_tensor_names.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Feature sample expects exactly one input tensor, got %zu", input_tensor_names.size());
        }

        auto input_dims = model->tensorShape(input_tensor_names.front());
        if (input_dims.nbDims != 4 || input_dims.d[1] != 3 || input_dims.d[2] <= 0 || input_dims.d[3] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Feature sample expects input shape Nx3xHxW, got %s", dimsToCsv(input_dims).c_str());
        }
        if (input_dims.d[0] != static_cast<int64_t>(images.size()))
        {
            input_dims.d[0] = static_cast<int64_t>(images.size());
            model->setTensorShape(input_tensor_names.front(), input_dims);
        }

        const auto preprocess_start = Clock::now();
        const auto input_data       = preprocessBatch(images, input_dims);
        const auto preprocess_end   = Clock::now();

        const bool uses_tensorrt = cli.runtime.backend() == irt::model::ModelRuntime::Backend::TensorRT;
        const auto stream        = uses_tensorrt ? model->resolveExecutionStream() : nullptr;

        DeviceBuffer d_input;
        if (uses_tensorrt)
        {
            d_input = DeviceBuffer(input_data.size(), nvinfer1::DataType::kFLOAT);
        }

        const auto              output_names = model->ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
        std::vector<TensorDump> dumps;
        dumps.reserve(output_names.size());

        std::vector<DeviceBuffer> device_outputs;
        device_outputs.reserve(output_names.size());
        std::vector<std::vector<std::max_align_t>> host_outputs;
        host_outputs.reserve(output_names.size());

        std::vector<void *> buffers;
        buffers.reserve(1 + output_names.size());
        buffers.push_back(uses_tensorrt ? d_input.data() : const_cast<float *>(input_data.data()));

        for (const auto &output_name : output_names)
        {
            TensorDump tensor;
            tensor.name      = output_name;
            tensor.dims      = model->tensorShape(output_name);
            tensor.data_type = model->tensorDataType(output_name);
            tensor.file_name = sanitizeFileStem(output_name) + ".bin";

            const size_t element_count = elementCount(tensor.dims);
            const size_t num_bytes     = element_count * elementSize(tensor.data_type);
            tensor.host_bytes.resize(num_bytes);

            if (uses_tensorrt)
            {
                device_outputs.emplace_back(element_count, tensor.data_type);
                buffers.push_back(device_outputs.back().data());
            }
            else
            {
                const size_t aligned_words = (num_bytes + sizeof(std::max_align_t) - 1) / sizeof(std::max_align_t);
                host_outputs.emplace_back(aligned_words);
                buffers.push_back(host_outputs.back().data());
            }
            dumps.push_back(std::move(tensor));
        }

        auto run_feature_once = [&]() -> IterationTiming
        {
            IterationTiming timing;

            const auto h2d_start = Clock::now();
            if (uses_tensorrt)
            {
                checkCuda(cudaMemcpyAsync(d_input.data(), input_data.data(), input_data.size() * sizeof(float),
                                          cudaMemcpyHostToDevice, stream),
                          "cudaMemcpyAsync(H2D input)");
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(H2D input)");
            }
            const auto h2d_end = Clock::now();
            timing.h2d_ms      = elapsedMs(h2d_start, h2d_end);

            const auto inference_start = Clock::now();
            model->forwardFeatures(buffers, stream, true);
            if (uses_tensorrt)
            {
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(feature forward)");
            }
            const auto inference_end = Clock::now();
            timing.inference_ms      = elapsedMs(inference_start, inference_end);

            const auto d2h_start = Clock::now();
            if (uses_tensorrt)
            {
                for (size_t i = 0; i < dumps.size(); ++i)
                {
                    checkCuda(cudaMemcpyAsync(dumps[i].host_bytes.data(), device_outputs[i].data(),
                                              dumps[i].host_bytes.size(), cudaMemcpyDeviceToHost, stream),
                              "cudaMemcpyAsync(D2H feature)");
                }
                checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(D2H feature)");
            }
            else
            {
                for (size_t i = 0; i < dumps.size(); ++i)
                {
                    std::memcpy(dumps[i].host_bytes.data(), host_outputs[i].data(), dumps[i].host_bytes.size());
                }
            }
            const auto d2h_end = Clock::now();
            timing.d2h_ms      = elapsedMs(d2h_start, d2h_end);
            return timing;
        };

        std::cout << "Running feature forward warmup=" << cli.warmup << ", repeat=" << cli.repeat << "..." << std::endl;
        for (int i = 0; i < cli.warmup; ++i)
        {
            (void)run_feature_once();
        }

        std::vector<double> h2d_times_ms;
        std::vector<double> inference_times_ms;
        std::vector<double> d2h_times_ms;
        std::vector<double> end_to_end_times_ms;
        h2d_times_ms.reserve(static_cast<size_t>(cli.repeat));
        inference_times_ms.reserve(static_cast<size_t>(cli.repeat));
        d2h_times_ms.reserve(static_cast<size_t>(cli.repeat));
        end_to_end_times_ms.reserve(static_cast<size_t>(cli.repeat));

        const auto infer_start = Clock::now();
        for (int i = 0; i < cli.repeat; ++i)
        {
            const auto timing = run_feature_once();
            h2d_times_ms.push_back(timing.h2d_ms);
            inference_times_ms.push_back(timing.inference_ms);
            d2h_times_ms.push_back(timing.d2h_ms);
            end_to_end_times_ms.push_back(timing.totalMs());
        }
        const auto infer_end = Clock::now();

        const auto postprocess_start = Clock::now();
        fs::create_directories(output_dir);
        writeFloatBinaryFile(output_dir / "input.bin", input_data);

        std::ofstream manifest(output_dir / "manifest.txt");
        if (!manifest)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open manifest file");
        }

        manifest << "version=1\n";
        manifest << "runtime=" << cli.runtime.toString() << "\n";
        manifest << "warmup=" << cli.warmup << "\n";
        manifest << "repeat=" << cli.repeat << "\n";
        manifest << "model_name=" << cli.model_name << "\n";
        manifest << "weights_file=" << fs::absolute(cli.weights_file).generic_string() << "\n";
        manifest << "image_count=" << image_paths.size() << "\n";
        manifest << "image_path=" << fs::absolute(image_paths.front()).generic_string() << "\n";
        for (size_t i = 0; i < image_paths.size(); ++i)
        {
            manifest << "image_path_" << i << "=" << fs::absolute(image_paths[i]).generic_string() << "\n";
        }
        manifest << "feature_only=true\n";
        manifest << "feature_tensor_names=";
        for (size_t i = 0; i < cli.feature_names.size(); ++i)
        {
            if (i > 0)
            {
                manifest << ",";
            }
            manifest << cli.feature_names[i];
        }
        manifest << "\n";
        manifest << "tensor|input|float32|" << dimsToCsv(input_dims) << "|input.bin\n";

        for (const auto &tensor : dumps)
        {
            writeBinaryFile(output_dir / tensor.file_name, tensor.host_bytes);
            manifest << "tensor|" << tensor.name << "|" << dataTypeToString(tensor.data_type) << "|"
                     << dimsToCsv(tensor.dims) << "|" << tensor.file_name << "\n";
        }
        const auto postprocess_end = Clock::now();

        std::cout << "Saved feature dump to: " << fs::absolute(output_dir).string() << std::endl;
        std::cout << "Runtime: " << cli.runtime.toString() << std::endl;
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
        std::cout << "Input dims=[" << dimsToCsv(input_dims) << "]" << std::endl;
        for (const auto &tensor : dumps)
        {
            printStats(tensor);
        }
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
