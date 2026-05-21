#include <cxxopts.hpp>
#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <inferrt/util/Path.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

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
    fs::path                 image_path;
    fs::path                 output_dir;
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

/**
 * @brief 简单的 CUDA 设备内存 RAII 封装。
 */
class CudaBuffer
{
public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t num_bytes)
    {
        allocate(num_bytes);
    }

    ~CudaBuffer()
    {
        if (ptr_)
        {
            cudaFree(ptr_);
        }
    }

    CudaBuffer(const CudaBuffer &)            = delete;
    CudaBuffer &operator=(const CudaBuffer &) = delete;

    CudaBuffer(CudaBuffer &&other) noexcept
        : ptr_(other.ptr_)
        , num_bytes_(other.num_bytes_)
    {
        other.ptr_       = nullptr;
        other.num_bytes_ = 0;
    }

    CudaBuffer &operator=(CudaBuffer &&other) noexcept
    {
        if (this != &other)
        {
            if (ptr_)
            {
                cudaFree(ptr_);
            }
            ptr_             = other.ptr_;
            num_bytes_       = other.num_bytes_;
            other.ptr_       = nullptr;
            other.num_bytes_ = 0;
        }
        return *this;
    }

    void allocate(size_t num_bytes)
    {
        if (ptr_)
        {
            cudaFree(ptr_);
            ptr_       = nullptr;
            num_bytes_ = 0;
        }

        if (num_bytes == 0)
        {
            return;
        }

        checkCuda(cudaMalloc(&ptr_, num_bytes), "cudaMalloc");
        num_bytes_ = num_bytes;
    }

    void *get() const noexcept
    {
        return ptr_;
    }

private:
    static void checkCuda(cudaError_t status, const char *op)
    {
        if (status != cudaSuccess)
        {
            throw irt::Exception(irt::Status::ERROR_INTERNAL, "%s failed: %s", op, cudaGetErrorString(status));
        }
    }

    void  *ptr_{nullptr};
    size_t num_bytes_{0};
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

/**
 * @brief 构造命令行选项定义。
 * @param program_name 可执行文件名。
 * @return cxxopts 选项对象。
 */
cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Dump intermediate feature tensors from InferRT models");
    options.add_options()("model,m", "Built-in model name (required)", cxxopts::value<std::string>())(
        "weights-file,w", "Weights file (.wts) (required)", cxxopts::value<std::string>())(
        "features,f", "Comma-separated feature tensor names (required)", cxxopts::value<std::string>())(
        "image-path,i", "Input image path", cxxopts::value<std::string>()->default_value(""))(
        "output-dir,o", "Output directory", cxxopts::value<std::string>()->default_value(""))("h,help",
                                                                                            "Show help");
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
    auto options = makeOptions(argv[0]);
    const auto result = options.parse(argc, argv);
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        std::cout << "Default image: " << irt::model::ImageNetUtil::kDefaultImagePath.generic_string() << std::endl;
        std::cout << "Default output dir: " << kDefaultOutputDir.generic_string() << std::endl;
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
    if (args.feature_names.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "At least one feature name is required in --features");
    }
    args.image_path = result["image-path"].as<std::string>();
    args.output_dir = result["output-dir"].as<std::string>();
    return args;
}

/**
 * @brief 返回 TensorRT 数据类型的单元素字节数。
 * @param data_type TensorRT 数据类型。
 * @return 单元素字节数。
 */
size_t elementSize(nvinfer1::DataType data_type)
{
    using nvinfer1::DataType;

    switch (data_type)
    {
    case DataType::kFLOAT:
    case DataType::kINT32:
        return 4;
    case DataType::kHALF:
        return 2;
    case DataType::kINT8:
    case DataType::kBOOL:
    case DataType::kUINT8:
        return 1;
    case DataType::kINT64:
        return 8;
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported TensorRT data type");
    }
}

/**
 * @brief 将 TensorRT 数据类型转换为便于写入 manifest 的字符串。
 * @param data_type TensorRT 数据类型。
 * @return 类型名字符串。
 */
std::string dataTypeToString(nvinfer1::DataType data_type)
{
    using nvinfer1::DataType;

    switch (data_type)
    {
    case DataType::kFLOAT:
        return "float32";
    case DataType::kHALF:
        return "float16";
    case DataType::kINT8:
        return "int8";
    case DataType::kUINT8:
        return "uint8";
    case DataType::kINT32:
        return "int32";
    case DataType::kINT64:
        return "int64";
    case DataType::kBOOL:
        return "bool";
    default:
        return "unknown";
    }
}

/**
 * @brief 将张量维度序列格式化为逗号分隔字符串。
 * @param dims TensorRT 维度对象。
 * @return 逗号分隔后的维度文本。
 */
std::string dimsToCsv(const nvinfer1::Dims &dims)
{
    std::string result;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (i > 0)
        {
            result += ",";
        }
        result += std::to_string(dims.d[i]);
    }
    return result;
}

/**
 * @brief 计算张量元素总数，并校验每一维均为正数。
 * @param dims TensorRT 维度对象。
 * @return 元素总数。
 */
size_t elementCount(const nvinfer1::Dims &dims)
{
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Tensor shape contains non-positive dimension: %d", dims.d[i]);
        }
        count *= static_cast<size_t>(dims.d[i]);
    }
    return count;
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
 * @brief 获取特征输出张量名。
 * @param config 模型配置。
 * @return 输出张量名列表。
 */
std::vector<std::string> featureOutputNames(const irt::model::IModelConfig &config)
{
    if (!config.featureOutputTensorNames().empty())
    {
        return config.featureOutputTensorNames();
    }
    return config.featureTensorNames();
}

/**
 * @brief 包装 CUDA 调用，失败时抛出带上下文的异常。
 * @param status CUDA 返回状态。
 * @param op 当前操作名。
 */
void checkCuda(cudaError_t status, const char *op)
{
    if (status != cudaSuccess)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "%s failed: %s", op, cudaGetErrorString(status));
    }
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
 * @brief 运行特征导出 sample，并将输入与中间张量保存到输出目录。
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
        const fs::path image_path = cli.image_path.empty() ? (project_root / irt::model::ImageNetUtil::kDefaultImagePath)
                                                           : cli.image_path;
        const fs::path output_dir = cli.output_dir.empty() ? (fs::current_path() / kDefaultOutputDir) : cli.output_dir;

        auto config = std::make_unique<irt::model::IModelConfig>();
        config->setFeatureTensorNames(cli.feature_names);
        config->setFeatureOnly(true);
        auto model = irt::model::CreateModel(cli.model_name, std::move(config));
        if (!model)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create model: %s",
                                 cli.model_name.c_str());
        }

        model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);
        std::cout << "Building or loading feature-only model..." << std::endl;
        model->buildOrLoad(cli.weights_file.string());

        cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (img.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                 image_path.string().c_str());
        }

        const auto preprocess_start = Clock::now();
        const auto preprocessed = irt::model::ImageNetUtil::preprocess(img);
        const auto input_data = irt::model::ImageNetUtil::imageToTensorCHW(preprocessed);
        const auto preprocess_end = Clock::now();

        CudaBuffer d_input(input_data.size() * sizeof(float));
        checkCuda(
            cudaMemcpy(d_input.get(), input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy(H2D input)");

        std::vector<std::string> output_names = featureOutputNames(model->modelConfig());
        std::vector<TensorDump>  dumps;
        dumps.reserve(output_names.size());

        std::vector<CudaBuffer> device_outputs;
        device_outputs.reserve(output_names.size());

        std::vector<void *> buffers;
        buffers.reserve(1 + output_names.size());
        buffers.push_back(d_input.get());

        for (const auto &output_name : output_names)
        {
            TensorDump tensor;
            tensor.name      = output_name;
            tensor.dims      = model->tensorShape(output_name);
            tensor.data_type = model->tensorDataType(output_name);
            tensor.file_name = sanitizeFileStem(output_name) + ".bin";

            const size_t num_bytes = elementCount(tensor.dims) * elementSize(tensor.data_type);
            tensor.host_bytes.resize(num_bytes);

            device_outputs.emplace_back(num_bytes);
            buffers.push_back(device_outputs.back().get());
            dumps.push_back(std::move(tensor));
        }

        std::cout << "Running feature forward..." << std::endl;
        const auto infer_start = Clock::now();
        model->forwardFeatures(buffers);
        const auto infer_end = Clock::now();

        const auto postprocess_start = Clock::now();
        for (size_t i = 0; i < dumps.size(); ++i)
        {
            checkCuda(cudaMemcpy(dumps[i].host_bytes.data(), device_outputs[i].get(), dumps[i].host_bytes.size(),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(D2H feature)");
        }

        fs::create_directories(output_dir);
        writeFloatBinaryFile(output_dir / "input.bin", input_data);

        std::ofstream manifest(output_dir / "manifest.txt");
        if (!manifest)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to open manifest file");
        }

        manifest << "version=1\n";
        manifest << "backend=inferrt\n";
        manifest << "model_name=" << cli.model_name << "\n";
        manifest << "weights_file=" << fs::absolute(cli.weights_file).generic_string() << "\n";
        manifest << "image_path=" << fs::absolute(image_path).generic_string() << "\n";
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
        manifest << "tensor|input|float32|1,3,224,224|input.bin\n";

        for (const auto &tensor : dumps)
        {
            writeBinaryFile(output_dir / tensor.file_name, tensor.host_bytes);
            manifest << "tensor|" << tensor.name << "|" << dataTypeToString(tensor.data_type) << "|"
                     << dimsToCsv(tensor.dims) << "|" << tensor.file_name << "\n";
        }
        const auto postprocess_end = Clock::now();

        std::cout << "Saved feature dump to: " << fs::absolute(output_dir).string() << std::endl;
        std::cout << "Timing: preprocess=" << elapsedMs(preprocess_start, preprocess_end)
                  << " ms, inference=" << elapsedMs(infer_start, infer_end)
                  << " ms, postprocess=" << elapsedMs(postprocess_start, postprocess_end) << " ms" << std::endl;
        std::cout << "Input dims=[1,3,224,224]" << std::endl;
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
