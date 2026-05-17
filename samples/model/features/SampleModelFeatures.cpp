#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct ModelSampleSpec
{
    const char *name;
};

struct Arguments
{
    std::string              model_name;
    fs::path                 weights_file;
    std::vector<std::string> feature_names;
    fs::path                 image_path;
    fs::path                 output_dir;
};

struct TensorDump
{
    std::string        name;
    nvinfer1::Dims     dims;
    nvinfer1::DataType data_type;
    std::vector<char>  host_bytes;
    std::string        file_name;
};

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
        : ptr_(other.ptr_), num_bytes_(other.num_bytes_)
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

inline constexpr std::array<ModelSampleSpec, 15> kSupportedModels = {
    {
     {"alexnet"},
     {"mobilenet_v2"},
     {"mobilenet_v3_large"},
     {"mobilenet_v3_small"},
     {"vgg11"},
     {"vgg13"},
     {"vgg16"},
     {"vgg19"},
     {"resnet18"},
     {"resnet34"},
     {"resnet50"},
     {"resnet101"},
     {"resnet152"},
     {"wide_resnet50_2"},
     {"wide_resnet101_2"},
     }
};

const fs::path kDefaultImagePath = "assets/pics/dog.jpg";
const fs::path kDefaultOutputDir = "feature_dump_cpp";

bool isSupportedModel(const std::string &model_name)
{
    return std::any_of(kSupportedModels.begin(), kSupportedModels.end(),
                       [&model_name](const auto &item) { return item.name == model_name; });
}

fs::path findProjectRoot(const char *program_name)
{
    std::vector<fs::path> starts;
    starts.push_back(fs::path(__FILE__).parent_path());
    starts.push_back(fs::current_path());
    if (program_name && *program_name)
    {
        starts.push_back(fs::absolute(program_name).parent_path());
    }

    for (auto start : starts)
    {
        for (fs::path path = fs::absolute(start); !path.empty(); path = path.parent_path())
        {
            if (fs::exists(path / kDefaultImagePath))
            {
                return path;
            }
            if (path == path.root_path())
            {
                break;
            }
        }
    }

    return fs::current_path();
}

std::string trim(std::string value)
{
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

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

void printUsage(const char *program_name)
{
    std::cerr << "Usage: " << program_name
              << " <model_name> <weights_file.wts> <feature_a,feature_b,...> [image_path] [output_dir]" << std::endl;
    std::cerr << "Default image: " << kDefaultImagePath.generic_string() << std::endl;
    std::cerr << "Default output dir: " << kDefaultOutputDir.generic_string() << std::endl;
    std::cerr << "Supported models:";
    for (const auto &model : kSupportedModels)
    {
        std::cerr << ' ' << model.name;
    }
    std::cerr << std::endl;
    std::cerr << "Example: " << program_name
              << " resnet18 samples/model/classification/resnet18.wts layer1,layer4 assets/pics/dog.jpg"
              << " build/feature_dump_cpp" << std::endl;
}

Arguments parseArguments(int argc, char *argv[])
{
    if (argc < 4 || argc > 6)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Expected 3 to 5 arguments after program name");
    }

    Arguments args;
    args.model_name    = argv[1];
    args.weights_file  = fs::path(argv[2]);
    args.feature_names = splitFeatureNames(argv[3]);
    if (args.feature_names.empty())
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "At least one feature name is required");
    }
    args.image_path = (argc >= 5) ? fs::path(argv[4]) : fs::path();
    args.output_dir = (argc >= 6) ? fs::path(argv[5]) : fs::path();
    return args;
}

cv::Mat preprocess(const cv::Mat &img)
{
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    cv::subtract(resized, mean, resized);
    cv::divide(resized, std, resized);
    return resized;
}

std::vector<float> makeInputTensor(const cv::Mat &preprocessed)
{
    std::vector<float>   input_data(1 * 3 * 224 * 224);
    std::vector<cv::Mat> channels(3);
    cv::split(preprocessed, channels);
    for (int c = 0; c < 3; ++c)
    {
        std::memcpy(input_data.data() + c * 224 * 224, channels[c].data, 224 * 224 * sizeof(float));
    }
    return input_data;
}

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

std::vector<std::string> featureOutputNames(const irt::model::IModelConfig &config)
{
    if (!config.featureOutputTensorNames().empty())
    {
        return config.featureOutputTensorNames();
    }
    return config.featureTensorNames();
}

void checkCuda(cudaError_t status, const char *op)
{
    if (status != cudaSuccess)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "%s failed: %s", op, cudaGetErrorString(status));
    }
}

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

void printStats(const TensorDump &tensor)
{
    if (tensor.data_type != nvinfer1::DataType::kFLOAT)
    {
        std::cout << tensor.name << " dtype=" << dataTypeToString(tensor.data_type) << " dims=[" << dimsToCsv(tensor.dims)
                  << "] bytes=" << tensor.host_bytes.size() << std::endl;
        return;
    }

    const auto *values = reinterpret_cast<const float *>(tensor.host_bytes.data());
    const auto  count  = tensor.host_bytes.size() / sizeof(float);
    const auto [min_it, max_it] = std::minmax_element(values, values + count);
    const double sum = std::accumulate(values, values + count, 0.0);
    std::cout << tensor.name << " dims=[" << dimsToCsv(tensor.dims) << "] min=" << *min_it << " max=" << *max_it
              << " mean=" << (sum / static_cast<double>(count)) << std::endl;
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        const Arguments cli = parseArguments(argc, argv);
        if (!isSupportedModel(cli.model_name))
        {
            std::cerr << "Unsupported model: " << cli.model_name << std::endl;
            printUsage(argv[0]);
            return -1;
        }

        const fs::path project_root = findProjectRoot(argv[0]);
        const fs::path image_path   = cli.image_path.empty() ? (project_root / kDefaultImagePath) : cli.image_path;
        const fs::path output_dir   = cli.output_dir.empty() ? (fs::current_path() / kDefaultOutputDir) : cli.output_dir;

        auto config = std::make_unique<irt::model::IModelConfig>();
        config->setFeatureTensorNames(cli.feature_names);
        auto model = irt::model::CreateModel(cli.model_name, std::move(config));
        if (!model)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to create model: %s",
                                 cli.model_name.c_str());
        }

        model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);
        std::cout << "Building or loading feature-enabled model..." << std::endl;
        model->buildOrLoad(cli.weights_file.string());

        cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
        if (img.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                 image_path.string().c_str());
        }

        const auto preprocessed = preprocess(img);
        const auto input_data   = makeInputTensor(preprocessed);

        CudaBuffer d_input(input_data.size() * sizeof(float));
        checkCuda(cudaMemcpy(d_input.get(), input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice),
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
        model->forwardFeatures(buffers);

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

        std::cout << "Saved feature dump to: " << fs::absolute(output_dir).string() << std::endl;
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
        printUsage(argv[0]);
        return -1;
    }
}
