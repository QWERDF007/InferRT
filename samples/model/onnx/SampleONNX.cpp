#include <cxxopts.hpp>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <inferrt/model/Utils.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <filesystem>
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
 * @brief 用于在显示帮助后中断主流程。
 */
struct HelpRequested
{
};

/**
 * @brief ONNX sample 的命令行参数集合。
 */
struct Arguments
{
    fs::path onnx_file;
    fs::path image_path;
    fs::path label_file;
};

/**
 * @brief 规范化后的图像输入张量形状描述。
 */
struct ImageTensorShape
{
    int batch{1};
    int channels{3};
    int height{224};
    int width{224};
};

/**
 * @brief 输出张量的设备侧与主机侧缓冲描述。
 */
struct OutputBuffer
{
    std::string         name;
    nvinfer1::Dims      dims;
    nvinfer1::DataType  data_type;
    void               *device_ptr{nullptr};
    std::vector<char>   host_bytes;
};

/**
 * @brief 构造命令行选项定义。
 * @param program_name 可执行文件名。
 * @return cxxopts 选项对象。
 */
cxxopts::Options makeOptions(const char *program_name)
{
    cxxopts::Options options(program_name, "Run ONNX models through the InferRT ONNX wrapper");
    options.add_options()("onnx-file,n", "Input ONNX model path (required)", cxxopts::value<std::string>())(
        "image-path,i", "Input image path (required)", cxxopts::value<std::string>())(
        "label-file,l", "ImageNet label file", cxxopts::value<std::string>()->default_value(""))("h,help",
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
        throw HelpRequested{};
    }

    if (!result.count("onnx-file") || !result.count("image-path"))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "--onnx-file and --image-path are required");
    }

    Arguments args;
    args.onnx_file  = result["onnx-file"].as<std::string>();
    args.image_path = result["image-path"].as<std::string>();
    args.label_file = result["label-file"].as<std::string>();
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
 * @brief 将 TensorRT 数据类型转换为便于输出的字符串。
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
 * @brief 将张量维度格式化为形如 `[a, b, c]` 的字符串。
 * @param dims TensorRT 维度对象。
 * @return 格式化后的维度文本。
 */
std::string dimsToString(const nvinfer1::Dims &dims)
{
    std::string result = "[";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (i > 0)
        {
            result += ", ";
        }
        result += std::to_string(dims.d[i]);
    }
    result += "]";
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
                                 "Tensor shape contains non-positive dimension: %d",
                                 dims.d[i]);
        }
        count *= static_cast<size_t>(dims.d[i]);
    }
    return count;
}

/**
 * @brief 将 3D/4D 输入张量维度解析为统一的图像形状描述。
 * @param dims 输入张量维度。
 * @return 规范化后的图像形状。
 */
ImageTensorShape parseImageTensorShape(const nvinfer1::Dims &dims)
{
    if (dims.nbDims == 4)
    {
        return {static_cast<int>(dims.d[0]), static_cast<int>(dims.d[1]), static_cast<int>(dims.d[2]),
                static_cast<int>(dims.d[3])};
    }

    if (dims.nbDims == 3)
    {
        return {1, static_cast<int>(dims.d[0]), static_cast<int>(dims.d[1]), static_cast<int>(dims.d[2])};
    }

    throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Expected 3D or 4D image tensor, got %d dims",
                         dims.nbDims);
}

/**
 * @brief 结合请求 batch 大小解析运行时输入形状。
 * @param engine_dims 引擎中的输入维度。
 * @param requested_batch 请求的 batch 大小。
 * @return 实际运行时维度。
 */
nvinfer1::Dims resolveInputDims(const nvinfer1::Dims &engine_dims, size_t requested_batch)
{
    nvinfer1::Dims resolved = engine_dims;

    if (resolved.nbDims == 4)
    {
        if (resolved.d[0] < 0)
        {
            resolved.d[0] = static_cast<int64_t>(requested_batch);
        }
        else if (static_cast<size_t>(resolved.d[0]) != requested_batch)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Model expects batch size %lld, but %zu images were provided", resolved.d[0],
                                 requested_batch);
        }
    }
    else if (resolved.nbDims == 3 && requested_batch != 1)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Model does not expose an explicit batch dimension; provide exactly one image");
    }

    return resolved;
}

/**
 * @brief 将单张图片预处理为匹配模型输入的 CHW float 数据。
 * @param img 输入图像。
 * @param shape 目标张量形状。
 * @return 单张图片的预处理结果。
 */
std::vector<float> preprocessOne(const cv::Mat &img, const ImageTensorShape &shape)
{
    if (shape.channels != 1 && shape.channels != 3)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "Only 1-channel or 3-channel image input is supported");
    }

    cv::Mat converted;
    if (shape.channels == 3)
    {
        converted = irt::model::ImageNetUtil::preprocess(img, cv::Size(shape.width, shape.height));
        return irt::model::ImageNetUtil::imageToTensorCHW(converted);
    }
    else
    {
        cv::cvtColor(img, converted, cv::COLOR_BGR2GRAY);
    }

    cv::Mat resized;
    cv::resize(converted, resized, cv::Size(shape.width, shape.height), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(resized, shape.channels == 3 ? CV_32FC3 : CV_32FC1, 1.0 / 255.0);

    std::vector<float> tensor(static_cast<size_t>(shape.channels) * shape.height * shape.width);
    std::memcpy(tensor.data(), resized.data, static_cast<size_t>(shape.height) * shape.width * sizeof(float));

    return tensor;
}

/**
 * @brief 批量加载并预处理输入图片。
 * @param image_paths 图片路径列表。
 * @param shape 目标张量形状。
 * @return 连续存储的 batch 输入数据。
 */
std::vector<float> preprocessBatch(const std::vector<fs::path> &image_paths, const ImageTensorShape &shape)
{
    const size_t image_elements = static_cast<size_t>(shape.channels) * shape.height * shape.width;
    std::vector<float> batch_data(image_paths.size() * image_elements);

    for (size_t i = 0; i < image_paths.size(); ++i)
    {
        cv::Mat image = cv::imread(fs::absolute(image_paths[i]).generic_string());
        if (image.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Failed to load image: %s",
                                 image_paths[i].string().c_str());
        }

        const auto single = preprocessOne(image, shape);
        std::memcpy(batch_data.data() + i * image_elements, single.data(), image_elements * sizeof(float));
    }

    return batch_data;
}

/**
 * @brief 按输入数据类型将 float 输入编码为字节缓冲。
 * @param input_data 预处理后的 float 数据。
 * @param data_type 模型输入数据类型。
 * @return 可直接拷贝到设备端的字节数组。
 */
std::vector<char> encodeInputBuffer(const std::vector<float> &input_data, nvinfer1::DataType data_type)
{
    std::vector<char> bytes(input_data.size() * elementSize(data_type));

    switch (data_type)
    {
    case nvinfer1::DataType::kFLOAT:
        std::memcpy(bytes.data(), input_data.data(), bytes.size());
        return bytes;
    case nvinfer1::DataType::kHALF:
    {
        auto *dst = reinterpret_cast<__half *>(bytes.data());
        for (size_t i = 0; i < input_data.size(); ++i)
        {
            dst[i] = __float2half(input_data[i]);
        }
        return bytes;
    }
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                             "This sample currently supports float32/float16 image input only");
    }
}

/**
 * @brief 将原始字节缓冲按目标类型转换为 double 数组，便于统一展示。
 * @tparam T 原始元素类型。
 * @param bytes 原始字节缓冲。
 * @return double 数组。
 */
template<typename T>
std::vector<double> castBufferToDouble(const std::vector<char> &bytes)
{
    const size_t count = bytes.size() / sizeof(T);
    const auto  *src   = reinterpret_cast<const T *>(bytes.data());
    std::vector<double> values(count);
    for (size_t i = 0; i < count; ++i)
    {
        values[i] = static_cast<double>(src[i]);
    }
    return values;
}

/**
 * @brief 按输出张量数据类型解码主机缓冲。
 * @param output 输出张量描述。
 * @return 统一转为 double 的输出值数组。
 */
std::vector<double> decodeOutputBuffer(const OutputBuffer &output)
{
    switch (output.data_type)
    {
    case nvinfer1::DataType::kFLOAT:
        return castBufferToDouble<float>(output.host_bytes);
    case nvinfer1::DataType::kHALF:
    {
        const size_t count = output.host_bytes.size() / sizeof(__half);
        const auto  *src   = reinterpret_cast<const __half *>(output.host_bytes.data());
        std::vector<double> values(count);
        for (size_t i = 0; i < count; ++i)
        {
            values[i] = static_cast<double>(__half2float(src[i]));
        }
        return values;
    }
    case nvinfer1::DataType::kINT8:
        return castBufferToDouble<int8_t>(output.host_bytes);
    case nvinfer1::DataType::kUINT8:
        return castBufferToDouble<uint8_t>(output.host_bytes);
    case nvinfer1::DataType::kINT32:
        return castBufferToDouble<int32_t>(output.host_bytes);
    case nvinfer1::DataType::kINT64:
        return castBufferToDouble<int64_t>(output.host_bytes);
    case nvinfer1::DataType::kBOOL:
        return castBufferToDouble<uint8_t>(output.host_bytes);
    default:
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Unsupported output tensor data type");
    }
}

/**
 * @brief 打印单个输出向量的 Top-K 结果。
 * @param tensor_name 输出张量名。
 * @param data 输出数值。
 * @param top_k 输出的前 K 个结果。
 * @param labels 可选标签列表。
 */
void printTopK(const std::string &tensor_name, const std::vector<double> &data, size_t top_k,
               const std::vector<std::string> &labels)
{
    if (data.empty())
    {
        std::cout << "Output tensor: " << tensor_name << " is empty" << std::endl;
        return;
    }

    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    const size_t actual_top_k = std::min(top_k, data.size());
    std::partial_sort(indices.begin(), indices.begin() + actual_top_k, indices.end(),
                      [&](size_t lhs, size_t rhs) { return data[lhs] > data[rhs]; });

    std::cout << "Output tensor: " << tensor_name << std::endl;
    for (size_t i = 0; i < actual_top_k; ++i)
    {
        const size_t idx   = indices[i];
        const double score = data[idx];
        if (idx < labels.size() && !labels[idx].empty())
        {
            std::cout << "  top: " << (i + 1) << ", confidence: " << score << ", label[" << idx
                      << "]: " << labels[idx] << std::endl;
        }
        else
        {
            std::cout << "  top: " << (i + 1) << ", confidence: " << score << ", index[" << idx << "]"
                      << std::endl;
        }
    }
}

/**
 * @brief 打印输出张量摘要；若包含 batch 维则按样本拆分展示。
 * @param output 输出张量描述。
 * @param batch_size 本次推理的 batch 大小。
 * @param labels 可选标签列表。
 */
void printOutputSummary(const OutputBuffer &output, size_t batch_size, const std::vector<std::string> &labels)
{
    std::cout << "Output tensor shape: " << dimsToString(output.dims) << ", dtype=" << dataTypeToString(output.data_type)
              << std::endl;

    if (output.host_bytes.empty())
    {
        return;
    }

    const auto values = decodeOutputBuffer(output);

    bool split_by_batch
        = output.dims.nbDims >= 1 && output.dims.d[0] > 0 && static_cast<size_t>(output.dims.d[0]) == batch_size;
    if (!split_by_batch)
    {
        printTopK(output.name, values, 3, labels);
        return;
    }

    const size_t per_sample_elements = values.size() / batch_size;
    for (size_t batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        const auto begin = values.begin() + static_cast<std::ptrdiff_t>(batch_index * per_sample_elements);
        const auto end   = begin + static_cast<std::ptrdiff_t>(per_sample_elements);
        std::vector<double> sample_values(begin, end);
        printTopK(output.name + " [batch " + std::to_string(batch_index) + "]", sample_values, 3, labels);
    }
}

} // namespace

/**
 * @brief 运行 ONNX 包装模型推理，并打印各输出张量摘要。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组。
 * @return 成功返回 0，失败返回非 0。
 */
int main(int argc, char *argv[])
{
    try
    {
        const Arguments args = parseArguments(argc, argv);

        auto model = irt::model::CreateModel("onnx");
        if (!model)
        {
            std::cerr << "Failed to create ONNX model wrapper." << std::endl;
            return -1;
        }

        model->setLogLevel(nvinfer1::ILogger::Severity::kINFO);

        std::cout << "Building or loading ONNX model..." << std::endl;
        model->buildOrLoad(args.onnx_file.string());
        std::cout << "Model loaded successfully." << std::endl;

        const auto input_names  = model->ioTensorNames(nvinfer1::TensorIOMode::kINPUT);
        const auto output_names = model->ioTensorNames(nvinfer1::TensorIOMode::kOUTPUT);
        if (input_names.empty() || output_names.empty())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION,
                                 "Engine must contain at least one input and one output");
        }
        if (input_names.size() != 1)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "This sample currently supports exactly one input tensor, got %zu", input_names.size());
        }

        const std::string &input_name        = input_names.front();
        const auto         input_engine_dims = model->tensorShape(input_name);
        const auto         input_type        = model->tensorDataType(input_name);
        const bool supports_image_input
            = input_type == nvinfer1::DataType::kFLOAT || input_type == nvinfer1::DataType::kHALF;
        if (!supports_image_input)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "This sample currently supports float32/float16 image input only");
        }

        const auto resolved_input_dims = resolveInputDims(input_engine_dims, 1);
        model->setTensorShape(input_name, resolved_input_dims);
        const auto input_runtime_dims = model->tensorShape(input_name);
        const auto image_shape        = parseImageTensorShape(input_runtime_dims);

        std::cout << "Input tensor: " << input_name << " engine_shape=" << dimsToString(input_engine_dims)
                  << " runtime_shape=" << dimsToString(input_runtime_dims)
                  << " dtype=" << dataTypeToString(input_type) << std::endl;
        for (const auto &output_name : output_names)
        {
            std::cout << "Output tensor: " << output_name << " shape=" << dimsToString(model->tensorShape(output_name))
                      << " dtype=" << dataTypeToString(model->tensorDataType(output_name))
                      << std::endl;
        }

        const auto preprocess_start = Clock::now();
        const std::vector<fs::path> image_paths = {args.image_path};
        const std::vector<float> input_data = preprocessBatch(image_paths, image_shape);
        const std::vector<char> input_host_bytes = encodeInputBuffer(input_data, input_type);
        const auto preprocess_end = Clock::now();

        std::vector<void *> device_buffers;
        device_buffers.reserve(input_names.size() + output_names.size());

        void *d_input      = nullptr;
        size_t input_bytes = input_host_bytes.size();
        const auto stream = model->resolveExecutionStream();
        cudaMalloc(&d_input, input_bytes);
        cudaMemcpyAsync(d_input, input_host_bytes.data(), input_bytes, cudaMemcpyHostToDevice, stream);
        device_buffers.push_back(d_input);

        std::vector<OutputBuffer> outputs;
        outputs.reserve(output_names.size());
        for (const auto &output_name : output_names)
        {
            const auto output_dims  = model->tensorShape(output_name);
            const auto output_type  = model->tensorDataType(output_name);
            const auto output_count = elementCount(output_dims);

            OutputBuffer output;
            output.name      = output_name;
            output.dims      = output_dims;
            output.data_type = output_type;
            output.host_bytes.resize(output_count * elementSize(output_type));

            const size_t output_bytes = output_count * elementSize(output_type);
            cudaMalloc(&output.device_ptr, output_bytes);
            device_buffers.push_back(output.device_ptr);
            outputs.push_back(std::move(output));
        }

        std::cout << "Running inference with batch size 1..." << std::endl;
        const auto infer_start = Clock::now();
        model->infer(device_buffers, stream, true);

        const auto postprocess_start = Clock::now();
        for (auto &output : outputs)
        {
            cudaMemcpyAsync(output.host_bytes.data(), output.device_ptr, output.host_bytes.size(),
                            cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
        const auto infer_end = Clock::now();

        std::vector<std::string> labels;
        if (!args.label_file.empty() && fs::exists(args.label_file))
        {
            labels = irt::model::readImagenetLabels(args.label_file.string());
        }
        const auto postprocess_end = Clock::now();

        std::cout << "Timing: preprocess=" << elapsedMs(preprocess_start, preprocess_end)
                  << " ms, inference=" << elapsedMs(infer_start, infer_end)
                  << " ms, postprocess=" << elapsedMs(postprocess_start, postprocess_end) << " ms" << std::endl;
        std::cout << "\nInference summary:" << std::endl;
        for (const auto &output : outputs)
        {
            printOutputSummary(output, 1, labels);
        }

        for (auto &output : outputs)
        {
            cudaFree(output.device_ptr);
        }
        cudaFree(d_input);

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
