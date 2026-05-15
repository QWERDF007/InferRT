#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
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

struct Arguments
{
    fs::path onnx_file;
    fs::path image_path;
    fs::path label_file;
};

struct ImageTensorShape
{
    int batch{1};
    int channels{3};
    int height{224};
    int width{224};
};

struct OutputBuffer
{
    std::string         name;
    nvinfer1::Dims      dims;
    nvinfer1::DataType  data_type;
    void               *device_ptr{nullptr};
    std::vector<char>   host_bytes;
};

void printUsage(const char *program_name)
{
    std::cerr << "Usage: " << program_name << " <model.onnx> <image_path> [label_file]" << std::endl;
    std::cerr << "Example: " << program_name
              << " samples/model/onnx/alexnet.onnx assets/pics/dog.jpg assets/imagenet1000_clsidx_to_labels.txt"
              << std::endl;
}

Arguments parseArguments(int argc, char *argv[])
{
    if (argc < 3 || argc > 4)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Expected 2 or 3 arguments after program name");
    }

    Arguments args;
    args.onnx_file  = fs::path(argv[1]);
    args.image_path = fs::path(argv[2]);
    args.label_file = (argc == 4) ? fs::path(argv[3]) : fs::path();
    return args;
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
        cv::cvtColor(img, converted, cv::COLOR_BGR2RGB);
    }
    else
    {
        cv::cvtColor(img, converted, cv::COLOR_BGR2GRAY);
    }

    cv::Mat resized;
    cv::resize(converted, resized, cv::Size(shape.width, shape.height), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(resized, shape.channels == 3 ? CV_32FC3 : CV_32FC1, 1.0 / 255.0);

    if (shape.channels == 3)
    {
        cv::Scalar mean(0.485, 0.456, 0.406);
        cv::Scalar std(0.229, 0.224, 0.225);
        cv::subtract(resized, mean, resized);
        cv::divide(resized, std, resized);
    }

    std::vector<float> tensor(static_cast<size_t>(shape.channels) * shape.height * shape.width);
    if (shape.channels == 3)
    {
        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);
        for (int c = 0; c < 3; ++c)
        {
            std::memcpy(tensor.data() + static_cast<size_t>(c) * shape.height * shape.width, channels[c].data,
                        static_cast<size_t>(shape.height) * shape.width * sizeof(float));
        }
    }
    else
    {
        std::memcpy(tensor.data(), resized.data, static_cast<size_t>(shape.height) * shape.width * sizeof(float));
    }

    return tensor;
}

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

int main(int argc, char *argv[])
{
    try
    {
        if (argc < 3)
        {
            printUsage(argv[0]);
            return -1;
        }

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

        model->setInputTensorNames(input_names);
        model->setOutputTensorNames(output_names);

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

        const std::vector<fs::path> image_paths   = {args.image_path};
        const std::vector<float>    input_data    = preprocessBatch(image_paths, image_shape);
        const std::vector<char>  input_host_bytes = encodeInputBuffer(input_data, input_type);

        std::vector<void *> device_buffers;
        device_buffers.reserve(input_names.size() + output_names.size());

        void *d_input      = nullptr;
        size_t input_bytes = input_host_bytes.size();
        cudaMalloc(&d_input, input_bytes);
        cudaMemcpy(d_input, input_host_bytes.data(), input_bytes, cudaMemcpyHostToDevice);
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
        model->infer(device_buffers);

        for (auto &output : outputs)
        {
            cudaMemcpy(output.host_bytes.data(), output.device_ptr, output.host_bytes.size(), cudaMemcpyDeviceToHost);
        }

        std::vector<std::string> labels;
        if (!args.label_file.empty() && fs::exists(args.label_file))
        {
            labels = irt::model::readImagenetLabels(args.label_file.string());
        }

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
}
