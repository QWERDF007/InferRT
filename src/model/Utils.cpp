#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/model/Utils.hpp>
#include <opencv2/imgproc.hpp>

#include <cstring>
#include <fstream>


namespace irt::model {

const std::filesystem::path ImageNetUtil::kDefaultImagePath = "assets/pics/dog.jpg";
const std::filesystem::path ImageNetUtil::kDefaultLabelPath = "assets/imagenet1000_clsidx_to_labels.txt";

/**
 * @brief 解析文本格式的 `.wts` 权重文件。
 * @param file 权重文件路径。
 * @return 权重映射表。
 */
WeightsMap loadWeights(const std::string &file)
{
    WeightsMap weights_map;

    std::ifstream input(file);
    if (!input.is_open())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open weights file: %s", file.c_str());
    }

    int32_t count;
    if (!(input >> count) || count <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to read valid count of weights from file: %s",
                             file.c_str());
    }

    while (count--)
    {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> wt.count;

        // Load blob
        auto *val = new uint32_t[wt.count];
        input >> std::hex;
        for (auto x = 0ll; x < wt.count; ++x)
        {
            input >> val[x];
        }
        wt.values         = val;
        weights_map[name] = wt;
    }

    return weights_map;
}

/**
 * @brief 读取 ImageNet 标签文件。
 * @param label_file 标签文件路径。
 * @return 长度为 1000 的标签数组。
 */
std::vector<std::string> readImagenetLabels(const std::string &label_file)
{
    std::vector<std::string> labels(1000);
    std::ifstream            file(label_file);
    if (!file.is_open())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open weights file: %s", label_file.c_str());
    }

    std::string line;
    while (std::getline(file, line))
    {
        size_t colon_pos = line.find(": ");
        if (colon_pos != std::string::npos)
        {
            int         idx   = std::stoi(line.substr(0, colon_pos));
            std::string label = line.substr(colon_pos + 2);
            if (label.size() >= 2 && label.front() == '\'' && label.back() == ',')
            {
                label = label.substr(1, label.size() - 3);
            }
            if (idx >= 0 && idx < 1000)
            {
                labels[idx] = label;
            }
        }
    }

    return labels;
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

size_t elementSize(nvinfer1::DataType data_type)
{
    switch (data_type)
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
        return 1;
    case nvinfer1::DataType::kINT64:
        return 8;
    default:
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED, "Unsupported TensorRT data type");
    }
}

size_t dataTypeSize(nvinfer1::DataType data_type)
{
    return elementSize(data_type);
}

std::string dataTypeToString(nvinfer1::DataType data_type)
{
    switch (data_type)
    {
    case nvinfer1::DataType::kFLOAT:
        return "float32";
    case nvinfer1::DataType::kHALF:
        return "float16";
    case nvinfer1::DataType::kINT8:
        return "int8";
    case nvinfer1::DataType::kUINT8:
        return "uint8";
    case nvinfer1::DataType::kINT32:
        return "int32";
    case nvinfer1::DataType::kINT64:
        return "int64";
    case nvinfer1::DataType::kBOOL:
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

void checkCuda(cudaError_t status, const char *op)
{
    if (status != cudaSuccess)
    {
        throw irt::Exception(irt::Status::ERROR_INTERNAL, "%s failed: %s", op, cudaGetErrorString(status));
    }
}

void setCudaDevice(int device_id)
{
    if (device_id < 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "CUDA device id must be non-negative, got %d", device_id);
    }
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
}

cv::Mat ImageNetUtil::preprocess(const cv::Mat &bgr_image, cv::Size target_size)
{
    if (bgr_image.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input image is empty");
    }
    if (target_size.width <= 0 || target_size.height <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid target size: %dx%d", target_size.width,
                             target_size.height);
    }

    cv::Mat rgb;
    cv::cvtColor(bgr_image, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, target_size, 0, 0, cv::INTER_LINEAR);
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

    const cv::Scalar mean(0.485, 0.456, 0.406);
    const cv::Scalar std(0.229, 0.224, 0.225);
    cv::subtract(resized, mean, resized);
    cv::divide(resized, std, resized);
    return resized;
}

std::vector<float> ImageNetUtil::imageToTensorCHW(const cv::Mat &image)
{
    if (image.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input image is empty");
    }
    if (image.type() != CV_32FC3)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Expected CV_32FC3 image, got type=%d", image.type());
    }

    const size_t         plane_size = static_cast<size_t>(image.rows) * static_cast<size_t>(image.cols);
    std::vector<float>   tensor(static_cast<size_t>(image.channels()) * plane_size);
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);
    for (int c = 0; c < 3; ++c)
    {
        std::memcpy(tensor.data() + static_cast<size_t>(c) * plane_size, channels[c].data, plane_size * sizeof(float));
    }
    return tensor;
}

} // namespace irt::model
