#include "TRTUtils.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Tensor.hpp>

#include <charconv>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>

namespace irt::model {

namespace priv {

std::vector<char> readBinaryFile(const std::string &file)
{
    std::ifstream input(file, std::ios::binary);
    if (!input.is_open())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to open binary file: %s", file.c_str());
    }

    input.seekg(0, std::ios::end);
    const auto file_size = input.tellg();
    if (file_size <= 0
        || static_cast<std::uintmax_t>(file_size) > static_cast<std::uintmax_t>((std::numeric_limits<size_t>::max)())
        || static_cast<std::uintmax_t>(file_size)
               > static_cast<std::uintmax_t>((std::numeric_limits<std::streamsize>::max)()))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Binary file is empty or too large: %s", file.c_str());
    }
    input.seekg(0, std::ios::beg);
    if (!input)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to seek binary file: %s", file.c_str());
    }

    const auto byte_count = static_cast<size_t>(file_size);
    std::vector<char> data(byte_count);
    input.read(data.data(), static_cast<std::streamsize>(byte_count));
    if (input.gcount() != static_cast<std::streamsize>(byte_count))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Failed to read binary file: %s", file.c_str());
    }
    return data;
}

std::shared_ptr<nvinfer1::ICudaEngine> deserializeCudaEngine(const void *data, const size_t byte_count,
                                                             nvinfer1::ILogger &logger)
{
    if (data == nullptr || byte_count == 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Serialized TensorRT engine is empty");
    }

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create InferRuntime");
    }

    auto engine = runtime->deserializeCudaEngine(data, byte_count);
    if (!engine)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to deserialize CUDA engine");
    }
    return std::shared_ptr<nvinfer1::ICudaEngine>(engine,
                                                  [](nvinfer1::ICudaEngine *value) { delete value; });
}

std::unique_ptr<nvinfer1::IExecutionContext>
createTensorRTExecutionContext(const std::shared_ptr<nvinfer1::ICudaEngine> &engine)
{
    if (!engine)
    {
        throw irt::Exception(Status::INVALID_OPERATION, "TensorRT engine is not initialized");
    }

    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
        throw irt::Exception(Status::ERROR_INTERNAL, "Failed to create execution context");
    }
    return context;
}

} // namespace priv

WeightsMap loadWeights(const std::string &file)
{
    WeightsMap weights_map;
    struct WeightsMapGuard
    {
        WeightsMap &map;
        bool        committed{false};

        ~WeightsMapGuard()
        {
            if (committed)
            {
                return;
            }
            for (auto &entry : map)
            {
                delete[] static_cast<const uint32_t *>(entry.second.values);
            }
        }
    } guard{weights_map};

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
        std::string name;
        int64_t     raw_element_count{0};
        if (!(input >> name >> std::dec >> raw_element_count) || name.empty() || raw_element_count < 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Invalid weight header in file: %s", file.c_str());
        }

        const size_t element_count = irt::checkedInt64ToSize(raw_element_count, "Weight element count");
        (void)irt::checkedSizeMul(element_count, sizeof(uint32_t), "Weight bytes");
        auto values = std::make_unique<uint32_t[]>(element_count);
        input >> std::hex;
        for (size_t index = 0; index < element_count; ++index)
        {
            std::string token;
            if (!(input >> token))
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                     "Truncated weight data for '%s' at element %zu in file: %s", name.c_str(),
                                     index, file.c_str());
            }

            uint64_t parsed{0};
            const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), parsed, 16);
            if (error != std::errc{} || end != token.data() + token.size()
                || parsed > (std::numeric_limits<uint32_t>::max)())
            {
                throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                     "Invalid hexadecimal weight data for '%s' at element %zu in file: %s",
                                     name.c_str(), index, file.c_str());
            }
            values[index] = static_cast<uint32_t>(parsed);
        }

        const nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT, values.get(), raw_element_count};
        if (!weights_map.emplace(name, weight).second)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Duplicate weight name '%s' in file: %s",
                                 name.c_str(), file.c_str());
        }
        values.release();
    }

    guard.committed = true;
    return weights_map;
}

size_t elementCount(const nvinfer1::Dims &dims)
{
    if (dims.nbDims < 0)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Tensor shape has negative rank: %d", dims.nbDims);
    }
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Tensor shape contains non-positive dimension: %d", dims.d[i]);
        }
        const size_t dim = static_cast<size_t>(dims.d[i]);
        if (dim > 0 && count > std::numeric_limits<size_t>::max() / dim)
        {
            throw irt::Exception(irt::Status::ERROR_OUT_OF_MEMORY, "Tensor element count calculation overflows size_t");
        }
        count *= dim;
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

} // namespace irt::model
