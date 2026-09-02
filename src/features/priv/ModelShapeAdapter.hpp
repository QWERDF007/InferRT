#pragma once

#include <NvInfer.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Tensor.hpp>

#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace irt::features::priv {

inline nvinfer1::Dims toTensorRtDims(const irt::Shape &shape)
{
    if (shape.rank() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Tensor rank exceeds TensorRT Dims capacity");
    }

    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(shape.rank());
    for (size_t index = 0; index < shape.rank(); ++index)
    {
        const auto value = shape[index];
        if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "Tensor dimension cannot be represented by TensorRT Dims");
        }
        dims.d[index] = static_cast<int32_t>(value);
    }
    return dims;
}

inline irt::Shape toCoreShape(const nvinfer1::Dims &dims)
{
    if (dims.nbDims < 0 || dims.nbDims > nvinfer1::Dims::MAX_DIMS)
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT, "Tensor rank is invalid");
    }

    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(dims.nbDims));
    for (int32_t index = 0; index < dims.nbDims; ++index)
    {
        shape.push_back(static_cast<int64_t>(dims.d[index]));
    }
    return irt::Shape{std::move(shape)};
}

inline size_t tensorElementCount(const nvinfer1::Dims &dims)
{
    return toCoreShape(dims).elementCount();
}

inline std::string tensorDimsToCsv(const nvinfer1::Dims &dims)
{
    std::string result;
    for (int32_t index = 0; index < dims.nbDims; ++index)
    {
        if (index > 0)
        {
            result += ',';
        }
        result += std::to_string(dims.d[index]);
    }
    return result;
}

} // namespace irt::features::priv
