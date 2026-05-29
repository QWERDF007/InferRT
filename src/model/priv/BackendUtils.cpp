#include "BackendUtils.hpp"

#include <inferrt/core/Exception.hpp>

#include <limits>

namespace irt::model::priv {

size_t TensorElementCount(const nvinfer1::Dims &dims, const std::string &tensor_name)
{
    if (dims.nbDims < 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Tensor rank is invalid: %s", tensor_name.c_str());
    }

    size_t count = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Tensor shape is not fully resolved: %s dim[%d]=%d",
                                 tensor_name.c_str(), i, dims.d[i]);
        }
        count *= static_cast<size_t>(dims.d[i]);
    }
    return count;
}

std::vector<int64_t> DimsToInt64Shape(const nvinfer1::Dims &dims, const std::string &tensor_name)
{
    if (dims.nbDims < 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Tensor rank is invalid: %s", tensor_name.c_str());
    }

    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(dims.nbDims));
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Tensor shape is not fully resolved: %s dim[%d]=%d",
                                 tensor_name.c_str(), i, dims.d[i]);
        }
        shape.push_back(static_cast<int64_t>(dims.d[i]));
    }
    return shape;
}

std::vector<size_t> DimsToSizeTShape(const nvinfer1::Dims &dims, const std::string &tensor_name)
{
    if (dims.nbDims < 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Tensor rank is invalid: %s", tensor_name.c_str());
    }

    std::vector<size_t> shape;
    shape.reserve(static_cast<size_t>(dims.nbDims));
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] <= 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_OPERATION, "Tensor shape is not fully resolved: %s dim[%d]=%d",
                                 tensor_name.c_str(), i, dims.d[i]);
        }
        shape.push_back(static_cast<size_t>(dims.d[i]));
    }
    return shape;
}

nvinfer1::Dims Int64ShapeToDims(const std::vector<int64_t> &shape)
{
    if (shape.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS))
    {
        throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Tensor rank %zu exceeds TensorRT Dims capacity",
                             shape.size());
    }

    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        const auto value = shape[i];
        if (value > std::numeric_limits<int32_t>::max())
        {
            throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Tensor dimension exceeds int32: %lld",
                                 static_cast<long long>(value));
        }
        dims.d[i] = static_cast<int32_t>(value);
    }
    return dims;
}

bool IsDefaultOutputConfig(const IModelConfig &config)
{
    const auto &outputs = config.outputTensorNames();
    return outputs.size() == 1 && outputs.front() == "output" && !config.featureOnly();
}

} // namespace irt::model::priv
