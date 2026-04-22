#pragma once

#include "Logging.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>

namespace irt::model {

static auto StreamDeleter = [](cudaStream_t *stream)
{
    if (stream)
    {
        static_cast<void>(cudaStreamDestroy(*stream));
        delete stream;
    }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> MakeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> stream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(stream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        stream.reset(nullptr);
    }

    return stream;
}

typedef struct TensorRTParams
{
    std::shared_ptr<nvinfer1::ICudaEngine>       engine{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> context{nullptr};

    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> stream{nullptr};

    std::unique_ptr<Logger>     logger{nullptr};
    nvinfer1::ILogger::Severity log_level{nvinfer1::ILogger::Severity::kWARNING};
} TRTParams;

} // namespace irt::model