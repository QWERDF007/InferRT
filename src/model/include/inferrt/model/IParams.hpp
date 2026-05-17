#pragma once

#include "Logging.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>

namespace irt::model {

/**
 * @brief CUDA stream 智能指针使用的删除器。
 *
 * 删除器负责销毁 cudaStream_t 并释放承载 stream 句柄的指针对象。
 */
static auto StreamDeleter = [](cudaStream_t *stream)
{
    if (stream)
    {
        static_cast<void>(cudaStreamDestroy(*stream));
        delete stream;
    }
};

/**
 * @brief 创建一个非阻塞 CUDA stream。
 * @return 成功时返回持有 stream 的智能指针，失败时返回空指针。
 */
inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> MakeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> stream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(stream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        stream.reset(nullptr);
    }

    return stream;
}

/**
 * @brief TensorRT 运行时对象集合。
 */
typedef struct TensorRTParams
{
    /// 已构建或已加载的 TensorRT engine。
    std::shared_ptr<nvinfer1::ICudaEngine> engine{nullptr};
    /// 与 engine 绑定的执行上下文。
    std::unique_ptr<nvinfer1::IExecutionContext> context{nullptr};

    /// 推理阶段使用的 CUDA stream。
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> stream{nullptr};

    /// TensorRT 日志对象。
    std::unique_ptr<Logger> logger{nullptr};
    /// 当前日志级别。
    nvinfer1::ILogger::Severity log_level{nvinfer1::ILogger::Severity::kWARNING};
} TRTParams;

} // namespace irt::model
