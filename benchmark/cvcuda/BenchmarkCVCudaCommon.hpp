/**
 * @file BenchmarkCVCudaCommon.hpp
 * @brief CVCUDA benchmark 公共辅助工具。
 *
 * 该文件集中放置 CUDA 资源管理、Event 手动计时和吞吐统计逻辑，
 * 让各算子的 benchmark 只保留参数解析与实际算子调用。
 */

#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace irt::cvcuda::bench {

/**
 * @brief 计算 OpenCV Mat 占用的总字节数。
 *
 * @param mat 待统计的 Mat。
 * @return Mat 数据区字节数。
 */
inline size_t MatBytes(const cv::Mat &mat)
{
    return mat.total() * mat.elemSize();
}

/**
 * @brief 创建随机 uint8 图像。
 *
 * @param width 图像宽度。
 * @param height 图像高度。
 * @param channels 图像通道数。
 * @return 填充随机数据的 HWC uint8 Mat。
 */
inline cv::Mat MakeRandomU8Mat(int width, int height, int channels)
{
    cv::Mat mat(height, width, CV_MAKETYPE(CV_8U, channels));
    cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
    return mat;
}

/**
 * @brief 轻量级 CUDA 设备缓冲区 RAII 封装。
 *
 * 仅负责 benchmark 中常见的分配、释放和主机/设备拷贝，
 * 避免每个算子重复编写 cudaMalloc/cudaFree 清理逻辑。
 */
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    DeviceBuffer(const DeviceBuffer &)            = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept
        : data_(other.data_)
        , bytes_(other.bytes_)
    {
        other.data_  = nullptr;
        other.bytes_ = 0;
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other)
        {
            Release();
            data_        = other.data_;
            bytes_       = other.bytes_;
            other.data_  = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer()
    {
        Release();
    }

    /**
     * @brief 分配指定字节数的设备内存。
     *
     * @param bytes 需要分配的字节数。
     * @return true 分配成功。
     * @return false 分配失败。
     */
    bool Allocate(size_t bytes)
    {
        Release();
        bytes_ = bytes;
        if (bytes == 0)
        {
            return true;
        }

        if (cudaMalloc(&data_, bytes) != cudaSuccess)
        {
            data_  = nullptr;
            bytes_ = 0;
            return false;
        }
        return true;
    }

    /**
     * @brief 释放当前持有的设备内存。
     */
    void Release()
    {
        if (data_ != nullptr)
        {
            cudaFree(data_);
            data_  = nullptr;
            bytes_ = 0;
        }
    }

    /**
     * @brief 从主机同步拷贝数据到设备。
     *
     * @param src 主机源地址。
     * @param bytes 拷贝字节数。
     * @return true 拷贝成功。
     * @return false 拷贝失败。
     */
    bool CopyFromHost(const void *src, size_t bytes)
    {
        return bytes == 0 || cudaMemcpy(data_, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    }

    /**
     * @brief 从主机异步拷贝数据到设备。
     *
     * @param src 主机源地址。
     * @param bytes 拷贝字节数。
     * @param stream CUDA stream。
     * @return true 拷贝提交成功。
     * @return false 拷贝提交失败。
     */
    bool CopyFromHostAsync(const void *src, size_t bytes, cudaStream_t stream)
    {
        return bytes == 0 || cudaMemcpyAsync(data_, src, bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }

    /**
     * @brief 从设备异步拷贝数据到主机。
     *
     * @param dst 主机目标地址。
     * @param bytes 拷贝字节数。
     * @param stream CUDA stream。
     * @return true 拷贝提交成功。
     * @return false 拷贝提交失败。
     */
    bool CopyToHostAsync(void *dst, size_t bytes, cudaStream_t stream) const
    {
        return bytes == 0 || cudaMemcpyAsync(dst, data_, bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    }

    template<typename T>
    T *As()
    {
        return static_cast<T *>(data_);
    }

    template<typename T>
    const T *As() const
    {
        return static_cast<const T *>(data_);
    }

private:
    void  *data_  = nullptr;
    size_t bytes_ = 0;
};

/**
 * @brief CUDA stream RAII 封装。
 */
class CudaStream
{
public:
    CudaStream() = default;

    CudaStream(const CudaStream &)            = delete;
    CudaStream &operator=(const CudaStream &) = delete;

    ~CudaStream()
    {
        if (stream_ != nullptr)
        {
            cudaStreamDestroy(stream_);
        }
    }

    /**
     * @brief 创建 CUDA stream。
     *
     * @return true 创建成功。
     * @return false 创建失败。
     */
    bool Create()
    {
        return cudaStreamCreate(&stream_) == cudaSuccess;
    }

    cudaStream_t Get() const
    {
        return stream_;
    }

private:
    cudaStream_t stream_ = nullptr;
};

/**
 * @brief 使用 CUDA Event 统计单次迭代耗时。
 */
class CudaEventTimer
{
public:
    CudaEventTimer() = default;

    CudaEventTimer(const CudaEventTimer &)            = delete;
    CudaEventTimer &operator=(const CudaEventTimer &) = delete;

    ~CudaEventTimer()
    {
        if (start_ != nullptr)
        {
            cudaEventDestroy(start_);
        }
        if (stop_ != nullptr)
        {
            cudaEventDestroy(stop_);
        }
    }

    /**
     * @brief 创建计时所需的 CUDA Event。
     *
     * @return true 创建成功。
     * @return false 创建失败。
     */
    bool Create()
    {
        return cudaEventCreate(&start_) == cudaSuccess && cudaEventCreate(&stop_) == cudaSuccess;
    }

    /**
     * @brief 记录开始事件。
     *
     * @param stream CUDA stream。
     * @return true 记录成功。
     * @return false 记录失败。
     */
    bool RecordStart(cudaStream_t stream)
    {
        return cudaEventRecord(start_, stream) == cudaSuccess;
    }

    /**
     * @brief 记录结束事件并返回毫秒耗时。
     *
     * @param stream CUDA stream。
     * @param elapsed_ms 输出毫秒耗时。
     * @return true 计时成功。
     * @return false 计时失败。
     */
    bool RecordStopAndElapsed(cudaStream_t stream, float &elapsed_ms)
    {
        return cudaEventRecord(stop_, stream) == cudaSuccess && cudaEventSynchronize(stop_) == cudaSuccess
            && cudaEventElapsedTime(&elapsed_ms, start_, stop_) == cudaSuccess;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_  = nullptr;
};

/**
 * @brief 将 CUDA 错误写入 benchmark 状态。
 *
 * @param state benchmark 状态对象。
 * @param ok 当前步骤是否成功。
 * @param message 失败时输出的错误信息。
 * @return true 当前步骤成功。
 * @return false 当前步骤失败。
 */
inline bool CheckStep(::benchmark::State &state, bool ok, const char *message)
{
    if (!ok)
    {
        state.SkipWithError(message);
        return false;
    }
    return true;
}

/**
 * @brief 检查 InferRT API 返回码。
 *
 * @param state benchmark 状态对象。
 * @param ret InferRT API 返回码。
 * @param op_name 算子名称。
 * @return true 调用成功。
 * @return false 调用失败。
 */
inline bool CheckStatus(::benchmark::State &state, int ret, const char *op_name)
{
    if (ret == IRT_SUCCESS)
    {
        return true;
    }

    const std::string message = std::string(op_name) + " failed";
    state.SkipWithError(message.c_str());
    return false;
}

/**
 * @brief 写入按输出元素数统计的吞吐指标。
 *
 * @param state benchmark 状态对象。
 * @param items_per_iteration 每次迭代处理的输出元素数。
 * @param label 当前参数组合的显示标签。
 */
inline void ApplyElementCounters(::benchmark::State &state, int64_t items_per_iteration, const char *label)
{
    state.counters["items/s"]
        = ::benchmark::Counter(static_cast<double>(items_per_iteration), ::benchmark::Counter::kIsRate);
    state.SetItemsProcessed(state.iterations() * items_per_iteration);
    if (label != nullptr)
    {
        state.SetLabel(label);
    }
}

/**
 * @brief 用 CUDA Event 计时单轮 GPU 工作。
 *
 * @tparam Callable 执行实际 GPU 工作的可调用对象。
 * @param state benchmark 状态对象。
 * @param timer CUDA Event 计时器。
 * @param stream CUDA stream。
 * @param callable 实际工作函数，返回 true 表示成功。
 * @return true 本轮执行成功。
 * @return false 本轮执行失败。
 */
template<typename Callable>
bool MeasureCudaIteration(::benchmark::State &state, CudaEventTimer &timer, cudaStream_t stream, Callable &&callable)
{
    if (!CheckStep(state, timer.RecordStart(stream), "Failed to record CUDA start event"))
    {
        return false;
    }

    if (!std::forward<Callable>(callable)())
    {
        return false;
    }

    float elapsed_ms = 0.0f;
    if (!CheckStep(state, timer.RecordStopAndElapsed(stream, elapsed_ms), "Failed to query CUDA event elapsed time"))
    {
        return false;
    }

    state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
    return true;
}

/**
 * @brief 运行只统计 GPU 算子耗时的通用 benchmark。
 *
 * @tparam SrcT 输入元素类型。
 * @tparam DstT 输出元素类型。
 * @tparam Operator 实际算子调用类型。
 * @param state benchmark 状态对象。
 * @param src 主机侧输入图像。
 * @param dst_count 输出元素数量。
 * @param items_per_iteration 每轮处理元素数量。
 * @param op_name 算子名称。
 * @param label 参数组合标签。
 * @param op 实际算子调用，签名为 (const SrcT*, DstT*, cudaStream_t) -> int。
 */
template<typename SrcT, typename DstT, typename Operator>
void RunDeviceOnly(::benchmark::State &state, const cv::Mat &src, size_t dst_count, int64_t items_per_iteration,
                   const char *op_name, const char *label, Operator &&op)
{
    const cv::Mat input     = src.isContinuous() ? src : src.clone();
    const size_t  src_bytes = MatBytes(input);
    const size_t  dst_bytes = dst_count * sizeof(DstT);

    DeviceBuffer d_src;
    DeviceBuffer d_dst;
    if (!CheckStep(state, d_src.Allocate(src_bytes), "Failed to allocate CUDA input buffer")
        || !CheckStep(state, d_dst.Allocate(dst_bytes), "Failed to allocate CUDA output buffer")
        || !CheckStep(state, d_src.CopyFromHost(input.data, src_bytes), "Failed to copy input data to device"))
    {
        return;
    }

    CudaEventTimer timer;
    if (!CheckStep(state, timer.Create(), "Failed to create CUDA events"))
    {
        return;
    }

    for (auto _ : state)
    {
        if (!MeasureCudaIteration(
                state, timer, nullptr,
                [&]() { return CheckStatus(state, op(d_src.As<SrcT>(), d_dst.As<DstT>(), nullptr), op_name); }))
        {
            break;
        }
        ::benchmark::ClobberMemory();
    }

    ApplyElementCounters(state, items_per_iteration, label);
}

/**
 * @brief 运行包含 H2D、算子执行和 D2H 的端到端 benchmark。
 *
 * @tparam SrcT 输入元素类型。
 * @tparam DstT 输出元素类型。
 * @tparam Operator 实际算子调用类型。
 * @param state benchmark 状态对象。
 * @param src 主机侧输入图像。
 * @param dst_count 输出元素数量。
 * @param items_per_iteration 每轮处理元素数量。
 * @param op_name 算子名称。
 * @param label 参数组合标签。
 * @param op 实际算子调用，签名为 (const SrcT*, DstT*, cudaStream_t) -> int。
 */
template<typename SrcT, typename DstT, typename Operator>
void RunEndToEnd(::benchmark::State &state, const cv::Mat &src, size_t dst_count, int64_t items_per_iteration,
                 const char *op_name, const char *label, Operator &&op)
{
    const cv::Mat input     = src.isContinuous() ? src : src.clone();
    const size_t  src_bytes = MatBytes(input);
    const size_t  dst_bytes = dst_count * sizeof(DstT);

    DeviceBuffer d_src;
    DeviceBuffer d_dst;
    if (!CheckStep(state, d_src.Allocate(src_bytes), "Failed to allocate CUDA input buffer")
        || !CheckStep(state, d_dst.Allocate(dst_bytes), "Failed to allocate CUDA output buffer"))
    {
        return;
    }

    CudaStream stream;
    if (!CheckStep(state, stream.Create(), "Failed to create CUDA stream"))
    {
        return;
    }

    CudaEventTimer timer;
    if (!CheckStep(state, timer.Create(), "Failed to create CUDA events"))
    {
        return;
    }

    std::vector<DstT> dst(dst_count);
    for (auto _ : state)
    {
        if (!MeasureCudaIteration(
                state, timer, stream.Get(),
                [&]()
                {
                    return CheckStep(state, d_src.CopyFromHostAsync(input.data, src_bytes, stream.Get()),
                                     "Failed to copy input data to device")
                        && CheckStatus(state, op(d_src.As<SrcT>(), d_dst.As<DstT>(), stream.Get()), op_name)
                        && CheckStep(state, d_dst.CopyToHostAsync(dst.data(), dst_bytes, stream.Get()),
                                     "Failed to copy output data to host");
                }))
        {
            break;
        }

        ::benchmark::DoNotOptimize(dst.data());
        ::benchmark::ClobberMemory();
    }

    ApplyElementCounters(state, items_per_iteration, label);
}

} // namespace irt::cvcuda::bench
