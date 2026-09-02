#pragma once

#include "EngineRuntimePlan.hpp"

#include <cuda_runtime_api.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/engine/InferenceEngine.hpp>
#include <inferrt/engine/Pipeline.hpp>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/Utils.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace irt::engine::priv {

inline std::exception_ptr cancelledError()
{
    return std::make_exception_ptr(
        irt::Exception(irt::Status::INVALID_OPERATION, "Inference request was cancelled"));
}

inline std::exception_ptr timeoutError()
{
    return std::make_exception_ptr(irt::Exception(irt::Status::NOT_READY, "Inference request deadline expired"));
}

inline std::exception_ptr droppedError()
{
    return std::make_exception_ptr(irt::Exception(irt::Status::NOT_READY, "Inference request was dropped"));
}

inline size_t alignDeviceOffset(const size_t value)
{
    constexpr size_t alignment = 256;
    const auto aligned = irt::checkedSizeAdd(value, alignment - 1, "Device arena alignment");
    return aligned & ~(alignment - 1);
}

inline uint64_t elapsedMicroseconds(const std::chrono::steady_clock::time_point begin,
                                    const std::chrono::steady_clock::time_point end)
{
    if (begin == std::chrono::steady_clock::time_point{} || end < begin)
    {
        return 0;
    }
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
}

inline void recordLatency(StageLatencySnapshot &snapshot, const uint64_t microseconds)
{
    ++snapshot.samples;
    snapshot.total_microseconds += microseconds;
    snapshot.max_microseconds = std::max(snapshot.max_microseconds, microseconds);

    size_t   bucket = 0;
    uint64_t limit  = 1;
    while (bucket + 1 < snapshot.histogram.size() && microseconds > limit)
    {
        limit *= 10;
        ++bucket;
    }
    ++snapshot.histogram[bucket];
}

inline bool isFatalCudaError(const cudaError_t error)
{
    return error == cudaErrorContextIsDestroyed || error == cudaErrorIllegalAddress || error == cudaErrorLaunchFailure;
}

/**
 * Owns one device arena and the stream used to retire it.
 *
 * The arena is deliberately a small private module: callers receive only a
 * stable byte pointer while allocation, device affinity and asynchronous
 * destruction stay in one RAII owner.  This keeps startup rollback and Slot
 * destruction identical and prevents a raw CUDA allocation from escaping.
 */
class DeviceArena final
{
public:
    DeviceArena() = default;
    ~DeviceArena() noexcept { reset(); }

    DeviceArena(const DeviceArena &)            = delete;
    DeviceArena &operator=(const DeviceArena &) = delete;

    DeviceArena(DeviceArena &&other) noexcept
        : data_(other.data_)
        , bytes_(other.bytes_)
        , device_id_(other.device_id_)
        , stream_(other.stream_)
    {
        other.data_      = nullptr;
        other.bytes_     = 0;
        other.device_id_ = -1;
        other.stream_    = nullptr;
    }

    DeviceArena &operator=(DeviceArena &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            data_      = other.data_;
            bytes_     = other.bytes_;
            device_id_ = other.device_id_;
            stream_    = other.stream_;
            other.data_      = nullptr;
            other.bytes_     = 0;
            other.device_id_ = -1;
            other.stream_    = nullptr;
        }
        return *this;
    }

    void allocate(const size_t bytes, const int device_id, cudaStream_t stream)
    {
        if (bytes == 0 || device_id < 0 || stream == nullptr)
        {
            throw irt::Exception(irt::Status::ERROR_INVALID_ARGUMENT,
                                 "DeviceArena requires positive bytes, a non-negative device and a stream");
        }
        reset();

        int previous_device = -1;
        irt::model::checkCuda(cudaGetDevice(&previous_device), "cudaGetDevice(DeviceArena)");
        irt::model::checkCuda(cudaSetDevice(device_id), "cudaSetDevice(DeviceArena)");

        void *candidate = nullptr;
        const auto allocation_status = cudaMallocAsync(&candidate, bytes, stream);
        if (allocation_status != cudaSuccess)
        {
            (void)cudaSetDevice(previous_device);
            throw irt::Exception(irt::Status::ERROR_OUT_OF_MEMORY,
                                 "cudaMallocAsync(DeviceArena) failed for %zu bytes: %s", bytes,
                                 cudaGetErrorString(allocation_status));
        }
        const auto sync_status = cudaStreamSynchronize(stream);
        if (sync_status != cudaSuccess)
        {
            (void)cudaFreeAsync(candidate, stream);
            (void)cudaStreamSynchronize(stream);
            (void)cudaSetDevice(previous_device);
            throw irt::Exception(irt::Status::ERROR_INTERNAL,
                                 "cudaStreamSynchronize(DeviceArena) failed: %s", cudaGetErrorString(sync_status));
        }

        data_      = candidate;
        bytes_     = bytes;
        device_id_ = device_id;
        stream_    = stream;
        (void)cudaSetDevice(previous_device);
    }

    void reset() noexcept
    {
        if (data_ == nullptr)
        {
            return;
        }
        int previous_device = -1;
        if (cudaGetDevice(&previous_device) == cudaSuccess && device_id_ >= 0)
        {
            (void)cudaSetDevice(device_id_);
        }
        if (stream_ != nullptr)
        {
            (void)cudaFreeAsync(data_, stream_);
            (void)cudaStreamSynchronize(stream_);
        }
        else
        {
            (void)cudaFree(data_);
        }
        if (previous_device >= 0)
        {
            (void)cudaSetDevice(previous_device);
        }
        data_      = nullptr;
        bytes_     = 0;
        device_id_ = -1;
        stream_    = nullptr;
    }

    [[nodiscard]] void *data() const noexcept { return data_; }
    [[nodiscard]] size_t size() const noexcept { return bytes_; }
    [[nodiscard]] int deviceId() const noexcept { return device_id_; }
    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

private:
    void          *data_{nullptr};
    size_t         bytes_{0};
    int            device_id_{-1};
    cudaStream_t   stream_{nullptr};
};

enum class FailureKind
{
    Failed,
    Cancelled,
    TimedOut,
    Dropped,
};

struct Request
{
    uint64_t                              id{0};
    cv::Mat                               image;
    std::chrono::steady_clock::time_point submitted;
    std::chrono::steady_clock::time_point deadline{std::chrono::steady_clock::time_point::max()};
    std::promise<InferenceResult>         promise;
    std::atomic_bool                      cancelled{false};
    std::atomic_bool                      finished{false};
    std::atomic_bool                      fulfilled{false};
    int                                   priority{0};
    std::string                           compatibility_key{"default"};
    TensorInputMap                        extra_inputs;
    std::string                           source_id{"default"};
    uint64_t                              source_sequence{0};
    bool                                  preserve_source_order{true};
};

using RequestPtr = std::shared_ptr<Request>;

struct InputTicket
{
    std::unordered_map<std::string, irt::model::PinnedHostBuffer> buffers;
    TensorViewMap                                                      tensors;
};

struct OutputTicket
{
    std::vector<irt::model::PinnedHostBuffer>                     model_outputs;
    std::unordered_map<std::string, irt::model::PinnedHostBuffer> result_outputs;
};

struct PendingCompletion
{
    RequestPtr         request;
    bool               success{false};
    InferenceResult    result;
    std::exception_ptr error;
};

struct BatchState
{
    std::vector<RequestPtr>               requests;
    int                                   actual_batch{0};
    int                                   execution_batch{0};
    InputTicket                          *input_ticket{nullptr};
    OutputTicket                         *output_ticket{nullptr};
    std::vector<size_t>                   output_elements_per_request;
    std::vector<std::string>              output_names;
    int                                   device_id{-1};
    std::chrono::steady_clock::time_point scheduled;
    std::chrono::steady_clock::time_point cpu_preprocess_started;
    std::chrono::steady_clock::time_point cpu_preprocess_finished;
    std::chrono::steady_clock::time_point gpu_submitted;
    std::chrono::steady_clock::time_point gpu_finished;
    std::chrono::steady_clock::time_point cpu_postprocess_started;
    std::chrono::steady_clock::time_point cpu_postprocess_finished;
    std::atomic_bool                      finalized{false};
};

using BatchPtr = std::shared_ptr<BatchState>;

struct Slot
{
    explicit Slot(const int device)
        : device_id(device)
    {
    }

    ~Slot()
    {
        cudaSetDevice(device_id);
        context.reset();
        device_arena.reset();
        if (h2d_done != nullptr)
        {
            cudaEventDestroy(h2d_done);
        }
        if (compute_done != nullptr)
        {
            cudaEventDestroy(compute_done);
        }
        if (d2h_done != nullptr)
        {
            cudaEventDestroy(d2h_done);
        }
        if (h2d_stream != nullptr)
        {
            cudaStreamDestroy(h2d_stream);
        }
        if (compute_stream != nullptr)
        {
            cudaStreamDestroy(compute_stream);
        }
        if (d2h_stream != nullptr)
        {
            cudaStreamDestroy(d2h_stream);
        }
    }

    int                                              device_id;
    std::shared_ptr<priv::IEngineRuntimePlan>        runtime_plan;
    std::unique_ptr<irt::ITensorRuntimeSession>      context;
    std::vector<std::pair<std::string, std::string>> model_inputs;
    std::vector<irt::TensorInfo>                      output_infos;
    cudaStream_t                                     h2d_stream{nullptr};
    cudaStream_t                                     compute_stream{nullptr};
    cudaStream_t                                     d2h_stream{nullptr};
    cudaEvent_t                                      h2d_done{nullptr};
    cudaEvent_t                                      compute_done{nullptr};
    cudaEvent_t                                      d2h_done{nullptr};
    DeviceArena                                      device_arena;
    size_t                                           device_arena_bytes{0};
    int                                              capacity_batch{0};
    std::unordered_map<std::string, TensorView>      device_tensors;
    std::vector<irt::BufferView>                    device_outputs;
    std::vector<size_t>                              output_capacity_bytes;
    std::vector<std::unique_ptr<IOperator>>          operators;
    std::mutex                                       mutex;
    BatchPtr                                         active_batch;
    bool                                             input_released{false};
};

} // namespace irt::engine::priv
