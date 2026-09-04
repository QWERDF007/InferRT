#pragma once

#include <inferrt/core/Status.hpp>
#include <inferrt/engine/EngineConfig.hpp>
#include <inferrt/engine/Export.h>
#include <inferrt/engine/Pipeline.hpp>
#include <opencv2/core.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace irt::engine {

/**
 * @brief 单张图像请求的推理输出。
 *
 * 首版仅接收和返回 float32 Tensor；每个输出以 TensorRT 的输出名称为 key，
 * 数据已经按 batch 维拆分为当前请求对应的一份。
 */
struct INFERRT_ENGINE_API InferenceResult
{
    uint64_t                                            request_id{0};
    std::string                                         source_id;
    uint64_t                                            source_sequence{0};
    std::unordered_map<std::string, std::vector<float>> outputs;
};

/**
 * @brief 单个请求的可选调度约束。
 *
 * deadline 从 submit 时刻起计算；零值表示不设置请求级 deadline，仍受 Engine
 * 的动态 batch 等待时间约束。
 */
struct INFERRT_ENGINE_API RequestOptions
{
    std::chrono::microseconds deadline{0};
    /** 优先级越大越优先；同优先级时优先调度最早提交的兼容组。 */
    int                       priority{0};
    /** 相同非空 key 才允许被 scheduler 组入同一 batch；空值表示默认兼容组。 */
    std::string               compatibility_key;
    /** 同一 source 的 future 按提交顺序兑现；空值归入 default source。 */
    std::string               source_id;
    bool                      preserve_source_order{true};
};

enum class FaultStage
{
    Start,
    Scheduler,
    CpuPreprocess,
    GpuDispatch,
    CompletionPoll,
    CpuPostprocess,
    Shutdown,
};

struct INFERRT_ENGINE_API FaultRecord
{
    std::chrono::system_clock::time_point timestamp;
    uint64_t                              request_id{0};
    std::string                           source_id;
    uint64_t                              source_sequence{0};
    int                                   device_id{-1};
    FaultStage                            stage{FaultStage::Start};
    irt::Status                           status{irt::Status::ERROR_INTERNAL};
    bool                                  fatal{false};
    std::string                           message;
};

/** @brief Engine 的生命周期状态。失败后应新建 Engine，而不是复用 CUDA/TensorRT 状态。 */
enum class EngineState
{
    Created,
    Starting,
    Running,
    Draining,
    Stopped,
    Failed,
};

/** @brief 单个阶段的聚合延迟；histogram 单位为微秒的对数桶。 */
struct INFERRT_ENGINE_API StageLatencySnapshot
{
    uint64_t                samples{0};
    uint64_t                total_microseconds{0};
    uint64_t                max_microseconds{0};
    std::array<uint64_t, 8> histogram{};
};

/** @brief 带 request id 的异步请求结果，用于取消尚未完成的请求。 */
class INFERRT_ENGINE_API RequestHandle final
{
public:
    RequestHandle()                                     = default;
    RequestHandle(const RequestHandle &)                = delete;
    RequestHandle &operator=(const RequestHandle &)     = delete;
    RequestHandle(RequestHandle &&) noexcept            = default;
    RequestHandle &operator=(RequestHandle &&) noexcept = default;

    [[nodiscard]] uint64_t                     id() const noexcept;
    [[nodiscard]] bool                         valid() const noexcept;
    [[nodiscard]] std::future<InferenceResult> takeFuture() &&;

private:
    friend class InferenceEngine;

    RequestHandle(uint64_t request_id, std::future<InferenceResult> future);

    uint64_t                     request_id_{0};
    std::future<InferenceResult> future_;
};

/** @brief Engine 的线程安全运行指标快照。 */
struct INFERRT_ENGINE_API EngineMetricsSnapshot
{
    uint64_t             accepted_requests{0};
    uint64_t             completed_requests{0};
    uint64_t             failed_requests{0};
    uint64_t             cancelled_requests{0};
    uint64_t             timed_out_requests{0};
    uint64_t             rejected_requests{0};
    uint64_t             dropped_requests{0};
    size_t               queued_requests{0};
    size_t               queued_high_watermark{0};
    size_t               inflight_batches{0};
    uint64_t             completed_batches{0};
    size_t               prepare_queue_high_watermark{0};
    size_t               gpu_queue_high_watermark{0};
    size_t               postprocess_queue_high_watermark{0};
    size_t               pinned_input_in_use{0};
    size_t               pinned_input_high_watermark{0};
    size_t               pinned_output_in_use{0};
    size_t               pinned_output_high_watermark{0};
    size_t               pinned_memory_capacity_bytes{0};
    size_t               device_arena_bytes{0};
    StageLatencySnapshot queue_latency;
    StageLatencySnapshot cpu_preprocess_latency;
    StageLatencySnapshot gpu_latency;
    StageLatencySnapshot cpu_postprocess_latency;

    // Lifecycle resource counts used to prove failed-start and shutdown
    // invariants without exposing Engine::Impl internals.
    size_t active_requests{0};
    size_t slot_count{0};
    size_t idle_slot_count{0};
    size_t active_slot_count{0};
    size_t prepare_queue_size{0};
    size_t gpu_queue_size{0};
    size_t postprocess_queue_size{0};
    size_t input_ticket_count{0};
    size_t output_ticket_count{0};
    size_t thread_count{0};
    size_t joinable_thread_count{0};
};

/**
 * @brief OpenCV 图像推理编排器。
 *
 * Engine 在启动时反序列化一次只读 TensorRT engine，并为每个 ExecutionSlot 创建
 * 独立 IExecutionContext。它负责图像所有权、动态组批、Pinned H2D/D2H、队列策略
 * 和异步结果分发。
 */
class INFERRT_ENGINE_API InferenceEngine final
{
public:
    /**
     * @brief 使用已编译的不可变 PipelinePlan 创建 Engine。
     *
     * Pipeline 由调用方在 C++ 中定义。需要改变 DAG 时，应构建新的 Plan 并创建
     * 新的 Engine，不支持运行期修改当前 Plan。
     */
    InferenceEngine(EngineConfig config, std::shared_ptr<const PipelinePlan> pipeline);

    ~InferenceEngine();

    InferenceEngine(const InferenceEngine &)            = delete;
    InferenceEngine &operator=(const InferenceEngine &) = delete;
    InferenceEngine(InferenceEngine &&)                 = delete;
    InferenceEngine &operator=(InferenceEngine &&)      = delete;

    /** @brief 创建但不启动使用代码定义 DAG 的 engine。 */
    static std::unique_ptr<InferenceEngine> create(EngineConfig config, std::shared_ptr<const PipelinePlan> pipeline);

    /** @brief 分配资源、加载每个执行槽并启动调度线程。 */
    void start();

    /**
     * @brief 提交一张 BGR/Gray OpenCV 图像。
     *
     * 函数在返回前 clone 输入，调用方可以立即复用或释放原始 cv::Mat。
     * 队列已满或 engine 未运行时抛出 irt::Exception。
     */
    std::future<InferenceResult> submit(const cv::Mat &image);
    std::future<InferenceResult> submit(const cv::Mat &image, TensorInputMap inputs);

    /**
     * @brief 提交带 request id 和 deadline 的异步请求。
     *
     * 通过返回的 id 调用 cancel() 可取消尚未完成的请求。已提交 GPU 的请求不会
     * 中断 kernel，但其结果会被丢弃且 future 以取消错误结束。
     */
    RequestHandle submit(const cv::Mat &image, RequestOptions options);
    RequestHandle submit(const cv::Mat &image, TensorInputMap inputs, RequestOptions options);

    /** @brief 提交并等待结果；超时不会中断已经提交到 GPU 的工作。 */
    InferenceResult infer(const cv::Mat &image, std::chrono::milliseconds timeout);

    /** @brief 请求取消；不存在或已完成的 id 返回 false。 */
    bool cancel(uint64_t request_id);

    /** @brief 停止接收新请求并排空已接受请求。可重复调用。 */
    void shutdown();

    /** @brief 当前尚未被 scheduler 组批的请求数量。 */
    size_t pendingRequests() const;

    /** @brief 获取队列、完成状态和背压统计。 */
    EngineMetricsSnapshot metrics() const;

    [[nodiscard]] std::vector<FaultRecord> faults() const;

    /** 返回当前生命周期状态；Failed 状态不会尝试原地恢复 CUDA context。 */
    EngineState state() const noexcept;

    const EngineConfig &config() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace irt::engine
