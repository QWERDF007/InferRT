#pragma once

#include <inferrt/core/PreprocessSpec.hpp>
#include <inferrt/engine/Export.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace irt::engine {

/**
 * @brief Engine 输入图像的预处理执行位置。
 */
using PreprocessBackend = irt::PreprocessBackend;

/** @brief 请求队列达到容量上限时的处理策略。 */
enum class QueuePolicy
{
    Reject,     ///< 拒绝新请求，并由 submit 抛出 ERROR_NOT_READY。
    Block,      ///< submit 等待 scheduler 腾出队列空间或 Engine 停止。
    DropOldest, ///< 丢弃最早尚未组 batch 的请求，接收新请求。
    DropNewest, ///< 接收调用，但返回一个已失败的 future，不影响队列中的旧请求。
};

/** @brief 固定 batch TensorRT engine 接收到不足 batch 的请求时的处理方式。 */
enum class StaticBatchPolicy
{
    Reject, ///< 不足固定 batch 时失败，不提交 TensorRT。
    Pad,    ///< 以零填充到固定 batch，并仅物化有效请求的结果。
};

/**
 * @brief Engine 的启动配置。
 *
 * 配置保持为单个具体对象：首版只支持一个图像输入、float32 NCHW TensorRT engine。
 * 这使数据流水线的内存大小、批处理范围和执行槽数量都能在启动前确定。
 */
struct INFERRT_ENGINE_API EngineConfig
{
    std::string              model_name;
    std::filesystem::path    engine_file;
    int                      device_id{0};
    /** 为空时仅使用 device_id；非空时 execution_slots 在这些 device 上轮转分配。 */
    std::vector<int>         device_ids;
    std::vector<std::string> output_tensor_names;
    std::vector<std::string> feature_tensor_names;
    bool                     feature_only{false};

    /**
     * Complete image preprocessing contract.  Input geometry, color/layout,
     * normalization and backend selection have one owner so model tools,
     * built-in operators and the legacy engine pipeline cannot drift apart.
     */
    irt::PreprocessSpec preprocess{};

    int                       min_batch_size{1};
    int                       opt_batch_size{1};
    int                       max_batch_size{1};
    std::chrono::microseconds max_wait{0};

    size_t      execution_slots{1};
    size_t      queue_capacity{64};
    QueuePolicy queue_policy{QueuePolicy::Reject};

    /** CPU 前处理和后处理的独立工作线程数。 */
    size_t cpu_preprocess_workers{1};
    size_t cpu_postprocess_workers{1};

    /**
     * Pinned ticket 数量；零值在启动时按 worker 与 slot 数量自动推导。
     * 输入和输出 pool 独立，避免 CPU 后处理占用 H2D staging buffer。
     */
    size_t pinned_input_tickets{0};
    size_t pinned_output_tickets{0};
    size_t pinned_memory_limit_bytes{0};
    size_t device_memory_limit_bytes{0};
    size_t fault_history_capacity{128};

    /** 动态 scheduler 优先使用的 batch 大小；为空时直接使用 max_batch_size。 */
    std::vector<int>  preferred_batch_sizes;
    StaticBatchPolicy static_batch_policy{StaticBatchPolicy::Pad};

    /** @brief 从 YAML 文件读取配置。 */
    static EngineConfig load(const std::filesystem::path &path);

    /** @brief 生成并校验唯一的图像预处理规格。 */
    [[nodiscard]] const irt::PreprocessSpec &preprocessSpec() const noexcept;

    /** @brief 校验字段之间的约束，不访问 CUDA 或模型文件。 */
    void validate() const;
};

} // namespace irt::engine
