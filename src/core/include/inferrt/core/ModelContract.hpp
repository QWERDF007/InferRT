#pragma once

#include <inferrt/core/Export.h>
#include <inferrt/core/Tensor.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace irt {

enum class TensorIOMode
{
    Input,
    Output,
};

struct TensorInfo
{
    std::string   name;
    TensorDesc    desc;
    TensorIOMode  mode{TensorIOMode::Input};
};

struct ExecuteOptions
{
    // The backend owns the stream. It is intentionally opaque at the core boundary.
    std::uintptr_t stream{0};
    bool           non_blocking{false};
};

/** Capabilities that affect how a plan can be scheduled and consumed. */
struct ExecutionCapabilities
{
    bool supports_dynamic_batch{false};
    bool supports_feature_outputs{false};
    int  fixed_batch_size{0};
};

class ITensorRuntimeSession;

/**
 * @brief Validate and canonicalize a complete execution binding set.
 *
 * Every buffer must carry the declared tensor name.  The returned views are
 * ordered as all declared inputs followed by all declared outputs, so backend
 * adapters never infer tensor identity from the caller's container order.
 */
INFERRT_CORE_API std::vector<BufferView> normalizeExecutionBuffers(
    std::span<const BufferView> buffers, std::span<const TensorInfo> inputs, std::span<const TensorInfo> outputs);

/** Backend-neutral tensor descriptor contract shared by model and engine adapters. */
class INFERRT_CORE_API IExecutionDescriptor
{
public:
    IExecutionDescriptor() = default;
    IExecutionDescriptor(const IExecutionDescriptor &) = default;
    IExecutionDescriptor &operator=(const IExecutionDescriptor &) = default;
    virtual ~IExecutionDescriptor();

    /** Return descriptors in the backend's canonical input order. */
    virtual std::vector<TensorInfo> inputs() const = 0;
    /** Return descriptors in the backend's canonical output order. */
    virtual std::vector<TensorInfo> outputs() const = 0;
};

/**
 * Backend-neutral plan contract shared by model and engine adapters.
 *
 * A plan owns immutable executable state.  Each concurrent consumer obtains a
 * private session for mutable shapes and binding addresses, then submits
 * buffers through this one execution seam.
 */
class INFERRT_CORE_API IExecutionPlan : public IExecutionDescriptor
{
public:
    IExecutionPlan() = default;
    IExecutionPlan(const IExecutionPlan &)            = delete;
    IExecutionPlan &operator=(const IExecutionPlan &) = delete;
    IExecutionPlan(IExecutionPlan &&) noexcept            = default;
    IExecutionPlan &operator=(IExecutionPlan &&) noexcept = default;
    virtual ~IExecutionPlan();

    /** Return scheduler-visible backend capabilities. */
    [[nodiscard]] virtual ExecutionCapabilities capabilities() const = 0;
    [[nodiscard]] virtual std::unique_ptr<ITensorRuntimeSession> createSession() const = 0;
    virtual void executeSession(ITensorRuntimeSession &session, std::span<const BufferView> buffers,
                                ExecuteOptions options = {}) const = 0;
};

/** Backend-neutral model execution contract shared by model and engine adapters. */
class INFERRT_CORE_API IExecutableModel : public IExecutionPlan
{
public:
    IExecutableModel() = default;
    IExecutableModel(const IExecutableModel &)            = delete;
    IExecutableModel &operator=(const IExecutableModel &) = delete;
    IExecutableModel(IExecutableModel &&) noexcept            = default;
    IExecutableModel &operator=(IExecutableModel &&) noexcept = default;
    ~IExecutableModel() override;

    virtual void setInputShape(const std::string &name, Shape shape) = 0;
    virtual void execute(std::span<const BufferView> buffers, ExecuteOptions options = {}) = 0;
};

/**
 * @brief Isolated execution state for one concurrent consumer.
 *
 * A backend runtime may be shared by several callers, while each session owns
 * its mutable shape and binding state.  The interface deliberately exposes
 * only core tensor contracts; TensorRT/ONNX/OpenVINO handles stay in private
 * adapters.
 */
class INFERRT_CORE_API ITensorRuntimeSession
{
public:
    ITensorRuntimeSession() = default;
    ITensorRuntimeSession(const ITensorRuntimeSession &) = delete;
    ITensorRuntimeSession &operator=(const ITensorRuntimeSession &) = delete;
    virtual ~ITensorRuntimeSession();

    [[nodiscard]] virtual Shape tensorShape(const std::string &tensor_name) const = 0;
    [[nodiscard]] virtual TensorDataType tensorDataType(const std::string &tensor_name) const = 0;
    virtual void setTensorShape(const std::string &tensor_name, const Shape &shape) = 0;
    virtual void execute(std::span<const BufferView> buffers, ExecuteOptions options = {}) = 0;
};

} // namespace irt
