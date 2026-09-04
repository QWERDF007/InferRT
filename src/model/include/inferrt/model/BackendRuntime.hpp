#pragma once

#include <inferrt/core/ModelContract.hpp>
#include <inferrt/model/Export.h>
#include <inferrt/model/IModelConfig.hpp>
#include <inferrt/model/Logging.hpp>
#include <inferrt/model/ModelRuntime.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace irt::model {

/**
 * Backend-neutral runtime contract shared by model and engine adapters.
 *
 * The interface contains no TensorRT, ONNX Runtime, or OpenVINO types.  The
 * concrete adapters remain private to the model library while Engine uses
 * this installed contract to load and execute a backend plan.
 */
class INFERRT_MODEL_API IBackendRuntime : public irt::IExecutableModel
{
public:
    virtual ~IBackendRuntime() = default;

    /** Describe backend I/O through the shared core execution contract. */
    std::vector<irt::TensorInfo> inputs() const override;
    std::vector<irt::TensorInfo> outputs() const override;
    irt::ExecutionCapabilities capabilities() const override;

    /** Return the selected backend. */
    virtual ModelRuntime::Backend backend() const noexcept = 0;

    /** Load a serialized model or graph using the supplied configuration. */
    virtual void load(const std::string &model_file, const IModelConfig &config,
                      const std::string &model_name) = 0;

    /** Save the current runtime when supported by the backend. */
    virtual void save(const std::string &engine_file) const;

    /** Enumerate input or output tensor names. */
    virtual std::vector<std::string> ioTensorNames(irt::TensorIOMode mode) const = 0;

    /** Return the address space required by execute() for the selected I/O. */
    virtual irt::MemoryKind ioMemoryKind(irt::TensorIOMode mode) const noexcept = 0;

    /** Return the currently resolved shape of a tensor. */
    virtual irt::Shape tensorShape(const std::string &tensor_name) const = 0;

    /** Preserve whether the input batch dimension was dynamic in source metadata. */
    virtual bool isInputBatchDynamic(const std::string &tensor_name) const;

    /** Return the core data type of a tensor. */
    virtual irt::TensorDataType tensorDataType(const std::string &tensor_name) const = 0;

    /** Resolve an input tensor shape for the next execution. */
    virtual void setTensorShape(const std::string &tensor_name, const irt::Shape &shape) = 0;

    /** Core contract spelling for setting an input shape. */
    void setInputShape(const std::string &tensor_name, irt::Shape shape) override;

    /**
     * Execute through the backend's direct model path.
     *
     * This entry point is the single binding boundary for direct execution:
     * it validates and canonicalizes named buffers before dispatching to the
     * backend adapter.
     */
    void execute(std::span<const irt::BufferView> buffers, irt::ExecuteOptions options = {}) final override;

    /** Create an isolated session for a concurrent consumer. */
    virtual std::unique_ptr<irt::ITensorRuntimeSession> createSession() const;

    /** Execute through an isolated session using the shared plan contract. */
    void executeSession(irt::ITensorRuntimeSession &session, std::span<const irt::BufferView> buffers,
                        irt::ExecuteOptions options = {}) const override;

    /** Register an external opaque execution stream. */
    virtual void setStream(std::uintptr_t stream);

    /** Clear the registered external execution stream. */
    virtual void clearStream();

    /** Resolve a temporary stream override or the registered stream. */
    virtual std::uintptr_t resolveExecutionStream(std::uintptr_t stream_override = 0);

    /** Return the backend-neutral log level. */
    virtual LogLevel logLevel() const noexcept;

    /** Set the backend-neutral log level. */
    virtual void setLogLevel(LogLevel level);

protected:
    /** Execute a buffer set already canonicalized by execute(). */
    virtual void executeNormalized(std::span<const irt::BufferView> buffers,
                                   irt::ExecuteOptions options) = 0;

    LogLevel         log_level_{LogLevel::Warning};
    std::uintptr_t   external_stream_{0};
};

/** Create the runtime adapter for a backend selected by the model runtime. */
INFERRT_MODEL_API std::unique_ptr<IBackendRuntime> CreateBackendRuntime(ModelRuntime::Backend backend);

} // namespace irt::model
