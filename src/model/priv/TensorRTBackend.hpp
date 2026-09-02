#pragma once

#include <inferrt/model/BackendRuntime.hpp>
#include "TRTParams.hpp"
#include "TRTUtils.hpp"

#include <NvInfer.h>

#include <functional>

namespace irt::model::priv {

/**
 * @brief TensorRT-specific backend adapter.
 *
 * This header is private to the model implementation.  Public model and core
 * headers depend only on ``IBackendRuntime`` and core tensor descriptors.
 */
class TensorRTBackend final : public irt::model::IBackendRuntime
{
public:
    using NetworkBuildFn = std::function<void(nvinfer1::INetworkDefinition *)>;

    ModelRuntime::Backend backend() const noexcept override;

    void load(const std::string &engine_file, const IModelConfig &config, const std::string &model_name) override;
    void save(const std::string &engine_file) const override;
    std::vector<std::string> ioTensorNames(irt::TensorIOMode mode) const override;
    irt::MemoryKind ioMemoryKind(irt::TensorIOMode mode) const noexcept override;
    irt::Shape tensorShape(const std::string &tensor_name) const override;
    bool isInputBatchDynamic(const std::string &tensor_name) const override;
    irt::TensorDataType tensorDataType(const std::string &tensor_name) const override;
    void setTensorShape(const std::string &tensor_name, const irt::Shape &shape) override;
    void execute(std::span<const irt::BufferView> buffers, irt::ExecuteOptions options = {}) override;
    std::unique_ptr<irt::ITensorRuntimeSession> createSession() const override;

    void setStream(std::uintptr_t stream) override;
    void clearStream() override;
    std::uintptr_t resolveExecutionStream(std::uintptr_t stream_override = 0) override;
    LogLevel logLevel() const noexcept override;
    void setLogLevel(LogLevel level) override;

    TRTParams &params() noexcept;
    const TRTParams &params() const noexcept;
    void initLogger(const std::string &model_name);

    void buildFromNetwork(const std::string &source_file, const std::string &model_name, const IModelConfig &config,
                          NetworkBuildFn build_fn);

    /** Mark whether this engine exposes feature outputs instead of primary outputs. */
    void setFeatureOnly(bool feature_only) noexcept;

private:
    TRTParams params_;
    int       device_id_{0};
};

} // namespace irt::model::priv
