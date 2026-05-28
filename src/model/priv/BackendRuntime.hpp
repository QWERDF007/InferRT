#pragma once

#include <inferrt/model/IModelConfig.hpp>
#include <inferrt/model/IParams.hpp>
#include <NvInfer.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace irt::model::priv {

class TensorRTBackend;

class IBackendRuntime
{
public:
    virtual ~IBackendRuntime() = default;

    virtual ModelBackend backend() const noexcept = 0;

    virtual void load(const std::string &model_file, const IModelConfig &config, const std::string &model_name) = 0;

    virtual void save(const std::string &engine_file) const;

    virtual std::vector<std::string> ioTensorNames(nvinfer1::TensorIOMode mode) const = 0;

    virtual nvinfer1::Dims tensorShape(const std::string &tensor_name) const = 0;

    virtual nvinfer1::DataType tensorDataType(const std::string &tensor_name) const = 0;

    virtual void setTensorShape(const std::string &tensor_name, const nvinfer1::Dims &dims) = 0;

    virtual void infer(const std::vector<void *> &buffers) = 0;

    virtual void setStream(cudaStream_t stream);

    virtual void clearStream();

    virtual cudaStream_t resolveExecutionStream(cudaStream_t stream_override = nullptr);

    virtual nvinfer1::ILogger::Severity logLevel() const noexcept;

    virtual void setLogLevel(nvinfer1::ILogger::Severity severity);

    virtual TensorRTBackend *asTensorRT() noexcept;

    virtual const TensorRTBackend *asTensorRT() const noexcept;

protected:
    nvinfer1::ILogger::Severity log_level_{nvinfer1::ILogger::Severity::kWARNING};
    cudaStream_t external_stream_{nullptr};
};

class TensorRTBackend final : public IBackendRuntime
{
public:
    using NetworkBuildFn = std::function<void(nvinfer1::INetworkDefinition *)>;

    ModelBackend backend() const noexcept override;

    void load(const std::string &engine_file, const IModelConfig &config, const std::string &model_name) override;

    void save(const std::string &engine_file) const override;

    std::vector<std::string> ioTensorNames(nvinfer1::TensorIOMode mode) const override;

    nvinfer1::Dims tensorShape(const std::string &tensor_name) const override;

    nvinfer1::DataType tensorDataType(const std::string &tensor_name) const override;

    void setTensorShape(const std::string &tensor_name, const nvinfer1::Dims &dims) override;

    void infer(const std::vector<void *> &buffers) override;

    void setStream(cudaStream_t stream) override;

    void clearStream() override;

    cudaStream_t resolveExecutionStream(cudaStream_t stream_override = nullptr) override;

    nvinfer1::ILogger::Severity logLevel() const noexcept override;

    void setLogLevel(nvinfer1::ILogger::Severity severity) override;

    TensorRTBackend *asTensorRT() noexcept override;

    const TensorRTBackend *asTensorRT() const noexcept override;

    TRTParams &params() noexcept;

    const TRTParams &params() const noexcept;

    void initLogger(const std::string &model_name);

    void buildFromNetwork(const std::string &source_file, const std::string &model_name, NetworkBuildFn build_fn);

    void execute(const std::vector<void *> &buffers, cudaStream_t stream_override, bool non_blocking);

    void setFeatureOnly(bool feature_only) noexcept;

private:
    void bindTensorAddresses(const std::vector<void *> &buffers);

    TRTParams params_;
};

std::unique_ptr<IBackendRuntime> CreateBackendRuntime(ModelBackend backend);

std::unique_ptr<IBackendRuntime> CreateONNXRuntimeBackend();
std::unique_ptr<IBackendRuntime> CreateOpenVINOBackend();

} // namespace irt::model::priv
