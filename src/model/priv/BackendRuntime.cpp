#include "BackendRuntime.hpp"

#include <inferrt/core/Exception.hpp>

namespace irt::model::priv {

void IBackendRuntime::save(const std::string &engine_file) const
{
    (void)engine_file;
    throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "save() is only supported by TensorRT backend");
}

void IBackendRuntime::setStream(cudaStream_t stream)
{
    if (!stream)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "stream must not be null; use clearStream to reset");
    }
    external_stream_ = stream;
}

void IBackendRuntime::clearStream()
{
    external_stream_ = nullptr;
}

cudaStream_t IBackendRuntime::resolveExecutionStream(cudaStream_t stream_override)
{
    return stream_override ? stream_override : external_stream_;
}

nvinfer1::ILogger::Severity IBackendRuntime::logLevel() const noexcept
{
    return log_level_;
}

void IBackendRuntime::setLogLevel(nvinfer1::ILogger::Severity severity)
{
    log_level_ = severity;
}

TensorRTBackend *IBackendRuntime::asTensorRT() noexcept
{
    return nullptr;
}

const TensorRTBackend *IBackendRuntime::asTensorRT() const noexcept
{
    return nullptr;
}

std::unique_ptr<IBackendRuntime> CreateBackendRuntime(ModelRuntime::Backend backend)
{
    switch (backend)
    {
    case ModelRuntime::Backend::TensorRT:
        return std::make_unique<TensorRTBackend>();
    case ModelRuntime::Backend::ONNXRuntime:
        return CreateONNXRuntimeBackend();
    case ModelRuntime::Backend::OpenVINO:
        return CreateOpenVINOBackend();
    }

    throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Unsupported model backend");
}

} // namespace irt::model::priv
