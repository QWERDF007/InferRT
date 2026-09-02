#include "BackendRuntime.hpp"
#include "TensorRTBackend.hpp"

#include <inferrt/core/Exception.hpp>

#include <atomic>

namespace irt::model {

namespace {

std::vector<irt::TensorInfo> describeIo(const IBackendRuntime &backend, const irt::TensorIOMode mode)
{
    const auto names = backend.ioTensorNames(mode);
    std::vector<irt::TensorInfo> result;
    result.reserve(names.size());
    for (const auto &name : names)
    {
        result.push_back(irt::TensorInfo{
            name,
            irt::TensorDesc{backend.tensorDataType(name), irt::TensorLayout::Opaque, backend.ioMemoryKind(mode),
                            backend.tensorShape(name)},
            mode == irt::TensorIOMode::Input ? irt::TensorIOMode::Input : irt::TensorIOMode::Output});
    }
    return result;
}

} // namespace

std::vector<irt::TensorInfo> IBackendRuntime::inputs() const
{
    return describeIo(*this, irt::TensorIOMode::Input);
}

std::vector<irt::TensorInfo> IBackendRuntime::outputs() const
{
    return describeIo(*this, irt::TensorIOMode::Output);
}

void IBackendRuntime::setInputShape(const std::string &tensor_name, irt::Shape shape)
{
    if (shape.empty())
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Input shape must not be empty");
    }
    (void)shape.elementCount();
    setTensorShape(tensor_name, shape);
}

std::unique_ptr<irt::ITensorRuntimeSession> IBackendRuntime::createSession() const
{
    throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED,
                         "Backend %s does not expose isolated execution sessions",
                         ModelRuntime::backendName(backend()));
}

void IBackendRuntime::executeSession(irt::ITensorRuntimeSession &session,
                                     const std::span<const irt::BufferView> buffers,
                                     const irt::ExecuteOptions options) const
{
    auto normalized = irt::normalizeExecutionBuffers(buffers, inputs(), outputs());
    session.execute(normalized, options);
}

bool IBackendRuntime::isInputBatchDynamic(const std::string &tensor_name) const
{
    const auto shape = tensorShape(tensor_name);
    return shape.rank() > 0 && shape[0] < 0;
}

namespace {

using BackendRuntimeFactory = std::unique_ptr<IBackendRuntime> (*)(ModelRuntime::Backend);
std::atomic<BackendRuntimeFactory> g_factory_override{nullptr};

} // namespace

void IBackendRuntime::save(const std::string &engine_file) const
{
    (void)engine_file;
    throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "save() is only supported by TensorRT backend");
}

void IBackendRuntime::setStream(const std::uintptr_t stream)
{
    if (stream == 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "stream must not be null; use clearStream to reset");
    }
    external_stream_ = stream;
}

void IBackendRuntime::clearStream()
{
    external_stream_ = 0;
}

std::uintptr_t IBackendRuntime::resolveExecutionStream(const std::uintptr_t stream_override)
{
    return stream_override != 0 ? stream_override : external_stream_;
}

LogLevel IBackendRuntime::logLevel() const noexcept
{
    return log_level_;
}

void IBackendRuntime::setLogLevel(const LogLevel level)
{
    log_level_ = level;
}

std::unique_ptr<IBackendRuntime> CreateBackendRuntime(ModelRuntime::Backend backend)
{
    if (const auto factory = g_factory_override.load(std::memory_order_acquire))
    {
        return factory(backend);
    }

    switch (backend)
    {
    case ModelRuntime::Backend::TensorRT:
        return std::make_unique<priv::TensorRTBackend>();
    case ModelRuntime::Backend::ONNXRuntime:
#if INFERRT_BUILD_ONNX
        return priv::CreateONNXRuntimeBackend();
#else
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                             "ONNX Runtime backend is disabled; configure with -DINFERRT_BUILD_ONNX=ON");
#endif
    case ModelRuntime::Backend::OpenVINO:
#if INFERRT_BUILD_OPENVINO
        return priv::CreateOpenVINOBackend();
#else
        throw irt::Exception(irt::Status::ERROR_NOT_IMPLEMENTED,
                             "OpenVINO backend is disabled; configure with -DINFERRT_BUILD_OPENVINO=ON");
#endif
    }

    throw irt::Exception(Status::ERROR_NOT_IMPLEMENTED, "Unsupported model backend");
}

void SetBackendRuntimeFactoryOverride(
    std::unique_ptr<irt::model::IBackendRuntime> (*factory)(ModelRuntime::Backend)) noexcept
{
    g_factory_override.store(factory, std::memory_order_release);
}

} // namespace irt::model
