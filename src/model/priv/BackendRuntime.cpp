#include "BackendRuntime.hpp"
#include "TensorRTBackend.hpp"

#include <inferrt/core/Exception.hpp>

#include <atomic>
#include <limits>

namespace irt::model {

namespace {

template <typename TensorDescriptor>
std::vector<irt::TensorInfo> describeIo(const IBackendRuntime &backend, const TensorDescriptor &descriptor,
                                        const irt::TensorIOMode mode)
{
    const auto names = backend.ioTensorNames(mode);
    std::vector<irt::TensorInfo> result;
    result.reserve(names.size());
    for (const auto &name : names)
    {
        result.push_back(irt::TensorInfo{
            name,
            irt::TensorDesc{descriptor.tensorDataType(name), irt::TensorLayout::Opaque, backend.ioMemoryKind(mode),
                            descriptor.tensorShape(name)},
            mode == irt::TensorIOMode::Input ? irt::TensorIOMode::Input : irt::TensorIOMode::Output});
    }
    return result;
}

} // namespace

std::vector<irt::TensorInfo> IBackendRuntime::inputs() const
{
    return describeIo(*this, *this, irt::TensorIOMode::Input);
}

std::vector<irt::TensorInfo> IBackendRuntime::outputs() const
{
    return describeIo(*this, *this, irt::TensorIOMode::Output);
}

irt::ExecutionCapabilities IBackendRuntime::capabilities() const
{
    const auto input_infos = inputs();
    if (input_infos.empty())
    {
        throw irt::Exception(Status::INVALID_OPERATION, "Backend runtime exposes no input tensors");
    }

    bool dynamic_batch = false;
    int  fixed_batch   = 0;
    for (const auto &info : input_infos)
    {
        if (info.desc.shape.rank() == 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Backend input tensor has no batch dimension: %s", info.name.c_str());
        }

        if (isInputBatchDynamic(info.name))
        {
            dynamic_batch = true;
            continue;
        }

        const auto batch = info.desc.shape[0];
        if (batch <= 0 || batch > static_cast<int64_t>((std::numeric_limits<int>::max)()))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Backend input tensor has an invalid fixed batch size: %s", info.name.c_str());
        }

        const int current_batch = static_cast<int>(batch);
        if (fixed_batch != 0 && fixed_batch != current_batch)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "Backend input tensors use different fixed batch sizes");
        }
        fixed_batch = current_batch;
    }

    return {.supports_dynamic_batch = dynamic_batch,
            .supports_feature_outputs = false,
            .fixed_batch_size = dynamic_batch ? 0 : fixed_batch};
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

void IBackendRuntime::execute(const std::span<const irt::BufferView> buffers, const irt::ExecuteOptions options)
{
    auto normalized = irt::normalizeExecutionBuffers(buffers, inputs(), outputs());
    executeNormalized(normalized, options);
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
    const auto session_inputs  = describeIo(*this, session, irt::TensorIOMode::Input);
    const auto session_outputs = describeIo(*this, session, irt::TensorIOMode::Output);
    auto       normalized      = irt::normalizeExecutionBuffers(buffers, session_inputs, session_outputs);
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
