#include "priv/IModelImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/model/IModel.h>

#include <utility>

namespace irt::model {

namespace {

void ensureValid(const IModel::ImplementationPtr &impl)
{
    if (!impl)
    {
        throw irt::Exception(Status::INVALID_OPERATION, "IModel handle is invalid (null implementation)");
    }
}

} // namespace

IModel::IModel()
    : impl_(nullptr)
{
}

void IModel::ImplementationDeleter::operator()(Implementation *implementation) const noexcept
{
    delete implementation;
}

IModel::~IModel() = default;

IModel::IModel(IModel &&) noexcept            = default;
IModel &IModel::operator=(IModel &&) noexcept = default;

bool IModel::isValid() const noexcept
{
    return impl_ != nullptr;
}

std::string IModel::name() const
{
    ensureValid(impl_);
    return impl_->name();
}

std::string IModel::wtsExtension() const
{
    ensureValid(impl_);
    return impl_->wtsExtension();
}

std::string IModel::engineExtension() const
{
    ensureValid(impl_);
    return impl_->engineExtension();
}

LogLevel IModel::logLevel() const
{
    ensureValid(impl_);
    return impl_->logLevel();
}

void IModel::build(const std::string &weights_file)
{
    ensureValid(impl_);
    impl_->build(weights_file);
}

void IModel::save(const std::string &weights_file)
{
    ensureValid(impl_);
    impl_->save(weights_file);
}

void IModel::load(const std::string &weights_file)
{
    ensureValid(impl_);
    impl_->load(weights_file);
}

void IModel::buildOrLoad(const std::string &weights_file)
{
    ensureValid(impl_);
    impl_->buildOrLoad(weights_file);
}

void IModel::infer(std::span<const irt::BufferView> buffers, std::uintptr_t stream, bool non_blocking)
{
    ensureValid(impl_);
    impl_->infer(buffers, stream, non_blocking);
}

void IModel::forwardFeatures(std::span<const irt::BufferView> buffers, std::uintptr_t stream, bool non_blocking)
{
    ensureValid(impl_);
    impl_->forwardFeatures(buffers, stream, non_blocking);
}

void IModel::setModelConfig(std::unique_ptr<IModelConfig> config)
{
    ensureValid(impl_);
    impl_->setModelConfig(std::move(config));
}

const IModelConfig &IModel::modelConfig() const
{
    ensureValid(impl_);
    return impl_->modelConfig();
}

const ModelRuntime &IModel::runtime() const
{
    ensureValid(impl_);
    return impl_->modelConfig().runtime();
}

std::vector<std::string> IModel::ioTensorNames(irt::TensorIOMode mode) const
{
    ensureValid(impl_);
    return impl_->ioTensorNames(mode);
}

irt::Shape IModel::tensorShape(const std::string &tensor_name) const
{
    ensureValid(impl_);
    return impl_->tensorShape(tensor_name);
}

irt::TensorDataType IModel::tensorDataType(const std::string &tensor_name) const
{
    ensureValid(impl_);
    return impl_->tensorDataType(tensor_name);
}

void IModel::setTensorShape(const std::string &tensor_name, irt::Shape shape)
{
    ensureValid(impl_);
    impl_->setTensorShape(tensor_name, shape);
}

void IModel::setStream(std::uintptr_t stream)
{
    ensureValid(impl_);
    impl_->setStream(stream);
}

void IModel::clearStream()
{
    ensureValid(impl_);
    impl_->clearStream();
}

std::uintptr_t IModel::resolveExecutionStream(std::uintptr_t stream_override) const
{
    ensureValid(impl_);
    return impl_->resolveExecutionStream(stream_override);
}

void IModel::setLogLevel(LogLevel level)
{
    ensureValid(impl_);
    impl_->setLogLevel(level);
}

std::vector<irt::TensorInfo> IModel::inputs() const
{
    ensureValid(impl_);
    return impl_->inputs();
}

std::vector<irt::TensorInfo> IModel::outputs() const
{
    ensureValid(impl_);
    return impl_->outputs();
}

irt::ExecutionCapabilities IModel::capabilities() const
{
    ensureValid(impl_);
    return impl_->capabilities();
}

void IModel::setInputShape(const std::string &name, irt::Shape shape)
{
    ensureValid(impl_);
    impl_->setInputShape(name, std::move(shape));
}

void IModel::execute(std::span<const irt::BufferView> buffers, irt::ExecuteOptions options)
{
    ensureValid(impl_);
    impl_->execute(buffers, options);
}

std::unique_ptr<irt::ITensorRuntimeSession> IModel::createSession() const
{
    ensureValid(impl_);
    return impl_->createSession();
}

void IModel::executeSession(irt::ITensorRuntimeSession &session, std::span<const irt::BufferView> buffers,
                            irt::ExecuteOptions options) const
{
    ensureValid(impl_);
    impl_->executeSession(session, buffers, options);
}

} // namespace irt::model
