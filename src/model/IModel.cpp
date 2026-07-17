#include "priv/IModelImpl.hpp"

#include <inferrt/model/IModel.h>


namespace irt::model {

IModel::IModel()
    : impl_(nullptr)
{
}

IModel::IModel(std::unique_ptr<priv::IModelImpl> impl)
    : impl_(std::move(impl))
{
}

IModel::~IModel() = default;

IModel::IModel(IModel &&) noexcept            = default;
IModel &IModel::operator=(IModel &&) noexcept = default;

std::string IModel::name() const noexcept
{
    return impl_ ? impl_->name() : "";
}

std::string IModel::wtsExtension() const noexcept
{
    return impl_ ? impl_->wtsExtension() : ".wts";
}

std::string IModel::engineExtension() const noexcept
{
    return impl_ ? impl_->engineExtension() : ".engine";
}

nvinfer1::ILogger::Severity IModel::logLevel() const noexcept
{
    return impl_->logLevel();
}

void IModel::build(const std::string &weights_file)
{
    impl_->build(weights_file);
}

void IModel::save(const std::string &weights_file)
{
    impl_->save(weights_file);
}

void IModel::load(const std::string &weights_file)
{
    impl_->load(weights_file);
}

void IModel::buildOrLoad(const std::string &weights_file)
{
    impl_->buildOrLoad(weights_file);
}

void IModel::buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map)
{
    impl_->buildNetwork(network, weights_map);
}

void IModel::infer(const std::vector<void *> &buffers, cudaStream_t stream, bool non_blocking)
{
    impl_->infer(buffers, stream, non_blocking);
}

void IModel::forwardFeatures(const std::vector<void *> &buffers, cudaStream_t stream, bool non_blocking)
{
    impl_->forwardFeatures(buffers, stream, non_blocking);
}

void IModel::setModelConfig(std::unique_ptr<IModelConfig> config)
{
    impl_->setModelConfig(std::move(config));
}

const IModelConfig &IModel::modelConfig() const noexcept
{
    return impl_->modelConfig();
}

ModelBackend IModel::backend() const noexcept
{
    return impl_->modelConfig().backend();
}

ModelDevice IModel::device() const noexcept
{
    return impl_->modelConfig().device();
}

int IModel::deviceId() const noexcept
{
    return impl_->modelConfig().deviceId();
}

std::vector<std::string> IModel::ioTensorNames(nvinfer1::TensorIOMode mode) const
{
    return impl_->ioTensorNames(mode);
}

nvinfer1::Dims IModel::tensorShape(const std::string &tensor_name) const
{
    return impl_->tensorShape(tensor_name);
}

nvinfer1::DataType IModel::tensorDataType(const std::string &tensor_name) const
{
    return impl_->tensorDataType(tensor_name);
}

void IModel::setTensorShape(const std::string &tensor_name, const nvinfer1::Dims &dims)
{
    impl_->setTensorShape(tensor_name, dims);
}

void IModel::setStream(cudaStream_t stream)
{
    impl_->setStream(stream);
}

void IModel::clearStream()
{
    impl_->clearStream();
}

cudaStream_t IModel::resolveExecutionStream(cudaStream_t stream_override) const
{
    return impl_->resolveExecutionStream(stream_override);
}

void IModel::setLogLevel(nvinfer1::ILogger::Severity severity)
{
    impl_->setLogLevel(severity);
}

} // namespace irt::model
