#include <inferrt/model/IModel.h>

#include "priv/IModelImpl.hpp"

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

void IModel::infer(const std::vector<void *> &buffers)
{
    impl_->infer(buffers);
}

void IModel::setModelConfig(std::unique_ptr<IModelConfig> config)
{
    impl_->setModelConfig(std::move(config));
}

void IModel::setNumClasses(int num_classes)
{
    impl_->setNumClasses(num_classes);
}

void IModel::setInputShape(const InputShape &shape)
{
    impl_->setInputShape(shape);
}

void IModel::setInputShape(int channels, int height, int width)
{
    impl_->setInputShape(channels, height, width);
}

void IModel::setInputTensorNames(std::vector<std::string> input_tensor_names)
{
    impl_->setInputTensorNames(std::move(input_tensor_names));
}

void IModel::setOutputTensorNames(std::vector<std::string> output_tensor_names)
{
    impl_->setOutputTensorNames(std::move(output_tensor_names));
}

const IModelConfig &IModel::modelConfig() const noexcept
{
    return impl_->modelConfig();
}

int IModel::numClasses() const noexcept
{
    return impl_->numClasses();
}

const InputShape &IModel::inputShape() const noexcept
{
    return impl_->inputShape();
}

const std::vector<std::string> &IModel::inputTensorNames() const noexcept
{
    return impl_->inputTensorNames();
}

const std::vector<std::string> &IModel::outputTensorNames() const noexcept
{
    return impl_->outputTensorNames();
}

void IModel::setLogLevel(nvinfer1::ILogger::Severity severity)
{
    impl_->setLogLevel(severity);
}

} // namespace irt::model
