#pragma once

#include "IModelConfig.hpp"
#include "IParams.hpp"
#include "Utils.hpp"

#include <memory>
#include <string>
#include <vector>

namespace irt::model::priv {
class IModelImpl;
}

namespace irt::model {

class INFERRT_MODEL_API IModel
{
public:
    IModel();
    explicit IModel(std::unique_ptr<priv::IModelImpl> impl);
    ~IModel();

    IModel(const IModel &)            = delete;
    IModel &operator=(const IModel &) = delete;
    IModel(IModel &&) noexcept;
    IModel &operator=(IModel &&) noexcept;

    virtual std::string name() const noexcept;

    virtual std::string wtsExtension() const noexcept;

    virtual std::string engineExtension() const noexcept;

    virtual nvinfer1::ILogger::Severity logLevel() const noexcept;

    virtual void build(const std::string &weights_file);
    virtual void save(const std::string &weights_file);
    virtual void load(const std::string &weights_file);
    virtual void buildOrLoad(const std::string &weights_file);

    virtual void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map);

    virtual void infer(const std::vector<void *> &buffers);

    virtual void setModelConfig(std::unique_ptr<IModelConfig> config);

    virtual void setNumClasses(int num_classes);

    virtual void setInputShape(const InputShape &shape);

    virtual void setInputShape(int channels, int height, int width);

    virtual void setInputTensorNames(std::vector<std::string> input_tensor_names);

    virtual void setOutputTensorNames(std::vector<std::string> output_tensor_names);

    virtual const IModelConfig &modelConfig() const noexcept;

    virtual int numClasses() const noexcept;

    virtual const InputShape &inputShape() const noexcept;

    virtual const std::vector<std::string> &inputTensorNames() const noexcept;

    virtual const std::vector<std::string> &outputTensorNames() const noexcept;

    virtual void setLogLevel(nvinfer1::ILogger::Severity severity);

private:
    std::unique_ptr<priv::IModelImpl> impl_;
};

} // namespace irt::model
