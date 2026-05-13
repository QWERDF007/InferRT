#pragma once

#include <inferrt/model/IModelConfig.hpp>
#include <inferrt/model/IParams.hpp>
#include <inferrt/model/Utils.hpp>

#include <memory>
#include <string>
#include <vector>

namespace irt::model::priv {

class IModelImpl
{
public:
    IModelImpl()
        : config_(std::make_unique<IModelConfig>())
    {
    }

    virtual ~IModelImpl() = default;

    virtual std::string name() const noexcept = 0;

    virtual std::string wtsExtension() const noexcept
    {
        return ".wts";
    }

    virtual std::string engineExtension() const noexcept
    {
        return ".engine";
    }

    virtual std::string generateSuffix(const IModelConfig &config) const noexcept;

    nvinfer1::ILogger::Severity logLevel() const noexcept;

    void build(const std::string &weights_file);
    void save(const std::string &engine_file);
    void load(const std::string &engine_file);
    void buildOrLoad(const std::string &weights_file);

    virtual void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) = 0;

    virtual void infer(const std::vector<void *> &buffers) = 0;

    void setModelConfig(std::unique_ptr<IModelConfig> config);
    void setNumClasses(int num_classes);
    void setInputShape(const InputShape &shape);
    void setInputShape(int channels, int height, int width);
    void setInputTensorNames(std::vector<std::string> input_tensor_names);
    void setOutputTensorNames(std::vector<std::string> output_tensor_names);

    const IModelConfig &modelConfig() const noexcept;
    int                 numClasses() const noexcept;
    const InputShape   &inputShape() const noexcept;
    const std::vector<std::string> &inputTensorNames() const noexcept;
    const std::vector<std::string> &outputTensorNames() const noexcept;

    void setLogLevel(nvinfer1::ILogger::Severity severity);

    nvinfer1::ITensor *addInputTensor(nvinfer1::INetworkDefinition *network, const nvinfer1::Dims &dims,
                                      nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT,
                                      size_t input_index = 0) const;
    void markOutputTensors(nvinfer1::INetworkDefinition *network, const std::vector<nvinfer1::ITensor *> &outputs) const;
    void bindTensorAddresses(const std::vector<void *> &buffers);

    TRTParams &trtParams() noexcept
    {
        return trt_params_;
    }

    const TRTParams &trtParams() const noexcept
    {
        return trt_params_;
    }

    void initLogger();

private:
    std::unique_ptr<IModelConfig> config_;
    TRTParams                     trt_params_;
};

} // namespace irt::model::priv
