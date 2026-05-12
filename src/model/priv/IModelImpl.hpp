#pragma once

#include <inferrt/model/IModelConfig.hpp>
#include <inferrt/model/IParams.hpp>
#include <inferrt/model/Utils.hpp>

#include <memory>
#include <string>

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

    const IModelConfig &modelConfig() const noexcept;
    int                 numClasses() const noexcept;
    const InputShape   &inputShape() const noexcept;

    void setLogLevel(nvinfer1::ILogger::Severity severity);

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
