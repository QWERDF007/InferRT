#pragma once

#include "IParams.hpp"
#include "Utils.hpp"

#include <inferrt/model/Export.h>

#include <string>
#include <vector>

namespace irt::model {

class INFERRT_MODEL_API IModel
{
public:
    IModel()          = default;
    virtual ~IModel() = default;

    virtual std::string name() const noexcept = 0;

    virtual std::string wtsExtension() const noexcept
    {
        return ".wts";
    }

    virtual std::string engineExtension() const noexcept
    {
        return ".engine";
    }

    virtual nvinfer1::ILogger::Severity logLevel() const noexcept
    {
        return trt_params_.log_level;
    }

    virtual void build(const std::string &weights_file);
    virtual void save(const std::string &weights_file);
    virtual void load(const std::string &weights_file);
    virtual void buildOrLoad(const std::string &weights_file);

    virtual void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) = 0;

    virtual void infer(const std::vector<void *> &buffers) = 0;

    void setLogLevel(nvinfer1::ILogger::Severity severity)
    {
        trt_params_.log_level = severity;
        if (trt_params_.logger)
        {
            trt_params_.logger->setReportableSeverity(severity);
        }
    }

protected:
    TRTParams trt_params_;

    void initLogger()
    {
        trt_params_.logger = std::make_unique<Logger>(name(), logLevel());
    }
};

} // namespace irt::model