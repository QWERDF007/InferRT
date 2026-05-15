#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

class MobileNet : public priv::IModelImpl
{
public:
    explicit MobileNet()
        : priv::IModelImpl() {};
    ~MobileNet() override = default;

    std::string name() const noexcept override
    {
        return "MobileNet";
    }

    void infer(const std::vector<void *> &buffers) override;
};

class MobileNetV2 : public MobileNet
{
public:
    static constexpr const char *key() noexcept
    {
        return "mobilenet_v2";
    }

    std::string name() const noexcept override
    {
        return "MobileNetV2";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

class MobileNetV3Large : public MobileNet
{
public:
    static constexpr const char *key() noexcept
    {
        return "mobilenet_v3_large";
    }

    std::string name() const noexcept override
    {
        return "MobileNetV3Large";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

class MobileNetV3Small : public MobileNet
{
public:
    static constexpr const char *key() noexcept
    {
        return "mobilenet_v3_small";
    }

    std::string name() const noexcept override
    {
        return "MobileNetV3Small";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

} // namespace irt::model
