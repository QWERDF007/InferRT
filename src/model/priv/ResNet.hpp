#pragma once

#include <inferrt/model/IModel.h>

namespace irt::model {

class ResNet : public IModel
{
public:
    explicit ResNet()
        : IModel() {};
    ~ResNet() override = default;

    std::string name() const noexcept override
    {
        return "ResNet";
    }

    void infer(const std::vector<void *> &buffers) override;
};

class ResNet18 : public ResNet
{
public:
    static constexpr const char *key() noexcept
    {
        return "resnet18";
    }

    explicit ResNet18()
        : ResNet() {};
    ~ResNet18() override = default;

    std::string name() const noexcept override
    {
        return "ResNet18";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

class ResNet34 : public ResNet
{
public:
    static constexpr const char *key() noexcept
    {
        return "resnet34";
    }

    explicit ResNet34()
        : ResNet() {};
    ~ResNet34() override = default;

    std::string name() const noexcept override
    {
        return "ResNet34";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

class ResNet50 : public ResNet
{
public:
    static constexpr const char *key() noexcept
    {
        return "resnet50";
    }

    explicit ResNet50()
        : ResNet() {};
    ~ResNet50() override = default;

    std::string name() const noexcept override
    {
        return "ResNet50";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

class ResNet101 : public ResNet
{
public:
    static constexpr const char *key() noexcept
    {
        return "resnet101";
    }

    explicit ResNet101()
        : ResNet() {};
    ~ResNet101() override = default;

    std::string name() const noexcept override
    {
        return "ResNet101";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

class ResNet152 : public ResNet
{
public:
    static constexpr const char *key() noexcept
    {
        return "resnet152";
    }

    explicit ResNet152()
        : ResNet() {};
    ~ResNet152() override = default;

    std::string name() const noexcept override
    {
        return "ResNet152";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

} // namespace irt::model
