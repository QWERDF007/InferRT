#pragma once

#include <inferrt/model/IModel.hpp>

namespace irt::model {

class ResNet : public IModel
{
public:
    explicit ResNet()
        : IModel() {};
    ~ResNet() override = default;

    std::string name() const noexcept
    {
        return "ResNet";
    }

    void infer(const std::vector<void *> &buffers) override;
};

class ResNet18 : public ResNet
{
public:
    explicit ResNet18()
        : ResNet() {};
    ~ResNet18() override = default;

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

} // namespace irt::model