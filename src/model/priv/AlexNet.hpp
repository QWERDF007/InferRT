#pragma once

#include <inferrt/model/IModel.hpp>

namespace irt::model {

class AlexNet : public IModel
{
public:
    explicit AlexNet()
        : IModel() {};
    ~AlexNet() override = default;

    std::string name() const noexcept
    {
        return "AlexNet";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;

    void infer(const std::vector<void *> &buffers) override;
};

} // namespace irt::model