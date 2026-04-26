#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

class AlexNet : public priv::IModelImpl
{
public:
    static constexpr const char *key() noexcept
    {
        return "alexnet";
    }

    explicit AlexNet()
        : priv::IModelImpl() {};
    ~AlexNet() override = default;

    std::string name() const noexcept
    {
        return "AlexNet";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;

    void infer(const std::vector<void *> &buffers) override;
};

} // namespace irt::model
