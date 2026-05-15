#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

class ONNXModel : public priv::IModelImpl
{
public:
    static constexpr const char *key() noexcept
    {
        return "onnx";
    }

    explicit ONNXModel()
        : priv::IModelImpl()
    {
    }

    ~ONNXModel() override = default;

    std::string name() const noexcept override
    {
        return "ONNX";
    }

    std::string wtsExtension() const noexcept override
    {
        return ".onnx";
    }

    void build(const std::string &onnx_file) override;
    void load(const std::string &engine_file) override;
    void buildOrLoad(const std::string &onnx_file) override;
    void infer(const std::vector<void *> &buffers) override;

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

} // namespace irt::model
