#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

/**
 * @brief ResNet 家族模型实现基类。
 */
class ResNet : public priv::IModelImpl
{
public:
    explicit ResNet()
        : priv::IModelImpl() {};
    ~ResNet() override = default;

    std::string name() const noexcept override
    {
        return "ResNet";
    }

    void infer(const std::vector<void *> &buffers) override;
};

/**
 * @brief torchvision 风格的 ResNet-18 实现。
 */
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

/**
 * @brief torchvision 风格的 ResNet-34 实现。
 */
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

/**
 * @brief torchvision 风格的 ResNet-50 实现。
 */
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

/**
 * @brief torchvision 风格的 ResNet-101 实现。
 */
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

/**
 * @brief torchvision 风格的 ResNet-152 实现。
 */
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

/**
 * @brief torchvision 风格的 Wide-ResNet-50-2 实现。
 */
class WideResNet50_2 : public ResNet
{
public:
    static constexpr const char *key() noexcept
    {
        return "wide_resnet50_2";
    }

    explicit WideResNet50_2()
        : ResNet() {};
    ~WideResNet50_2() override = default;

    std::string name() const noexcept override
    {
        return "WideResNet50_2";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 Wide-ResNet-101-2 实现。
 */
class WideResNet101_2 : public ResNet
{
public:
    static constexpr const char *key() noexcept
    {
        return "wide_resnet101_2";
    }

    explicit WideResNet101_2()
        : ResNet() {};
    ~WideResNet101_2() override = default;

    std::string name() const noexcept override
    {
        return "WideResNet101_2";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

} // namespace irt::model
