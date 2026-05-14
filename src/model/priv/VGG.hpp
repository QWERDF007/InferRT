#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

/**
 * @brief VGG 家族模型实现基类。
 */
class VGG : public priv::IModelImpl
{
public:
    explicit VGG()
        : priv::IModelImpl() {};
    ~VGG() override = default;

    std::string name() const noexcept override
    {
        return "VGG";
    }

    void infer(const std::vector<void *> &buffers) override;
};

/**
 * @brief torchvision 风格的 VGG11 实现。
 */
class VGG11 : public VGG
{
public:
    static constexpr const char *key() noexcept
    {
        return "vgg11";
    }

    explicit VGG11()
        : VGG() {};
    ~VGG11() override = default;

    std::string name() const noexcept override
    {
        return "VGG11";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 VGG13 实现。
 */
class VGG13 : public VGG
{
public:
    static constexpr const char *key() noexcept
    {
        return "vgg13";
    }

    explicit VGG13()
        : VGG() {};
    ~VGG13() override = default;

    std::string name() const noexcept override
    {
        return "VGG13";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 VGG16 实现。
 */
class VGG16 : public VGG
{
public:
    static constexpr const char *key() noexcept
    {
        return "vgg16";
    }

    explicit VGG16()
        : VGG() {};
    ~VGG16() override = default;

    std::string name() const noexcept override
    {
        return "VGG16";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 VGG19 实现。
 */
class VGG19 : public VGG
{
public:
    static constexpr const char *key() noexcept
    {
        return "vgg19";
    }

    explicit VGG19()
        : VGG() {};
    ~VGG19() override = default;

    std::string name() const noexcept override
    {
        return "VGG19";
    }

    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

} // namespace irt::model
