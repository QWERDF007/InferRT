#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

/**
 * @brief VGG 家族模型实现基类。
 */
class VGG : public priv::IModelImpl
{
public:
    /**
     * @brief 构造 VGG 基类实现。
     */
    explicit VGG()
        : priv::IModelImpl() {};

    /**
     * @brief 析构 VGG 基类实现。
     */
    ~VGG() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `VGG`。
     */
    std::string name() const noexcept override
    {
        return "VGG";
    }

};

/**
 * @brief torchvision 风格的 VGG11 实现。
 */
class VGG11 : public VGG
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `vgg11`。
     */
    static constexpr const char *key() noexcept
    {
        return "vgg11";
    }

    /**
     * @brief 构造 VGG11 模型实现。
     */
    explicit VGG11()
        : VGG() {};

    /**
     * @brief 析构 VGG11 模型实现。
     */
    ~VGG11() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `VGG11`。
     */
    std::string name() const noexcept override
    {
        return "VGG11";
    }

    /**
     * @brief 构建 VGG11 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 VGG13 实现。
 */
class VGG13 : public VGG
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `vgg13`。
     */
    static constexpr const char *key() noexcept
    {
        return "vgg13";
    }

    /**
     * @brief 构造 VGG13 模型实现。
     */
    explicit VGG13()
        : VGG() {};

    /**
     * @brief 析构 VGG13 模型实现。
     */
    ~VGG13() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `VGG13`。
     */
    std::string name() const noexcept override
    {
        return "VGG13";
    }

    /**
     * @brief 构建 VGG13 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 VGG16 实现。
 */
class VGG16 : public VGG
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `vgg16`。
     */
    static constexpr const char *key() noexcept
    {
        return "vgg16";
    }

    /**
     * @brief 构造 VGG16 模型实现。
     */
    explicit VGG16()
        : VGG() {};

    /**
     * @brief 析构 VGG16 模型实现。
     */
    ~VGG16() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `VGG16`。
     */
    std::string name() const noexcept override
    {
        return "VGG16";
    }

    /**
     * @brief 构建 VGG16 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 VGG19 实现。
 */
class VGG19 : public VGG
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `vgg19`。
     */
    static constexpr const char *key() noexcept
    {
        return "vgg19";
    }

    /**
     * @brief 构造 VGG19 模型实现。
     */
    explicit VGG19()
        : VGG() {};

    /**
     * @brief 析构 VGG19 模型实现。
     */
    ~VGG19() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `VGG19`。
     */
    std::string name() const noexcept override
    {
        return "VGG19";
    }

    /**
     * @brief 构建 VGG19 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

} // namespace irt::model
