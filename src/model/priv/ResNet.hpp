#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

/**
 * @brief ResNet 家族模型实现基类。
 */
class ResNet : public priv::IModelImpl
{
public:
    /**
     * @brief 构造 ResNet 基类实现。
     */
    explicit ResNet()
        : priv::IModelImpl() {};

    /**
     * @brief 析构 ResNet 基类实现。
     */
    ~ResNet() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `ResNet`。
     */
    std::string name() const noexcept override
    {
        return "ResNet";
    }

    /**
     * @brief 在指定 CUDA stream 上执行 ResNet 推理。
     * @param buffers 输入输出缓冲区地址列表。
     * @param stream 调用方提供的 CUDA stream；为空时使用模型当前默认 stream。
     */
    void infer(const std::vector<void *> &buffers, cudaStream_t stream = nullptr,
               bool non_blocking = false) override;

};

/**
 * @brief torchvision 风格的 ResNet-18 实现。
 */
class ResNet18 : public ResNet
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `resnet18`。
     */
    static constexpr const char *key() noexcept
    {
        return "resnet18";
    }

    /**
     * @brief 构造 ResNet-18 模型实现。
     */
    explicit ResNet18()
        : ResNet() {};

    /**
     * @brief 析构 ResNet-18 模型实现。
     */
    ~ResNet18() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `ResNet18`。
     */
    std::string name() const noexcept override
    {
        return "ResNet18";
    }

    /**
     * @brief 构建 ResNet-18 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 ResNet-34 实现。
 */
class ResNet34 : public ResNet
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `resnet34`。
     */
    static constexpr const char *key() noexcept
    {
        return "resnet34";
    }

    /**
     * @brief 构造 ResNet-34 模型实现。
     */
    explicit ResNet34()
        : ResNet() {};

    /**
     * @brief 析构 ResNet-34 模型实现。
     */
    ~ResNet34() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `ResNet34`。
     */
    std::string name() const noexcept override
    {
        return "ResNet34";
    }

    /**
     * @brief 构建 ResNet-34 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 ResNet-50 实现。
 */
class ResNet50 : public ResNet
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `resnet50`。
     */
    static constexpr const char *key() noexcept
    {
        return "resnet50";
    }

    /**
     * @brief 构造 ResNet-50 模型实现。
     */
    explicit ResNet50()
        : ResNet() {};

    /**
     * @brief 析构 ResNet-50 模型实现。
     */
    ~ResNet50() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `ResNet50`。
     */
    std::string name() const noexcept override
    {
        return "ResNet50";
    }

    /**
     * @brief 构建 ResNet-50 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 ResNet-101 实现。
 */
class ResNet101 : public ResNet
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `resnet101`。
     */
    static constexpr const char *key() noexcept
    {
        return "resnet101";
    }

    /**
     * @brief 构造 ResNet-101 模型实现。
     */
    explicit ResNet101()
        : ResNet() {};

    /**
     * @brief 析构 ResNet-101 模型实现。
     */
    ~ResNet101() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `ResNet101`。
     */
    std::string name() const noexcept override
    {
        return "ResNet101";
    }

    /**
     * @brief 构建 ResNet-101 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 ResNet-152 实现。
 */
class ResNet152 : public ResNet
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `resnet152`。
     */
    static constexpr const char *key() noexcept
    {
        return "resnet152";
    }

    /**
     * @brief 构造 ResNet-152 模型实现。
     */
    explicit ResNet152()
        : ResNet() {};

    /**
     * @brief 析构 ResNet-152 模型实现。
     */
    ~ResNet152() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `ResNet152`。
     */
    std::string name() const noexcept override
    {
        return "ResNet152";
    }

    /**
     * @brief 构建 ResNet-152 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 Wide-ResNet-50-2 实现。
 */
class WideResNet50_2 : public ResNet
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `wide_resnet50_2`。
     */
    static constexpr const char *key() noexcept
    {
        return "wide_resnet50_2";
    }

    /**
     * @brief 构造 Wide-ResNet-50-2 模型实现。
     */
    explicit WideResNet50_2()
        : ResNet() {};

    /**
     * @brief 析构 Wide-ResNet-50-2 模型实现。
     */
    ~WideResNet50_2() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `WideResNet50_2`。
     */
    std::string name() const noexcept override
    {
        return "WideResNet50_2";
    }

    /**
     * @brief 构建 Wide-ResNet-50-2 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief torchvision 风格的 Wide-ResNet-101-2 实现。
 */
class WideResNet101_2 : public ResNet
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `wide_resnet101_2`。
     */
    static constexpr const char *key() noexcept
    {
        return "wide_resnet101_2";
    }

    /**
     * @brief 构造 Wide-ResNet-101-2 模型实现。
     */
    explicit WideResNet101_2()
        : ResNet() {};

    /**
     * @brief 析构 Wide-ResNet-101-2 模型实现。
     */
    ~WideResNet101_2() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `WideResNet101_2`。
     */
    std::string name() const noexcept override
    {
        return "WideResNet101_2";
    }

    /**
     * @brief 构建 Wide-ResNet-101-2 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

} // namespace irt::model
