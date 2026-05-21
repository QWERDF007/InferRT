#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

/**
 * @brief MobileNet 家族模型实现基类。
 */
class MobileNet : public priv::IModelImpl
{
public:
    /**
     * @brief 构造 MobileNet 基类实现。
     */
    explicit MobileNet()
        : priv::IModelImpl() {};

    /**
     * @brief 析构 MobileNet 基类实现。
     */
    ~MobileNet() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `MobileNet`。
     */
    std::string name() const noexcept override
    {
        return "MobileNet";
    }

    /**
     * @brief 在指定 CUDA stream 上执行 MobileNet 推理。
     * @param buffers 输入输出缓冲区地址列表。
     * @param stream 调用方提供的 CUDA stream；为空时使用模型当前默认 stream。
     */
    void infer(const std::vector<void *> &buffers, cudaStream_t stream = nullptr,
               bool non_blocking = false) override;

};

/**
 * @brief MobileNetV2 模型实现。
 */
class MobileNetV2 : public MobileNet
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `mobilenet_v2`。
     */
    static constexpr const char *key() noexcept
    {
        return "mobilenet_v2";
    }

    /**
     * @brief 获取模型显示名称。
     * @return `MobileNetV2`。
     */
    std::string name() const noexcept override
    {
        return "MobileNetV2";
    }

    /**
     * @brief 构建 MobileNetV2 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief MobileNetV3 Large 模型实现。
 */
class MobileNetV3Large : public MobileNet
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `mobilenet_v3_large`。
     */
    static constexpr const char *key() noexcept
    {
        return "mobilenet_v3_large";
    }

    /**
     * @brief 获取模型显示名称。
     * @return `MobileNetV3Large`。
     */
    std::string name() const noexcept override
    {
        return "MobileNetV3Large";
    }

    /**
     * @brief 构建 MobileNetV3 Large 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

/**
 * @brief MobileNetV3 Small 模型实现。
 */
class MobileNetV3Small : public MobileNet
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `mobilenet_v3_small`。
     */
    static constexpr const char *key() noexcept
    {
        return "mobilenet_v3_small";
    }

    /**
     * @brief 获取模型显示名称。
     * @return `MobileNetV3Small`。
     */
    std::string name() const noexcept override
    {
        return "MobileNetV3Small";
    }

    /**
     * @brief 构建 MobileNetV3 Small 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

} // namespace irt::model
