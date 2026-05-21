#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

/**
 * @brief AlexNet 模型实现。
 */
class AlexNet : public priv::IModelImpl
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `alexnet`。
     */
    static constexpr const char *key() noexcept
    {
        return "alexnet";
    }

    /**
     * @brief 构造 AlexNet 模型实现。
     */
    explicit AlexNet()
        : priv::IModelImpl() {};

    /**
     * @brief 析构 AlexNet 模型实现。
     */
    ~AlexNet() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `AlexNet`。
     */
    std::string name() const noexcept
    {
        return "AlexNet";
    }

    /**
     * @brief 构建 AlexNet 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;

    /**
     * @brief 在指定 CUDA stream 上执行 AlexNet 推理。
     * @param buffers 输入输出缓冲区地址列表。
     * @param stream 调用方提供的 CUDA stream；为空时使用模型当前默认 stream。
     */
    void infer(const std::vector<void *> &buffers, cudaStream_t stream = nullptr,
               bool non_blocking = false) override;
};

} // namespace irt::model
