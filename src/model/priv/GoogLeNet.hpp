#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

/**
 * @brief GoogLeNet / Inception v1 模型实现。
 *
 * 该实现对齐 torchvision.models.googlenet 的主分类分支，并复用项目内
 * IModelImpl 的权重读取、engine 构建、I/O 命名和 featureOnly 特征导出能力。
 */
class GoogLeNet : public priv::IModelImpl
{
public:
    /**
     * @brief 返回模型注册 key。
     * @return 小写模型名称 `googlenet`。
     */
    static constexpr const char *key() noexcept
    {
        return "googlenet";
    }

    /**
     * @brief 构造 GoogLeNet 模型实现。
     */
    explicit GoogLeNet()
        : priv::IModelImpl()
    {
    }

    /**
     * @brief 析构 GoogLeNet 模型实现。
     */
    ~GoogLeNet() override = default;

    /**
     * @brief 获取模型显示名称。
     * @return `GoogLeNet`。
     */
    std::string name() const noexcept override
    {
        return "GoogLeNet";
    }

    /**
     * @brief GoogLeNet 主干支持仅 batch 维动态的 TensorRT profile。
     * @return 始终返回 true。
     */
    bool supportsDynamicBatch() const noexcept override
    {
        return true;
    }

    /**
     * @brief 构建 GoogLeNet 的 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 权重映射表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;
};

} // namespace irt::model
