#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

/**
 * @brief DINO Transformer 主干版本。
 */
enum class DINOVersion
{
    V2, ///< DINOv2：学习式位置编码，register token 可选。
    V3, ///< DINOv3/LingBot-Vision：RoPE 位置编码，storage token 固定用于主干表征。
};

/**
 * @brief Transformer FFN/MLP 子层类型。
 */
enum class DINOMlpKind
{
    Mlp,          ///< 标准 Linear + GeLU + Linear。
    PackedSwiGLU, ///< DINOv2 Giant 使用的 w12/w3 打包 SwiGLU。
    SplitSwiGLU,  ///< DINOv3 Plus/7B 使用的 w1/w2/w3 拆分 SwiGLU。
};

/**
 * @brief DINO/LingBot-Vision Transformer 结构参数。
 *
 * 该结构只保存推理建网所需的静态超参数。权重命名差异会在构建阶段根据
 * `.wts` 中实际存在的 key 自动适配，便于同时支持官方仓库和 timm 转换后的权重。
 */
struct DINOTransformerSpec
{
    const char *display_name; ///< 模型显示名称。
    int         image_size;   ///< 默认输入图像边长。
    int         patch_size;   ///< Patch embedding 卷积核与步幅。
    int         embed_dim;    ///< Token embedding 维度。
    int         depth;        ///< Transformer block 数量。
    int         num_heads;    ///< 注意力 head 数量。
    int         extra_tokens; ///< register/storage token 数量，不包含 cls token。
    float       mlp_ratio;    ///< FFN 隐藏层相对 embedding 的扩展比例。
    DINOVersion version;      ///< DINO 主干版本。
    DINOMlpKind mlp_kind;     ///< FFN 子层类型。
    int         swiglu_align; ///< SwiGLU 隐藏维度对齐粒度；标准 MLP 设为 1。
    float       norm_epsilon; ///< LayerNorm epsilon。
    bool        exact_gelu{false}; ///< 标准 MLP 是否使用精确 GeLU；默认保持 DINO 兼容的近似实现。
};

/**
 * @brief DINOv2/DINOv3/LingBot-Vision ViT 主干的统一 TensorRT 实现。
 *
 * 主输出为归一化后的 CLS token 特征，形状为 `[N, embed_dim]`。当启用
 * feature-only 构建时，可导出 `tokens`、`blocks.N`、`x_norm_clstoken`、
 * `x_norm_regtokens`/`x_storage_tokens`、`x_norm_patchtokens` 等中间张量。
 */
class DINOTransformer : public priv::IModelImpl
{
public:
    /**
     * @brief 构造 DINO Transformer 主干。
     * @param spec 结构参数。
     */
    explicit DINOTransformer(DINOTransformerSpec spec)
        : spec_(spec)
    {
    }

    /**
     * @brief 获取模型显示名称。
     * @return 例如 `DINOv2ViTB14`。
     */
    std::string name() const noexcept override
    {
        return spec_.display_name;
    }

    /**
     * @brief DINO 手写 TensorRT 网络的第 0 维使用动态 shape 构建。
     * @return 始终支持 TensorRT 动态 batch。
     */
    bool supportsDynamicBatch() const noexcept override
    {
        return true;
    }

    /**
     * @brief 构建 DINO Transformer TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map 从 `.wts` 读取的权重表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;

protected:
    /**
     * @brief 根据模型变体补齐默认输入尺寸。
     * @param config 待规整的模型配置。
     */
    void normalizeModelConfig(IModelConfig &config) const override;

private:
    DINOTransformerSpec spec_; ///< 当前 DINO 变体的结构参数。
};

} // namespace irt::model
