#pragma once

#include "IModelImpl.hpp"

namespace irt::model {

/**
 * @brief Vision Transformer 变体的结构参数。
 *
 * 仅覆盖 timm 中标准 ViT Block 的推理参数：patch 大小、embedding 维度、
 * Transformer block 深度、注意力头数量和 MLP 扩展比例。
 */
struct VisionTransformerSpec
{
    const char *display_name; ///< 模型显示名称。
    int         image_size;   ///< 默认输入图像边长。
    int         patch_size;   ///< patch embedding 的卷积核与步幅。
    int         embed_dim;    ///< token embedding 维度。
    int         depth;        ///< Transformer block 数量。
    int         num_heads;    ///< 多头注意力 head 数量。
    float       mlp_ratio;    ///< MLP 隐藏层相对 embedding 维度的扩展比例。
};

/**
 * @brief timm 风格 Vision Transformer 的通用实现。
 *
 * 该基类实现 patch embedding、class token、位置编码、Pre-LN Transformer
 * blocks、最终 LayerNorm 与分类头。派生类只提供不同的结构参数并注册名称。
 */
class VisionTransformer : public priv::IModelImpl
{
public:
    /**
     * @brief 构造 Vision Transformer 基类。
     * @param spec ViT 结构参数。
     */
    explicit VisionTransformer(VisionTransformerSpec spec)
        : spec_(spec)
    {
    }

    /**
     * @brief 获取模型显示名称。
     * @return 例如 `ViTBasePatch16_224`。
     */
    std::string name() const noexcept override
    {
        return spec_.display_name;
    }

    /**
     * @brief 构建 Vision Transformer TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map PyTorch `.wts` 权重表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;

protected:
    /**
     * @brief 根据 ViT 变体名补齐默认输入尺寸。
     * @param config 待规整的模型配置。
     */
    void normalizeModelConfig(IModelConfig &config) const override;

    /**
     * @brief 获取当前变体结构参数。
     * @return ViT 结构参数。
     */
    const VisionTransformerSpec &spec() const noexcept
    {
        return spec_;
    }

private:
    VisionTransformerSpec spec_; ///< 当前 ViT 变体的结构参数。
};

/**
 * @brief ViT-Base/16 的简短别名，便于兼容 tensorrtx/vit 示例命名。
 */
class ViT : public VisionTransformer
{
public:
    /**
     * @brief 构造 ViT-Base/16 兼容别名。
     */
    ViT()
        : VisionTransformer({"ViTBasePatch16_224", 224, 16, 768, 12, 12, 4.0F})
    {
    }

    /**
     * @brief 模型注册 key。
     * @return `vit`。
     */
    static const char *key() noexcept
    {
        return "vit";
    }
};

/**
 * @brief ViT-Tiny/16 224x224 变体。
 */
class ViTTinyPatch16_224 : public VisionTransformer
{
public:
    ViTTinyPatch16_224()
        : VisionTransformer({"ViTTinyPatch16_224", 224, 16, 192, 12, 3, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_tiny_patch16_224";
    }
};

/**
 * @brief ViT-Tiny/16 384x384 变体。
 */
class ViTTinyPatch16_384 : public VisionTransformer
{
public:
    ViTTinyPatch16_384()
        : VisionTransformer({"ViTTinyPatch16_384", 384, 16, 192, 12, 3, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_tiny_patch16_384";
    }
};

/**
 * @brief ViT-Small/32 224x224 变体。
 */
class ViTSmallPatch32_224 : public VisionTransformer
{
public:
    ViTSmallPatch32_224()
        : VisionTransformer({"ViTSmallPatch32_224", 224, 32, 384, 12, 6, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_small_patch32_224";
    }
};

/**
 * @brief ViT-Small/32 384x384 变体。
 */
class ViTSmallPatch32_384 : public VisionTransformer
{
public:
    ViTSmallPatch32_384()
        : VisionTransformer({"ViTSmallPatch32_384", 384, 32, 384, 12, 6, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_small_patch32_384";
    }
};

/**
 * @brief ViT-Small/16 224x224 变体。
 */
class ViTSmallPatch16_224 : public VisionTransformer
{
public:
    ViTSmallPatch16_224()
        : VisionTransformer({"ViTSmallPatch16_224", 224, 16, 384, 12, 6, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_small_patch16_224";
    }
};

/**
 * @brief ViT-Small/16 384x384 变体。
 */
class ViTSmallPatch16_384 : public VisionTransformer
{
public:
    ViTSmallPatch16_384()
        : VisionTransformer({"ViTSmallPatch16_384", 384, 16, 384, 12, 6, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_small_patch16_384";
    }
};

/**
 * @brief ViT-Small/8 224x224 变体。
 */
class ViTSmallPatch8_224 : public VisionTransformer
{
public:
    ViTSmallPatch8_224()
        : VisionTransformer({"ViTSmallPatch8_224", 224, 8, 384, 12, 6, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_small_patch8_224";
    }
};

/**
 * @brief ViT-Base/32 224x224 变体。
 */
class ViTBasePatch32_224 : public VisionTransformer
{
public:
    ViTBasePatch32_224()
        : VisionTransformer({"ViTBasePatch32_224", 224, 32, 768, 12, 12, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_base_patch32_224";
    }
};

/**
 * @brief ViT-Base/32 384x384 变体。
 */
class ViTBasePatch32_384 : public VisionTransformer
{
public:
    ViTBasePatch32_384()
        : VisionTransformer({"ViTBasePatch32_384", 384, 32, 768, 12, 12, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_base_patch32_384";
    }
};

/**
 * @brief ViT-Base/16 224x224 变体。
 */
class ViTBasePatch16_224 : public VisionTransformer
{
public:
    ViTBasePatch16_224()
        : VisionTransformer({"ViTBasePatch16_224", 224, 16, 768, 12, 12, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_base_patch16_224";
    }
};

/**
 * @brief ViT-Base/16 384x384 变体。
 */
class ViTBasePatch16_384 : public VisionTransformer
{
public:
    ViTBasePatch16_384()
        : VisionTransformer({"ViTBasePatch16_384", 384, 16, 768, 12, 12, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_base_patch16_384";
    }
};

/**
 * @brief ViT-Base/8 224x224 变体。
 */
class ViTBasePatch8_224 : public VisionTransformer
{
public:
    ViTBasePatch8_224()
        : VisionTransformer({"ViTBasePatch8_224", 224, 8, 768, 12, 12, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_base_patch8_224";
    }
};

/**
 * @brief ViT-Large/32 224x224 变体。
 */
class ViTLargePatch32_224 : public VisionTransformer
{
public:
    ViTLargePatch32_224()
        : VisionTransformer({"ViTLargePatch32_224", 224, 32, 1024, 24, 16, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_large_patch32_224";
    }
};

/**
 * @brief ViT-Large/32 384x384 变体。
 */
class ViTLargePatch32_384 : public VisionTransformer
{
public:
    ViTLargePatch32_384()
        : VisionTransformer({"ViTLargePatch32_384", 384, 32, 1024, 24, 16, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_large_patch32_384";
    }
};

/**
 * @brief ViT-Large/16 224x224 变体。
 */
class ViTLargePatch16_224 : public VisionTransformer
{
public:
    ViTLargePatch16_224()
        : VisionTransformer({"ViTLargePatch16_224", 224, 16, 1024, 24, 16, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_large_patch16_224";
    }
};

/**
 * @brief ViT-Large/16 384x384 变体。
 */
class ViTLargePatch16_384 : public VisionTransformer
{
public:
    ViTLargePatch16_384()
        : VisionTransformer({"ViTLargePatch16_384", 384, 16, 1024, 24, 16, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_large_patch16_384";
    }
};

/**
 * @brief ViT-Large/14 224x224 变体。
 */
class ViTLargePatch14_224 : public VisionTransformer
{
public:
    ViTLargePatch14_224()
        : VisionTransformer({"ViTLargePatch14_224", 224, 14, 1024, 24, 16, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_large_patch14_224";
    }
};

/**
 * @brief ViT-Huge/14 224x224 变体。
 */
class ViTHugePatch14_224 : public VisionTransformer
{
public:
    ViTHugePatch14_224()
        : VisionTransformer({"ViTHugePatch14_224", 224, 14, 1280, 32, 16, 4.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_huge_patch14_224";
    }
};

/**
 * @brief ViT-Giant/14 224x224 变体。
 */
class ViTGiantPatch14_224 : public VisionTransformer
{
public:
    ViTGiantPatch14_224()
        : VisionTransformer({"ViTGiantPatch14_224", 224, 14, 1408, 40, 16, 48.0F / 11.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_giant_patch14_224";
    }
};

/**
 * @brief ViT-Gigantic/14 224x224 变体。
 */
class ViTGiganticPatch14_224 : public VisionTransformer
{
public:
    ViTGiganticPatch14_224()
        : VisionTransformer({"ViTGiganticPatch14_224", 224, 14, 1664, 48, 16, 64.0F / 13.0F})
    {
    }

    static const char *key() noexcept
    {
        return "vit_gigantic_patch14_224";
    }
};

} // namespace irt::model
