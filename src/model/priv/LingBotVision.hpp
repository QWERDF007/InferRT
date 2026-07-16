#pragma once

#include "DINO.hpp"

namespace irt::model {

inline DINOTransformerSpec makeLingBotVisionSpec(const char *display_name, int embed_dim, int depth, int num_heads,
                                                 DINOMlpKind mlp_kind = DINOMlpKind::Mlp,
                                                 int          swiglu_align = 1)
{
    return {display_name, 512, 16, embed_dim, depth, num_heads, 4, 4.0F, DINOVersion::V3, mlp_kind, swiglu_align,
            1e-5F};
}

/**
 * @brief LingBot-Vision ViT-S/16 TensorRT 主干。
 *
 * LingBot-Vision 与 DINOv3 使用相同的 token、RoPE 和 Transformer block
 * 拓扑；具体差异通过 Transformer spec 配置，网络实现复用 DINOTransformer。
 */
class LingBotVisionViTS16 : public DINOTransformer
{
public:
    LingBotVisionViTS16()
        : DINOTransformer(makeLingBotVisionSpec("LingBotVisionViTS16", 384, 12, 6))
    {
    }

    static const char *key() noexcept
    {
        return "lingbot_vision_vits16";
    }
};

/**
 * @brief LingBot-Vision ViT-B/16 TensorRT 主干。
 */
class LingBotVisionViTB16 : public DINOTransformer
{
public:
    LingBotVisionViTB16()
        : DINOTransformer(makeLingBotVisionSpec("LingBotVisionViTB16", 768, 12, 12))
    {
    }

    static const char *key() noexcept
    {
        return "lingbot_vision_vitb16";
    }
};

/**
 * @brief LingBot-Vision ViT-L/16 TensorRT 主干。
 */
class LingBotVisionViTL16 : public DINOTransformer
{
public:
    LingBotVisionViTL16()
        : DINOTransformer(makeLingBotVisionSpec("LingBotVisionViTL16", 1024, 24, 16))
    {
    }

    static const char *key() noexcept
    {
        return "lingbot_vision_vitl16";
    }
};

/**
 * @brief LingBot-Vision ViT-G/16 TensorRT 主干。
 */
class LingBotVisionViTG16 : public DINOTransformer
{
public:
    LingBotVisionViTG16()
        : DINOTransformer(makeLingBotVisionSpec("LingBotVisionViTG16", 1536, 40, 24,
                                                 DINOMlpKind::SplitSwiGLU, 8))
    {
    }

    static const char *key() noexcept
    {
        return "lingbot_vision_vitg16";
    }
};

} // namespace irt::model
