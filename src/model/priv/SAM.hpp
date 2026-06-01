#pragma once

#include "IModelImpl.hpp"

#include <set>
#include <string>
#include <vector>

namespace irt::model {

/**
 * @brief SAM 系列来源版本。
 */
enum class SAMFamily
{
    SAM,     ///< Segment Anything v1，ViT image encoder + prompt/mask decoder。
    EdgeSAM, ///< EdgeSAM，RepViT image encoder + SAM prompt/mask heads。
    SAM2,    ///< SAM2，Hiera image encoder + SAM prompt/mask heads。
    SAM3,    ///< SAM3，视觉语言/交互分割体系中的图像分割入口。
};

/**
 * @brief SAM 系列 TensorRT 构建参数。
 */
struct SAMSpec
{
    const char      *display_name;        ///< 模型显示名称。
    int              image_size;          ///< 输入图像边长。
    int              mask_size;           ///< 低分辨率 mask 边长。
    int              max_points;          ///< 固定点提示容量。
    int              multimask_outputs;   ///< 输出候选 mask 数量。
    int              encoder_embed_dim;   ///< SAM v1 ViT token 维度。
    int              encoder_depth;       ///< SAM v1 ViT block 数量。
    int              encoder_num_heads;   ///< SAM v1 ViT 注意力头数量。
    std::set<int>    global_attn_indexes; ///< 使用全局注意力的 block 下标。
    std::vector<int> hiera_stages;        ///< SAM2 Hiera 各 stage 的 block 数量。
    std::vector<int> hiera_window_spec;   ///< SAM2 Hiera 各 stage 的窗口大小。
    std::vector<int> backbone_channels;   ///< SAM2 FPN neck 由低分辨率到高分辨率的输入通道。
    int              pos_embed_size;      ///< SAM2 Hiera 背景位置编码的基准边长。
    int              q_pool;              ///< SAM2 Hiera 执行 q pooling 的 stage 数。
    SAMFamily        family;              ///< SAM 系列版本。
};

/**
 * @brief SAM/EdgeSAM/SAM2/SAM3 的手写 TensorRT 分割模型基类。
 *
 * 当前实现固定采用官方 SAM 常见的五输入契约：
 * `image`、`point_coords`、`point_labels`、`mask_input` 和 `has_mask_input`。
 * 输出契约为 `masks`、`iou_predictions`、`low_res_masks`。SAM v1、EdgeSAM 和 SAM2 使用
 * 手写 TensorRT 子图接入官方 image encoder、prompt encoder 和 mask decoder；
 * SAM3 保留工厂入口，未完成原生主干前会在构建期显式报错。
 */
class SAMSegmentationModel : public priv::IModelImpl
{
public:
    /**
     * @brief 构造 SAM 系列模型。
     * @param spec 模型结构参数。
     */
    explicit SAMSegmentationModel(SAMSpec spec);

    /**
     * @brief 获取显示名称。
     * @return 模型名称。
     */
    std::string name() const noexcept override;

    /**
     * @brief 构建 SAM 系列 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map `.wts` 权重表，SAM v1 需要官方 state_dict 导出的权重。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;

    /**
     * @brief 查询 SAM 手写 TensorRT 网络是否支持动态 batch。
     * @return SAM/SAM2/EdgeSAM 支持；SAM3 原生主干尚未接入，保持不支持。
     */
    bool supportsDynamicBatch() const noexcept override
    {
        return spec_.family != SAMFamily::SAM3;
    }

    /**
     * @brief 根据 SAM 变体补齐默认输入/输出配置。
     * @param config 待规整配置。
     */
    void normalizeModelConfig(IModelConfig &config) const override;

protected:
    SAMSpec spec_; ///< 当前 SAM 变体参数。
};

/**
 * @brief SAM ViT-H 默认别名。
 */
class SAM : public SAMSegmentationModel
{
public:
    SAM();

    static const char *key() noexcept
    {
        return "sam";
    }
};

/**
 * @brief SAM ViT-B 变体。
 */
class SAMViTB : public SAMSegmentationModel
{
public:
    SAMViTB();

    static const char *key() noexcept
    {
        return "sam_vit_b";
    }
};

/**
 * @brief SAM ViT-L 变体。
 */
class SAMViTL : public SAMSegmentationModel
{
public:
    SAMViTL();

    static const char *key() noexcept
    {
        return "sam_vit_l";
    }
};

/**
 * @brief SAM ViT-H 变体。
 */
class SAMViTH : public SAMSegmentationModel
{
public:
    SAMViTH();

    static const char *key() noexcept
    {
        return "sam_vit_h";
    }
};

/**
 * @brief EdgeSAM RepViT-M1 变体。
 */
class EdgeSAM : public SAMSegmentationModel
{
public:
    EdgeSAM();

    static const char *key() noexcept
    {
        return "edge_sam";
    }
};

/**
 * @brief SAM2 Hiera-L 默认别名。
 */
class SAM2 : public SAMSegmentationModel
{
public:
    SAM2();

    static const char *key() noexcept
    {
        return "sam2";
    }
};

/**
 * @brief SAM2 Hiera Tiny 变体。
 */
class SAM2HieraTiny : public SAMSegmentationModel
{
public:
    SAM2HieraTiny();

    static const char *key() noexcept
    {
        return "sam2_hiera_tiny";
    }
};

/**
 * @brief SAM2 Hiera Small 变体。
 */
class SAM2HieraSmall : public SAMSegmentationModel
{
public:
    SAM2HieraSmall();

    static const char *key() noexcept
    {
        return "sam2_hiera_small";
    }
};

/**
 * @brief SAM2 Hiera Base+ 变体。
 */
class SAM2HieraBasePlus : public SAMSegmentationModel
{
public:
    SAM2HieraBasePlus();

    static const char *key() noexcept
    {
        return "sam2_hiera_base_plus";
    }
};

/**
 * @brief SAM2 Hiera Large 变体。
 */
class SAM2HieraLarge : public SAMSegmentationModel
{
public:
    SAM2HieraLarge();

    static const char *key() noexcept
    {
        return "sam2_hiera_large";
    }
};

/**
 * @brief SAM2.1 Hiera Tiny 变体。
 */
class SAM21HieraTiny : public SAMSegmentationModel
{
public:
    SAM21HieraTiny();

    static const char *key() noexcept
    {
        return "sam2_1_hiera_tiny";
    }
};

/**
 * @brief SAM2.1 Hiera Small 变体。
 */
class SAM21HieraSmall : public SAMSegmentationModel
{
public:
    SAM21HieraSmall();

    static const char *key() noexcept
    {
        return "sam2_1_hiera_small";
    }
};

/**
 * @brief SAM2.1 Hiera Base+ 变体。
 */
class SAM21HieraBasePlus : public SAMSegmentationModel
{
public:
    SAM21HieraBasePlus();

    static const char *key() noexcept
    {
        return "sam2_1_hiera_base_plus";
    }
};

/**
 * @brief SAM2.1 Hiera Large 变体。
 */
class SAM21HieraLarge : public SAMSegmentationModel
{
public:
    SAM21HieraLarge();

    static const char *key() noexcept
    {
        return "sam2_1_hiera_large";
    }
};

/**
 * @brief SAM3 图像分割默认入口。
 */
class SAM3 : public SAMSegmentationModel
{
public:
    SAM3();

    static const char *key() noexcept
    {
        return "sam3";
    }
};

/**
 * @brief SAM3 image 模型显式别名。
 */
class SAM3Image : public SAMSegmentationModel
{
public:
    SAM3Image();

    static const char *key() noexcept
    {
        return "sam3_image";
    }
};

} // namespace irt::model
