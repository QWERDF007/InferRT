#pragma once

#include "IModelImpl.hpp"

#include <string>
#include <vector>

namespace irt::model {

/**
 * @brief RF-DETR 原生 TensorRT 变体结构描述。
 *
 * 该结构保存官方 RF-DETR 配置中会影响手写 TensorRT 建图的静态参数，包括 DINOv2
 * windowed backbone、projector、two-stage decoder 和可选实例分割头。
 */
struct RFDETRSpec
{
    const char              *display_name;          ///< 模型显示名称。
    int                      resolution;            ///< 官方正方形输入分辨率。
    int                      patch_size;            ///< DINOv2 patch 大小。
    int                      num_windows;           ///< window attention 的窗口划分数。
    int                      encoder_dim;           ///< DINOv2 backbone hidden size。
    int                      encoder_heads;         ///< DINOv2 backbone self-attention head 数。
    int                      hidden_dim;            ///< DETR decoder hidden size。
    int                      decoder_layers;        ///< decoder 层数。
    int                      self_attention_heads;  ///< decoder self-attention head 数。
    int                      cross_attention_heads; ///< MSDeformAttn head 数。
    int                      deform_points;         ///< 每个 feature level 的采样点数。
    int                      num_queries;           ///< query 数。
    int                      num_select;            ///< 默认后处理保留数量。
    std::vector<int>         out_feature_indexes;   ///< DINOv2 输出 stage 索引。
    std::vector<std::string> projector_scales;      ///< P3/P4/P5 projector 输出层级。
    bool                     segmentation;          ///< 是否构建实例分割 mask head。
};

/**
 * @brief RF-DETR 原生 TensorRT 模型实现。
 *
 * 构图覆盖官方导出路径：DINOv2 windowed backbone、MultiScaleProjector、two-stage
 * proposal 初始化、Transformer decoder、检测头以及可选 segmentation head。权重来自
 * `samples/model/python/rfdetr_gen_wts.py` 导出的 `.wts` 文件。
 */
class RFDETRModel : public priv::IModelImpl
{
public:
    /**
     * @brief 构造 RF-DETR 原生 TensorRT 模型。
     * @param spec 变体结构参数。
     */
    explicit RFDETRModel(RFDETRSpec spec)
        : spec_(std::move(spec))
    {
    }

    /**
     * @brief 获取模型显示名称。
     * @return 例如 `RFDETRMedium` 或 `RFDETRSegLarge`。
     */
    std::string name() const noexcept override
    {
        return spec_.display_name;
    }

    /**
     * @brief RF-DETR 建图中所有 batch 相关 reshape 均使用 TensorRT 动态 batch 语义。
     * @return 始终返回 true。
     */
    bool supportsDynamicBatch() const noexcept override
    {
        return true;
    }

    /**
     * @brief 构建 RF-DETR 原生 TensorRT 网络。
     * @param network TensorRT 网络定义。
     * @param weights_map `.wts` 权重表。
     */
    void buildNetwork(nvinfer1::INetworkDefinition *network, const WeightsMap &weights_map) override;

protected:
    /**
     * @brief 按 RF-DETR 官方变体补齐默认输入、类别数和输出张量名。
     * @param config 待规整的模型配置。
     */
    void normalizeModelConfig(IModelConfig &config) const override;

private:
    RFDETRSpec spec_; ///< 当前 RF-DETR 变体结构参数。
};

} // namespace irt::model
