#pragma once

/**
 * @file DinoBackbone.hpp
 * @brief 冻结 DINO 骨干适配器：patch token 网格、valid_area、提取器签名。
 */

#include "DinoTypes.hpp"
#include "DinoViews.hpp"

#include <NvInfer.h>
#include <inferrt/features/DinoRegionSearch.hpp>
#include <inferrt/model/Buffers.hpp>
#include <inferrt/model/IModel.h>

#include <cstdint>
#include <memory>
#include <string>

namespace irt::features::priv {

/** @brief 提取器签名：权重、代码、提取层、前处理与精度的唯一记录。 */
struct DinoExtractorSignature
{
    std::string model_name{};
    std::string weights_path{};
    uintmax_t   weights_size{0};
    int64_t     weights_mtime{0};
    std::string runtime{};
    std::string precision{};
    std::string input_tensor{};
    std::string output_tensor{};
    std::string input_shape{};
    std::string output_shape{};
    std::string extractor_layer{};
    std::string preprocess{};
    std::string revision{};
    int         patch_size{0};
    int         channels{0};
    int         encoder_edge{0};
    int         grid_height{0};
    int         grid_width{0};

    /** @brief 稳定的配置/输出签名，用于进程缓存隔离。 */
    std::string cacheKey() const;
};

/**
 * @brief 冻结骨干适配器。
 *
 * 默认 DINOv3 ViT-S/16，对照 DINOv2 ViT-S/14 register；两者不能共用索引向量。
 * 骨干以 ``eval()`` 语义构建，索引与查询都不做反向传播、PCA 拟合或码本训练。
 */
class DinoBackbone
{
public:
    explicit DinoBackbone(const DinoRegionSearchConfig &config);

    ~DinoBackbone();

    DinoBackbone(const DinoBackbone &)            = delete;
    DinoBackbone &operator=(const DinoBackbone &) = delete;

    const DinoExtractorSignature &signature() const noexcept
    {
        return signature_;
    }

    int patchSize() const noexcept
    {
        return signature_.patch_size;
    }

    int channels() const noexcept
    {
        return signature_.channels;
    }

    int encoderEdge() const noexcept
    {
        return signature_.encoder_edge;
    }

    int gridHeight() const noexcept
    {
        return signature_.grid_height;
    }

    int gridWidth() const noexcept
    {
        return signature_.grid_width;
    }

    size_t maxBatchSize() const noexcept
    {
        return max_batch_size_;
    }

    size_t forwardCount() const noexcept
    {
        return forward_count_;
    }

    /**
     * @brief 对一批视图光栅提取 patch token 网格。
     *
     * 所有视图共享同一输入光栅尺寸与 patch 网格；批量超过骨干上限时自动分批。
     */
    std::vector<DinoFeatureGrid> extract(const std::vector<DinoViewRaster> &rasters);

private:
    std::vector<DinoFeatureGrid> extractChunk(const std::vector<DinoViewRaster> &rasters, size_t begin, size_t count);

    DinoRegionSearchConfig              config_{};
    irt::PreprocessSpec                 preprocess_spec_{};
    std::unique_ptr<irt::model::IModel> model_{};
    DinoExtractorSignature              signature_{};
    std::string                         input_name_{};
    std::string                         output_name_{};
    nvinfer1::Dims                      input_shape_{};
    nvinfer1::Dims                      output_shape_{};
    size_t                              max_batch_size_{1};
    size_t                              input_elements_per_sample_{0};
    size_t                              token_count_{0};
    size_t                              forward_count_{0};
    bool                                use_device_buffers_{false};

    irt::model::DeviceBuffer device_input_{};
    irt::model::DeviceBuffer device_output_{};
    std::vector<float>       host_input_{};
    std::vector<float>       host_output_{};
};

/**
 * @brief 获取当前进程复用的冻结骨干实例。
 *
 * 同一骨干配置共享模型与执行缓冲；配置变化时替换 bundle，调用方持有的旧实例保持有效。
 */
std::shared_ptr<DinoBackbone> dinoAcquireBackbone(const DinoRegionSearchConfig &config);

/** @brief 释放当前进程持有的骨干实例。 */
void dinoResetBackbones();

} // namespace irt::features::priv
