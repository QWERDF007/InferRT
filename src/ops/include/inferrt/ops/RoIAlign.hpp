#pragma once

#include <inferrt/core/Status.h>
#include <inferrt/ops/Export.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace irt::ops {

/**
 * @brief RoIAlign 特征池化算子。
 */
class INFERRT_OPS_API RoIAlign final
{
public:
    /**
     * @brief 使用显式输出高宽构造 RoIAlign 算子。
     * @param pooled_height 输出特征图高度。
     * @param pooled_width 输出特征图宽度。
     * @param spatial_scale RoI 坐标到输入特征图坐标的缩放系数。
     * @param sampling_ratio 每个输出 bin 的采样点数量，-1 表示自适应采样。
     * @param aligned 是否使用 torchvision aligned 坐标对齐语义。
     */
    RoIAlign(int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned);

    /**
     * @brief 使用输出尺寸二元组构造 RoIAlign 算子。
     * @param output_size 输出特征图尺寸，格式为 {height, width}。
     * @param spatial_scale RoI 坐标到输入特征图坐标的缩放系数。
     * @param sampling_ratio 每个输出 bin 的采样点数量，-1 表示自适应采样。
     * @param aligned 是否使用 torchvision aligned 坐标对齐语义。
     */
    explicit RoIAlign(std::pair<int, int> output_size, float spatial_scale = 1.0f, int sampling_ratio = -1,
                      bool aligned = false);

    /**
     * @brief 获取输出特征图高度。
     * @return 输出特征图高度。
     */
    [[nodiscard]] int pooledHeight() const noexcept;

    /**
     * @brief 获取输出特征图宽度。
     * @return 输出特征图宽度。
     */
    [[nodiscard]] int pooledWidth() const noexcept;

    /**
     * @brief 获取空间缩放系数。
     * @return RoI 坐标到输入特征图坐标的缩放系数。
     */
    [[nodiscard]] float spatialScale() const noexcept;

    /**
     * @brief 获取采样比例。
     * @return 每个输出 bin 的采样点数量，-1 表示自适应采样。
     */
    [[nodiscard]] int samplingRatio() const noexcept;

    /**
     * @brief 查询是否使用 aligned 坐标语义。
     * @return 若使用 aligned 坐标语义则返回 true，否则返回 false。
     */
    [[nodiscard]] bool aligned() const noexcept;

    /**
     * @brief 获取输出特征图尺寸。
     * @return 输出尺寸二元组 {height, width}。
     */
    [[nodiscard]] std::pair<int, int> outputSize() const noexcept;

    /**
     * @brief 执行 RoIAlign 前向计算并写入外部输出缓冲区。
     * @param input 输入特征图，形状为 [N, C, H, W]，按行主序 NCHW 存储。
     * @param input_shape 输入特征图形状数组，顺序为 [N, C, H, W]。
     * @param rois RoI 数组，形状为 [num_rois, 5]，格式为 [batch_index, x1, y1, x2, y2]。
     * @param num_rois RoI 数量。
     * @param output 输出缓冲区，形状为 [num_rois, C, pooled_height, pooled_width]。
     * @return 无。
     */
    void forward(const float *input, const int64_t input_shape[4], const float *rois, int64_t num_rois,
                 float *output) const;

    /**
     * @brief 执行 RoIAlign 前向计算并返回输出向量。
     * @param input 输入特征图，形状为 [N, C, H, W]，按行主序 NCHW 存储。
     * @param input_shape 输入特征图形状数组，顺序为 [N, C, H, W]。
     * @param rois RoI 数组，形状为 [num_rois, 5]，格式为 [batch_index, x1, y1, x2, y2]。
     * @param num_rois RoI 数量。
     * @return 输出特征向量，形状为 [num_rois, C, pooled_height, pooled_width]，按行主序存储。
     */
    [[nodiscard]] std::vector<float> forward(const float *input, const int64_t input_shape[4], const float *rois,
                                             int64_t num_rois) const;

private:
    int   pooled_height_{0};    ///< 输出特征图高度。
    int   pooled_width_{0};     ///< 输出特征图宽度。
    float spatial_scale_{1.0f}; ///< RoI 坐标到输入特征图坐标的缩放系数。
    int   sampling_ratio_{-1};  ///< 每个输出 bin 的采样点数量，-1 表示自适应采样。
    bool  aligned_{false};      ///< 是否使用 aligned 坐标对齐语义。
};

} // namespace irt::ops
