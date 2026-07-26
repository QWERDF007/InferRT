/**
 * @file OpNormalize.h
 * @brief HWC uint8 图像到 NCHW float 张量的逐通道归一化算子。
 */

#pragma once

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>
#include <opencv2/core.hpp>

#include <cstdint>

namespace irt::cvcuda {

/**
 * @brief 将连续的 HWC uint8 图像转换为 NCHW float 并完成归一化。
 *
 * 每个输出元素为 `(src / 255 - mean[channel]) / stddev[channel]`。
 * 输入通道顺序保持不变；例如输入 RGB 时输出平面也为 RGB。`mean` 和
 * `stddev` 是长度至少为 `channels` 的主机内存数组，算子在启动 kernel 时
 * 将其作为值参数传入。
 */
[[nodiscard]] INFERRT_CVCUDA_API IRTStatus normalize(const uint8_t *d_src, float *d_dst, cv::Size size,
                                                      int channels, const float *mean, const float *stddev,
                                                      cudaStream_t stream = nullptr);

} // namespace irt::cvcuda
