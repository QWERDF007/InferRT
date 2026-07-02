#pragma once

#include <inferrt/ops/Export.h>

#include <cstdint>
#include <vector>

namespace irt::ops {

/**
 * @brief 对检测框执行非极大值抑制。
 * @param boxes 输入检测框，形状为 [num_boxes, 4]，格式为 xyxy，按行主序存储。
 * @param scores 每个检测框的置信度分数，形状为 [num_boxes]。
 * @param num_boxes 检测框数量。
 * @param iou_threshold 抑制重叠框的 IoU 阈值。
 * @return 保留的检测框索引，按分数从高到低排列。
 */
[[nodiscard]] INFERRT_OPS_API std::vector<int64_t> nms(const float *boxes, const float *scores, int64_t num_boxes,
                                                       float iou_threshold);

} // namespace irt::ops
