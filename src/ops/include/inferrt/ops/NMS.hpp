#pragma once

#include <inferrt/ops/Export.h>

#include <cstdint>
#include <vector>

namespace irt::ops {

[[nodiscard]] INFERRT_OPS_API std::vector<int64_t> nms(const float *boxes, const float *scores, int64_t num_boxes,
                                                       float iou_threshold);

} // namespace irt::ops
