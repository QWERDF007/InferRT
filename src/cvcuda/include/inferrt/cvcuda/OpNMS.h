#pragma once

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>

#include <cstdint>

namespace irt::cvcuda {

[[nodiscard]] INFERRT_CVCUDA_API IRTStatus nms(const float *d_boxes, const float *d_scores, int64_t *d_keep,
                                               int *d_keep_count, int num_boxes, float iou_threshold,
                                               cudaStream_t stream = nullptr);

} // namespace irt::cvcuda
