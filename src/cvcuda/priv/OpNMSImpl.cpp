#include "OpNMSImpl.hpp"

#include <inferrt/core/Exception.hpp>

#include <cmath>

namespace irt::cvcuda::priv {

void NMSImpl::operator()(const float *d_boxes, const float *d_scores, int64_t *d_keep, int *d_keep_count,
                         int num_boxes, float iou_threshold, cudaStream_t stream)
{
    if (num_boxes < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_boxes must be non-negative");
    }
    if (!std::isfinite(iou_threshold))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "iou_threshold must be finite");
    }
    if (d_keep_count == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Keep count pointer is null");
    }
    if (num_boxes > 0 && d_boxes == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Boxes pointer is null");
    }
    if (num_boxes > 0 && d_scores == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Scores pointer is null");
    }
    if (num_boxes > 0 && d_keep == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Keep pointer is null");
    }

    RunNMS(d_boxes, d_scores, d_keep, d_keep_count, num_boxes, iou_threshold, stream);
}

} // namespace irt::cvcuda::priv
