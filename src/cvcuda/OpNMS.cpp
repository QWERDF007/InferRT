#include "priv/OpNMSImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpNMS.h>
#include <inferrt/cvcuda/OpNMS.hpp>

namespace irt::cvcuda {

using irt::ProtectCall;

IRTStatus nms(const float *d_boxes, const float *d_scores, int64_t *d_keep, int *d_keep_count, int num_boxes,
              float iou_threshold, cudaStream_t stream)
{
    NMS nms_op;
    return nms_op(d_boxes, d_scores, d_keep, d_keep_count, num_boxes, iou_threshold, stream);
}

NMS::NMS()
{
    impl_ = new priv::NMSImpl();
}

NMS::~NMS()
{
    if (impl_)
    {
        delete impl_;
        impl_ = nullptr;
    }
}

IRTStatus NMS::operator()(const float *d_boxes, const float *d_scores, int64_t *d_keep, int *d_keep_count,
                          int num_boxes, float iou_threshold, cudaStream_t stream)
{
    IRTStatus status = ProtectCall(
        [&]
        {
            if (impl_ == nullptr)
            {
                throw Exception(Status::ERROR_NOT_IMPLEMENTED, "Operator not implemented");
            }

            auto *nmsImpl = static_cast<priv::NMSImpl *>(impl_);
            (*nmsImpl)(d_boxes, d_scores, d_keep, d_keep_count, num_boxes, iou_threshold, stream);
        });
    return status;
}

} // namespace irt::cvcuda
