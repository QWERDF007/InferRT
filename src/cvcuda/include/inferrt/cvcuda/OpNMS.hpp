#pragma once

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/Export.h>

#include <cstdint>

namespace irt::cvcuda {

class INFERRT_CVCUDA_API NMS final : public IOperator
{
public:
    explicit NMS();

    ~NMS();

    [[nodiscard]] IRTStatus operator()(const float *d_boxes, const float *d_scores, int64_t *d_keep,
                                       int *d_keep_count, int num_boxes, float iou_threshold,
                                       cudaStream_t stream = nullptr);

    virtual OperatorHandle handle() const noexcept override
    {
        return impl_;
    }

private:
    OperatorHandle impl_;
};

} // namespace irt::cvcuda
