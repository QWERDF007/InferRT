#pragma once

#include "IOperatorImpl.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace irt::cvcuda::priv {

class NMSImpl final : public IOperatorImpl
{
public:
    explicit NMSImpl()  = default;
    ~NMSImpl() override = default;

    void operator()(const float *d_boxes, const float *d_scores, int64_t *d_keep, int *d_keep_count, int num_boxes,
                    float iou_threshold, cudaStream_t stream);

private:
    void RunNMS(const float *d_boxes, const float *d_scores, int64_t *d_keep, int *d_keep_count, int num_boxes,
                float iou_threshold, cudaStream_t stream);
};

} // namespace irt::cvcuda::priv
