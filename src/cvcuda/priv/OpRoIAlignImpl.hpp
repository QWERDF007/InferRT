#pragma once

#include "IOperatorImpl.hpp"

#include <cuda_runtime.h>

namespace irt::cvcuda::priv {

class RoIAlignImpl final : public IOperatorImpl
{
public:
    explicit RoIAlignImpl()  = default;
    ~RoIAlignImpl() override = default;

    void operator()(const float *d_input, const float *d_rois, float *d_output, int batches, int channels,
                    const int2 input_size, int num_rois, const int2 output_size, float spatial_scale,
                    int sampling_ratio, bool aligned, cudaStream_t stream);

private:
    void RunRoIAlign(const float *d_input, const float *d_rois, float *d_output, int batches, int channels,
                     const int2 input_size, int num_rois, const int2 output_size, float spatial_scale,
                     int sampling_ratio, bool aligned, cudaStream_t stream);
};

} // namespace irt::cvcuda::priv
