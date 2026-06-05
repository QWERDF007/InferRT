#pragma once

#include "IOperatorImpl.hpp"

#include <cuda_runtime.h>

namespace irt::cvcuda::priv {

template<typename T>
class AdaptiveThresholdImpl final : public IOperatorImpl
{
public:
    explicit AdaptiveThresholdImpl()  = default;
    ~AdaptiveThresholdImpl() override = default;

    void operator()(const T *d_src, T *d_dst, const int2 size, const int sstride, const int dstride, const int CH,
                    const double maxval, const int adaptive_method, const int threshold_type, const int block_size,
                    const double param, const float *d_weights, cudaStream_t stream);

private:
    void RunAdaptiveThreshold(const T *d_src, T *d_dst, const int2 size, const int sstride, const int dstride,
                              const int CH, const double maxval, const int adaptive_method, const int threshold_type,
                              const int block_size, const double param, const float *d_weights, cudaStream_t stream);
};

} // namespace irt::cvcuda::priv
