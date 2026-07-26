#pragma once

#include "IOperatorImpl.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace irt::cvcuda::priv {

class NormalizeImpl final : public IOperatorImpl
{
public:
    void operator()(const uint8_t *d_src, float *d_dst, int2 size, int channels, const float *mean,
                    const float *stddev, cudaStream_t stream);

private:
    void run(const uint8_t *d_src, float *d_dst, int2 size, int channels, const float *mean, const float *stddev,
             cudaStream_t stream);
};

} // namespace irt::cvcuda::priv
