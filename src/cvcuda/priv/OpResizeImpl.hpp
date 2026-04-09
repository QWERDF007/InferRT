#pragma once

#include "IOperatorImpl.hpp"

#include <cuda_runtime.h>

namespace irt::cvcuda::priv {

template<typename T, typename CT>
void resize_bilinear(const T *d_src, T *d_dst, const int2 ssize, const int sstride, const int2 dsize, const int dstride,
                     const int CH, cudaStream_t stream);

class ResizeImpl final : public IOperatorImpl
{
public:
    explicit ResizeImpl()  = default;
    ~ResizeImpl() override = default;

    template<typename T>
    void operator()(const T *d_src, T *d_dst, const int2 ssize, const int sstride, const int2 dsize, const int dstride,
                    const int CH, const int interpolation, cudaStream_t stream);

private:
    template<typename T>
    void RunResize(const T *d_src, T *d_dst, const int2 ssize, const int sstride, const int2 dsize, const int dstride,
                   const int CH, const int interpolation, cudaStream_t stream);
};

} // namespace irt::cvcuda::priv