#pragma once

#include "IOperatorImpl.hpp"

namespace irt::cvcuda::priv {

class ResizeImpl final : public IOperatorImpl
{
public:
    explicit ResizeImpl() {}

    template<typename T>
    [[nodiscard]] int operator()(const T *d_src, T *d_dst, const int2 ssize, const int sstride, const int2 dsize,
                                 const int dstride, const int CH, const int interpolation, cudaStream_t stream);

private:
    template<typename T>
    void RunResize(const T *d_src, T *d_dst, const int2 ssize, const int sstride, const int2 dsize, const int dstride,
                   const int CH, const int interpolation, cudaStream_t stream);
};

} // namespace irt::cvcuda::priv