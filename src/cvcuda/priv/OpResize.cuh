#pragma once

#include <cuda_runtime.h>

namespace irt::cvcuda {

template<typename T, typename CT>
void resize_bilinear(const T *d_src, T *d_dst, const int2 ssize, const int sstride, const int2 dsize, const int dstride,
                     const int CH, cudaStream_t stream);

} // namespace irt::cvcuda