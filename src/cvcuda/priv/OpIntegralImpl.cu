#include "OpIntegralImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/util/CheckError.hpp>

#include <cstdint>

namespace irt::cvcuda::priv {

template<typename T, typename CT, int CH>
__global__ void integral_row_prefix_kernel(const T *src, CT *dst, const int2 ssize, const int sstride,
                                           const int dst_w)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ssize.y)
    {
        return;
    }

    CT *out_row = dst + (y + 1) * dst_w * CH;
    const T *in_row = src + y * sstride;

#pragma unroll
    for (int ch = 0; ch < CH; ++ch)
    {
        out_row[ch] = 0;
    }

    for (int x = 0; x < ssize.x; ++x)
    {
#pragma unroll
        for (int ch = 0; ch < CH; ++ch)
        {
            out_row[(x + 1) * CH + ch] = out_row[x * CH + ch] + static_cast<CT>(in_row[x * CH + ch]);
        }
    }
}

template<typename CT, int CH>
__global__ void integral_col_prefix_kernel(CT *dst, const int2 ssize, const int dst_w)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= dst_w)
    {
        return;
    }

#pragma unroll
    for (int ch = 0; ch < CH; ++ch)
    {
        CT sum = 0;
        for (int y = 0; y <= ssize.y; ++y)
        {
            const int idx = (y * dst_w + x) * CH + ch;
            sum += dst[idx];
            dst[idx] = sum;
        }
    }
}

template<typename T, typename CT, int CH>
void launch_integral_kernels(const T *d_src, CT *d_dst, const int2 &ssize, const int sstride, const int dst_w,
                             const int block_size, cudaStream_t stream)
{
    const int row_grid_size = (ssize.y + block_size - 1) / block_size;
    integral_row_prefix_kernel<T, CT, CH><<<row_grid_size, block_size, 0, stream>>>(d_src, d_dst, ssize, sstride,
                                                                                   dst_w);
    IRT_CHECK_THROW(cudaPeekAtLastError(), "Integral row-prefix kernel launch failed: ch=%d src=%dx%d", CH, ssize.x,
                    ssize.y);

    const int col_grid_size = (dst_w + block_size - 1) / block_size;
    integral_col_prefix_kernel<CT, CH><<<col_grid_size, block_size, 0, stream>>>(d_dst, ssize, dst_w);
    IRT_CHECK_THROW(cudaPeekAtLastError(), "Integral col-prefix kernel launch failed: ch=%d src=%dx%d", CH, ssize.x,
                    ssize.y);
}

template<typename T, typename CT>
void IntegralImpl<T, CT>::RunIntegral(const T *d_src, CT *d_dst, const int2 ssize, const int sstride, const int CH,
                                      cudaStream_t stream)
{
    const int dst_w = ssize.x + 1;
    const int block_size = 256;

    IRT_CHECK_THROW(cudaMemsetAsync(d_dst, 0, static_cast<size_t>(dst_w) * CH * sizeof(CT), stream),
                    "Integral top-row initialization failed: ch=%d src=%dx%d", CH, ssize.x, ssize.y);

    switch (CH)
    {
    case 1:
        launch_integral_kernels<T, CT, 1>(d_src, d_dst, ssize, sstride, dst_w, block_size, stream);
        break;
    case 3:
        launch_integral_kernels<T, CT, 3>(d_src, d_dst, ssize, sstride, dst_w, block_size, stream);
        break;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Channels must be 1 or 3");
    }
}

template void IntegralImpl<uint8_t, uint32_t>::RunIntegral(const uint8_t *, uint32_t *, const int2, const int,
                                                           const int, cudaStream_t);
template void IntegralImpl<float, float>::RunIntegral(const float *, float *, const int2, const int, const int,
                                                      cudaStream_t);

} // namespace irt::cvcuda::priv
