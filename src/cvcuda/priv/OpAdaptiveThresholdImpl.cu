#include "OpAdaptiveThresholdImpl.hpp"

#include "saturate.cuh"

#include <inferrt/core/Exception.hpp>
#include <inferrt/cvcuda/OpAdaptiveThreshold.h>
#include <inferrt/util/CheckError.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstdint>

namespace irt::cvcuda::priv {

__device__ __forceinline__ int border_replicate(int p, int len)
{
    return min(max(p, 0), len - 1);
}

template<typename T>
__device__ __forceinline__ T adaptive_max_value(double maxval)
{
    return saturate_cast<T>(maxval);
}

template<typename T>
__device__ __forceinline__ T binary_threshold_value(T src, T mean, T maxval, int delta, bool inverse)
{
    const bool foreground = static_cast<int>(src) > static_cast<int>(mean) - delta;
    return (foreground != inverse) ? maxval : T{0};
}

template<typename T, typename CT, int CH, bool INVERSE>
__global__ void adaptive_threshold_mean_kernel(const T *src, T *dst, const double maxval, const int delta,
                                               const int block_size, const int2 size, const int sstride,
                                               const int dstride, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
    {
        return;
    }

    const int x      = idx % size.x;
    const int y      = idx / size.x;
    const int radius = block_size / 2;
    const CT  count  = static_cast<CT>(block_size * block_size);

    const int src_base  = y * sstride + x * CH;
    const int dst_base  = y * dstride + x * CH;
    const T   max_value = adaptive_max_value<T>(maxval);

#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        CT sum = 0;
        for (int ky = -radius; ky <= radius; ++ky)
        {
            const int yy = border_replicate(y + ky, size.y);
            for (int kx = -radius; kx <= radius; ++kx)
            {
                const int xx = border_replicate(x + kx, size.x);
                sum += src[yy * sstride + xx * CH + c];
            }
        }

        const T mean      = saturate_cast<T>(sum / count);
        dst[dst_base + c] = binary_threshold_value(src[src_base + c], mean, max_value, delta, INVERSE);
    }
}

template<typename T, typename CT, typename WT, int CH, bool INVERSE>
__global__ void adaptive_threshold_gaussian_kernel(const T *src, T *dst, const WT *weights, const double maxval,
                                                   const int delta, const int block_size, const int2 size,
                                                   const int sstride, const int dstride, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
    {
        return;
    }

    const int x      = idx % size.x;
    const int y      = idx / size.x;
    const int radius = block_size / 2;

    const int src_base  = y * sstride + x * CH;
    const int dst_base  = y * dstride + x * CH;
    const T   max_value = adaptive_max_value<T>(maxval);

#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        CT sum = 0;
        for (int ky = -radius; ky <= radius; ++ky)
        {
            const int yy = border_replicate(y + ky, size.y);
            for (int kx = -radius; kx <= radius; ++kx)
            {
                const int xx = border_replicate(x + kx, size.x);
                const int wy = ky + radius;
                const int wx = kx + radius;
                sum += weights[wy * block_size + wx] * src[yy * sstride + xx * CH + c];
            }
        }

        const T mean      = saturate_cast<T>(sum);
        dst[dst_base + c] = binary_threshold_value(src[src_base + c], mean, max_value, delta, INVERSE);
    }
}

template<typename T, typename CT, int CH, bool INVERSE>
__global__ void adaptive_threshold_percentage_kernel(const T *src, T *dst, const double maxval, const double percentage,
                                                     const int block_size, const int2 size, const int sstride,
                                                     const int dstride, const int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
    {
        return;
    }

    const int x      = idx % size.x;
    const int y      = idx / size.x;
    const int radius = block_size / 2;

    const int src_base  = y * sstride + x * CH;
    const int dst_base  = y * dstride + x * CH;
    const T   max_value = adaptive_max_value<T>(maxval);

    const int y1    = max(y - radius, 0);
    const int y2    = min(y + radius, size.y - 1);
    const int x1    = max(x - radius, 0);
    const int x2    = min(x + radius, size.x - 1);
    const int count = (x2 - x1 + 1) * (y2 - y1 + 1);

#pragma unroll
    for (int c = 0; c < CH; ++c)
    {
        CT sum = 0;
        for (int yy = y1; yy <= y2; ++yy)
        {
            for (int xx = x1; xx <= x2; ++xx)
            {
                sum += src[yy * sstride + xx * CH + c];
            }
        }

        const bool below  = static_cast<double>(src[src_base + c]) * count < static_cast<double>(sum) * percentage;
        dst[dst_base + c] = (below == INVERSE) ? max_value : T{0};
    }
}

template<typename T, typename CT, int CH>
void launch_mean_kernel(const T *d_src, T *d_dst, const double maxval, const int delta, const int block_size,
                        const int2 &size, const int sstride, const int dstride, const int N, const int grid_size,
                        const int cuda_block_size, const bool inverse, cudaStream_t stream)
{
    if (inverse)
    {
        adaptive_threshold_mean_kernel<T, CT, CH, true><<<grid_size, cuda_block_size, 0, stream>>>(
            d_src, d_dst, maxval, delta, block_size, size, sstride, dstride, N);
    }
    else
    {
        adaptive_threshold_mean_kernel<T, CT, CH, false><<<grid_size, cuda_block_size, 0, stream>>>(
            d_src, d_dst, maxval, delta, block_size, size, sstride, dstride, N);
    }
}

template<typename T, typename CT, typename WT, int CH>
void launch_gaussian_kernel(const T *d_src, T *d_dst, const WT *d_weights, const double maxval, const int delta,
                            const int block_size, const int2 &size, const int sstride, const int dstride, const int N,
                            const int grid_size, const int cuda_block_size, const bool inverse, cudaStream_t stream)
{
    if (inverse)
    {
        adaptive_threshold_gaussian_kernel<T, CT, WT, CH, true><<<grid_size, cuda_block_size, 0, stream>>>(
            d_src, d_dst, d_weights, maxval, delta, block_size, size, sstride, dstride, N);
    }
    else
    {
        adaptive_threshold_gaussian_kernel<T, CT, WT, CH, false><<<grid_size, cuda_block_size, 0, stream>>>(
            d_src, d_dst, d_weights, maxval, delta, block_size, size, sstride, dstride, N);
    }
}

template<typename T, typename CT, int CH>
void launch_percentage_kernel(const T *d_src, T *d_dst, const double maxval, const double percentage,
                              const int block_size, const int2 &size, const int sstride, const int dstride, const int N,
                              const int grid_size, const int cuda_block_size, const bool inverse, cudaStream_t stream)
{
    if (inverse)
    {
        adaptive_threshold_percentage_kernel<T, CT, CH, true><<<grid_size, cuda_block_size, 0, stream>>>(
            d_src, d_dst, maxval, percentage, block_size, size, sstride, dstride, N);
    }
    else
    {
        adaptive_threshold_percentage_kernel<T, CT, CH, false><<<grid_size, cuda_block_size, 0, stream>>>(
            d_src, d_dst, maxval, percentage, block_size, size, sstride, dstride, N);
    }
}

template<typename T>
void AdaptiveThresholdImpl<T>::RunAdaptiveThreshold(const T *d_src, T *d_dst, const int2 size, const int sstride,
                                                    const int dstride, const int CH, const double maxval,
                                                    const int adaptive_method, const int threshold_type,
                                                    const int block_size, const double param, const float *d_weights,
                                                    cudaStream_t stream)
{
    const int  N               = size.x * size.y;
    const int  cuda_block_size = 256;
    const int  grid_size       = (N + cuda_block_size - 1) / cuda_block_size;
    const bool inverse         = threshold_type == cv::THRESH_BINARY_INV;
    const int  delta           = inverse ? static_cast<int>(std::floor(param)) : static_cast<int>(std::ceil(param));

    if (adaptive_method == cv::ADAPTIVE_THRESH_MEAN_C)
    {
        switch (CH)
        {
        case 1:
            launch_mean_kernel<T, float, 1>(d_src, d_dst, maxval, delta, block_size, size, sstride, dstride, N,
                                            grid_size, cuda_block_size, inverse, stream);
            break;
        case 3:
            launch_mean_kernel<T, float, 3>(d_src, d_dst, maxval, delta, block_size, size, sstride, dstride, N,
                                            grid_size, cuda_block_size, inverse, stream);
            break;
        default:
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Channels must be 1 or 3");
        }
    }
    else if (adaptive_method == cv::ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        switch (CH)
        {
        case 1:
            launch_gaussian_kernel<T, float, float, 1>(d_src, d_dst, d_weights, maxval, delta, block_size, size,
                                                       sstride, dstride, N, grid_size, cuda_block_size, inverse,
                                                       stream);
            break;
        case 3:
            launch_gaussian_kernel<T, float, float, 3>(d_src, d_dst, d_weights, maxval, delta, block_size, size,
                                                       sstride, dstride, N, grid_size, cuda_block_size, inverse,
                                                       stream);
            break;
        default:
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Channels must be 1 or 3");
        }
    }
    else if (adaptive_method == irt::cvcuda::ADAPTIVE_THRESH_PERCENTAGE)
    {
        switch (CH)
        {
        case 1:
            launch_percentage_kernel<T, uint32_t, 1>(d_src, d_dst, maxval, param, block_size, size, sstride, dstride, N,
                                                     grid_size, cuda_block_size, inverse, stream);
            break;
        case 3:
            launch_percentage_kernel<T, uint32_t, 3>(d_src, d_dst, maxval, param, block_size, size, sstride, dstride, N,
                                                     grid_size, cuda_block_size, inverse, stream);
            break;
        default:
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Channels must be 1 or 3");
        }
    }
    else
    {
        throw Exception(Status::ERROR_NOT_IMPLEMENTED, "Adaptive threshold method not implemented: %d",
                        adaptive_method);
    }

    IRT_CHECK_THROW(cudaPeekAtLastError(),
                    "AdaptiveThreshold kernel launch failed: method=%d type=%d ch=%d size=%dx%d block=%d",
                    adaptive_method, threshold_type, CH, size.x, size.y, block_size);
}

template void AdaptiveThresholdImpl<uint8_t>::RunAdaptiveThreshold(const uint8_t *, uint8_t *, const int2, const int,
                                                                   const int, const int, const double, const int,
                                                                   const int, const int, const double, const float *,
                                                                   cudaStream_t);

} // namespace irt::cvcuda::priv
