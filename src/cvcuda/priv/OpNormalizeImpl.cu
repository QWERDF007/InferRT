#include "OpNormalizeImpl.hpp"

#include <inferrt/core/Exception.hpp>
#include <inferrt/util/CheckError.hpp>

#include <cstdint>

namespace irt::cvcuda::priv {
namespace {

template<int Channels>
__global__ void normalizeKernel(const uint8_t *src, float *dst, int pixels, float3 mean, float3 inverse_stddev)
{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels)
    {
        return;
    }

    if constexpr (Channels == 1)
    {
        dst[pixel] = (static_cast<float>(src[pixel]) / 255.0F - mean.x) * inverse_stddev.x;
    }
    else
    {
        const int source_offset = pixel * Channels;
        dst[pixel] = (static_cast<float>(src[source_offset]) / 255.0F - mean.x) * inverse_stddev.x;
        dst[pixels + pixel]
            = (static_cast<float>(src[source_offset + 1]) / 255.0F - mean.y) * inverse_stddev.y;
        dst[2 * pixels + pixel]
            = (static_cast<float>(src[source_offset + 2]) / 255.0F - mean.z) * inverse_stddev.z;
    }
}

template<int Channels>
void launchNormalizeKernel(const uint8_t *d_src, float *d_dst, int pixels, float3 mean, float3 inverse_stddev,
                           cudaStream_t stream)
{
    constexpr int block_size = 256;
    const int     grid_size  = (pixels + block_size - 1) / block_size;
    normalizeKernel<Channels><<<grid_size, block_size, 0, stream>>>(d_src, d_dst, pixels, mean, inverse_stddev);
}

} // namespace

void NormalizeImpl::run(const uint8_t *d_src, float *d_dst, const int2 size, const int channels, const float *mean,
                        const float *stddev, cudaStream_t stream)
{
    const float3 mean_values{mean[0], channels == 3 ? mean[1] : 0.0F, channels == 3 ? mean[2] : 0.0F};
    const float3 inverse_stddev{1.0F / stddev[0], channels == 3 ? 1.0F / stddev[1] : 0.0F,
                                channels == 3 ? 1.0F / stddev[2] : 0.0F};
    const int    pixels = size.x * size.y;

    switch (channels)
    {
    case 1:
        launchNormalizeKernel<1>(d_src, d_dst, pixels, mean_values, inverse_stddev, stream);
        break;
    case 3:
        launchNormalizeKernel<3>(d_src, d_dst, pixels, mean_values, inverse_stddev, stream);
        break;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Channel count must be 1 or 3");
    }

    IRT_CHECK_THROW(cudaPeekAtLastError(), "Normalize kernel launch failed: channels=%d size=%dx%d", channels,
                    size.x, size.y);
}

} // namespace irt::cvcuda::priv
