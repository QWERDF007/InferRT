#include "OpLetterBoxImpl.hpp"

#include "saturate.cuh"

#include <inferrt/core/Exception.hpp>
#include <inferrt/util/CheckError.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace irt::cvcuda::priv {

static const int INTER_RESIZE_COEF_BITS  = 11;
static const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

template<int CH>
__global__ void letter_box_kernel(const uint8_t *src, float *dst, const double2 scale, const int2 ssize,
                                  const int sstride, const int2 dsize, const int dst_N, const int4 valid_rect)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dst_N)
    {
        return;
    }

    int2 dst_coord;
    dst_coord.x = idx % dsize.x;
    dst_coord.y = idx / dsize.x;

    const int dst_idx = dst_coord.y * dsize.x + dst_coord.x;

    if (dst_coord.x < valid_rect.x || dst_coord.x >= valid_rect.z || dst_coord.y < valid_rect.y
        || dst_coord.y >= valid_rect.w)
    {
#pragma unroll
        for (int ch = 0; ch < CH; ++ch)
        {
            dst[dst_idx + ch * dst_N] = 114.0f / 255.0f;
        }
        return;
    }

    int2 temp_coord;
    temp_coord.x = dst_coord.x - valid_rect.x;
    temp_coord.y = dst_coord.y - valid_rect.y;

    float2 src_coord;
    src_coord.x = static_cast<float>((temp_coord.x + 0.5) * scale.x - 0.5);
    src_coord.y = static_cast<float>((temp_coord.y + 0.5) * scale.y - 0.5);

    int2 s;
    s.x = __float2int_rd(src_coord.x);
    s.y = __float2int_rd(src_coord.y);

    float2 f;
    f.x = src_coord.x - s.x;
    f.y = src_coord.y - s.y;

    if (s.x < 0)
    {
        s.x = 0;
        f.x = 0.0f;
    }
    else if (s.x >= ssize.x - 1)
    {
        s.x = ssize.x - 1;
        f.x = 0.0f;
    }

    if (s.y < 0)
    {
        s.y = 0;
        f.y = 0.0f;
    }
    else if (s.y >= ssize.y - 1)
    {
        s.y = ssize.y - 1;
        f.y = 0.0f;
    }

    int2 s1;
    s1.x = min(s.x + 1, ssize.x - 1);
    s1.y = min(s.y + 1, ssize.y - 1);

    short2 alpha;
    alpha.x = saturate_cast<short>((1.0f - f.x) * static_cast<float>(INTER_RESIZE_COEF_SCALE));
    alpha.y = INTER_RESIZE_COEF_SCALE - alpha.x;

    short2 beta;
    beta.x = saturate_cast<short>((1.0f - f.y) * static_cast<float>(INTER_RESIZE_COEF_SCALE));
    beta.y = INTER_RESIZE_COEF_SCALE - beta.x;

    const uint8_t *row0 = src + s.y * sstride;
    const uint8_t *row1 = src + s1.y * sstride;

#pragma unroll
    for (int ch = 0; ch < CH; ++ch)
    {
        int2 hval;
        hval.x = row0[s.x * CH + ch] * alpha.x + row0[s1.x * CH + ch] * alpha.y;
        hval.y = row1[s.x * CH + ch] * alpha.x + row1[s1.x * CH + ch] * alpha.y;

        int2 term;
        term.x           = (beta.x * (hval.x >> 4)) >> 16;
        term.y           = (beta.y * (hval.y >> 4)) >> 16;
        const int result = (term.x + term.y + 2) >> 2;

        const int channel_idx              = (CH == 3 && ch < 3) ? (2 - ch) : ch;
        dst[dst_idx + channel_idx * dst_N] = result / 255.0f;
    }
}

template<int CH>
void launch_letter_box_kernel(const uint8_t *d_src, float *d_dst, const double2 &scale, const int2 &ssize,
                              const int sstride, const int2 &dsize, const int dst_N, const int4 &valid_rect,
                              const int grid_size, const int block_size, cudaStream_t stream)
{
    letter_box_kernel<CH>
        <<<grid_size, block_size, 0, stream>>>(d_src, d_dst, scale, ssize, sstride, dsize, dst_N, valid_rect);
}

void LetterBoxImpl::RunLetterBox(const uint8_t *d_src, float *d_dst, const int2 ssize, const int sstride,
                                 const int2 dsize, const int CH, cudaStream_t stream)
{
    const double r_w   = static_cast<double>(dsize.x) / ssize.x;
    const double r_h   = static_cast<double>(dsize.y) / ssize.y;
    const double ratio = std::min(r_w, r_h);

    int2 resized;
    resized.x = std::max(1, static_cast<int>(std::round(ssize.x * ratio)));
    resized.y = std::max(1, static_cast<int>(std::round(ssize.y * ratio)));
    resized.x = std::min(resized.x, dsize.x);
    resized.y = std::min(resized.y, dsize.y);

    double2 scale;
    scale.x = static_cast<double>(ssize.x) / resized.x;
    scale.y = static_cast<double>(ssize.y) / resized.y;

    int2 pad;
    pad.x = (dsize.x - resized.x) / 2;
    pad.y = (dsize.y - resized.y) / 2;

    int4 valid_rect;
    valid_rect.x = static_cast<int>(std::round(pad.x - 0.1));
    valid_rect.y = static_cast<int>(std::round(pad.y - 0.1));
    valid_rect.z = valid_rect.x + resized.x;
    valid_rect.w = valid_rect.y + resized.y;

    const int dst_N      = dsize.x * dsize.y;
    const int block_size = 256;
    const int grid_size  = (dst_N + block_size - 1) / block_size;

    switch (CH)
    {
    case 1:
        launch_letter_box_kernel<1>(d_src, d_dst, scale, ssize, sstride, dsize, dst_N, valid_rect, grid_size,
                                    block_size, stream);
        break;
    case 3:
        launch_letter_box_kernel<3>(d_src, d_dst, scale, ssize, sstride, dsize, dst_N, valid_rect, grid_size,
                                    block_size, stream);
        break;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Channels must be 1 or 3");
    }

    IRT_CHECK_THROW(cudaPeekAtLastError(), "LetterBox kernel launch failed: ch=%d src=%dx%d dst=%dx%d", CH, ssize.x,
                    ssize.y, dsize.x, dsize.y);
}

} // namespace irt::cvcuda::priv
