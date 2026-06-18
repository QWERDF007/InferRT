#include "OpRoIAlignImpl.hpp"

#include <inferrt/util/CheckError.hpp>

#include <cmath>
#include <cstddef>

namespace irt::cvcuda::priv {

__device__ float roi_align_bilinear_interpolate(const float *input, int batch, int channel, int channels, int height,
                                                int width, float y, float x)
{
    y = fmaxf(y, 0.0f);
    x = fmaxf(x, 0.0f);

    int y_low = static_cast<int>(y);
    int x_low = static_cast<int>(x);

    int y_high = 0;
    if (y_low >= height - 1)
    {
        y_low  = height - 1;
        y_high = height - 1;
    }
    else
    {
        y_high = y_low + 1;
    }

    int x_high = 0;
    if (x_low >= width - 1)
    {
        x_low  = width - 1;
        x_high = width - 1;
    }
    else
    {
        x_high = x_low + 1;
    }

    const float ly = y - static_cast<float>(y_low);
    const float lx = x - static_cast<float>(x_low);
    const float hy = 1.0f - ly;
    const float hx = 1.0f - lx;

    const size_t base = static_cast<size_t>((batch * channels + channel) * height * width);
    const float  v1   = input[base + static_cast<size_t>(y_low * width + x_low)];
    const float  v2   = input[base + static_cast<size_t>(y_low * width + x_high)];
    const float  v3   = input[base + static_cast<size_t>(y_high * width + x_low)];
    const float  v4   = input[base + static_cast<size_t>(y_high * width + x_high)];

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

__global__ void roi_align_kernel(const float *input, const float *rois, float *output, int batches, int channels,
                                 int height, int width, int num_rois, int pooled_height, int pooled_width,
                                 float spatial_scale, int sampling_ratio, bool aligned, size_t output_count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_count)
    {
        return;
    }

    const int pw        = static_cast<int>(index % pooled_width);
    const int ph        = static_cast<int>((index / pooled_width) % pooled_height);
    const int channel   = static_cast<int>((index / pooled_width / pooled_height) % channels);
    const int roi_index = static_cast<int>(index / pooled_width / pooled_height / channels);

    const float *roi = rois + roi_index * 5;
    const int    roi_batch_ind = static_cast<int>(roi[0]);
    if (roi_batch_ind < 0 || roi_batch_ind >= batches)
    {
        output[index] = 0.0f;
        return;
    }

    const float offset      = aligned ? 0.5f : 0.0f;
    const float roi_start_w = roi[1] * spatial_scale - offset;
    const float roi_start_h = roi[2] * spatial_scale - offset;
    const float roi_end_w   = roi[3] * spatial_scale - offset;
    const float roi_end_h   = roi[4] * spatial_scale - offset;

    float roi_width  = roi_end_w - roi_start_w;
    float roi_height = roi_end_h - roi_start_h;
    if (!aligned)
    {
        roi_width  = fmaxf(roi_width, 1.0f);
        roi_height = fmaxf(roi_height, 1.0f);
    }

    const float bin_size_h = roi_height / static_cast<float>(pooled_height);
    const float bin_size_w = roi_width / static_cast<float>(pooled_width);

    const bool exact_sampling = sampling_ratio > 0;
    const int  grid_h = exact_sampling ? sampling_ratio : static_cast<int>(ceilf(roi_height / pooled_height));
    const int  grid_w = exact_sampling ? sampling_ratio : static_cast<int>(ceilf(roi_width / pooled_width));
    const int  grid_count = grid_h * grid_w;
    const int  count      = grid_count > 1 ? grid_count : 1;

    const int loop_grid_h = grid_h > 0 ? grid_h : 0;
    const int loop_grid_w = grid_w > 0 ? grid_w : 0;

    float sum = 0.0f;
    for (int iy = 0; iy < loop_grid_h; ++iy)
    {
        const float y = roi_start_h + static_cast<float>(ph) * bin_size_h
                      + (static_cast<float>(iy) + 0.5f) * bin_size_h / static_cast<float>(grid_h);
        for (int ix = 0; ix < loop_grid_w; ++ix)
        {
            const float x = roi_start_w + static_cast<float>(pw) * bin_size_w
                          + (static_cast<float>(ix) + 0.5f) * bin_size_w / static_cast<float>(grid_w);
            sum += roi_align_bilinear_interpolate(input, roi_batch_ind, channel, channels, height, width, y, x);
        }
    }

    output[index] = sum / static_cast<float>(count);
}

void RoIAlignImpl::RunRoIAlign(const float *d_input, const float *d_rois, float *d_output, int batches, int channels,
                               const int2 input_size, int num_rois, const int2 output_size, float spatial_scale,
                               int sampling_ratio, bool aligned, cudaStream_t stream)
{
    if (num_rois == 0)
    {
        return;
    }

    const size_t output_count = static_cast<size_t>(num_rois) * channels * output_size.y * output_size.x;
    const int    block_size   = 256;
    const int    grid_size    = static_cast<int>((output_count + block_size - 1) / block_size);

    roi_align_kernel<<<grid_size, block_size, 0, stream>>>(d_input, d_rois, d_output, batches, channels, input_size.y,
                                                           input_size.x, num_rois, output_size.y, output_size.x,
                                                           spatial_scale, sampling_ratio, aligned, output_count);
    IRT_CHECK_THROW(cudaPeekAtLastError(), "RoIAlign kernel launch failed: input=%dx%d rois=%d output=%dx%d",
                    input_size.x, input_size.y, num_rois, output_size.x, output_size.y);
}

} // namespace irt::cvcuda::priv
