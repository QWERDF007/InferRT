/**
 * @file TestRoIAlign.cpp
 * @brief CUDA RoIAlign operator tests.
 */

#include "TestCVCudaCommon.hpp"

#include <gtest/gtest.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/OpRoIAlign.h>
#include <inferrt/cvcuda/OpRoIAlign.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using irt::cvcuda::test::AssertInferRTSuccess;

float bilinearInterpolate(const std::vector<float> &input, int batch, int channel, int batches, int channels,
                          int height, int width, float y, float x)
{
    (void)batches;
    y = std::max(y, 0.0f);
    x = std::max(x, 0.0f);

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
    const auto   at   = [&](int iy, int ix) { return input[base + static_cast<size_t>(iy * width + ix)]; };

    const float v1 = at(y_low, x_low);
    const float v2 = at(y_low, x_high);
    const float v3 = at(y_high, x_low);
    const float v4 = at(y_high, x_high);

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

std::vector<float> makeRoIAlignReference(const std::vector<float> &input, int batches, int channels, int height,
                                         int width, const std::vector<float> &rois, int pooled_height,
                                         int pooled_width, float spatial_scale, int sampling_ratio, bool aligned)
{
    const int          num_rois = static_cast<int>(rois.size() / 5);
    std::vector<float> output(static_cast<size_t>(num_rois) * channels * pooled_height * pooled_width, 0.0f);

    const bool  exact_sampling = sampling_ratio > 0;
    const float offset         = aligned ? 0.5f : 0.0f;

    for (int roi_index = 0; roi_index < num_rois; ++roi_index)
    {
        const float *roi = rois.data() + roi_index * 5;
        const int    roi_batch_ind = static_cast<int>(roi[0]);

        const float roi_start_w = roi[1] * spatial_scale - offset;
        const float roi_start_h = roi[2] * spatial_scale - offset;
        const float roi_end_w   = roi[3] * spatial_scale - offset;
        const float roi_end_h   = roi[4] * spatial_scale - offset;

        float roi_width  = roi_end_w - roi_start_w;
        float roi_height = roi_end_h - roi_start_h;
        if (!aligned)
        {
            roi_width  = std::max(roi_width, 1.0f);
            roi_height = std::max(roi_height, 1.0f);
        }

        const float bin_size_h = roi_height / static_cast<float>(pooled_height);
        const float bin_size_w = roi_width / static_cast<float>(pooled_width);

        const int grid_h = exact_sampling ? sampling_ratio : static_cast<int>(std::ceil(roi_height / pooled_height));
        const int grid_w = exact_sampling ? sampling_ratio : static_cast<int>(std::ceil(roi_width / pooled_width));
        const int count  = std::max(grid_h * grid_w, 1);

        const int loop_grid_h = std::max(grid_h, 0);
        const int loop_grid_w = std::max(grid_w, 0);

        for (int channel = 0; channel < channels; ++channel)
        {
            for (int ph = 0; ph < pooled_height; ++ph)
            {
                for (int pw = 0; pw < pooled_width; ++pw)
                {
                    float sum = 0.0f;
                    for (int iy = 0; iy < loop_grid_h; ++iy)
                    {
                        const float y = roi_start_h + static_cast<float>(ph) * bin_size_h
                                      + (static_cast<float>(iy) + 0.5f) * bin_size_h / static_cast<float>(grid_h);
                        for (int ix = 0; ix < loop_grid_w; ++ix)
                        {
                            const float x = roi_start_w + static_cast<float>(pw) * bin_size_w
                                          + (static_cast<float>(ix) + 0.5f) * bin_size_w / static_cast<float>(grid_w);
                            sum += bilinearInterpolate(input, roi_batch_ind, channel, batches, channels, height, width,
                                                       y, x);
                        }
                    }

                    const size_t out_index = static_cast<size_t>(
                        ((roi_index * channels + channel) * pooled_height + ph) * pooled_width + pw);
                    output[out_index] = sum / static_cast<float>(count);
                }
            }
        }
    }

    return output;
}

template<typename Caller>
void runRoIAlignTest(int batches, int channels, int height, int width, const std::vector<float> &rois,
                     cv::Size output_size, float spatial_scale, int sampling_ratio, bool aligned, Caller caller)
{
    std::vector<float> input(static_cast<size_t>(batches) * channels * height * width);
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = static_cast<float>(i) / 10.0f;
    }

    const std::vector<float> ref = makeRoIAlignReference(input, batches, channels, height, width, rois,
                                                         output_size.height, output_size.width, spatial_scale,
                                                         sampling_ratio, aligned);

    float *d_input  = nullptr;
    float *d_rois   = nullptr;
    float *d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, input.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_rois, rois.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, ref.size() * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_rois, rois.data(), rois.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    const int ret = caller(d_input, d_rois, d_output, batches, channels, cv::Size(width, height),
                           static_cast<int>(rois.size() / 5), output_size, spatial_scale, sampling_ratio, aligned,
                           nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> output(ref.size());
    ASSERT_EQ(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_input);
    cudaFree(d_rois);
    cudaFree(d_output);

    float max_diff = 0.0f;
    for (size_t i = 0; i < output.size(); ++i)
    {
        max_diff = std::max(max_diff, std::abs(output[i] - ref[i]));
    }
    EXPECT_LE(max_diff, 1e-5f);
}

} // namespace

TEST(RoIAlignFunctionTest, MatchesReferenceLegacySampling)
{
    const std::vector<float> rois{
        0.0f, 0.0f, 0.0f, 4.0f, 4.0f,
        1.0f, 1.0f, 0.5f, 5.0f, 4.5f,
        0.0f, -0.25f, 1.0f, 3.25f, 5.0f,
    };

    runRoIAlignTest(2, 3, 5, 6, rois, cv::Size(2, 2), 1.0f, 1, false,
                    [](const float *input, const float *rois, float *output, int batches, int channels,
                       cv::Size input_size, int num_rois, cv::Size output_size, float spatial_scale,
                       int sampling_ratio, bool aligned, cudaStream_t stream)
                    {
                        return irt::cvcuda::roiAlign(input, rois, output, batches, channels, input_size, num_rois,
                                                     output_size, spatial_scale, sampling_ratio, aligned, stream);
                    });
}

TEST(RoIAlignFunctionTest, MatchesReferenceAdaptiveAligned)
{
    const std::vector<float> rois{
        0.0f, 0.5f, 0.5f, 4.5f, 4.5f,
        0.0f, 1.0f, 0.0f, 5.0f, 3.5f,
    };

    runRoIAlignTest(1, 2, 6, 6, rois, cv::Size(3, 2), 1.0f, -1, true,
                    [](const float *input, const float *rois, float *output, int batches, int channels,
                       cv::Size input_size, int num_rois, cv::Size output_size, float spatial_scale,
                       int sampling_ratio, bool aligned, cudaStream_t stream)
                    {
                        return irt::cvcuda::roi_align(input, rois, output, batches, channels, input_size, num_rois,
                                                      output_size, spatial_scale, sampling_ratio, aligned, stream);
                    });
}

TEST(RoIAlignClassTest, MatchesReferenceWithClassWrapper)
{
    const std::vector<float> rois{
        0.0f, 0.0f, 0.0f, 3.0f, 3.0f,
    };

    irt::cvcuda::RoIAlign op(cv::Size(2, 2), 1.0f, 2, true);
    runRoIAlignTest(1, 1, 4, 4, rois, cv::Size(2, 2), 1.0f, 2, true,
                    [&op](const float *input, const float *rois, float *output, int batches, int channels,
                          cv::Size input_size, int num_rois, cv::Size output_size, float spatial_scale,
                          int sampling_ratio, bool aligned, cudaStream_t stream)
                    {
                        (void)output_size;
                        (void)spatial_scale;
                        (void)sampling_ratio;
                        (void)aligned;
                        return op(input, rois, output, batches, channels, input_size, num_rois, stream);
                    });
}

TEST(RoIAlignFunctionEdgeCaseTest, RejectsInvalidOutputSize)
{
    float *d_input  = nullptr;
    float *d_rois   = nullptr;
    float *d_output = nullptr;
    ASSERT_EQ(cudaMalloc(&d_input, 16 * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_rois, 5 * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_output, 4 * sizeof(float)), cudaSuccess);

    const int ret = irt::cvcuda::roiAlign(d_input, d_rois, d_output, 1, 1, cv::Size(4, 4), 1, cv::Size(0, 2), 1.0f,
                                          1, false, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_input);
    cudaFree(d_rois);
    cudaFree(d_output);
}
