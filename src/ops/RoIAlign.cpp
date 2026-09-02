#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Tensor.hpp>
#include <inferrt/ops/RoIAlign.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace irt::ops {
namespace {

void validateConstructorArgs(int pooled_height, int pooled_width, float spatial_scale)
{
    if (pooled_height <= 0 || pooled_width <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "RoIAlign output_size must be positive, got %dx%d",
                        pooled_height, pooled_width);
    }
    if (!std::isfinite(spatial_scale))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "RoIAlign spatial_scale must be finite");
    }
}

void validateInputShape(const int64_t input_shape[4])
{
    if (input_shape == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "input_shape must not be null");
    }
    for (int i = 0; i < 4; ++i)
    {
        if (input_shape[i] <= 0)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "input_shape[%d] must be positive, got %lld", i,
                            static_cast<long long>(input_shape[i]));
        }
    }
}

size_t outputElementCount(int64_t num_rois, int64_t channels, int pooled_height, int pooled_width)
{
    size_t count = irt::checkedInt64ToSize(num_rois, "RoIAlign ROI count");
    count = irt::checkedSizeMul(count, irt::checkedInt64ToSize(channels, "RoIAlign channel count"),
                                "RoIAlign output");
    count = irt::checkedSizeMul(count, static_cast<size_t>(pooled_height), "RoIAlign output");
    count = irt::checkedSizeMul(count, static_cast<size_t>(pooled_width), "RoIAlign output");
    return count;
}

float bilinearInterpolate(const float *input, int64_t batch, int64_t channel, int64_t channels, int64_t height,
                          int64_t width, float y, float x)
{
    y = std::max(y, 0.0f);
    x = std::max(x, 0.0f);

    int64_t y_low = static_cast<int64_t>(y);
    int64_t x_low = static_cast<int64_t>(x);

    int64_t y_high = 0;
    if (y_low >= height - 1)
    {
        y_low  = height - 1;
        y_high = height - 1;
    }
    else
    {
        y_high = y_low + 1;
    }

    int64_t x_high = 0;
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

    const size_t batch_offset = irt::checkedSizeMul(irt::checkedInt64ToSize(batch, "RoIAlign batch index"),
                                                    irt::checkedInt64ToSize(channels, "RoIAlign channels"),
                                                    "RoIAlign input batch offset");
    const size_t channel_offset = irt::checkedSizeAdd(batch_offset,
                                                      irt::checkedInt64ToSize(channel, "RoIAlign channel index"),
                                                      "RoIAlign input channel offset");
    const size_t spatial = irt::checkedSizeMul(irt::checkedInt64ToSize(height, "RoIAlign height"),
                                               irt::checkedInt64ToSize(width, "RoIAlign width"),
                                               "RoIAlign input plane");
    const size_t base = irt::checkedSizeMul(channel_offset, spatial, "RoIAlign input offset");
    const auto   at   = [&](int64_t iy, int64_t ix) -> float
    {
        const size_t row = irt::checkedSizeMul(irt::checkedInt64ToSize(iy, "RoIAlign y index"),
                                               irt::checkedInt64ToSize(width, "RoIAlign width"),
                                               "RoIAlign row offset");
        const size_t offset = irt::checkedSizeAdd(row, irt::checkedInt64ToSize(ix, "RoIAlign x index"),
                                                  "RoIAlign pixel offset");
        return input[irt::checkedSizeAdd(base, offset, "RoIAlign input index")];
    };

    const float v1 = at(y_low, x_low);
    const float v2 = at(y_low, x_high);
    const float v3 = at(y_high, x_low);
    const float v4 = at(y_high, x_high);

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

int adaptiveGridSize(float roi_size, int pooled_size)
{
    const double grid = std::ceil(static_cast<double>(roi_size) / static_cast<double>(pooled_size));
    if (!std::isfinite(grid) || grid > static_cast<double>(std::numeric_limits<int>::max()))
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "RoIAlign sampling grid exceeds int range");
    }
    return static_cast<int>(grid);
}

} // namespace

RoIAlign::RoIAlign(int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned)
    : pooled_height_(pooled_height)
    , pooled_width_(pooled_width)
    , spatial_scale_(spatial_scale)
    , sampling_ratio_(sampling_ratio)
    , aligned_(aligned)
{
    validateConstructorArgs(pooled_height_, pooled_width_, spatial_scale_);
}

RoIAlign::RoIAlign(std::pair<int, int> output_size, float spatial_scale, int sampling_ratio, bool aligned)
    : RoIAlign(output_size.first, output_size.second, spatial_scale, sampling_ratio, aligned)
{
}

int RoIAlign::pooledHeight() const noexcept
{
    return pooled_height_;
}

int RoIAlign::pooledWidth() const noexcept
{
    return pooled_width_;
}

float RoIAlign::spatialScale() const noexcept
{
    return spatial_scale_;
}

int RoIAlign::samplingRatio() const noexcept
{
    return sampling_ratio_;
}

bool RoIAlign::aligned() const noexcept
{
    return aligned_;
}

std::pair<int, int> RoIAlign::outputSize() const noexcept
{
    return {pooled_height_, pooled_width_};
}

void RoIAlign::forward(const float *input, const int64_t input_shape[4], const float *rois, int64_t num_rois,
                       float *output) const
{
    validateInputShape(input_shape);
    if (num_rois < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_rois must be non-negative, got %lld",
                        static_cast<long long>(num_rois));
    }

    const int64_t batches  = input_shape[0];
    const int64_t channels = input_shape[1];
    const int64_t height   = input_shape[2];
    const int64_t width    = input_shape[3];

    const size_t output_count = outputElementCount(num_rois, channels, pooled_height_, pooled_width_);
    if (output_count > 0 && input == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "input must not be null");
    }
    if (num_rois > 0 && rois == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "rois must not be null");
    }
    if (output_count > 0 && output == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "output must not be null");
    }
    if (output_count == 0)
    {
        return;
    }

    const bool  exact_sampling = sampling_ratio_ > 0;
    const float offset         = aligned_ ? 0.5f : 0.0f;

    for (int64_t roi_index = 0; roi_index < num_rois; ++roi_index)
    {
        const float *roi = rois
                         + irt::checkedSizeMul(irt::checkedInt64ToSize(roi_index, "RoIAlign ROI index"), 5U,
                                               "RoIAlign ROI offset");
        for (int i = 0; i < 5; ++i)
        {
            if (!std::isfinite(roi[i]))
            {
                throw Exception(Status::ERROR_INVALID_ARGUMENT, "rois[%lld, %d] must be finite",
                                static_cast<long long>(roi_index), i);
            }
        }

        const int64_t roi_batch_ind = static_cast<int64_t>(roi[0]);
        if (roi_batch_ind < 0 || roi_batch_ind >= batches)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT,
                            "ROI batch index out of range: roi=%lld batch=%lld valid=[0,%lld)",
                            static_cast<long long>(roi_index), static_cast<long long>(roi_batch_ind),
                            static_cast<long long>(batches));
        }

        const float roi_start_w = roi[1] * spatial_scale_ - offset;
        const float roi_start_h = roi[2] * spatial_scale_ - offset;
        const float roi_end_w   = roi[3] * spatial_scale_ - offset;
        const float roi_end_h   = roi[4] * spatial_scale_ - offset;

        float roi_width  = roi_end_w - roi_start_w;
        float roi_height = roi_end_h - roi_start_h;
        if (!aligned_)
        {
            roi_width  = std::max(roi_width, 1.0f);
            roi_height = std::max(roi_height, 1.0f);
        }

        const float bin_size_h = roi_height / static_cast<float>(pooled_height_);
        const float bin_size_w = roi_width / static_cast<float>(pooled_width_);

        const int grid_h = exact_sampling ? sampling_ratio_ : adaptiveGridSize(roi_height, pooled_height_);
        const int grid_w = exact_sampling ? sampling_ratio_ : adaptiveGridSize(roi_width, pooled_width_);
        const size_t grid_count = irt::checkedSizeProduct({static_cast<size_t>(std::max(grid_h, 0)),
                                                           static_cast<size_t>(std::max(grid_w, 0))},
                                                          "RoIAlign sampling grid");
        const size_t count = std::max(grid_count, static_cast<size_t>(1));

        const int loop_grid_h = std::max(grid_h, 0);
        const int loop_grid_w = std::max(grid_w, 0);

        for (int64_t channel = 0; channel < channels; ++channel)
        {
            for (int ph = 0; ph < pooled_height_; ++ph)
            {
                for (int pw = 0; pw < pooled_width_; ++pw)
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
                            sum += bilinearInterpolate(input, roi_batch_ind, channel, channels, height, width, y, x);
                        }
                    }

                    size_t out_index = irt::checkedSizeMul(
                        irt::checkedInt64ToSize(roi_index, "RoIAlign ROI index"),
                        irt::checkedInt64ToSize(channels, "RoIAlign channels"), "RoIAlign output index");
                    out_index = irt::checkedSizeAdd(out_index, irt::checkedInt64ToSize(channel, "RoIAlign channel"),
                                                    "RoIAlign output index");
                    out_index = irt::checkedSizeMul(out_index, static_cast<size_t>(pooled_height_),
                                                    "RoIAlign output index");
                    out_index = irt::checkedSizeAdd(out_index, static_cast<size_t>(ph), "RoIAlign output index");
                    out_index = irt::checkedSizeMul(out_index, static_cast<size_t>(pooled_width_),
                                                    "RoIAlign output index");
                    out_index = irt::checkedSizeAdd(out_index, static_cast<size_t>(pw), "RoIAlign output index");
                    output[out_index] = sum / static_cast<float>(count);
                }
            }
        }
    }
}

std::vector<float> RoIAlign::forward(const float *input, const int64_t input_shape[4], const float *rois,
                                     int64_t num_rois) const
{
    validateInputShape(input_shape);
    if (num_rois < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_rois must be non-negative, got %lld",
                        static_cast<long long>(num_rois));
    }
    const size_t       count = outputElementCount(num_rois, input_shape[1], pooled_height_, pooled_width_);
    std::vector<float> output(count);
    forward(input, input_shape, rois, num_rois, output.data());
    return output;
}

} // namespace irt::ops
