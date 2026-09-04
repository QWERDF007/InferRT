#pragma once

#include <inferrt/core/Exception.hpp>
#include <inferrt/core/Status.hpp>
#include <inferrt/core/Tensor.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace irt {

enum class ColorFormat
{
    BGR,
    RGB,
    GRAY,
    BGRA,
    RGBA,
};

[[nodiscard]] inline constexpr int colorChannels(const ColorFormat format) noexcept
{
    switch (format)
    {
    case ColorFormat::GRAY:
        return 1;
    case ColorFormat::BGR:
    case ColorFormat::RGB:
        return 3;
    case ColorFormat::BGRA:
    case ColorFormat::RGBA:
        return 4;
    }
    return 0;
}

enum class Interpolation
{
    Nearest,
    Linear,
    Cubic,
    Area,
};

enum class PaddingMode
{
    DirectResize,
    Letterbox,
    CenterCrop,
};

enum class PaddingAlignment
{
    Center,
    TopLeft,
};

enum class PreprocessBackend
{
    CPU,
    CUDA,
};

struct PreprocessSpec
{
    int             input_width{0};
    int             input_height{0};
    int             input_channels{3};
    // Zero means the source channel count is derived from src_color.
    int             source_channels{0};
    ColorFormat     src_color{ColorFormat::BGR};
    ColorFormat     dst_color{ColorFormat::RGB};
    Interpolation   interpolation{Interpolation::Linear};
    PaddingMode     padding_mode{PaddingMode::DirectResize};
    PaddingAlignment padding_alignment{PaddingAlignment::Center};
    PreprocessBackend backend{PreprocessBackend::CPU};
    int             source_width{0};
    int             source_height{0};
    float           pad_value{114.0f};
    // When true, letterbox padding is inserted after normalization and is zero in tensor space.
    bool            pad_after_normalize{false};
    std::vector<float> mean{0.485f, 0.456f, 0.406f};
    std::vector<float> stddev{0.229f, 0.224f, 0.225f};
    float           scale{1.0f / 255.0f};
    TensorLayout    output_layout{TensorLayout::NCHW};
    TensorDataType  output_dtype{TensorDataType::F32};

    void validate(bool require_geometry = true) const
    {
        if (input_channels <= 0
            || (require_geometry && (input_width <= 0 || input_height <= 0))
            || (!require_geometry && ((input_width == 0) != (input_height == 0)
                                      || input_width < 0 || input_height < 0)))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "PreprocessSpec dimensions must be positive, got width=%d, height=%d, channels=%d",
                                 input_width, input_height, input_channels);
        }
        if ((source_width == 0) != (source_height == 0) || source_width < 0 || source_height < 0)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "PreprocessSpec source width and height must both be positive or omitted");
        }
        if (mean.size() != std::size_t(input_channels) || stddev.size() != std::size_t(input_channels))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "PreprocessSpec mean/stddev size (%zu/%zu) must match input channels (%d)",
                                 mean.size(), stddev.size(), input_channels);
        }
        const int expected_output_channels = colorChannels(dst_color);
        if (expected_output_channels != input_channels)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "PreprocessSpec destination color format requires %d channels, got %d",
                                 expected_output_channels, input_channels);
        }
        const int expected_source_channels = colorChannels(src_color);
        if (expected_source_channels == 0
            || (source_channels != 0 && source_channels != expected_source_channels))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "PreprocessSpec source color format and channel count are inconsistent");
        }
        if (output_layout != TensorLayout::NCHW && output_layout != TensorLayout::CHW)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "PreprocessSpec output layout must be NCHW or CHW");
        }
        if (padding_alignment != PaddingAlignment::Center && padding_alignment != PaddingAlignment::TopLeft)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Unsupported PreprocessSpec padding alignment");
        }
        if (output_dtype != TensorDataType::F32 || !std::isfinite(scale) || !std::isfinite(pad_value))
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "PreprocessSpec requires finite scale/pad and float32 output");
        }
        if (pad_after_normalize && padding_mode != PaddingMode::Letterbox)
        {
            throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                 "PreprocessSpec pad_after_normalize requires Letterbox padding");
        }
        for (std::size_t i = 0; i < stddev.size(); ++i)
        {
            if (!std::isfinite(mean[i]) || !std::isfinite(stddev[i]) || std::abs(stddev[i]) < 1e-7f)
            {
                throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                                     "PreprocessSpec stddev[%zu] is too close to zero: %f", i, stddev[i]);
            }
        }
    }
};

/**
 * @brief Resolved image geometry shared by all preprocessing backends.
 *
 * The resize dimensions are the concrete raster dimensions after rounding.
 * Padding and crop offsets are measured in the target raster coordinate
 * system.  `scale` is the isotropic source-to-resized scale used by
 * letterbox and center-crop; direct resize keeps the historical minimum-axis
 * value because its two axes may be stretched independently.
 */
struct PreprocessGeometry
{
    int   original_width{0};
    int   original_height{0};
    int   resized_width{0};
    int   resized_height{0};
    int   pad_left{0};
    int   pad_top{0};
    int   crop_left{0};
    int   crop_top{0};
    float scale{1.0F};
};

namespace detail {

[[nodiscard]] inline int roundPreprocessDimension(const double value, const char *axis)
{
    if (!std::isfinite(value) || value <= 0.0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Preprocess %s resize dimension is invalid: %.9g", axis, value);
    }
    if (value < 1.0)
    {
        return 1;
    }
    const double rounded = std::round(value);
    if (rounded < 1.0 || rounded > static_cast<double>(std::numeric_limits<int>::max()))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Preprocess %s resize dimension is out of range: %.9g", axis, rounded);
    }
    return static_cast<int>(rounded);
}

} // namespace detail

/** Resolve concrete resize, padding, and crop geometry for one source image. */
[[nodiscard]] inline PreprocessGeometry resolvePreprocessGeometry(const PreprocessSpec &spec, const int source_width,
                                                                  const int source_height)
{
    spec.validate();
    if (source_width <= 0 || source_height <= 0)
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Preprocess source dimensions must be positive, got width=%d, height=%d", source_width,
                             source_height);
    }
    if (spec.source_width != 0
        && (spec.source_width != source_width || spec.source_height != source_height))
    {
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT,
                             "Preprocess source dimensions do not match spec: expected %dx%d, got %dx%d",
                             spec.source_width, spec.source_height, source_width, source_height);
    }

    PreprocessGeometry geometry;
    geometry.original_width  = source_width;
    geometry.original_height = source_height;

    switch (spec.padding_mode)
    {
    case PaddingMode::DirectResize:
        geometry.resized_width  = spec.input_width;
        geometry.resized_height = spec.input_height;
        geometry.scale          = std::min(static_cast<float>(spec.input_width) / source_width,
                                           static_cast<float>(spec.input_height) / source_height);
        break;
    case PaddingMode::Letterbox:
    {
        const float scale = std::min(static_cast<float>(spec.input_width) / source_width,
                                     static_cast<float>(spec.input_height) / source_height);
        geometry.resized_width = std::min(
            detail::roundPreprocessDimension(static_cast<double>(source_width) * scale, "width"), spec.input_width);
        geometry.resized_height = std::min(
            detail::roundPreprocessDimension(static_cast<double>(source_height) * scale, "height"), spec.input_height);
        if (spec.padding_alignment == PaddingAlignment::Center)
        {
            geometry.pad_left = (spec.input_width - geometry.resized_width) / 2;
            geometry.pad_top  = (spec.input_height - geometry.resized_height) / 2;
        }
        geometry.scale = scale;
        break;
    }
    case PaddingMode::CenterCrop:
    {
        const float scale = std::max(static_cast<float>(spec.input_width) / source_width,
                                     static_cast<float>(spec.input_height) / source_height);
        geometry.resized_width = std::max(
            spec.input_width, detail::roundPreprocessDimension(static_cast<double>(source_width) * scale, "width"));
        geometry.resized_height = std::max(
            spec.input_height, detail::roundPreprocessDimension(static_cast<double>(source_height) * scale, "height"));
        geometry.crop_left = (geometry.resized_width - spec.input_width) / 2;
        geometry.crop_top  = (geometry.resized_height - spec.input_height) / 2;
        geometry.scale      = scale;
        break;
    }
    default:
        throw irt::Exception(Status::ERROR_INVALID_ARGUMENT, "Unsupported PreprocessSpec padding mode");
    }
    return geometry;
}

} // namespace irt
