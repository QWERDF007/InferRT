/**
 * @file TestLetterBox.cpp
 * @brief LetterBox 算子的单元测试
 *
 * 覆盖内容：
 * - C API（letter_box）与 C++ API（letterBox）的 BGR HWC → RGB CHW 转换
 * - 单通道灰度图 letterbox
 * - LetterBox 类的封装调用
 * - 非法通道数等边界错误
 */

#include "TestCVCudaCommon.hpp"

#include <gtest/gtest.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/OpLetterBox.h>
#include <inferrt/cvcuda/OpLetterBox.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using irt::cvcuda::test::AssertInferRTSuccess;

int cvInterpolation(const irt::Interpolation interpolation)
{
    switch (interpolation)
    {
    case irt::Interpolation::Nearest:
        return cv::INTER_NEAREST;
    case irt::Interpolation::Linear:
        return cv::INTER_LINEAR;
    case irt::Interpolation::Cubic:
        return cv::INTER_CUBIC;
    case irt::Interpolation::Area:
        return cv::INTER_AREA;
    }
    throw std::invalid_argument("unsupported interpolation");
}

int colorConversionCode(const irt::ColorFormat source, const irt::ColorFormat destination)
{
    if (source == destination)
    {
        return -1;
    }

    using irt::ColorFormat;
    switch (source)
    {
    case ColorFormat::BGR:
        switch (destination)
        {
        case ColorFormat::RGB:  return cv::COLOR_BGR2RGB;
        case ColorFormat::GRAY: return cv::COLOR_BGR2GRAY;
        case ColorFormat::BGRA: return cv::COLOR_BGR2BGRA;
        case ColorFormat::RGBA: return cv::COLOR_BGR2RGBA;
        default:                break;
        }
        break;
    case ColorFormat::RGB:
        switch (destination)
        {
        case ColorFormat::BGR:  return cv::COLOR_RGB2BGR;
        case ColorFormat::GRAY: return cv::COLOR_RGB2GRAY;
        case ColorFormat::BGRA: return cv::COLOR_RGB2BGRA;
        case ColorFormat::RGBA: return cv::COLOR_RGB2RGBA;
        default:                break;
        }
        break;
    case ColorFormat::GRAY:
        switch (destination)
        {
        case ColorFormat::BGR:  return cv::COLOR_GRAY2BGR;
        case ColorFormat::RGB:  return cv::COLOR_GRAY2RGB;
        case ColorFormat::BGRA: return cv::COLOR_GRAY2BGRA;
        case ColorFormat::RGBA: return cv::COLOR_GRAY2RGBA;
        default:                break;
        }
        break;
    case ColorFormat::BGRA:
        switch (destination)
        {
        case ColorFormat::BGR:  return cv::COLOR_BGRA2BGR;
        case ColorFormat::RGB:  return cv::COLOR_BGRA2RGB;
        case ColorFormat::GRAY: return cv::COLOR_BGRA2GRAY;
        case ColorFormat::RGBA: return cv::COLOR_BGRA2RGBA;
        default:                break;
        }
        break;
    case ColorFormat::RGBA:
        switch (destination)
        {
        case ColorFormat::BGR:  return cv::COLOR_RGBA2BGR;
        case ColorFormat::RGB:  return cv::COLOR_RGBA2RGB;
        case ColorFormat::GRAY: return cv::COLOR_RGBA2GRAY;
        case ColorFormat::BGRA: return cv::COLOR_RGBA2BGRA;
        default:                break;
        }
        break;
    }
    throw std::invalid_argument("unsupported color conversion");
}

std::vector<float> makePreprocessSpecReference(const cv::Mat &src, const irt::PreprocessSpec &spec)
{
    spec.validate();
    const int expected_source_channels = irt::colorChannels(spec.src_color);
    if (src.depth() != CV_8U || src.channels() != expected_source_channels)
    {
        throw std::invalid_argument("reference preprocessing source format is invalid");
    }

    cv::Mat converted;
    const int conversion = colorConversionCode(spec.src_color, spec.dst_color);
    if (conversion < 0)
    {
        converted = src;
    }
    else
    {
        cv::cvtColor(src, converted, conversion);
    }

    const auto geometry = irt::resolvePreprocessGeometry(spec, src.cols, src.rows);
    const cv::Size target(spec.input_width, spec.input_height);
    const int interpolation = cvInterpolation(spec.interpolation);
    cv::Mat resized;
    cv::Rect content_rect;
    switch (spec.padding_mode)
    {
    case irt::PaddingMode::DirectResize:
        cv::resize(converted, resized, target, 0.0, 0.0, interpolation);
        break;
    case irt::PaddingMode::Letterbox:
        cv::resize(converted, resized, cv::Size(geometry.resized_width, geometry.resized_height), 0.0, 0.0,
                   interpolation);
        content_rect = cv::Rect(geometry.pad_left, geometry.pad_top, resized.cols, resized.rows);
        if (!spec.pad_after_normalize)
        {
            cv::Mat canvas(target.height, target.width, resized.type(),
                           cv::Scalar(spec.pad_value, spec.pad_value, spec.pad_value, spec.pad_value));
            resized.copyTo(canvas(content_rect));
            resized = std::move(canvas);
        }
        break;
    case irt::PaddingMode::CenterCrop:
        cv::resize(converted, resized, cv::Size(geometry.resized_width, geometry.resized_height), 0.0, 0.0,
                   interpolation);
        resized = resized(cv::Rect(geometry.crop_left, geometry.crop_top, target.width, target.height)).clone();
        break;
    }

    cv::Mat normalized;
    resized.convertTo(normalized, CV_MAKETYPE(CV_32F, spec.input_channels), spec.scale);
    std::vector<cv::Mat> channels;
    cv::split(normalized, channels);
    for (size_t channel = 0; channel < channels.size(); ++channel)
    {
        channels[channel].convertTo(channels[channel], CV_32F, 1.0, -spec.mean[channel]);
        channels[channel] /= spec.stddev[channel];
    }
    cv::merge(channels, normalized);

    if (spec.pad_after_normalize)
    {
        cv::Mat padded = cv::Mat::zeros(target.height, target.width, normalized.type());
        normalized.copyTo(padded(content_rect));
        normalized = std::move(padded);
    }

    const size_t plane_size = normalized.total();
    std::vector<float> tensor(plane_size * static_cast<size_t>(normalized.channels()));
    cv::split(normalized, channels);
    for (size_t channel = 0; channel < channels.size(); ++channel)
    {
        std::memcpy(tensor.data() + channel * plane_size, channels[channel].ptr<float>(),
                    plane_size * sizeof(float));
    }
    return tensor;
}

template<typename Caller>
void runPreprocessSpecParity(const cv::Mat &source, const irt::PreprocessSpec &spec, Caller caller)
{
    const std::vector<float> reference = makePreprocessSpecReference(source, spec);
    const size_t row_bytes = source.cols * source.elemSize();
    const size_t source_bytes = source.step * static_cast<size_t>(source.rows);
    const size_t destination_bytes = reference.size() * sizeof(float);

    uint8_t *d_source = nullptr;
    float   *d_destination = nullptr;
    ASSERT_EQ(cudaMalloc(&d_source, source_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_destination, destination_bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy2D(d_source, source.step, source.data, source.step, row_bytes,
                           static_cast<size_t>(source.rows), cudaMemcpyHostToDevice), cudaSuccess);

    const int ret = caller(d_source, d_destination, cv::Size(source.cols, source.rows),
                           cv::Size(spec.input_width, spec.input_height), source.channels(), source.step, spec,
                           nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> actual(reference.size());
    ASSERT_EQ(cudaMemcpy(actual.data(), d_destination, destination_bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaFree(d_source), cudaSuccess);
    ASSERT_EQ(cudaFree(d_destination), cudaSuccess);

    float  max_diff = 0.0F;
    size_t max_index = 0;
    for (size_t index = 0; index < actual.size(); ++index)
    {
        const float diff = std::abs(actual[index] - reference[index]);
        if (diff > max_diff)
        {
            max_diff  = diff;
            max_index = index;
        }
    }
    const float minimum_stddev = *std::min_element(spec.stddev.begin(), spec.stddev.end());
    EXPECT_LE(max_diff, 3.0F / 255.0F / minimum_stddev + 1e-6F)
        << "max_index=" << max_index << " actual=" << actual[max_index]
        << " reference=" << reference[max_index];
}

irt::PreprocessSpec makeSpec(const int width, const int height, const int channels,
                             const irt::ColorFormat source_color, const irt::ColorFormat destination_color)
{
    irt::PreprocessSpec spec;
    spec.input_width     = width;
    spec.input_height    = height;
    spec.input_channels  = channels;
    spec.source_channels = channels;
    spec.src_color       = source_color;
    spec.dst_color       = destination_color;
    spec.padding_mode    = irt::PaddingMode::Letterbox;
    spec.mean.assign(static_cast<size_t>(channels), 0.0F);
    spec.stddev.assign(static_cast<size_t>(channels), 1.0F);
    return spec;
}

auto runSharedSpecLetterBox = [](const uint8_t *source, float *destination, const cv::Size source_size,
                                 const cv::Size destination_size, const int channels, const size_t source_stride,
                                 const irt::PreprocessSpec &spec, cudaStream_t stream) {
    return irt::cvcuda::letterBox(source, destination, source_size, destination_size, channels, spec, source_stride,
                                   stream);
};

/**
 * @brief 公共 LetterBox 测试逻辑
 *
 * 流程：
 * 1. 生成随机源图（HWC uint8）
 * 2. CPU 参考实现得到期望输出
 * 3. 分配 GPU 缓冲并 H2D 上传
 * 4. 通过 Caller 调用 GPU LetterBox
 * 5. D2H 下载并与参考逐元素比较（允许 1/255 量化误差）
 *
 * @tparam Caller 可调用对象，签名为 (const uint8_t*, float*, cv::Size, cv::Size, int, cudaStream_t) -> int
 * @param src_w 源图宽度
 * @param src_h 源图高度
 * @param ch 通道数（1、3 或 4）
 * @param dsize 目标画布尺寸
 * @param caller LetterBox 实现（函数或类 operator()）
 */
template<typename Caller>
void runLetterBoxTest(int src_w, int src_h, int ch, cv::Size dsize, Caller caller)
{
    cv::Mat src(src_h, src_w, CV_MAKETYPE(CV_8U, ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));

    const auto source_color = ch == 1 ? irt::ColorFormat::GRAY
                                      : ch == 4 ? irt::ColorFormat::BGRA : irt::ColorFormat::BGR;
    const auto destination_color = ch == 1 ? irt::ColorFormat::GRAY
                                           : ch == 4 ? irt::ColorFormat::RGBA : irt::ColorFormat::RGB;
    const auto spec = makeSpec(dsize.width, dsize.height, ch, source_color, destination_color);
    const std::vector<float> ref = makePreprocessSpecReference(src, spec);

    const size_t src_bytes = static_cast<size_t>(src_w) * src_h * ch * sizeof(uint8_t);
    const size_t dst_bytes = static_cast<size_t>(dsize.width) * dsize.height * ch * sizeof(float);

    uint8_t *d_src = nullptr;
    float   *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_bytes), cudaSuccess);

    cv::Mat src_cont = src.isContinuous() ? src : src.clone();
    ASSERT_EQ(cudaMemcpy(d_src, src_cont.data, src_bytes, cudaMemcpyHostToDevice), cudaSuccess);

    const int ret = caller(d_src, d_dst, cv::Size(src_w, src_h), dsize, ch, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> dst(ref.size());
    ASSERT_EQ(cudaMemcpy(dst.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    float max_diff = 0.0f;
    for (size_t i = 0; i < dst.size(); ++i)
    {
        max_diff = std::max(max_diff, std::abs(dst[i] - ref[i]));
    }
    EXPECT_LE(max_diff, 1.0f / 255.0f + 1e-6f);
}

} // namespace

// ============================================================================
// 功能正确性测试
// ============================================================================

TEST(PreprocessSpecParityTest, BgrSpecUsesResolvedOddPadding)
{
    const cv::Mat source(3, 5, CV_8UC3, cv::Scalar(10, 20, 30));
    const auto spec = makeSpec(9, 8, 3, irt::ColorFormat::BGR, irt::ColorFormat::RGB);

    const auto geometry = irt::resolvePreprocessGeometry(spec, source.cols, source.rows);
    ASSERT_EQ(geometry.resized_width, 9);
    ASSERT_EQ(geometry.resized_height, 5);
    ASSERT_EQ(geometry.pad_left, 0);
    ASSERT_EQ(geometry.pad_top, 1);
    runPreprocessSpecParity(source, spec, runSharedSpecLetterBox);
}

TEST(PreprocessSpecParityTest, GraySpecSupportsExtremeAspectRatio)
{
    cv::Mat source(1, 31, CV_8UC1);
    for (int x = 0; x < source.cols; ++x)
    {
        source.at<uint8_t>(0, x) = static_cast<uint8_t>(x * 7);
    }
    const auto spec = makeSpec(7, 5, 1, irt::ColorFormat::GRAY, irt::ColorFormat::GRAY);

    const auto geometry = irt::resolvePreprocessGeometry(spec, source.cols, source.rows);
    ASSERT_EQ(geometry.resized_width, 7);
    ASSERT_EQ(geometry.resized_height, 1);
    runPreprocessSpecParity(source, spec, runSharedSpecLetterBox);
}

TEST(PreprocessSpecParityTest, BgraSpecPadsAfterNormalization)
{
    cv::Mat source(2, 4, CV_8UC4);
    for (int y = 0; y < source.rows; ++y)
    {
        for (int x = 0; x < source.cols; ++x)
        {
            source.at<cv::Vec4b>(y, x) = cv::Vec4b(static_cast<uint8_t>(x + 1), static_cast<uint8_t>(y + 2),
                                                   30, 255);
        }
    }

    auto spec = makeSpec(7, 7, 4, irt::ColorFormat::BGRA, irt::ColorFormat::RGBA);
    spec.pad_value         = 77.0F;
    spec.pad_after_normalize = true;
    spec.scale              = 1.0F / 255.0F;
    spec.mean               = {0.1F, 0.2F, 0.3F, 0.4F};
    spec.stddev             = {0.5F, 0.6F, 0.7F, 0.8F};
    runPreprocessSpecParity(source, spec, runSharedSpecLetterBox);
}

TEST(PreprocessSpecParityTest, NonContinuousBgrSourceUsesItsRowStride)
{
    cv::Mat backing(5, 9, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat source = backing(cv::Rect(1, 1, 5, 3));
    ASSERT_FALSE(source.isContinuous());
    for (int y = 0; y < source.rows; ++y)
    {
        for (int x = 0; x < source.cols; ++x)
        {
            source.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uint8_t>(x + 1), static_cast<uint8_t>(y + 3), 42);
        }
    }

    const auto spec = makeSpec(9, 8, 3, irt::ColorFormat::BGR, irt::ColorFormat::RGB);
    runPreprocessSpecParity(source, spec, runSharedSpecLetterBox);
}

/**
 * @brief 测试 letterBox 函数：BGR HWC → RGB CHW，带 letterbox 填充
 *
 * 非正方形源图（37×19）缩放至 64×64 画布，验证与 CPU 参考一致。
 */
TEST(LetterBoxFunctionTest, BgrHwcToRgbChwWithPadding)
{
    runLetterBoxTest(37, 19, 3, cv::Size(64, 64),
                     [](const uint8_t *src, float *dst, cv::Size ssize, cv::Size dsize, int ch, cudaStream_t stream)
                     { return irt::cvcuda::letterBox(src, dst, ssize, dsize, ch, stream); });
}

/**
 * @brief 测试 letter_box C API：灰度 HWC → CHW，带 letterbox 填充
 *
 * 单通道（17×41 → 64×64），验证 C 风格命名接口。
 */
TEST(LetterBoxFunctionTest, GrayHwcToChwWithPadding)
{
    runLetterBoxTest(17, 41, 1, cv::Size(64, 64),
                     [](const uint8_t *src, float *dst, cv::Size ssize, cv::Size dsize, int ch, cudaStream_t stream)
                     { return irt::cvcuda::letter_box(src, dst, ssize, dsize, ch, stream); });
}

/**
 * @brief 测试 LetterBox 类：BGR HWC → RGB CHW，非正方形目标画布
 *
 * 源图 80×23，目标 96×48，通过类实例 operator() 调用。
 */
TEST(LetterBoxClassTest, BgrHwcToRgbChwWithPadding)
{
    irt::cvcuda::LetterBox op;
    runLetterBoxTest(80, 23, 3, cv::Size(96, 48),
                     [&op](const uint8_t *src, float *dst, cv::Size ssize, cv::Size dsize, int ch, cudaStream_t stream)
                     { return op(src, dst, ssize, dsize, ch, stream); });
}

// ============================================================================
// 边界与错误处理测试
// ============================================================================

/**
 * @brief 测试非法通道数：应返回 IRT_ERROR_INVALID_ARGUMENT
 *
 * 仅支持 1、3 或 4 通道；传入 ch=5 时期望失败且不崩溃。
 */
TEST(LetterBoxFunctionEdgeCaseTest, RejectsInvalidChannels)
{
    uint8_t *d_src = nullptr;
    float   *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 10 * 10 * 5), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, 32 * 32 * 5 * sizeof(float)), cudaSuccess);

    const int ret = irt::cvcuda::letterBox(d_src, d_dst, cv::Size(10, 10), cv::Size(32, 32), 5, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_src);
    cudaFree(d_dst);
}

TEST(LetterBoxSpecTest, RespectsTopLeftPaddingAlignment)
{
    constexpr int source_width  = 4;
    constexpr int source_height = 2;
    constexpr int target_width  = 4;
    constexpr int target_height = 4;
    constexpr int channels      = 3;

    cv::Mat source(source_height, source_width, CV_8UC3, cv::Scalar(10, 20, 30));
    const size_t source_bytes = source.total() * source.elemSize();
    const size_t target_bytes = static_cast<size_t>(target_width) * target_height * channels * sizeof(float);

    uint8_t *d_source = nullptr;
    float   *d_target = nullptr;
    ASSERT_EQ(cudaMalloc(&d_source, source_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_target, target_bytes), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_source, source.data, source_bytes, cudaMemcpyHostToDevice), cudaSuccess);

    irt::PreprocessSpec spec;
    spec.input_width       = target_width;
    spec.input_height      = target_height;
    spec.input_channels    = channels;
    spec.source_channels   = channels;
    spec.src_color         = irt::ColorFormat::BGR;
    spec.dst_color         = irt::ColorFormat::RGB;
    spec.padding_mode      = irt::PaddingMode::Letterbox;
    spec.padding_alignment = irt::PaddingAlignment::TopLeft;
    spec.mean              = {0.0F, 0.0F, 0.0F};
    spec.stddev            = {1.0F, 1.0F, 1.0F};

    irt::cvcuda::LetterBox letter_box;
    const int ret = letter_box(d_source, d_target, cv::Size(source_width, source_height),
                               cv::Size(target_width, target_height), channels, spec, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<float> target(static_cast<size_t>(target_width) * target_height * channels);
    ASSERT_EQ(cudaMemcpy(target.data(), d_target, target_bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_source);
    cudaFree(d_target);

    const size_t plane_size = static_cast<size_t>(target_width) * target_height;
    EXPECT_NEAR(target[0], 30.0F / 255.0F, 1.0F / 255.0F);
    EXPECT_NEAR(target[plane_size], 20.0F / 255.0F, 1.0F / 255.0F);
    EXPECT_NEAR(target[2 * plane_size], 10.0F / 255.0F, 1.0F / 255.0F);
    EXPECT_FLOAT_EQ(target[target_width * 2], 114.0F / 255.0F);
    EXPECT_FLOAT_EQ(target[plane_size + target_width * 2], 114.0F / 255.0F);
    EXPECT_FLOAT_EQ(target[2 * plane_size + target_width * 2], 114.0F / 255.0F);
}
