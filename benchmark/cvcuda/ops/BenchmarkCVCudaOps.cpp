/**
 * @file BenchmarkCVCudaOps.cpp
 * @brief CVCUDA 其他算子的 benchmark。
 *
 * 本文件覆盖 cvcuda 中的 resize、cvtColor、letterBox、integral 和
 * adaptiveThreshold。每个 GPU 算子均提供纯 GPU 计时与端到端计时。
 */

#include "../BenchmarkCVCudaCommon.hpp"

#include <benchmark/benchmark.h>
#include <inferrt/cvcuda/OpAdaptiveThreshold.h>
#include <inferrt/cvcuda/OpCvtColor.h>
#include <inferrt/cvcuda/OpIntegral.h>
#include <inferrt/cvcuda/OpLetterBox.h>
#include <inferrt/cvcuda/OpResize.h>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

namespace bench = irt::cvcuda::bench;

/**
 * @brief 根据缩放倍率计算 Resize 输出尺寸。
 *
 * @param src_w 输入图像宽度。
 * @param src_h 输入图像高度。
 * @param scale 缩放倍率。
 * @return 输出图像尺寸。
 */
cv::Size ComputeScaledSize(int src_w, int src_h, double scale)
{
    return cv::Size(static_cast<int>(std::lround(static_cast<double>(src_w) * scale)),
                    static_cast<int>(std::lround(static_cast<double>(src_h) * scale)));
}

/**
 * @brief 返回 Resize 插值方法名称。
 *
 * @param interpolation OpenCV 插值方法。
 * @return 插值方法名称。
 */
const char *InterpolationName(int interpolation)
{
    switch (interpolation)
    {
    case cv::INTER_NEAREST:
        return "nearest";
    case cv::INTER_LINEAR:
        return "linear";
    case cv::INTER_CUBIC:
        return "cubic";
    default:
        return "unknown";
    }
}

/**
 * @brief 写入 Resize benchmark 的公共统计指标。
 *
 * @param state benchmark 状态对象。
 * @param items_per_iteration 每轮输出元素数量。
 * @param scale 缩放倍率。
 * @param interpolation 插值方法。
 */
void ApplyResizeCounters(benchmark::State &state, int64_t items_per_iteration, double scale, int interpolation)
{
    state.counters["scale"] = scale;
    bench::ApplyElementCounters(state, items_per_iteration, InterpolationName(interpolation));
}

/**
 * @brief 返回 cvtColor 转换码对应的输出通道数。
 *
 * @param code OpenCV 颜色转换码。
 * @return 输出图像通道数。
 */
int CvtColorDstChannels(int code)
{
    switch (code)
    {
    case cv::COLOR_BGR2GRAY:
        return 1;
    case cv::COLOR_GRAY2BGRA:
        return 4;
    case cv::COLOR_BGR2RGB:
    default:
        return 3;
    }
}

/**
 * @brief 返回 cvtColor 转换码名称。
 *
 * @param code OpenCV 颜色转换码。
 * @return 转换名称。
 */
const char *CvtColorName(int code)
{
    switch (code)
    {
    case cv::COLOR_BGR2GRAY:
        return "bgr_to_gray";
    case cv::COLOR_GRAY2BGRA:
        return "gray_to_bgra";
    case cv::COLOR_BGR2RGB:
        return "bgr_to_rgb";
    default:
        return "unknown";
    }
}

/**
 * @brief 返回 LetterBox 参数组合名称。
 *
 * @param channels 输入图像通道数。
 * @return 参数组合名称。
 */
const char *LetterBoxName(int channels)
{
    return channels == 1 ? "gray_hwc_to_chw" : "bgr_hwc_to_rgb_chw";
}

/**
 * @brief 返回 AdaptiveThreshold 方法名称。
 *
 * @param adaptive_method 自适应阈值方法。
 * @return 方法名称。
 */
const char *AdaptiveMethodName(int adaptive_method)
{
    switch (adaptive_method)
    {
    case cv::ADAPTIVE_THRESH_MEAN_C:
        return "mean";
    case cv::ADAPTIVE_THRESH_GAUSSIAN_C:
        return "gaussian";
    case irt::cvcuda::ADAPTIVE_THRESH_PERCENTAGE:
        return "percentage";
    default:
        return "unknown";
    }
}

/**
 * @brief 生成 adaptiveThreshold gaussian 分支使用的二维权重。
 *
 * @param block_size 自适应窗口尺寸。
 * @return 行优先布局的二维 Gaussian 权重。
 */
std::vector<float> MakeGaussianWeights(int block_size)
{
    cv::Mat            kernel = cv::getGaussianKernel(block_size, 0.0, CV_32F);
    std::vector<float> weights(static_cast<size_t>(block_size) * block_size);

    for (int y = 0; y < block_size; ++y)
    {
        for (int x = 0; x < block_size; ++x)
        {
            weights[static_cast<size_t>(y) * block_size + x] = kernel.at<float>(y, 0) * kernel.at<float>(x, 0);
        }
    }

    return weights;
}

/**
 * @brief 为 gaussian adaptiveThreshold 准备设备端权重。
 *
 * @param state benchmark 状态对象。
 * @param adaptive_method 自适应阈值方法。
 * @param block_size 自适应窗口尺寸。
 * @param weights_buffer 输出设备权重缓冲区。
 * @return 设备端权重指针；非 gaussian 分支返回 nullptr。
 */
const float *PrepareAdaptiveWeights(benchmark::State &state, int adaptive_method, int block_size,
                                    bench::DeviceBuffer &weights_buffer)
{
    if (adaptive_method != cv::ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        return nullptr;
    }

    const std::vector<float> weights = MakeGaussianWeights(block_size);
    const size_t             bytes   = weights.size() * sizeof(float);
    if (!bench::CheckStep(state, weights_buffer.Allocate(bytes), "Failed to allocate adaptive weights buffer")
        || !bench::CheckStep(state, weights_buffer.CopyFromHost(weights.data(), bytes),
                             "Failed to copy adaptive weights to device"))
    {
        return nullptr;
    }

    return weights_buffer.As<float>();
}

/**
 * @brief 注册 Resize benchmark 参数组合。
 *
 * 固定输入为 1024x1024x3，覆盖多组缩放倍率与插值方法。
 *
 * @param bench_obj benchmark 注册对象。
 */
void RegisterResizeArguments(benchmark::Benchmark *bench_obj)
{
    constexpr std::array<int, 4> kScales         = {300, 500, 1300, 2000};
    constexpr std::array<int, 3> kInterpolations = {
        cv::INTER_LINEAR,
        cv::INTER_CUBIC,
        cv::INTER_NEAREST,
    };

    for (const int scale_x1000 : kScales)
    {
        for (const int interpolation : kInterpolations)
        {
            bench_obj->Args({1024, 1024, 3, scale_x1000, interpolation});
        }
    }
}

/**
 * @brief 注册 cvtColor benchmark 参数。
 *
 * 覆盖常见的通道交换、彩色转灰度和灰度扩展到 BGRA。
 *
 * @param bench_obj benchmark 注册对象。
 */
void RegisterCvtColorArguments(benchmark::Benchmark *bench_obj)
{
    bench_obj->Args({1920, 1080, 3, cv::COLOR_BGR2RGB});
    bench_obj->Args({1920, 1080, 3, cv::COLOR_BGR2GRAY});
    bench_obj->Args({1920, 1080, 1, cv::COLOR_GRAY2BGRA});
}

/**
 * @brief 注册 LetterBox benchmark 参数。
 *
 * 覆盖 BGR 与灰度输入，以及常见检测模型输入尺寸 640x640。
 *
 * @param bench_obj benchmark 注册对象。
 */
void RegisterLetterBoxArguments(benchmark::Benchmark *bench_obj)
{
    bench_obj->Args({640, 480, 3, 640, 640});
    bench_obj->Args({1920, 1080, 3, 640, 640});
    bench_obj->Args({1024, 768, 1, 640, 640});
}

/**
 * @brief 注册 Integral benchmark 参数。
 *
 * 覆盖单通道和三通道 uint8 输入到 uint32 积分图的场景。
 *
 * @param bench_obj benchmark 注册对象。
 */
void RegisterIntegralArguments(benchmark::Benchmark *bench_obj)
{
    bench_obj->Args({1024, 1024, 1});
    bench_obj->Args({1024, 1024, 3});
    bench_obj->Args({1920, 1080, 3});
}

/**
 * @brief 注册 AdaptiveThreshold benchmark 参数。
 *
 * 参数中的 param_x1000 使用整数表达浮点参数，避免 benchmark 参数列表中出现浮点值。
 *
 * @param bench_obj benchmark 注册对象。
 */
void RegisterAdaptiveThresholdArguments(benchmark::Benchmark *bench_obj)
{
    bench_obj->Args({1024, 1024, 1, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, 3000});
    bench_obj->Args({1024, 1024, 1, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 1000});
    bench_obj->Args({1024, 1024, 1, irt::cvcuda::ADAPTIVE_THRESH_PERCENTAGE, cv::THRESH_BINARY, 7, 850});
    bench_obj->Args({1024, 1024, 3, irt::cvcuda::ADAPTIVE_THRESH_PERCENTAGE, cv::THRESH_BINARY_INV, 7, 850});
}

/**
 * @brief OpenCV Resize CPU 基线 benchmark。
 *
 * 仅统计 cv::resize 本身的执行开销，作为 GPU 实现的对照组。
 *
 * @param state benchmark 状态对象。
 */
static void BM_OpenCVResize(benchmark::State &state)
{
    const int      src_w         = static_cast<int>(state.range(0));
    const int      src_h         = static_cast<int>(state.range(1));
    const int      channels      = static_cast<int>(state.range(2));
    const double   scale         = static_cast<double>(state.range(3)) / 1000.0;
    const int      interpolation = static_cast<int>(state.range(4));
    const cv::Size dst_size      = ComputeScaledSize(src_w, src_h, scale);
    const int64_t  items         = static_cast<int64_t>(dst_size.width) * dst_size.height * channels;

    cv::Mat src = bench::MakeRandomU8Mat(src_w, src_h, channels);
    cv::Mat dst(dst_size.height, dst_size.width, CV_MAKETYPE(CV_8U, channels));

    for (auto _ : state)
    {
        cv::resize(src, dst, dst_size, 0.0, 0.0, interpolation);
        benchmark::DoNotOptimize(dst.data);
        benchmark::ClobberMemory();
    }

    ApplyResizeCounters(state, items, scale, interpolation);
}

/**
 * @brief InferRT CVCUDA Resize 纯 GPU benchmark。
 *
 * 使用 CUDA Event 手动计时，仅统计 GPU resize 执行时间。
 *
 * @param state benchmark 状态对象。
 */
static void BM_InferRTResize(benchmark::State &state)
{
    const int      src_w         = static_cast<int>(state.range(0));
    const int      src_h         = static_cast<int>(state.range(1));
    const int      channels      = static_cast<int>(state.range(2));
    const double   scale         = static_cast<double>(state.range(3)) / 1000.0;
    const int      interpolation = static_cast<int>(state.range(4));
    const cv::Size dst_size      = ComputeScaledSize(src_w, src_h, scale);
    const int64_t  items         = static_cast<int64_t>(dst_size.width) * dst_size.height * channels;
    cv::Mat        src           = bench::MakeRandomU8Mat(src_w, src_h, channels);

    bench::RunDeviceOnly<uint8_t, uint8_t>(
        state, src, static_cast<size_t>(items), items, "inferrt_cvcuda::resize", InterpolationName(interpolation),
        [&](const uint8_t *d_src, uint8_t *d_dst, cudaStream_t stream)
        {
            return irt::cvcuda::resize<uint8_t>(d_src, d_dst, cv::Size(src_w, src_h), dst_size, channels, interpolation,
                                                stream);
        });
    state.counters["scale"] = scale;
}

/**
 * @brief InferRT CVCUDA Resize 端到端 benchmark。
 *
 * 每轮计时覆盖 H2D、GPU resize 和 D2H，更接近实际预处理链路。
 *
 * @param state benchmark 状态对象。
 */
static void BM_InferRTResizeEndToEnd(benchmark::State &state)
{
    const int      src_w         = static_cast<int>(state.range(0));
    const int      src_h         = static_cast<int>(state.range(1));
    const int      channels      = static_cast<int>(state.range(2));
    const double   scale         = static_cast<double>(state.range(3)) / 1000.0;
    const int      interpolation = static_cast<int>(state.range(4));
    const cv::Size dst_size      = ComputeScaledSize(src_w, src_h, scale);
    const int64_t  items         = static_cast<int64_t>(dst_size.width) * dst_size.height * channels;
    cv::Mat        src           = bench::MakeRandomU8Mat(src_w, src_h, channels);

    bench::RunEndToEnd<uint8_t, uint8_t>(
        state, src, static_cast<size_t>(items), items, "inferrt_cvcuda::resize", InterpolationName(interpolation),
        [&](const uint8_t *d_src, uint8_t *d_dst, cudaStream_t stream)
        {
            return irt::cvcuda::resize<uint8_t>(d_src, d_dst, cv::Size(src_w, src_h), dst_size, channels, interpolation,
                                                stream);
        });
    state.counters["scale"] = scale;
}

/**
 * @brief InferRT CVCUDA cvtColor 纯 GPU benchmark。
 *
 * 仅统计 GPU 颜色转换本身的执行耗时。
 *
 * @param state benchmark 状态对象。
 */
static void BM_InferRTCvtColor(benchmark::State &state)
{
    const int     width  = static_cast<int>(state.range(0));
    const int     height = static_cast<int>(state.range(1));
    const int     src_ch = static_cast<int>(state.range(2));
    const int     code   = static_cast<int>(state.range(3));
    const int     dst_ch = CvtColorDstChannels(code);
    const int64_t items  = static_cast<int64_t>(width) * height * dst_ch;
    cv::Mat       src    = bench::MakeRandomU8Mat(width, height, src_ch);

    bench::RunDeviceOnly<uint8_t, uint8_t>(
        state, src, static_cast<size_t>(items), items, "inferrt_cvcuda::cvtColor", CvtColorName(code),
        [&](const uint8_t *d_src, uint8_t *d_dst, cudaStream_t stream)
        { return irt::cvcuda::cvtColor<uint8_t>(d_src, d_dst, cv::Size(width, height), code, stream); });
}

/**
 * @brief InferRT CVCUDA cvtColor 端到端 benchmark。
 *
 * 每轮计时包含 H2D、GPU cvtColor 和 D2H。
 *
 * @param state benchmark 状态对象。
 */
static void BM_InferRTCvtColorEndToEnd(benchmark::State &state)
{
    const int     width  = static_cast<int>(state.range(0));
    const int     height = static_cast<int>(state.range(1));
    const int     src_ch = static_cast<int>(state.range(2));
    const int     code   = static_cast<int>(state.range(3));
    const int     dst_ch = CvtColorDstChannels(code);
    const int64_t items  = static_cast<int64_t>(width) * height * dst_ch;
    cv::Mat       src    = bench::MakeRandomU8Mat(width, height, src_ch);

    bench::RunEndToEnd<uint8_t, uint8_t>(
        state, src, static_cast<size_t>(items), items, "inferrt_cvcuda::cvtColor", CvtColorName(code),
        [&](const uint8_t *d_src, uint8_t *d_dst, cudaStream_t stream)
        { return irt::cvcuda::cvtColor<uint8_t>(d_src, d_dst, cv::Size(width, height), code, stream); });
}

/**
 * @brief InferRT CVCUDA LetterBox 纯 GPU benchmark。
 *
 * 仅统计 GPU letterBox 预处理本身的执行耗时。
 *
 * @param state benchmark 状态对象。
 */
static void BM_InferRTLetterBox(benchmark::State &state)
{
    const int     src_w    = static_cast<int>(state.range(0));
    const int     src_h    = static_cast<int>(state.range(1));
    const int     channels = static_cast<int>(state.range(2));
    const int     dst_w    = static_cast<int>(state.range(3));
    const int     dst_h    = static_cast<int>(state.range(4));
    const int64_t items    = static_cast<int64_t>(dst_w) * dst_h * channels;
    cv::Mat       src      = bench::MakeRandomU8Mat(src_w, src_h, channels);

    bench::RunDeviceOnly<uint8_t, float>(state, src, static_cast<size_t>(items), items, "inferrt_cvcuda::letterBox",
                                         LetterBoxName(channels),
                                         [&](const uint8_t *d_src, float *d_dst, cudaStream_t stream)
                                         {
                                             return irt::cvcuda::letterBox(d_src, d_dst, cv::Size(src_w, src_h),
                                                                           cv::Size(dst_w, dst_h), channels, stream);
                                         });
}

/**
 * @brief InferRT CVCUDA LetterBox 端到端 benchmark。
 *
 * 每轮计时包含 H2D、GPU letterBox 和 D2H。
 *
 * @param state benchmark 状态对象。
 */
static void BM_InferRTLetterBoxEndToEnd(benchmark::State &state)
{
    const int     src_w    = static_cast<int>(state.range(0));
    const int     src_h    = static_cast<int>(state.range(1));
    const int     channels = static_cast<int>(state.range(2));
    const int     dst_w    = static_cast<int>(state.range(3));
    const int     dst_h    = static_cast<int>(state.range(4));
    const int64_t items    = static_cast<int64_t>(dst_w) * dst_h * channels;
    cv::Mat       src      = bench::MakeRandomU8Mat(src_w, src_h, channels);

    bench::RunEndToEnd<uint8_t, float>(state, src, static_cast<size_t>(items), items, "inferrt_cvcuda::letterBox",
                                       LetterBoxName(channels),
                                       [&](const uint8_t *d_src, float *d_dst, cudaStream_t stream)
                                       {
                                           return irt::cvcuda::letterBox(d_src, d_dst, cv::Size(src_w, src_h),
                                                                         cv::Size(dst_w, dst_h), channels, stream);
                                       });
}

/**
 * @brief InferRT CVCUDA Integral 纯 GPU benchmark。
 *
 * 仅统计 GPU 积分图生成本身的执行耗时。
 *
 * @param state benchmark 状态对象。
 */
static void BM_InferRTIntegral(benchmark::State &state)
{
    const int     width    = static_cast<int>(state.range(0));
    const int     height   = static_cast<int>(state.range(1));
    const int     channels = static_cast<int>(state.range(2));
    const int64_t items    = static_cast<int64_t>(width + 1) * (height + 1) * channels;
    cv::Mat       src      = bench::MakeRandomU8Mat(width, height, channels);

    bench::RunDeviceOnly<uint8_t, uint32_t>(
        state, src, static_cast<size_t>(items), items, "inferrt_cvcuda::integral", "u8_to_u32",
        [&](const uint8_t *d_src, uint32_t *d_dst, cudaStream_t stream)
        { return irt::cvcuda::integral<uint8_t, uint32_t>(d_src, d_dst, cv::Size(width, height), channels, stream); });
}

/**
 * @brief InferRT CVCUDA Integral 端到端 benchmark。
 *
 * 每轮计时包含 H2D、GPU integral 和 D2H。
 *
 * @param state benchmark 状态对象。
 */
static void BM_InferRTIntegralEndToEnd(benchmark::State &state)
{
    const int     width    = static_cast<int>(state.range(0));
    const int     height   = static_cast<int>(state.range(1));
    const int     channels = static_cast<int>(state.range(2));
    const int64_t items    = static_cast<int64_t>(width + 1) * (height + 1) * channels;
    cv::Mat       src      = bench::MakeRandomU8Mat(width, height, channels);

    bench::RunEndToEnd<uint8_t, uint32_t>(
        state, src, static_cast<size_t>(items), items, "inferrt_cvcuda::integral", "u8_to_u32",
        [&](const uint8_t *d_src, uint32_t *d_dst, cudaStream_t stream)
        { return irt::cvcuda::integral<uint8_t, uint32_t>(d_src, d_dst, cv::Size(width, height), channels, stream); });
}

/**
 * @brief InferRT CVCUDA AdaptiveThreshold 纯 GPU benchmark。
 *
 * 仅统计 GPU 自适应阈值本身的执行耗时，gaussian 分支的权重在计时前上传。
 *
 * @param state benchmark 状态对象。
 */
static void BM_InferRTAdaptiveThreshold(benchmark::State &state)
{
    const int     width           = static_cast<int>(state.range(0));
    const int     height          = static_cast<int>(state.range(1));
    const int     channels        = static_cast<int>(state.range(2));
    const int     adaptive_method = static_cast<int>(state.range(3));
    const int     threshold_type  = static_cast<int>(state.range(4));
    const int     block_size      = static_cast<int>(state.range(5));
    const double  param           = static_cast<double>(state.range(6)) / 1000.0;
    const int64_t items           = static_cast<int64_t>(width) * height * channels;
    cv::Mat       src             = bench::MakeRandomU8Mat(width, height, channels);

    bench::DeviceBuffer weights_buffer;
    const float        *d_weights = PrepareAdaptiveWeights(state, adaptive_method, block_size, weights_buffer);
    if (state.skipped())
    {
        return;
    }

    bench::RunDeviceOnly<uint8_t, uint8_t>(state, src, static_cast<size_t>(items), items,
                                           "inferrt_cvcuda::adaptiveThreshold", AdaptiveMethodName(adaptive_method),
                                           [&](const uint8_t *d_src, uint8_t *d_dst, cudaStream_t stream)
                                           {
                                               return irt::cvcuda::adaptiveThreshold<uint8_t>(
                                                   d_src, d_dst, cv::Size(width, height), channels, 255.0,
                                                   adaptive_method, threshold_type, block_size, param, d_weights,
                                                   stream);
                                           });
}

/**
 * @brief InferRT CVCUDA AdaptiveThreshold 端到端 benchmark。
 *
 * 每轮计时包含 H2D、GPU adaptiveThreshold 和 D2H，gaussian 权重不重复上传。
 *
 * @param state benchmark 状态对象。
 */
static void BM_InferRTAdaptiveThresholdEndToEnd(benchmark::State &state)
{
    const int     width           = static_cast<int>(state.range(0));
    const int     height          = static_cast<int>(state.range(1));
    const int     channels        = static_cast<int>(state.range(2));
    const int     adaptive_method = static_cast<int>(state.range(3));
    const int     threshold_type  = static_cast<int>(state.range(4));
    const int     block_size      = static_cast<int>(state.range(5));
    const double  param           = static_cast<double>(state.range(6)) / 1000.0;
    const int64_t items           = static_cast<int64_t>(width) * height * channels;
    cv::Mat       src             = bench::MakeRandomU8Mat(width, height, channels);

    bench::DeviceBuffer weights_buffer;
    const float        *d_weights = PrepareAdaptiveWeights(state, adaptive_method, block_size, weights_buffer);
    if (state.skipped())
    {
        return;
    }

    bench::RunEndToEnd<uint8_t, uint8_t>(state, src, static_cast<size_t>(items), items,
                                         "inferrt_cvcuda::adaptiveThreshold", AdaptiveMethodName(adaptive_method),
                                         [&](const uint8_t *d_src, uint8_t *d_dst, cudaStream_t stream)
                                         {
                                             return irt::cvcuda::adaptiveThreshold<uint8_t>(
                                                 d_src, d_dst, cv::Size(width, height), channels, 255.0,
                                                 adaptive_method, threshold_type, block_size, param, d_weights, stream);
                                         });
}

BENCHMARK(BM_OpenCVResize)->Apply(RegisterResizeArguments);
BENCHMARK(BM_InferRTResize)->Apply(RegisterResizeArguments)->UseManualTime();
BENCHMARK(BM_InferRTResizeEndToEnd)->Apply(RegisterResizeArguments)->UseManualTime();
BENCHMARK(BM_InferRTCvtColor)->Apply(RegisterCvtColorArguments)->UseManualTime();
BENCHMARK(BM_InferRTCvtColorEndToEnd)->Apply(RegisterCvtColorArguments)->UseManualTime();
BENCHMARK(BM_InferRTLetterBox)->Apply(RegisterLetterBoxArguments)->UseManualTime();
BENCHMARK(BM_InferRTLetterBoxEndToEnd)->Apply(RegisterLetterBoxArguments)->UseManualTime();
BENCHMARK(BM_InferRTIntegral)->Apply(RegisterIntegralArguments)->UseManualTime();
BENCHMARK(BM_InferRTIntegralEndToEnd)->Apply(RegisterIntegralArguments)->UseManualTime();
BENCHMARK(BM_InferRTAdaptiveThreshold)->Apply(RegisterAdaptiveThresholdArguments)->UseManualTime();
BENCHMARK(BM_InferRTAdaptiveThresholdEndToEnd)->Apply(RegisterAdaptiveThresholdArguments)->UseManualTime();

} // namespace

BENCHMARK_MAIN();
