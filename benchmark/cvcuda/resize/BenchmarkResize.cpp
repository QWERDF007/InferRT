#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <inferrt/cvcuda/OpResize.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <cstdint>

namespace {

/**
 * @brief Resize 基准测试使用的共享上下文。
 *
 * 同时保存主机侧 OpenCV 图像和设备侧 CUDA 缓冲区，
 * 便于 CPU 基线测试与 GPU 实现测试复用相同的数据组织方式。
 */
struct ResizeBenchmarkContext
{
    cv::Mat  src;
    cv::Mat  dst;
    uint8_t *d_src = nullptr;
    uint8_t *d_dst = nullptr;
};

/**
 * @brief 初始化 Resize 基准测试上下文。
 *
 * 创建输入输出图像，生成随机输入数据，并为 CUDA 版本分配、
 * 填充对应的设备缓冲区。
 *
 * @param ctx 基准测试上下文。
 * @param src_w 输入图像宽度。
 * @param src_h 输入图像高度。
 * @param channels 图像通道数。
 * @param dst_w 输出图像宽度。
 * @param dst_h 输出图像高度。
 * @return true 初始化成功。
 * @return false 初始化失败。
 */
bool InitContext(ResizeBenchmarkContext &ctx, int src_w, int src_h, int channels, int dst_w, int dst_h)
{
    ctx.src = cv::Mat(src_h, src_w, CV_MAKETYPE(CV_8U, channels));
    ctx.dst = cv::Mat(dst_h, dst_w, CV_MAKETYPE(CV_8U, channels));

    cv::randu(ctx.src, cv::Scalar::all(0), cv::Scalar::all(255));

    const size_t src_bytes = ctx.src.total() * ctx.src.elemSize();
    const size_t dst_bytes = ctx.dst.total() * ctx.dst.elemSize();

    if (cudaMalloc(&ctx.d_src, src_bytes) != cudaSuccess)
    {
        return false;
    }

    if (cudaMalloc(&ctx.d_dst, dst_bytes) != cudaSuccess)
    {
        cudaFree(ctx.d_src);
        ctx.d_src = nullptr;
        return false;
    }

    if (cudaMemcpy(ctx.d_src, ctx.src.data, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(ctx.d_src);
        cudaFree(ctx.d_dst);
        ctx.d_src = nullptr;
        ctx.d_dst = nullptr;
        return false;
    }

    return true;
}

/**
 * @brief 计算上下文中输入图像对应的字节数。
 *
 * @param ctx 基准测试上下文。
 * @return size_t 输入图像字节数。
 */
size_t GetSrcBytes(const ResizeBenchmarkContext &ctx)
{
    return ctx.src.total() * ctx.src.elemSize();
}

/**
 * @brief 计算上下文中输出图像对应的字节数。
 *
 * @param ctx 基准测试上下文。
 * @return size_t 输出图像字节数。
 */
size_t GetDstBytes(const ResizeBenchmarkContext &ctx)
{
    return ctx.dst.total() * ctx.dst.elemSize();
}

/**
 * @brief 释放 Resize 基准测试上下文中的 CUDA 资源。
 *
 * @param ctx 基准测试上下文。
 */
void ReleaseContext(ResizeBenchmarkContext &ctx)
{
    if (ctx.d_src != nullptr)
    {
        cudaFree(ctx.d_src);
        ctx.d_src = nullptr;
    }

    if (ctx.d_dst != nullptr)
    {
        cudaFree(ctx.d_dst);
        ctx.d_dst = nullptr;
    }
}

/**
 * @brief 根据缩放倍率计算输出尺寸。
 *
 * @param src_w 输入图像宽度。
 * @param src_h 输入图像高度。
 * @param scale 缩放倍率。
 * @return cv::Size 输出图像尺寸。
 */
cv::Size ComputeScaledSize(int src_w, int src_h, double scale)
{
    return cv::Size(static_cast<int>(std::lround(static_cast<double>(src_w) * scale)),
                    static_cast<int>(std::lround(static_cast<double>(src_h) * scale)));
}

/**
 * @brief 返回插值方法对应的名称。
 *
 * @param interpolation OpenCV 插值方法枚举值。
 * @return const char * 插值方法名称。
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
 * @brief 填充 CPU/GPU 基准共享的统计指标。
 *
 * 用于在 benchmark 输出中展示缩放倍率、插值方法和按输出元素数估算的吞吐率。
 *
 * @param state benchmark 状态对象。
 * @param channels 图像通道数。
 * @param dst_w 输出图像宽度。
 * @param dst_h 输出图像高度。
 * @param scale 缩放倍率。
 * @param interpolation 插值方法。
 */
void ApplyCommonCounters(benchmark::State &state, int channels, int dst_w, int dst_h, double scale, int interpolation)
{
    const double dst_pixels = static_cast<double>(dst_w) * static_cast<double>(dst_h);
    const double elements   = dst_pixels * static_cast<double>(channels);

    state.counters["scale"] = scale;
    // state.counters["interpolation"] = static_cast<double>(interpolation);
    state.counters["items/s"] = benchmark::Counter(elements, benchmark::Counter::kIsRate);
    state.SetLabel(InterpolationName(interpolation));
}

/**
 * @brief 注册 Resize 基准测试参数组合。
 *
 * 固定输入为 1024x1024x3，覆盖多组缩放倍率与插值方法，
 * 用于观察不同参数组合下的性能表现。
 *
 * @param bench benchmark 注册对象。
 */
void RegisterResizeArguments(benchmark::Benchmark *bench)
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
            bench->Args({1024, 1024, 3, scale_x1000, interpolation});
        }
    }
}

/**
 * @brief OpenCV Resize CPU 基线测试。
 *
 * 仅统计 `cv::resize` 本身的执行开销，作为 InferRT CVCUDA 实现的对照组。
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
    const int      dst_w         = dst_size.width;
    const int      dst_h         = dst_size.height;

    ResizeBenchmarkContext ctx;
    ctx.src = cv::Mat(src_h, src_w, CV_MAKETYPE(CV_8U, channels));
    ctx.dst = cv::Mat(dst_h, dst_w, CV_MAKETYPE(CV_8U, channels));
    cv::randu(ctx.src, cv::Scalar::all(0), cv::Scalar::all(255));

    for (auto _ : state)
    {
        cv::resize(ctx.src, ctx.dst, dst_size, 0.0, 0.0, interpolation);
        benchmark::DoNotOptimize(ctx.dst.data);
        benchmark::ClobberMemory();
    }

    ApplyCommonCounters(state, channels, dst_w, dst_h, scale, interpolation);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(dst_w) * dst_h * channels);
}

/**
 * @brief InferRT CVCUDA Resize GPU 基准测试。
 *
 * 使用 CUDA Event 手动计时，仅统计 GPU 侧 resize 执行时间，
 * 避免主机端调用与调度开销对结果造成干扰。
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
    const int      dst_w         = dst_size.width;
    const int      dst_h         = dst_size.height;

    ResizeBenchmarkContext ctx;
    if (!InitContext(ctx, src_w, src_h, channels, dst_w, dst_h))
    {
        state.SkipWithError("Failed to initialize CUDA benchmark buffers");
        ReleaseContext(ctx);
        return;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
    {
        if (start != nullptr)
        {
            cudaEventDestroy(start);
        }
        if (stop != nullptr)
        {
            cudaEventDestroy(stop);
        }
        ReleaseContext(ctx);
        state.SkipWithError("Failed to create CUDA events");
        return;
    }

    for (auto _ : state)
    {
        cudaEventRecord(start);
        const int ret = irt::cvcuda::resize<uint8_t>(ctx.d_src, ctx.d_dst, cv::Size(src_w, src_h), dst_size, channels,
                                                     interpolation, nullptr);
        cudaEventRecord(stop);

        if (ret != IRT_SUCCESS)
        {
            state.SkipWithError("inferrt_cvcuda::resize failed");
            break;
        }

        if (cudaEventSynchronize(stop) != cudaSuccess)
        {
            state.SkipWithError("Failed to synchronize CUDA event");
            break;
        }

        float elapsed_ms = 0.0f;
        if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess)
        {
            state.SkipWithError("Failed to query CUDA event elapsed time");
            break;
        }

        state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
        benchmark::ClobberMemory();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ReleaseContext(ctx);

    ApplyCommonCounters(state, channels, dst_w, dst_h, scale, interpolation);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(dst_w) * dst_h * channels);
}

/**
 * @brief InferRT CVCUDA Resize 端到端 GPU 基准测试。
 *
 * 每轮计时均覆盖主机到设备拷贝、GPU 侧 resize 以及设备到主机拷贝，
 * 用于评估更接近实际预处理链路的总耗时。
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
    const int      dst_w         = dst_size.width;
    const int      dst_h         = dst_size.height;

    ResizeBenchmarkContext ctx;
    if (!InitContext(ctx, src_w, src_h, channels, dst_w, dst_h))
    {
        state.SkipWithError("Failed to initialize CUDA benchmark buffers");
        ReleaseContext(ctx);
        return;
    }

    const size_t src_bytes = GetSrcBytes(ctx);
    const size_t dst_bytes = GetDstBytes(ctx);

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        ReleaseContext(ctx);
        state.SkipWithError("Failed to create CUDA stream");
        return;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
    {
        if (start != nullptr)
        {
            cudaEventDestroy(start);
        }
        if (stop != nullptr)
        {
            cudaEventDestroy(stop);
        }
        cudaStreamDestroy(stream);
        ReleaseContext(ctx);
        state.SkipWithError("Failed to create CUDA events");
        return;
    }

    for (auto _ : state)
    {
        cudaEventRecord(start, stream);

        if (cudaMemcpyAsync(ctx.d_src, ctx.src.data, src_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            state.SkipWithError("Failed to copy input data to device");
            break;
        }

        const int ret = irt::cvcuda::resize<uint8_t>(ctx.d_src, ctx.d_dst, cv::Size(src_w, src_h), dst_size, channels,
                                                     interpolation, stream);
        if (ret != IRT_SUCCESS)
        {
            state.SkipWithError("inferrt_cvcuda::resize failed");
            break;
        }

        if (cudaMemcpyAsync(ctx.dst.data, ctx.d_dst, dst_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        {
            state.SkipWithError("Failed to copy output data to host");
            break;
        }

        cudaEventRecord(stop, stream);

        if (cudaEventSynchronize(stop) != cudaSuccess)
        {
            state.SkipWithError("Failed to synchronize CUDA event");
            break;
        }

        float elapsed_ms = 0.0f;
        if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess)
        {
            state.SkipWithError("Failed to query CUDA event elapsed time");
            break;
        }

        state.SetIterationTime(static_cast<double>(elapsed_ms) / 1000.0);
        benchmark::DoNotOptimize(ctx.dst.data);
        benchmark::ClobberMemory();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    ReleaseContext(ctx);

    ApplyCommonCounters(state, channels, dst_w, dst_h, scale, interpolation);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(dst_w) * dst_h * channels);
}

BENCHMARK(BM_OpenCVResize)->Apply(RegisterResizeArguments);
BENCHMARK(BM_InferRTResize)->Apply(RegisterResizeArguments)->UseManualTime();
BENCHMARK(BM_InferRTResizeEndToEnd)->Apply(RegisterResizeArguments)->UseManualTime();

} // namespace

BENCHMARK_MAIN();
