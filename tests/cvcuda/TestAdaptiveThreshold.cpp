/**
 * @file TestAdaptiveThreshold.cpp
 * @brief AdaptiveThreshold 自适应阈值算子的单元测试。
 *
 * 覆盖 mean、gaussian、percentage 三种自适应阈值策略，以及普通二值和反向二值两种
 * threshold 类型；同时验证关键非法输入会返回预期的 InferRT 错误码。
 */

#include "TestCVCudaCommon.hpp"

#include <gtest/gtest.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/OpAdaptiveThreshold.h>
#include <inferrt/cvcuda/OpAdaptiveThreshold.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using irt::cvcuda::test::AssertInferRTSuccess;

/**
 * @brief 使用 BORDER_REPLICATE 规则裁剪一维坐标。
 *
 * @param p 原始坐标。
 * @param len 有效长度。
 * @return 裁剪后的合法坐标。
 */
int borderReplicate(int p, int len)
{
    return std::min(std::max(p, 0), len - 1);
}

/**
 * @brief 生成二维 Gaussian 自适应阈值权重。
 *
 * 使用 OpenCV 的一维高斯核做外积，得到与 CUDA gaussian 分支输入一致的二维权重表。
 *
 * @param block_size 自适应窗口尺寸，必须为奇数。
 * @return 行优先布局的二维高斯权重。
 */
std::vector<float> makeGaussianWeights(int block_size)
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
 * @brief 在 CPU 上生成 AdaptiveThreshold 参考结果。
 *
 * 参考实现显式复现 CUDA kernel 的边界处理、delta 取整策略和 percentage 判断规则，
 * 用于逐元素验证 GPU 输出。
 *
 * @param src HWC 布局的 uint8 输入图像。
 * @param maxval 阈值命中时写入的最大值。
 * @param adaptive_method 自适应阈值方法。
 * @param threshold_type 二值化类型。
 * @param block_size 自适应窗口尺寸。
 * @param param mean/gaussian 分支表示 C，percentage 分支表示比例。
 * @param weights gaussian 分支使用的二维权重表。
 * @return 与输入尺寸和通道数一致的 HWC uint8 参考输出。
 */
std::vector<uint8_t> makeAdaptiveThresholdReference(const cv::Mat &src, double maxval, int adaptive_method,
                                                    int threshold_type, int block_size, double param,
                                                    const std::vector<float> &weights = {})
{
    const int     width     = src.cols;
    const int     height    = src.rows;
    const int     channels  = src.channels();
    const int     radius    = block_size / 2;
    const uint8_t max_value = cv::saturate_cast<uint8_t>(maxval);
    const bool    inverse   = threshold_type == cv::THRESH_BINARY_INV;
    const int     delta     = inverse ? static_cast<int>(std::floor(param)) : static_cast<int>(std::ceil(param));

    std::vector<uint8_t> ref(static_cast<size_t>(width) * height * channels);

    for (int y = 0; y < height; ++y)
    {
        const uint8_t *src_row = src.ptr<uint8_t>(y);
        for (int x = 0; x < width; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                const int     base      = (y * width + x) * channels + c;
                const uint8_t src_value = src_row[x * channels + c];

                if (adaptive_method == irt::cvcuda::ADAPTIVE_THRESH_PERCENTAGE)
                {
                    const int y1    = std::max(y - radius, 0);
                    const int y2    = std::min(y + radius, height - 1);
                    const int x1    = std::max(x - radius, 0);
                    const int x2    = std::min(x + radius, width - 1);
                    const int count = (x2 - x1 + 1) * (y2 - y1 + 1);

                    uint32_t sum = 0;
                    for (int yy = y1; yy <= y2; ++yy)
                    {
                        const uint8_t *row = src.ptr<uint8_t>(yy);
                        for (int xx = x1; xx <= x2; ++xx)
                        {
                            sum += row[xx * channels + c];
                        }
                    }

                    const bool below = static_cast<double>(src_value) * count < static_cast<double>(sum) * param;
                    ref[base]        = (below == inverse) ? max_value : 0;
                    continue;
                }

                double sum = 0.0;
                for (int ky = -radius; ky <= radius; ++ky)
                {
                    const int      yy  = borderReplicate(y + ky, height);
                    const uint8_t *row = src.ptr<uint8_t>(yy);
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        const int     xx    = borderReplicate(x + kx, width);
                        const uint8_t value = row[xx * channels + c];
                        if (adaptive_method == cv::ADAPTIVE_THRESH_GAUSSIAN_C)
                        {
                            sum += weights[static_cast<size_t>(ky + radius) * block_size + (kx + radius)] * value;
                        }
                        else
                        {
                            sum += value;
                        }
                    }
                }

                const double  mean_value = adaptive_method == cv::ADAPTIVE_THRESH_GAUSSIAN_C
                                             ? sum
                                             : sum / static_cast<double>(block_size * block_size);
                const uint8_t mean       = cv::saturate_cast<uint8_t>(mean_value);
                const bool    foreground = static_cast<int>(src_value) > static_cast<int>(mean) - delta;
                ref[base]                = (foreground != inverse) ? max_value : 0;
            }
        }
    }

    return ref;
}

/**
 * @brief 运行 AdaptiveThreshold 的通用正确性测试。
 *
 * 测试流程包括随机输入生成、CPU 参考结果生成、GPU 内存分配、算子调用、结果下载和逐元素比较。
 *
 * @tparam Caller 可调用对象类型，签名与 adaptiveThreshold/AdaptiveThreshold::operator() 保持一致。
 */
template<typename Caller>
void runAdaptiveThresholdTest(int width, int height, int channels, double maxval, int adaptive_method,
                              int threshold_type, int block_size, double param, Caller caller,
                              const std::vector<float> &weights = {})
{
    cv::Mat src(height, width, CV_MAKETYPE(CV_8U, channels));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));

    const std::vector<uint8_t> ref
        = makeAdaptiveThresholdReference(src, maxval, adaptive_method, threshold_type, block_size, param, weights);

    const size_t bytes         = static_cast<size_t>(width) * height * channels * sizeof(uint8_t);
    const size_t weights_bytes = weights.size() * sizeof(float);

    uint8_t *d_src     = nullptr;
    uint8_t *d_dst     = nullptr;
    float   *d_weights = nullptr;

    ASSERT_EQ(cudaMalloc(&d_src, bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, bytes), cudaSuccess);
    if (!weights.empty())
    {
        ASSERT_EQ(cudaMalloc(&d_weights, weights_bytes), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_weights, weights.data(), weights_bytes, cudaMemcpyHostToDevice), cudaSuccess);
    }

    cv::Mat src_cont = src.isContinuous() ? src : src.clone();
    ASSERT_EQ(cudaMemcpy(d_src, src_cont.data, bytes, cudaMemcpyHostToDevice), cudaSuccess);

    const int ret = caller(d_src, d_dst, cv::Size(width, height), channels, maxval, adaptive_method, threshold_type,
                           block_size, param, d_weights, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint8_t> dst(ref.size());
    ASSERT_EQ(cudaMemcpy(dst.data(), d_dst, bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);
    if (d_weights != nullptr)
    {
        cudaFree(d_weights);
    }

    EXPECT_EQ(dst, ref);
}

} // namespace

/**
 * @brief 验证 mean 自适应阈值的单通道普通二值输出。
 */
TEST(AdaptiveThresholdFunctionTest, MeanGrayBinary)
{
    runAdaptiveThresholdTest(
        31, 17, 1, 255.0, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 5, 3.0,
        [](const uint8_t *src, uint8_t *dst, cv::Size size, int ch, double maxval, int adaptive_method,
           int threshold_type, int block_size, double param, const float *weights, cudaStream_t stream)
        {
            return irt::cvcuda::adaptiveThreshold<uint8_t>(src, dst, size, ch, maxval, adaptive_method, threshold_type,
                                                           block_size, param, weights, stream);
        });
}

/**
 * @brief 验证 mean 自适应阈值的三通道反向二值输出。
 */
TEST(AdaptiveThresholdFunctionTest, MeanBgrBinaryInv)
{
    runAdaptiveThresholdTest(
        23, 19, 3, 200.0, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 3, 2.0,
        [](const uint8_t *src, uint8_t *dst, cv::Size size, int ch, double maxval, int adaptive_method,
           int threshold_type, int block_size, double param, const float *weights, cudaStream_t stream)
        {
            return irt::cvcuda::adaptiveThreshold<uint8_t>(src, dst, size, ch, maxval, adaptive_method, threshold_type,
                                                           block_size, param, weights, stream);
        });
}

/**
 * @brief 验证 AdaptiveThreshold 类封装的 gaussian 单通道普通二值输出。
 */
TEST(AdaptiveThresholdClassTest, GaussianGrayBinary)
{
    const std::vector<float>                weights = makeGaussianWeights(5);
    irt::cvcuda::AdaptiveThreshold<uint8_t> op;

    runAdaptiveThresholdTest(
        29, 21, 1, 255.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 5, 1.0,
        [&op](const uint8_t *src, uint8_t *dst, cv::Size size, int ch, double maxval, int adaptive_method,
              int threshold_type, int block_size, double param, const float *weights_ptr, cudaStream_t stream)
        {
            return op(src, dst, size, ch, maxval, adaptive_method, threshold_type, block_size, param, weights_ptr,
                      stream);
        },
        weights);
}

/**
 * @brief 验证 percentage 自适应阈值的三通道反向二值输出。
 */
TEST(AdaptiveThresholdFunctionTest, PercentageBgrBinaryInv)
{
    runAdaptiveThresholdTest(
        27, 25, 3, 180.0, irt::cvcuda::ADAPTIVE_THRESH_PERCENTAGE, cv::THRESH_BINARY_INV, 7, 0.85,
        [](const uint8_t *src, uint8_t *dst, cv::Size size, int ch, double maxval, int adaptive_method,
           int threshold_type, int block_size, double param, const float *weights, cudaStream_t stream)
        {
            return irt::cvcuda::adaptiveThreshold<uint8_t>(src, dst, size, ch, maxval, adaptive_method, threshold_type,
                                                           block_size, param, weights, stream);
        });
}

/**
 * @brief gaussian 分支缺少权重指针时应返回 IRT_ERROR_INVALID_ARGUMENT。
 */
TEST(AdaptiveThresholdFunctionEdgeCaseTest, RejectsGaussianWithoutWeights)
{
    uint8_t *d_src = nullptr;
    uint8_t *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 10 * 10), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, 10 * 10), cudaSuccess);

    const int ret = irt::cvcuda::adaptiveThreshold<uint8_t>(d_src, d_dst, cv::Size(10, 10), 1, 255.0,
                                                            cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 3, 1.0,
                                                            nullptr, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_src);
    cudaFree(d_dst);
}

/**
 * @brief 非法通道数应返回 IRT_ERROR_INVALID_ARGUMENT。
 */
TEST(AdaptiveThresholdFunctionEdgeCaseTest, RejectsInvalidChannels)
{
    uint8_t *d_src = nullptr;
    uint8_t *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 10 * 10 * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, 10 * 10 * 4), cudaSuccess);

    const int ret
        = irt::cvcuda::adaptiveThreshold<uint8_t>(d_src, d_dst, cv::Size(10, 10), 4, 255.0, cv::ADAPTIVE_THRESH_MEAN_C,
                                                  cv::THRESH_BINARY, 3, 1.0, nullptr, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_src);
    cudaFree(d_dst);
}
