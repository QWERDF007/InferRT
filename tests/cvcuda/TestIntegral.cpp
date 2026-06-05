/**
 * @file TestIntegral.cpp
 * @brief Integral（积分图 / 前缀和）算子的单元测试
 *
 * 覆盖内容：
 * - 模板函数 integral<T, CT>：uint8→uint32、float→float，灰度与 BGR
 * - Integral 类封装调用
 * - 输出尺寸为 (src_w+1)×(src_h+1)，首行首列为 0 的 OpenCV 风格积分图
 * - 非法通道数等边界错误
 */

#include "TestCVCudaCommon.hpp"

#include <gtest/gtest.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/OpIntegral.h>
#include <inferrt/cvcuda/OpIntegral.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using irt::cvcuda::test::AssertInferRTSuccess;
using irt::cvcuda::test::CvDepth;

/**
 * @brief 在 CPU 上生成积分图参考结果
 *
 * 与 OpenCV cv::integral 一致：输出 (w+1)×(h+1)，第 0 行、第 0 列为 0；
 * 先对每行做水平前缀和，再对每列做垂直前缀和。
 *
 * @tparam T 源元素类型
 * @tparam CT 积分图元素类型（累加结果类型）
 * @param src 源图像（HWC，连续存储）
 * @return 长度为 (src.cols+1)×(src.rows+1)×channels 的积分图
 */
template<typename T, typename CT>
std::vector<CT> makeIntegralReference(const cv::Mat &src)
{
    const int src_w = src.cols;
    const int src_h = src.rows;
    const int ch    = src.channels();
    const int dst_w = src_w + 1;
    const int dst_h = src_h + 1;

    std::vector<CT> ref(static_cast<size_t>(dst_w) * dst_h * ch, CT{0});

    for (int y = 0; y < src_h; ++y)
    {
        const T *in_row  = src.ptr<T>(y);
        CT      *out_row = ref.data() + static_cast<size_t>(y + 1) * dst_w * ch;

        for (int c = 0; c < ch; ++c)
        {
            out_row[c] = 0;
        }

        for (int x = 0; x < src_w; ++x)
        {
            for (int c = 0; c < ch; ++c)
            {
                out_row[(x + 1) * ch + c] = out_row[x * ch + c] + static_cast<CT>(in_row[x * ch + c]);
            }
        }
    }

    for (int x = 0; x < dst_w; ++x)
    {
        for (int c = 0; c < ch; ++c)
        {
            CT sum = 0;
            for (int y = 0; y < dst_h; ++y)
            {
                CT *row = ref.data() + static_cast<size_t>(y) * dst_w * ch;
                sum += row[x * ch + c];
                row[x * ch + c] = sum;
            }
        }
    }

    return ref;
}

/**
 * @brief 公共 Integral 测试逻辑
 *
 * 流程：
 * 1. 按 T 类型生成随机源图
 * 2. CPU 参考实现得到期望积分图
 * 3. 分配 GPU 缓冲并 H2D 上传
 * 4. 通过 Caller 调用 GPU integral
 * 5. D2H 下载并与参考比较（允许 max_diff 误差）
 *
 * @tparam T 源数据类型
 * @tparam CT 积分图数据类型
 * @tparam Caller 可调用对象，签名为 (const T*, CT*, cv::Size, int, cudaStream_t) -> int
 * @param src_w 源图宽度
 * @param src_h 源图高度
 * @param ch 通道数（1 或 3）
 * @param max_diff 允许的最大绝对误差
 * @param caller integral 实现（函数或类 operator()）
 */
template<typename T, typename CT, typename Caller>
void runIntegralTest(int src_w, int src_h, int ch, double max_diff, Caller caller)
{
    cv::Mat src(src_h, src_w, CV_MAKETYPE(CvDepth<T>::value, ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(CvDepth<T>::range));

    const std::vector<CT> ref = makeIntegralReference<T, CT>(src);

    const size_t src_bytes = static_cast<size_t>(src_w) * src_h * ch * sizeof(T);
    const size_t dst_bytes = static_cast<size_t>(src_w + 1) * (src_h + 1) * ch * sizeof(CT);

    T  *d_src = nullptr;
    CT *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_bytes), cudaSuccess);

    cv::Mat src_cont = src.isContinuous() ? src : src.clone();
    ASSERT_EQ(cudaMemcpy(d_src, src_cont.data, src_bytes, cudaMemcpyHostToDevice), cudaSuccess);

    const int ret = caller(d_src, d_dst, cv::Size(src_w, src_h), ch, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<CT> dst(ref.size());
    ASSERT_EQ(cudaMemcpy(dst.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    double actual_max_diff = 0.0;
    for (size_t i = 0; i < dst.size(); ++i)
    {
        actual_max_diff
            = std::max(actual_max_diff, std::abs(static_cast<double>(dst[i]) - static_cast<double>(ref[i])));
    }
    EXPECT_LE(actual_max_diff, max_diff);
}

} // namespace

// ============================================================================
// 功能正确性测试
// ============================================================================

/**
 * @brief 测试 integral 函数：uint8 灰度 → uint32 积分图
 *
 * 31×17 单通道，整数累加要求与参考完全一致（max_diff=0）。
 */
TEST(IntegralFunctionTest, U8GrayToU32)
{
    runIntegralTest<uint8_t, uint32_t>(
        31, 17, 1, 0.0, [](const uint8_t *src, uint32_t *dst, cv::Size ssize, int ch, cudaStream_t stream)
        { return irt::cvcuda::integral<uint8_t, uint32_t>(src, dst, ssize, ch, stream); });
}

/**
 * @brief 测试 integral 函数：uint8 BGR → uint32 积分图
 *
 * 29×23 三通道，逐通道独立前缀和，误差阈值为 0。
 */
TEST(IntegralFunctionTest, U8BgrToU32)
{
    runIntegralTest<uint8_t, uint32_t>(
        29, 23, 3, 0.0, [](const uint8_t *src, uint32_t *dst, cv::Size ssize, int ch, cudaStream_t stream)
        { return irt::cvcuda::integral<uint8_t, uint32_t>(src, dst, ssize, ch, stream); });
}

/**
 * @brief 测试 integral 函数：float BGR → float 积分图
 *
 * 37×19 三通道浮点源，允许 1e-5 浮点误差。
 */
TEST(IntegralFunctionTest, F32BgrToF32)
{
    runIntegralTest<float, float>(37, 19, 3, 1e-5,
                                  [](const float *src, float *dst, cv::Size ssize, int ch, cudaStream_t stream)
                                  { return irt::cvcuda::integral<float, float>(src, dst, ssize, ch, stream); });
}

/**
 * @brief 测试 Integral 类：float 灰度 → float 积分图
 *
 * 41×13 单通道，通过 Integral<float,float> 实例 operator() 调用。
 */
TEST(IntegralClassTest, F32GrayToF32)
{
    irt::cvcuda::Integral<float, float> op;
    runIntegralTest<float, float>(41, 13, 1, 1e-5,
                                  [&op](const float *src, float *dst, cv::Size ssize, int ch, cudaStream_t stream)
                                  { return op(src, dst, ssize, ch, stream); });
}

// ============================================================================
// 边界与错误处理测试
// ============================================================================

/**
 * @brief 测试非法通道数：应返回 IRT_ERROR_INVALID_ARGUMENT
 *
 * integral 仅支持 1 或 3 通道；传入 ch=4 时期望失败。
 */
TEST(IntegralFunctionEdgeCaseTest, RejectsInvalidChannels)
{
    uint8_t  *d_src = nullptr;
    uint32_t *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 10 * 10 * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, 11 * 11 * 4 * sizeof(uint32_t)), cudaSuccess);

    const int ret = irt::cvcuda::integral<uint8_t, uint32_t>(d_src, d_dst, cv::Size(10, 10), 4, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_src);
    cudaFree(d_dst);
}
