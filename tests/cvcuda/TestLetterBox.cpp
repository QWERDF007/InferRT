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
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using irt::cvcuda::test::AssertInferRTSuccess;

/**
 * @brief 使用 OpenCV 在 CPU 上生成 LetterBox 参考输出
 *
 * 算法与 GPU 实现一致：等比缩放至目标画布内，居中粘贴，空白区域填充 114/255，
 * 输出为 CHW 布局的 float [0,1]；彩色通道由 BGR 转为 RGB。
 *
 * @param src 源图像（HWC，uint8）
 * @param dsize 目标画布尺寸（宽 × 高）
 * @return CHW 浮点张量，长度 ch × dsize.width × dsize.height
 */
std::vector<float> makeLetterBoxReference(const cv::Mat &src, cv::Size dsize)
{
    const int src_w = src.cols;
    const int src_h = src.rows;
    const int ch    = src.channels();

    const double r_w   = static_cast<double>(dsize.width) / src_w;
    const double r_h   = static_cast<double>(dsize.height) / src_h;
    const double ratio = std::min(r_w, r_h);

    int resized_w = std::max(1, static_cast<int>(std::round(src_w * ratio)));
    int resized_h = std::max(1, static_cast<int>(std::round(src_h * ratio)));
    resized_w     = std::min(resized_w, dsize.width);
    resized_h     = std::min(resized_h, dsize.height);

    const int left       = static_cast<int>(std::round((dsize.width - resized_w) / 2 - 0.1));
    const int top        = static_cast<int>(std::round((dsize.height - resized_h) / 2 - 0.1));
    const int plane_size = dsize.width * dsize.height;

    std::vector<float> ref(static_cast<size_t>(ch) * plane_size, 114.0f / 255.0f);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

    for (int y = 0; y < resized_h; ++y)
    {
        for (int x = 0; x < resized_w; ++x)
        {
            const int dst_idx = (top + y) * dsize.width + (left + x);
            if (ch == 1)
            {
                ref[dst_idx] = resized.at<uint8_t>(y, x) / 255.0f;
            }
            else
            {
                const cv::Vec3b pixel         = resized.at<cv::Vec3b>(y, x);
                ref[0 * plane_size + dst_idx] = pixel[2] / 255.0f;
                ref[1 * plane_size + dst_idx] = pixel[1] / 255.0f;
                ref[2 * plane_size + dst_idx] = pixel[0] / 255.0f;
            }
        }
    }

    return ref;
}

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
 * @param ch 通道数（1 或 3）
 * @param dsize 目标画布尺寸
 * @param caller LetterBox 实现（函数或类 operator()）
 */
template<typename Caller>
void runLetterBoxTest(int src_w, int src_h, int ch, cv::Size dsize, Caller caller)
{
    cv::Mat src(src_h, src_w, CV_MAKETYPE(CV_8U, ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));

    const std::vector<float> ref = makeLetterBoxReference(src, dsize);

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
 * 仅支持 1 或 3 通道；传入 ch=4 时期望失败且不崩溃。
 */
TEST(LetterBoxFunctionEdgeCaseTest, RejectsInvalidChannels)
{
    uint8_t *d_src = nullptr;
    float   *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 10 * 10 * 4), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, 32 * 32 * 4 * sizeof(float)), cudaSuccess);

    const int ret = irt::cvcuda::letterBox(d_src, d_dst, cv::Size(10, 10), cv::Size(32, 32), 4, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_src);
    cudaFree(d_dst);
}
