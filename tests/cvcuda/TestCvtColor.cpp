#include <gtest/gtest.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/OpCvtColor.h>
#include <inferrt/cvcuda/OpCvtColor.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

/**
 * @brief 断言 InferRT 调用成功
 *
 * 若返回值非 IRT_SUCCESS，则读取并附加最近一次错误信息。
 *
 * @param ret InferRT 状态码
 * @return 成功时返回 AssertionSuccess，否则返回带错误详情的 AssertionFailure
 */
static ::testing::AssertionResult AssertInferRTSuccess(int ret)
{
    if (ret == IRT_SUCCESS)
    {
        return ::testing::AssertionSuccess();
    }

    char msg[IRT_MAX_STATUS_MESSAGE_LENGTH] = {};
    irt::PeekAtLastErrorMessage(msg, sizeof(msg));
    return ::testing::AssertionFailure() << "ret=" << ret << " (" << irt::StatusGetName(static_cast<IRTStatus>(ret))
                                         << "), last_error=" << msg;
}

// ============================================================================
// 类型映射
// ============================================================================

/**
 * @brief 将 C++ 类型映射到对应的 OpenCV 深度
 *
 * @tparam T 数据类型（uint8_t 或 float）
 */
template<typename T>
struct CvDepth;

/**
 * @brief uint8_t 类型的 OpenCV 深度映射
 *
 * - value: CV_8U（8 位无符号整数）
 * - range: 255.0（随机值生成的上限）
 */
template<>
struct CvDepth<uint8_t>
{
    static constexpr int value = CV_8U;
    static constexpr double range = 255.0;
};

/**
 * @brief float 类型的 OpenCV 深度映射
 *
 * - value: CV_32F（32 位浮点数）
 * - range: 1.0（随机值生成的上限）
 */
template<>
struct CvDepth<float>
{
    static constexpr int value = CV_32F;
    static constexpr double range = 1.0;
};

// ============================================================================
// 公共测试逻辑
// ============================================================================

/**
 * @brief 公共 cvtColor 测试逻辑
 *
 * 该函数执行完整的 GPU 颜色空间转换测试流程：
 * 1. 创建随机源图像
 * 2. 使用 OpenCV 生成参考结果
 * 3. 分配 GPU 内存并上传数据
 * 4. 执行 GPU cvtColor 操作（通过 Caller 调用）
 * 5. 下载结果并与参考结果比较
 *
 * @tparam T 数据类型（uint8_t 或 float）
 * @tparam Caller 调用器类型（函数指针或 lambda，签名与 cvtColor 一致）
 * @param width 源图像宽度
 * @param height 源图像高度
 * @param src_ch 源图像通道数
 * @param code OpenCV 颜色转换码（如 cv::COLOR_BGR2RGB）
 * @param max_diff 允许的最大像素差异
 * @param caller 实际执行 cvtColor 的可调用对象
 */
template<typename T, typename Caller>
void runCvtColorTest(int width, int height, int src_ch, int code, double max_diff, Caller caller)
{
    cv::Mat src(height, width, CV_MAKETYPE(CvDepth<T>::value, src_ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(CvDepth<T>::range));

    cv::Mat ref;
    cv::cvtColor(src, ref, code);

    const size_t src_bytes = src.total() * src.channels() * sizeof(T);
    const size_t dst_bytes = ref.total() * ref.channels() * sizeof(T);

    T *d_src = nullptr;
    T *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_bytes), cudaSuccess);

    cv::Mat src_cont = src.isContinuous() ? src : src.clone();
    ASSERT_EQ(cudaMemcpy(d_src, src_cont.data, src_bytes, cudaMemcpyHostToDevice), cudaSuccess);

    const int ret = caller(d_src, d_dst, cv::Size(width, height), code, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<T> dst(dst_bytes / sizeof(T));
    ASSERT_EQ(cudaMemcpy(dst.data(), d_dst, dst_bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    const T *ref_data = reinterpret_cast<const T *>(ref.data);
    double actual_max_diff = 0.0;
    for (size_t i = 0; i < dst.size(); ++i)
    {
        actual_max_diff = std::max(actual_max_diff, std::abs(static_cast<double>(dst[i]) - static_cast<double>(ref_data[i])));
    }
    EXPECT_LE(actual_max_diff, max_diff);
}

} // namespace

// ============================================================================
// 函数版本测试
// ============================================================================

/**
 * @brief 测试 uint8_t 类型 BGR 转 RGB
 *
 * 验证 8 位无符号整数图像的通道交换，期望与 OpenCV 结果完全一致（max_diff = 0）。
 */
TEST(CvtColorFunctionTest, U8BgrToRgb)
{
    runCvtColorTest<uint8_t>(31, 17, 3, cv::COLOR_BGR2RGB, 0.0,
                             [](const uint8_t *src, uint8_t *dst, cv::Size size, int code, cudaStream_t stream)
                             { return irt::cvcuda::cvtColor<uint8_t>(src, dst, size, code, stream); });
}

/**
 * @brief 测试 uint8_t 类型 BGR 转灰度
 *
 * 验证 8 位 BGR 转灰度图，允许最大像素差异为 1.0（量化舍入误差）。
 */
TEST(CvtColorFunctionTest, U8BgrToGray)
{
    runCvtColorTest<uint8_t>(29, 23, 3, cv::COLOR_BGR2GRAY, 1.0,
                             [](const uint8_t *src, uint8_t *dst, cv::Size size, int code, cudaStream_t stream)
                             { return irt::cvcuda::cvtColor<uint8_t>(src, dst, size, code, stream); });
}

/**
 * @brief 测试 uint8_t 类型灰度转 BGRA
 *
 * 验证单通道灰度图扩展为 4 通道 BGRA，期望与 OpenCV 结果完全一致。
 */
TEST(CvtColorFunctionTest, U8GrayToBgra)
{
    runCvtColorTest<uint8_t>(19, 37, 1, cv::COLOR_GRAY2BGRA, 0.0,
                             [](const uint8_t *src, uint8_t *dst, cv::Size size, int code, cudaStream_t stream)
                             { return irt::cvcuda::cvtColor<uint8_t>(src, dst, size, code, stream); });
}

/**
 * @brief 测试 float 类型 BGR 转 YCrCb
 *
 * 验证 32 位浮点图像的颜色空间转换，允许最大像素差异为 1e-3（浮点精度误差）。
 */
TEST(CvtColorFunctionTest, F32BgrToYCrCb)
{
    runCvtColorTest<float>(21, 25, 3, cv::COLOR_BGR2YCrCb, 1e-3,
                           [](const float *src, float *dst, cv::Size size, int code, cudaStream_t stream)
                           { return irt::cvcuda::cvtColor<float>(src, dst, size, code, stream); });
}

// ============================================================================
// 类版本测试
// ============================================================================

/**
 * @brief 测试 float 类型的 CvtColor 类
 *
 * 通过 irt::cvcuda::CvtColor<float> 的 operator() 执行 BGR 转 RGB，
 * 验证类封装与函数版本行为一致。
 */
TEST(CvtColorClassTest, F32BgrToRgb)
{
    irt::cvcuda::CvtColor<float> op;
    runCvtColorTest<float>(33, 11, 3, cv::COLOR_BGR2RGB, 0.0,
                           [&op](const float *src, float *dst, cv::Size size, int code, cudaStream_t stream)
                           { return op(src, dst, size, code, stream); });
}

// ============================================================================
// 边界与异常测试
// ============================================================================

/**
 * @brief 测试不支持的颜色转换码
 *
 * 传入无效 code（-1），期望返回 IRT_ERROR_NOT_IMPLEMENTED。
 */
TEST(CvtColorFunctionEdgeCaseTest, RejectsUnsupportedCode)
{
    uint8_t *d_src = nullptr;
    uint8_t *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 10 * 10 * 3), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, 10 * 10 * 3), cudaSuccess);

    const int ret = irt::cvcuda::cvtColor<uint8_t>(d_src, d_dst, cv::Size(10, 10), -1, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_NOT_IMPLEMENTED);

    cudaFree(d_src);
    cudaFree(d_dst);
}
