#include <gtest/gtest.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/OpResize.h>
#include <inferrt/cvcuda/OpResize.hpp>
#include <opencv2/opencv.hpp>
#include <vtest/common/ValueTests.hpp>

#include <type_traits>

/**
 * @brief 多参数组合测试套件
 * 
 * 测试参数组合：
 * - src_width: 源图像宽度 {1, 2, 10, 100, 256}
 * - src_height: 源图像高度 {1, 2, 10, 100, 256}
 * - channels: 通道数 {1, 3, 4}
 * - scale: 缩放比例 {0.1, 0.3, 0.5, 0.7, 1.3, 2.0, 3.0, 5.0, 10.0}
 * - interp: 插值方法 {cv::INTER_LINEAR}
 *
 * @note 此组合测试覆盖：极小图像、相同尺寸、极端上/下采样率(0.1,10)、多通道
 */
_TEST_SUITE_P(MultiParamTest,
              vtest::ValueList<int>{10, 25, 100} * vtest::ValueList<int>{10, 25, 100} * vtest::ValueList<int>{1, 3, 4}
                  * vtest::ValueList<double>{0.1, 0.3, 0.5, 1.3, 2.0, 10.0}
                  * vtest::ValueList<int>{cv::INTER_NEAREST, cv::INTER_LINEAR, cv::INTER_CUBIC, cv::INTER_AREA,
                                          cv::INTER_LANCZOS4, cv::INTER_NEAREST_EXACT, cv::INTER_LINEAR_EXACT});

// ============================================================================
// 类型映射和调用器抽象
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
 * - value: CV_8U（8位无符号整数）
 * - range: 255.0（随机值生成的上限）
 */
template<>
struct CvDepth<uint8_t>
{
    static constexpr int    value = CV_8U;
    static constexpr double range = 255.0;
};

/**
 * @brief float 类型的 OpenCV 深度映射
 * 
 * - value: CV_32F（32位浮点数）
 * - range: 1.0（随机值生成的上限）
 */
template<>
struct CvDepth<float>
{
    static constexpr int    value = CV_32F;
    static constexpr double range = 1.0;
};

/**
 * @brief Resize 操作调用器 - 函数版本
 * 
 * 封装 resize 函数调用，用于统一测试接口。
 */
template<typename T>
struct ResizeFunctionCaller
{
    int operator()(const T *d_src, T *d_dst, cv::Size ssize, cv::Size dsize, int ch, int interp,
                   cudaStream_t stream) const
    {
        return irt::cvcuda::resize<T>(d_src, d_dst, ssize, dsize, ch, interp, stream);
    }
};

/**
 * @brief Resize 操作调用器 - 类版本
 * 
 * 封装 Resize 类的 operator() 调用，用于统一测试接口。
 */
template<typename T>
struct ResizeClassCaller
{
    int operator()(const T *d_src, T *d_dst, cv::Size ssize, cv::Size dsize, int ch, int interp,
                   cudaStream_t stream) const
    {
        irt::cvcuda::Resize<T> resize_op;
        return resize_op(d_src, d_dst, ssize, dsize, ch, interp, stream);
    }
};

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

template<typename T>
bool ShouldSkipExactInterpolation(int interp)
{
    return std::is_same_v<T, float> && (interp == cv::INTER_NEAREST_EXACT || interp == cv::INTER_LINEAR_EXACT);
}

// ============================================================================
// 公共测试逻辑
// ============================================================================

/**
 * @brief 公共 resize 测试逻辑
 * 
 * 该函数执行完整的 GPU resize 测试流程：
 * 1. 创建随机源图像
 * 2. 使用 OpenCV 生成参考结果
 * 3. 分配 GPU 内存并上传数据
 * 4. 执行 GPU resize 操作（通过 Caller 调用）
 * 5. 下载结果并与参考结果比较
 * 
 * @tparam T 数据类型（uint8_t 或 float）
 * @tparam Caller 调用器类型（ResizeFunctionCaller 或 ResizeClassCaller）
 * @param src_w 源图像宽度
 * @param src_h 源图像高度
 * @param ch 通道数
 * @param scale 缩放比例
 * @param interp 插值方法（如 cv::INTER_LINEAR）
 * @param max_diff 允许的最大像素差异
 */
template<typename T, typename Caller>
void runResizeTest(int src_w, int src_h, int ch, double scale, int interp, double max_diff)
{
    if (ShouldSkipExactInterpolation<T>(interp))
    {
        GTEST_SKIP() << "Skipping *_EXACT interpolation for float";
    }

    const int dst_w = std::max(1, static_cast<int>(src_w * scale));
    const int dst_h = std::max(1, static_cast<int>(src_h * scale));

    // 创建随机源图像
    cv::Mat src(src_h, src_w, CV_MAKETYPE(CvDepth<T>::value, ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(CvDepth<T>::range));

    // OpenCV 参考结果
    cv::Mat ref;
    cv::resize(src, ref, cv::Size(dst_w, dst_h), 0, 0, interp);

    // 分配 GPU 内存
    const size_t src_bytes = src_h * src_w * ch * sizeof(T);
    const size_t dst_bytes = dst_h * dst_w * ch * sizeof(T);

    T *d_src = nullptr;
    T *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_bytes), cudaSuccess);

    // 上传源数据（确保连续）
    cv::Mat src_cont = src.isContinuous() ? src : src.clone();
    ASSERT_EQ(cudaMemcpy(d_src, src_cont.data, src_bytes, cudaMemcpyHostToDevice), cudaSuccess);

    // 执行 resize（通过 Caller 调用）
    Caller caller;
    int    ret = caller(d_src, d_dst, cv::Size(src_w, src_h), cv::Size(dst_w, dst_h), ch, interp, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // 下载结果
    cv::Mat dst(dst_h, dst_w, CV_MAKETYPE(CvDepth<T>::value, ch));
    ASSERT_EQ(cudaMemcpy(dst.data, d_dst, dst_bytes, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    // 比较最大差异
    cv::Mat diff;
    cv::absdiff(dst, ref, diff);
    double max_val = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_val);
    EXPECT_LE(max_val, max_diff) << "src=" << src_w << "x" << src_h << " ch=" << ch << " scale=" << scale
                                 << " dst=" << dst_w << "x" << dst_h;
}

// ============================================================================
// 参数化测试 - 函数版本
// ============================================================================

/**
 * @brief 测试 uint8_t 类型的 resize 函数
 * 
 * 使用参数化测试验证 8 位无符号整数图像的 resize 功能。
 * 允许的最大像素差异为 1.0（由于量化误差）。
 */
TEST_P(MultiParamTest, CFunc8UTest)
{
    runResizeTest<uint8_t, ResizeFunctionCaller<uint8_t>>(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                                          GetParamValue<3>(), GetParamValue<4>(), 1.0);
}

/**
 * @brief 测试 float 类型的 resize 函数
 *
 * 使用参数化测试验证 32 位浮点数图像的 resize 功能。
 * 允许的最大像素差异为 1e-3（浮点精度误差）。
 */
TEST_P(MultiParamTest, CFunc32FTest)
{
    runResizeTest<float, ResizeFunctionCaller<float>>(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                                      GetParamValue<3>(), GetParamValue<4>(), 1e-3);
}

// ============================================================================
// 参数化测试 - 类版本
// ============================================================================

/**
 * @brief 测试 uint8_t 类型的 Resize 类
 *
 * 使用参数化测试验证 8 位无符号整数图像的 Resize 类功能。
 * 允许的最大像素差异为 1.0（由于量化误差）。
 */
TEST_P(MultiParamTest, CppClass8UTest)
{
    runResizeTest<uint8_t, ResizeClassCaller<uint8_t>>(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                                       GetParamValue<3>(), GetParamValue<4>(), 1.0);
}

/**
 * @brief 测试 float 类型的 Resize 类
 *
 * 使用参数化测试验证 32 位浮点数图像的 Resize 类功能。
 * 允许的最大像素差异为 1e-3（浮点精度误差）。
 */
TEST_P(MultiParamTest, CppClass32FTest)
{
    runResizeTest<float, ResizeClassCaller<float>>(GetParamValue<0>(), GetParamValue<1>(), GetParamValue<2>(),
                                                   GetParamValue<3>(), GetParamValue<4>(), 1e-3);
}

// ============================================================================
// 边界值和异常输入测试（函数版本）
// ============================================================================

/**
 * @brief 测试未实现的插值方法
 *
 * 验证当插值方法为 -1（无效值）时，函数返回 IRT_ERROR_NOT_IMPLEMENTED。
 */
TEST(ResizeFunctionEdgeCaseTest, NotImplementedMethod)
{
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 100 * 100 * 3), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, 100 * 100 * 3), cudaSuccess);

    int ret = irt::cvcuda::resize<uint8_t>(d_src, d_dst, cv::Size(100, 100), cv::Size(100, 100), 3, -1, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_NOT_IMPLEMENTED);

    cudaFree(d_src);
    cudaFree(d_dst);
}

/**
 * @brief 测试空指针输入 - 源指针为空
 *
 * 验证当源图像指针为 nullptr 时，函数返回 IRT_ERROR_INVALID_ARGUMENT。
 */
TEST(ResizeFunctionEdgeCaseTest, NullSourcePointer)
{
    uint8_t *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_dst, 100 * 100 * 3), cudaSuccess);

    int ret = irt::cvcuda::resize<uint8_t>(nullptr, d_dst, cv::Size(10, 10), cv::Size(100, 100), 3, cv::INTER_LINEAR,
                                           nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    EXPECT_EQ(irt::GetLastError(), IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_dst);
}

/**
 * @brief 测试空指针输入 - 目标指针为空
 *
 * 验证当目标图像指针为 nullptr 时，函数返回 IRT_ERROR_INVALID_ARGUMENT。
 */
TEST(ResizeFunctionEdgeCaseTest, NullDestinationPointer)
{
    uint8_t *d_src = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 10 * 10 * 3), cudaSuccess);

    int ret = irt::cvcuda::resize<uint8_t>(d_src, nullptr, cv::Size(10, 10), cv::Size(100, 100), 3, cv::INTER_LINEAR,
                                           nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_src);
}

/**
 * @brief 测试零尺寸输入 - 源图像宽度为 0
 *
 * 验证当源图像宽度为 0 时，函数返回 IRT_ERROR_INVALID_ARGUMENT。
 */
TEST(ResizeFunctionEdgeCaseTest, ZeroSourceWidth)
{
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 100), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, 100), cudaSuccess);

    int ret
        = irt::cvcuda::resize<uint8_t>(d_src, d_dst, cv::Size(0, 10), cv::Size(10, 10), 1, cv::INTER_LINEAR, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_src);
    cudaFree(d_dst);
}

/**
 * @brief 测试无效通道数 - 通道数为 0
 *
 * 验证当通道数为 0 时，函数返回 IRT_ERROR_INVALID_ARGUMENT。
 */
TEST(ResizeFunctionEdgeCaseTest, ZeroChannels)
{
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 100), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, 100), cudaSuccess);

    int ret
        = irt::cvcuda::resize<uint8_t>(d_src, d_dst, cv::Size(10, 10), cv::Size(10, 10), 0, cv::INTER_LINEAR, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_src);
    cudaFree(d_dst);
}

// ============================================================================
// 边界值和异常输入测试（类版本）
// ============================================================================

/**
 * @brief 测试空指针输入 - 源指针为空（类版本）
 *
 * 验证当源图像指针为 nullptr 时，Resize 类返回 IRT_ERROR_INVALID_ARGUMENT。
 */
TEST(ResizeClassEdgeCaseTest, NullSourcePointer)
{
    uint8_t *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_dst, 100 * 100 * 3), cudaSuccess);

    irt::cvcuda::Resize<uint8_t> resize_op;
    int ret = resize_op(nullptr, d_dst, cv::Size(10, 10), cv::Size(100, 100), 3, cv::INTER_LINEAR, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_dst);
}

/**
 * @brief 测试空指针输入 - 目标指针为空（类版本）
 *
 * 验证当目标图像指针为 nullptr 时，Resize 类返回 IRT_ERROR_INVALID_ARGUMENT。
 */
TEST(ResizeClassEdgeCaseTest, NullDestinationPointer)
{
    uint8_t *d_src = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 10 * 10 * 3), cudaSuccess);

    irt::cvcuda::Resize<uint8_t> resize_op;
    int ret = resize_op(d_src, nullptr, cv::Size(10, 10), cv::Size(100, 100), 3, cv::INTER_LINEAR, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_src);
}

/**
 * @brief 测试零尺寸输入 - 源图像宽度为 0（类版本）
 *
 * 验证当源图像宽度为 0 时，Resize 类返回 IRT_ERROR_INVALID_ARGUMENT。
 */
TEST(ResizeClassEdgeCaseTest, ZeroSourceWidth)
{
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 100), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, 100), cudaSuccess);

    irt::cvcuda::Resize<uint8_t> resize_op;
    int ret = resize_op(d_src, d_dst, cv::Size(0, 10), cv::Size(10, 10), 1, cv::INTER_LINEAR, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_src);
    cudaFree(d_dst);
}

/**
 * @brief 测试无效通道数 - 通道数为 0（类版本）
 *
 * 验证当通道数为 0 时，Resize 类返回 IRT_ERROR_INVALID_ARGUMENT。
 */
TEST(ResizeClassEdgeCaseTest, ZeroChannels)
{
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 100), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, 100), cudaSuccess);

    irt::cvcuda::Resize<uint8_t> resize_op;
    int ret = resize_op(d_src, d_dst, cv::Size(10, 10), cv::Size(10, 10), 0, cv::INTER_LINEAR, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);

    cudaFree(d_src);
    cudaFree(d_dst);
}

/**
 * @brief 测试未实现的插值方法（类版本）
 *
 * 验证当插值方法为 -1（无效值）时，Resize 类返回 IRT_ERROR_NOT_IMPLEMENTED。
 */
TEST(ResizeClassEdgeCaseTest, NotImplementedMethod)
{
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, 100 * 100 * 3), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, 100 * 100 * 3), cudaSuccess);

    irt::cvcuda::Resize<uint8_t> resize_op;
    int                          ret = resize_op(d_src, d_dst, cv::Size(100, 100), cv::Size(100, 100), 3, -1, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_NOT_IMPLEMENTED);

    cudaFree(d_src);
    cudaFree(d_dst);
}

// ============================================================================
// 极端缩放和非对称缩放测试（函数版本）
// ============================================================================

/**
 * @brief 测试极端下采样 - 100x -> 1x（函数版本）
 *
 * 验证从大图像缩小到极小图像时的正确性。
 * 这测试了下采样算法在极端情况下的稳定性。
 */
TEST(ResizeFunctionEdgeCaseTest, ExtremeDownscale)
{
    const int src_w = 100, src_h = 100, dst_w = 1, dst_h = 1, ch = 3;
    cv::Mat   src(src_h, src_w, CV_8UC(ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::Mat ref;
    cv::resize(src, ref, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_w * src_h * ch), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_w * dst_h * ch), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, src.data, src_w * src_h * ch, cudaMemcpyHostToDevice), cudaSuccess);

    int ret = irt::cvcuda::resize<uint8_t>(d_src, d_dst, cv::Size(src_w, src_h), cv::Size(dst_w, dst_h), ch,
                                           cv::INTER_LINEAR, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cv::Mat dst(dst_h, dst_w, CV_8UC(ch));
    ASSERT_EQ(cudaMemcpy(dst.data, d_dst, dst_w * dst_h * ch, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    cv::Mat diff;
    cv::absdiff(dst, ref, diff);
    double max_val = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_val);
    EXPECT_LE(max_val, 1.0);
}

/**
 * @brief 测试极端上采样 - 1x -> 100x（函数版本）
 *
 * 验证从极小图像放大到大图像时的正确性。
 * 这测试了上采样算法在极端情况下的稳定性。
 */
TEST(ResizeFunctionEdgeCaseTest, ExtremeUpscale)
{
    const int src_w = 1, src_h = 1, dst_w = 100, dst_h = 100, ch = 3;
    cv::Mat   src(src_h, src_w, CV_8UC(ch), cv::Scalar(128, 64, 32));

    cv::Mat ref;
    cv::resize(src, ref, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_w * src_h * ch), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_w * dst_h * ch), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, src.data, src_w * src_h * ch, cudaMemcpyHostToDevice), cudaSuccess);

    int ret = irt::cvcuda::resize<uint8_t>(d_src, d_dst, cv::Size(src_w, src_h), cv::Size(dst_w, dst_h), ch,
                                           cv::INTER_LINEAR, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cv::Mat dst(dst_h, dst_w, CV_8UC(ch));
    ASSERT_EQ(cudaMemcpy(dst.data, d_dst, dst_w * dst_h * ch, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    cv::Mat diff;
    cv::absdiff(dst, ref, diff);
    double max_val = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_val);
    EXPECT_LE(max_val, 1.0);
}

/**
 * @brief 测试非对称缩放 - 宽度放大，高度缩小（函数版本）
 *
 * 验证宽度和高度使用不同缩放比例时的正确性。
 * 这测试了算法在非均匀缩放下的表现。
 */
TEST(ResizeFunctionEdgeCaseTest, AsymmetricScale)
{
    const int src_w = 50, src_h = 100, dst_w = 200, dst_h = 25, ch = 3;
    cv::Mat   src(src_h, src_w, CV_8UC(ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::Mat ref;
    cv::resize(src, ref, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_w * src_h * ch), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_w * dst_h * ch), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, src.data, src_w * src_h * ch, cudaMemcpyHostToDevice), cudaSuccess);

    int ret = irt::cvcuda::resize<uint8_t>(d_src, d_dst, cv::Size(src_w, src_h), cv::Size(dst_w, dst_h), ch,
                                           cv::INTER_LINEAR, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cv::Mat dst(dst_h, dst_w, CV_8UC(ch));
    ASSERT_EQ(cudaMemcpy(dst.data, d_dst, dst_w * dst_h * ch, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    cv::Mat diff;
    cv::absdiff(dst, ref, diff);
    double max_val = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_val);
    EXPECT_LE(max_val, 1.0);
}

/**
 * @brief 测试非对称缩放 - 宽度缩小，高度放大（函数版本）
 *
 * 验证宽度缩小、高度放大的非对称缩放。
 * 测试算法在相反方向缩放时的稳定性。
 */
TEST(ResizeFunctionEdgeCaseTest, AsymmetricScaleReverse)
{
    const int src_w = 200, src_h = 25, dst_w = 50, dst_h = 100, ch = 3;
    cv::Mat   src(src_h, src_w, CV_8UC(ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::Mat ref;
    cv::resize(src, ref, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_w * src_h * ch), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_w * dst_h * ch), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, src.data, src_w * src_h * ch, cudaMemcpyHostToDevice), cudaSuccess);

    int ret = irt::cvcuda::resize<uint8_t>(d_src, d_dst, cv::Size(src_w, src_h), cv::Size(dst_w, dst_h), ch,
                                           cv::INTER_LINEAR, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cv::Mat dst(dst_h, dst_w, CV_8UC(ch));
    ASSERT_EQ(cudaMemcpy(dst.data, d_dst, dst_w * dst_h * ch, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    cv::Mat diff;
    cv::absdiff(dst, ref, diff);
    double max_val = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_val);
    EXPECT_LE(max_val, 1.0);
}

// ============================================================================
// 极端缩放和非对称缩放测试（类版本）
// ============================================================================

/**
 * @brief 测试极端下采样 - 100x -> 1x（类版本）
 *
 * 验证从大图像缩小到极小图像时的正确性。
 * 这测试了 Resize 类在极端下采样情况下的稳定性。
 */
TEST(ResizeClassEdgeCaseTest, ExtremeDownscale)
{
    const int src_w = 100, src_h = 100, dst_w = 1, dst_h = 1, ch = 3;
    cv::Mat   src(src_h, src_w, CV_8UC(ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::Mat ref;
    cv::resize(src, ref, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_w * src_h * ch), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_w * dst_h * ch), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, src.data, src_w * src_h * ch, cudaMemcpyHostToDevice), cudaSuccess);

    irt::cvcuda::Resize<uint8_t> resize_op;
    int ret = resize_op(d_src, d_dst, cv::Size(src_w, src_h), cv::Size(dst_w, dst_h), ch, cv::INTER_LINEAR, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cv::Mat dst(dst_h, dst_w, CV_8UC(ch));
    ASSERT_EQ(cudaMemcpy(dst.data, d_dst, dst_w * dst_h * ch, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    cv::Mat diff;
    cv::absdiff(dst, ref, diff);
    double max_val = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_val);
    EXPECT_LE(max_val, 1.0);
}

/**
 * @brief 测试极端上采样 - 1x -> 100x（类版本）
 *
 * 验证从极小图像放大到大图像时的正确性。
 * 这测试了 Resize 类在极端上采样情况下的稳定性。
 */
TEST(ResizeClassEdgeCaseTest, ExtremeUpscale)
{
    const int src_w = 1, src_h = 1, dst_w = 100, dst_h = 100, ch = 3;
    cv::Mat   src(src_h, src_w, CV_8UC(ch), cv::Scalar(128, 64, 32));

    cv::Mat ref;
    cv::resize(src, ref, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_w * src_h * ch), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_w * dst_h * ch), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, src.data, src_w * src_h * ch, cudaMemcpyHostToDevice), cudaSuccess);

    irt::cvcuda::Resize<uint8_t> resize_op;
    int ret = resize_op(d_src, d_dst, cv::Size(src_w, src_h), cv::Size(dst_w, dst_h), ch, cv::INTER_LINEAR, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cv::Mat dst(dst_h, dst_w, CV_8UC(ch));
    ASSERT_EQ(cudaMemcpy(dst.data, d_dst, dst_w * dst_h * ch, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    cv::Mat diff;
    cv::absdiff(dst, ref, diff);
    double max_val = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_val);
    EXPECT_LE(max_val, 1.0);
}

/**
 * @brief 测试非对称缩放 - 宽度放大，高度缩小（类版本）
 *
 * 验证宽度和高度使用不同缩放比例时的正确性。
 * 这测试了 Resize 类在非均匀缩放下的表现。
 */
TEST(ResizeClassEdgeCaseTest, AsymmetricScale)
{
    const int src_w = 50, src_h = 100, dst_w = 200, dst_h = 25, ch = 3;
    cv::Mat   src(src_h, src_w, CV_8UC(ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::Mat ref;
    cv::resize(src, ref, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_w * src_h * ch), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_w * dst_h * ch), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, src.data, src_w * src_h * ch, cudaMemcpyHostToDevice), cudaSuccess);

    irt::cvcuda::Resize<uint8_t> resize_op;
    int ret = resize_op(d_src, d_dst, cv::Size(src_w, src_h), cv::Size(dst_w, dst_h), ch, cv::INTER_LINEAR, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cv::Mat dst(dst_h, dst_w, CV_8UC(ch));
    ASSERT_EQ(cudaMemcpy(dst.data, d_dst, dst_w * dst_h * ch, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    cv::Mat diff;
    cv::absdiff(dst, ref, diff);
    double max_val = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_val);
    EXPECT_LE(max_val, 1.0);
}

/**
 * @brief 测试非对称缩放 - 宽度缩小，高度放大（类版本）
 *
 * 验证宽度缩小、高度放大的非对称缩放。
 * 测试 Resize 类在相反方向缩放时的稳定性。
 */
TEST(ResizeClassEdgeCaseTest, AsymmetricScaleReverse)
{
    const int src_w = 200, src_h = 25, dst_w = 50, dst_h = 100, ch = 3;
    cv::Mat   src(src_h, src_w, CV_8UC(ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(255));

    cv::Mat ref;
    cv::resize(src, ref, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_w * src_h * ch), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_w * dst_h * ch), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, src.data, src_w * src_h * ch, cudaMemcpyHostToDevice), cudaSuccess);

    irt::cvcuda::Resize<uint8_t> resize_op;
    int ret = resize_op(d_src, d_dst, cv::Size(src_w, src_h), cv::Size(dst_w, dst_h), ch, cv::INTER_LINEAR, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cv::Mat dst(dst_h, dst_w, CV_8UC(ch));
    ASSERT_EQ(cudaMemcpy(dst.data, d_dst, dst_w * dst_h * ch, cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    cv::Mat diff;
    cv::absdiff(dst, ref, diff);
    double max_val = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_val);
    EXPECT_LE(max_val, 1.0);
}

// ============================================================================
// Float 类型的极端缩放测试
// ============================================================================

/**
 * @brief 测试 float 类型的极端下采样（函数版本）
 *
 * 验证 float 类型图像在极端下采样时的精度。
 */
TEST(ResizeFunctionEdgeCaseTest, ExtremeDownscaleFloat)
{
    const int src_w = 100, src_h = 100, dst_w = 1, dst_h = 1, ch = 3;
    cv::Mat   src(src_h, src_w, CV_32FC(ch));
    cv::randu(src, cv::Scalar::all(0), cv::Scalar::all(1));

    cv::Mat ref;
    cv::resize(src, ref, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

    float *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_w * src_h * ch * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_w * dst_h * ch * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, src.data, src_w * src_h * ch * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    int ret = irt::cvcuda::resize<float>(d_src, d_dst, cv::Size(src_w, src_h), cv::Size(dst_w, dst_h), ch,
                                         cv::INTER_LINEAR, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cv::Mat dst(dst_h, dst_w, CV_32FC(ch));
    ASSERT_EQ(cudaMemcpy(dst.data, d_dst, dst_w * dst_h * ch * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    cv::Mat diff;
    cv::absdiff(dst, ref, diff);
    double max_val = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_val);
    EXPECT_LE(max_val, 1e-3);
}

/**
 * @brief 测试 float 类型的极端上采样（类版本）
 *
 * 验证 float 类型图像在极端上采样时的精度。
 */
TEST(ResizeClassEdgeCaseTest, ExtremeUpscaleFloat)
{
    const int src_w = 1, src_h = 1, dst_w = 100, dst_h = 100, ch = 3;
    cv::Mat   src(src_h, src_w, CV_32FC(ch), cv::Scalar(0.5f, 0.3f, 0.7f));

    cv::Mat ref;
    cv::resize(src, ref, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);

    float *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(cudaMalloc(&d_src, src_w * src_h * ch * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_dst, dst_w * dst_h * ch * sizeof(float)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_src, src.data, src_w * src_h * ch * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    irt::cvcuda::Resize<float> resize_op;
    int ret = resize_op(d_src, d_dst, cv::Size(src_w, src_h), cv::Size(dst_w, dst_h), ch, cv::INTER_LINEAR, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cv::Mat dst(dst_h, dst_w, CV_32FC(ch));
    ASSERT_EQ(cudaMemcpy(dst.data, d_dst, dst_w * dst_h * ch * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_src);
    cudaFree(d_dst);

    cv::Mat diff;
    cv::absdiff(dst, ref, diff);
    double max_val = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_val);
    EXPECT_LE(max_val, 1e-3);
}
