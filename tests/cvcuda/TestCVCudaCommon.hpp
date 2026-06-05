/**
 * @file TestCVCudaCommon.hpp
 * @brief cvcuda 单元测试公共辅助定义。
 *
 * 该文件集中放置多个 cvcuda 测试文件共享的小型工具，避免每个测试文件重复实现相同的
 * InferRT 状态断言和 OpenCV 类型映射逻辑。
 */

#pragma once

#include <gtest/gtest.h>
#include <inferrt/core/Status.h>
#include <opencv2/core.hpp>

#include <cstdint>

namespace irt::cvcuda::test {

/**
 * @brief 断言 InferRT 调用成功。
 *
 * 当返回值不是 IRT_SUCCESS 时，会读取 InferRT 最近一次错误消息，并把状态码、状态名称和
 * 错误消息写入 Google Test 的失败输出，便于定位具体失败原因。
 *
 * @param ret InferRT API 返回的状态码。
 * @return 成功时返回 AssertionSuccess，否则返回包含错误详情的 AssertionFailure。
 */
inline ::testing::AssertionResult AssertInferRTSuccess(int ret)
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

/**
 * @brief 将 C++ 像素类型映射到 OpenCV Mat 深度和随机测试取值范围。
 *
 * @tparam T C++ 像素类型。
 */
template<typename T>
struct CvDepth;

/**
 * @brief uint8_t 图像类型的 OpenCV 深度和随机数据范围。
 */
template<>
struct CvDepth<uint8_t>
{
    static constexpr int    value = CV_8U;
    static constexpr double range = 255.0;
};

/**
 * @brief float 图像类型的 OpenCV 深度和随机数据范围。
 */
template<>
struct CvDepth<float>
{
    static constexpr int    value = CV_32F;
    static constexpr double range = 1.0;
};

} // namespace irt::cvcuda::test
