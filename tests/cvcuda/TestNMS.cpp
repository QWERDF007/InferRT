/**
 * @file TestNMS.cpp
 * @brief CUDA NMS 算子的单元测试。
 */

#include "TestCVCudaCommon.hpp"

#include <gtest/gtest.h>
#include <inferrt/core/Status.h>
#include <inferrt/cvcuda/OpNMS.h>
#include <inferrt/cvcuda/OpNMS.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using irt::cvcuda::test::AssertInferRTSuccess;

/**
 * @brief 计算 xyxy 框面积，退化边长按 0 处理。
 * @param box 指向长度为 4 的 xyxy 框坐标数组。
 * @return 非负框面积。
 */
float area(const float *box)
{
    const float width  = std::max(box[2] - box[0], 0.0f);
    const float height = std::max(box[3] - box[1], 0.0f);
    return width * height;
}

/**
 * @brief 计算两个 xyxy 框的 IoU，作为 CPU 参考实现的一部分。
 * @param lhs 左侧 xyxy 框坐标数组。
 * @param rhs 右侧 xyxy 框坐标数组。
 * @return 两个框的 IoU；并集面积为 0 时返回 0。
 */
float iou(const float *lhs, const float *rhs)
{
    const float xx1   = std::max(lhs[0], rhs[0]);
    const float yy1   = std::max(lhs[1], rhs[1]);
    const float xx2   = std::min(lhs[2], rhs[2]);
    const float yy2   = std::min(lhs[3], rhs[3]);
    const float width = std::max(xx2 - xx1, 0.0f);
    const float height = std::max(yy2 - yy1, 0.0f);
    const float inter  = width * height;
    const float uni    = area(lhs) + area(rhs) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

/**
 * @brief 使用朴素 CPU NMS 生成期望索引序列。
 * @param boxes 按 xyxy 连续存储的框数组，长度为 ``scores.size() * 4``。
 * @param scores 每个框对应的分数数组。
 * @param iou_threshold 抑制重叠框使用的 IoU 阈值。
 * @return CPU 参考实现保留下来的原始框索引。
 */
std::vector<int64_t> makeNMSReference(const std::vector<float> &boxes, const std::vector<float> &scores,
                                      float iou_threshold)
{
    const int n = static_cast<int>(scores.size());
    std::vector<uint8_t> suppressed(static_cast<size_t>(n), 0);
    std::vector<int64_t> keep;
    keep.reserve(scores.size());

    for (int iter = 0; iter < n; ++iter)
    {
        int   best_index = -1;
        float best_score = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n; ++i)
        {
            if (!suppressed[static_cast<size_t>(i)] && (best_index < 0 || scores[static_cast<size_t>(i)] > best_score))
            {
                best_index = i;
                best_score = scores[static_cast<size_t>(i)];
            }
        }
        if (best_index < 0)
        {
            break;
        }

        keep.push_back(best_index);
        suppressed[static_cast<size_t>(best_index)] = 1;
        const float *best_box = boxes.data() + best_index * 4;
        for (int i = 0; i < n; ++i)
        {
            if (!suppressed[static_cast<size_t>(i)] && iou(best_box, boxes.data() + i * 4) > iou_threshold)
            {
                suppressed[static_cast<size_t>(i)] = 1;
            }
        }
    }

    return keep;
}

/**
 * @brief 将输入拷到 GPU，调用待测 NMS 接口，并与 CPU 参考结果比较。
 * @tparam Caller 可调用对象类型，签名与 ``nms`` 或 ``NMS::operator()`` 保持一致。
 * @param boxes 按 xyxy 连续存储的框数组，长度为 ``scores.size() * 4``。
 * @param scores 每个框对应的分数数组。
 * @param iou_threshold 抑制重叠框使用的 IoU 阈值。
 * @param caller 实际待测的 CUDA NMS 调用器。
 */
template<typename Caller>
void runNMSTest(const std::vector<float> &boxes, const std::vector<float> &scores, float iou_threshold, Caller caller)
{
    const int                 num_boxes = static_cast<int>(scores.size());
    const std::vector<int64_t> ref      = makeNMSReference(boxes, scores, iou_threshold);

    float   *d_boxes      = nullptr;
    float   *d_scores     = nullptr;
    int64_t *d_keep       = nullptr;
    int     *d_keep_count = nullptr;
    ASSERT_EQ(cudaMalloc(&d_boxes, boxes.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_scores, scores.size() * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_keep, scores.size() * sizeof(int64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_keep_count, sizeof(int)), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(d_boxes, boxes.data(), boxes.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_scores, scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);

    const int ret = caller(d_boxes, d_scores, d_keep, d_keep_count, num_boxes, iou_threshold, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int keep_count = 0;
    ASSERT_EQ(cudaMemcpy(&keep_count, d_keep_count, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    std::vector<int64_t> keep(static_cast<size_t>(keep_count));
    ASSERT_EQ(cudaMemcpy(keep.data(), d_keep, keep.size() * sizeof(int64_t), cudaMemcpyDeviceToHost), cudaSuccess);

    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_keep);
    cudaFree(d_keep_count);

    EXPECT_EQ(keep, ref);
}

} // namespace

/**
 * @brief 验证函数式 NMS 会抑制低分重叠框。
 */
TEST(NMSFunctionTest, SuppressesLowerScoringOverlaps)
{
    const std::vector<float> boxes{
        0.0f, 0.0f, 10.0f, 10.0f,
        1.0f, 1.0f, 11.0f, 11.0f,
        20.0f, 20.0f, 30.0f, 30.0f,
        21.0f, 21.0f, 31.0f, 31.0f,
    };
    const std::vector<float> scores{0.9f, 0.8f, 0.7f, 0.6f};

    runNMSTest(boxes, scores, 0.5f,
               [](const float *boxes, const float *scores, int64_t *keep, int *keep_count, int num_boxes,
                  float threshold, cudaStream_t stream)
               { return irt::cvcuda::nms(boxes, scores, keep, keep_count, num_boxes, threshold, stream); });
}

/**
 * @brief 验证类封装返回的索引仍按分数降序选择。
 */
TEST(NMSClassTest, ReturnsIndicesSortedByScore)
{
    const std::vector<float> boxes{
        0.0f, 0.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 3.0f, 3.0f,
        4.0f, 4.0f, 5.0f, 5.0f,
    };
    const std::vector<float> scores{0.1f, 0.9f, 0.2f};

    irt::cvcuda::NMS op;
    runNMSTest(boxes, scores, 0.0f,
               [&op](const float *boxes, const float *scores, int64_t *keep, int *keep_count, int num_boxes,
                     float threshold, cudaStream_t stream)
               { return op(boxes, scores, keep, keep_count, num_boxes, threshold, stream); });
}

/**
 * @brief 构造超过单个 bitmask block 的输入，覆盖 block-wise 并行路径。
 */
TEST(NMSFunctionTest, HandlesMultipleBitmaskBlocks)
{
    constexpr int num_boxes = 160;
    std::vector<float> boxes;
    std::vector<float> scores;
    boxes.reserve(static_cast<size_t>(num_boxes) * 4);
    scores.reserve(num_boxes);

    for (int i = 0; i < num_boxes; ++i)
    {
        const int   group = i / 5;
        const float base  = static_cast<float>(group * 20);
        const float delta = static_cast<float>(i % 5) * 0.35f;
        boxes.insert(boxes.end(), {base + delta, base + delta, base + 10.0f + delta, base + 10.0f + delta});
        scores.push_back(static_cast<float>((num_boxes - ((i * 37) % num_boxes))) / static_cast<float>(num_boxes));
    }

    runNMSTest(boxes, scores, 0.5f,
               [](const float *boxes, const float *scores, int64_t *keep, int *keep_count, int num_boxes,
                  float threshold, cudaStream_t stream)
               { return irt::cvcuda::nms(boxes, scores, keep, keep_count, num_boxes, threshold, stream); });
}

/**
 * @brief 验证空输入只写出 keep_count=0。
 */
TEST(NMSFunctionEdgeCaseTest, SupportsEmptyInput)
{
    int *d_keep_count = nullptr;
    ASSERT_EQ(cudaMalloc(&d_keep_count, sizeof(int)), cudaSuccess);

    const int ret = irt::cvcuda::nms(nullptr, nullptr, nullptr, d_keep_count, 0, 0.5f, nullptr);
    ASSERT_TRUE(AssertInferRTSuccess(ret));
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    int keep_count = -1;
    ASSERT_EQ(cudaMemcpy(&keep_count, d_keep_count, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(keep_count, 0);

    cudaFree(d_keep_count);
}

/**
 * @brief 验证 keep_count 指针为空时返回参数错误。
 */
TEST(NMSFunctionEdgeCaseTest, RejectsNullCount)
{
    const int ret = irt::cvcuda::nms(nullptr, nullptr, nullptr, nullptr, 0, 0.5f, nullptr);
    EXPECT_EQ(ret, IRT_ERROR_INVALID_ARGUMENT);
}
