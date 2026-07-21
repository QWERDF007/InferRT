/**
 * @file ShapeTemplateMatcherImpl.cpp
 * @brief v0 原始形状模板匹配器的具体实现。
 */

#include "ShapeTemplateMatcherImpl.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// v0 的热点实现完全收敛在 v0::detail；仅通过明确的 common 别名访问共享流程协议。
namespace irt::features::v0::detail {
namespace common = ::irt::features::detail;

using common::ShapeTemplateCandidate;
using common::ShapeTemplateMatcherKernel;
using common::ShapeTemplateQuantizedGradient;
using common::ShapeTemplateResponseMaps;
using common::ShapeTemplateResponseTable;
using common::ShapeTemplateScoredPosition;
using common::kShapeTemplateInvalidLabel;
using common::kShapeTemplateOrientationBins;
using common::quantizeShapeTemplateAngle;
using common::sortShapeTemplateCandidates;

namespace {

/**
 * @brief v0 的参考标量评分：逐滑窗、逐特征累加，并按理论上界提前淘汰。
 *
 * 这条路径刻意保持为基线实现；v1 的任何得分图/SIMD 优化均不应改变它。
 */
bool similarityAtLeast(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                       int x, int y, float threshold, float &score)
{
    std::uint64_t sum = 0;
    const std::uint64_t maximum_per_feature = static_cast<std::uint64_t>(response_maps.denominator_per_feature);
    const size_t feature_count = templ.features.size();
    for (size_t index = 0; index < feature_count; ++index)
    {
        const auto &feature = templ.features[index];
        const auto &map = response_maps.maps[static_cast<size_t>(feature.label)];
        sum += map.ptr<unsigned char>(y + feature.y)[x + feature.x];
        const std::uint64_t remaining = static_cast<std::uint64_t>(feature_count - index - 1);
        const float upper_bound = 100.0f * static_cast<float>(sum + remaining * maximum_per_feature)
                                / static_cast<float>(maximum_per_feature * feature_count);
        if (upper_bound < threshold)
            return false;
    }
    score = 100.0f * static_cast<float>(sum) / static_cast<float>(maximum_per_feature * feature_count);
    return score >= threshold;
}

std::vector<ShapeTemplateScoredPosition>
scanTemplateReference(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                      const cv::Mat &search_mask, bool /* full_search_mask */, cv::Size image_size,
                      int scan_step, float threshold)
{
    std::vector<ShapeTemplateScoredPosition> positions;
    if (templ.features.empty() || templ.width > image_size.width || templ.height > image_size.height)
        return positions;

    const int max_x = image_size.width - templ.width;
    const int max_y = image_size.height - templ.height;

    for (int y = 0; y <= max_y; y += scan_step)
    {
        const int center_y = std::min(image_size.height - 1, y + templ.height / 2);
        const auto *mask_row = search_mask.ptr<unsigned char>(center_y);
        for (int x = 0; x <= max_x; x += scan_step)
        {
            const int center_x = std::min(image_size.width - 1, x + templ.width / 2);
            if (mask_row[center_x] == 0)
                continue;

            float score = 0.0f;
            if (similarityAtLeast(response_maps, templ, x, y, threshold, score))
                positions.push_back(ShapeTemplateScoredPosition{x, y, score});
        }
    }
    return positions;
}

void fillQuantizedLabelsReference(const cv::Mat &magnitude, const cv::Mat &angle, const cv::Mat &mask,
                                  float threshold, cv::Mat &labels)
{
    for (int y = 0; y < labels.rows; ++y)
    {
        const auto *mag_row   = magnitude.ptr<float>(y);
        const auto *angle_row = angle.ptr<float>(y);
        const auto *mask_row  = mask.ptr<unsigned char>(y);
        auto       *label_row = labels.ptr<unsigned char>(y);
#pragma loop(no_vector)
        for (int x = 0; x < labels.cols; ++x)
        {
            if (mask_row[x] != 0 && mag_row[x] >= threshold)
            {
                label_row[x] = static_cast<unsigned char>(quantizeShapeTemplateAngle(angle_row[x]));
            }
        }
    }
}

std::vector<ShapeTemplateCandidate>
collectCandidatesReference(const ShapeTemplateQuantizedGradient &gradient, const cv::Mat &mask, float threshold)
{
    std::vector<ShapeTemplateCandidate> candidates;
    for (int y = 1; y < gradient.labels.rows - 1; ++y)
    {
        const auto *mag_row   = gradient.magnitude.ptr<float>(y);
        const auto *angle_row = gradient.angle.ptr<float>(y);
        const auto *mask_row  = mask.ptr<unsigned char>(y);
        const auto *label_row = gradient.labels.ptr<unsigned char>(y);
#pragma loop(no_vector)
        for (int x = 1; x < gradient.labels.cols - 1; ++x)
        {
            if (mask_row[x] == 0 || label_row[x] == kShapeTemplateInvalidLabel || mag_row[x] < threshold)
            {
                continue;
            }
            candidates.push_back(ShapeTemplateCandidate{x, y, static_cast<int>(label_row[x]), angle_row[x],
                                                        std::max(mag_row[x], 0.0f)});
        }
    }

    sortShapeTemplateCandidates(candidates);
    return candidates;
}

void fillResponseMapReference(const cv::Mat &labels, cv::Mat &response, int template_label,
                              const ShapeTemplateResponseTable &table)
{
    for (int y = 0; y < labels.rows; ++y)
    {
        const auto *src = labels.ptr<unsigned char>(y);
        auto       *dst = response.ptr<unsigned char>(y);
#pragma loop(no_vector)
        for (int x = 0; x < labels.cols; ++x)
        {
            dst[x] = src[x] < kShapeTemplateOrientationBins ? table[template_label][src[x]] : 0;
        }
    }
}

/** @brief v0 参考热点内核，严格保留逐位置、逐特征评分语义。 */
class ShapeTemplateMatcherKernelImpl final : public ShapeTemplateMatcherKernel
{
public:
    bool needsMaterializedResponseMaps() const noexcept override
    {
        return true;
    }

    bool usesOptimizedTemplateTraining() const noexcept override
    {
        return false;
    }

    void fillQuantizedLabels(const cv::Mat &magnitude, const cv::Mat &angle, const cv::Mat &mask,
                             float threshold, cv::Mat &labels) const override
    {
        fillQuantizedLabelsReference(magnitude, angle, mask, threshold, labels);
    }

    std::vector<ShapeTemplateCandidate>
    collectCandidates(const ShapeTemplateQuantizedGradient &gradient, const cv::Mat &mask, float threshold) const override
    {
        return collectCandidatesReference(gradient, mask, threshold);
    }

    void fillResponseMap(const cv::Mat &labels, cv::Mat &response, int template_label,
                         const ShapeTemplateResponseTable &table) const override
    {
        fillResponseMapReference(labels, response, template_label, table);
    }

    std::vector<ShapeTemplateScoredPosition>
    scanTemplate(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                 const cv::Mat &search_mask, bool full_search_mask, cv::Size image_size, int scan_step,
                 float threshold) const override
    {
        return scanTemplateReference(response_maps, templ, search_mask, full_search_mask, image_size, scan_step,
                                     threshold);
    }
};

} // namespace

/** @brief 创建只属于 v0 的参考热点内核。 */
static std::unique_ptr<ShapeTemplateMatcherKernel> makeV0Kernel()
{
    return std::make_unique<ShapeTemplateMatcherKernelImpl>();
}

ShapeTemplateMatcherImpl::ShapeTemplateMatcherImpl(ShapeTemplateMatcherConfig config)
    : common::ShapeTemplateMatcherEngine(std::move(config), makeV0Kernel())
{
}

} // namespace irt::features::v0::detail
