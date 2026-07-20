/**
 * @file ShapeTemplateMatcherScalarKernel.cpp
 * @brief v0 的原始标量热点内核。
 */

#include "ShapeTemplateMatcherEngine.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace irt::features::detail {
namespace {

/**
 * @brief v0 的参考标量评分：逐滑窗、逐特征累加，并按理论上界提前淘汰。
 *
 * 这条路径刻意保持为基线实现；v1 的任何得分图/SIMD 优化均不应改变它。
 */
bool similarityAtLeastScalar(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
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
scanTemplateScalar(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                   const cv::Mat &search_mask, cv::Size image_size, int scan_step, float threshold)
{
    std::vector<ShapeTemplateScoredPosition> positions;
    if (templ.width > image_size.width || templ.height > image_size.height || templ.features.empty())
        return positions;

    const int max_y = image_size.height - templ.height;
    const int max_x = image_size.width - templ.width;
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
            if (similarityAtLeastScalar(response_maps, templ, x, y, threshold, score))
                positions.push_back(ShapeTemplateScoredPosition{x, y, score});
        }
    }
    return positions;
}

void fillQuantizedLabelsScalar(const cv::Mat &magnitude, const cv::Mat &angle, const cv::Mat &mask,
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
collectCandidatesScalar(const ShapeTemplateQuantizedGradient &gradient, const cv::Mat &mask, float threshold)
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

void fillResponseMapScalar(const cv::Mat &labels, cv::Mat &response, int template_label,
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

class ScalarShapeTemplateMatcherKernel final : public ShapeTemplateMatcherKernel
{
public:
    bool needsMaterializedResponseMaps() const noexcept override
    {
        return true;
    }

    void fillQuantizedLabels(const cv::Mat &magnitude, const cv::Mat &angle, const cv::Mat &mask,
                             float threshold, cv::Mat &labels) const override
    {
        fillQuantizedLabelsScalar(magnitude, angle, mask, threshold, labels);
    }

    std::vector<ShapeTemplateCandidate>
    collectCandidates(const ShapeTemplateQuantizedGradient &gradient, const cv::Mat &mask, float threshold) const override
    {
        return collectCandidatesScalar(gradient, mask, threshold);
    }

    void fillResponseMap(const cv::Mat &labels, cv::Mat &response, int template_label,
                         const ShapeTemplateResponseTable &table) const override
    {
        fillResponseMapScalar(labels, response, template_label, table);
    }

    std::vector<ShapeTemplateScoredPosition>
    scanTemplate(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                 const cv::Mat &search_mask, cv::Size image_size, int scan_step, float threshold) const override
    {
        return scanTemplateScalar(response_maps, templ, search_mask, image_size, scan_step, threshold);
    }
};

} // namespace

std::unique_ptr<ShapeTemplateMatcherKernel> createScalarShapeTemplateMatcherKernel()
{
    return std::make_unique<ScalarShapeTemplateMatcherKernel>();
}

} // namespace irt::features::detail
