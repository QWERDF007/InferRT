/**
 * @file ShapeTemplateMatcherAvx512Impl.cpp
 * @brief v2 AVX512 形状模板匹配器的具体实现与热点内核。
 */

#include "ShapeTemplateMatcherAvx512Impl.hpp"

#include <inferrt/core/Exception.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#if defined(INFERRT_SHAPE_TEMPLATE_AVX512_ENABLED)
#include <immintrin.h>
#endif

// v2 的 AVX512 实现完全位于 v2::detail；只通过局部 common 别名访问版本无关协议。
namespace irt::features::v2::detail {
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

#if defined(INFERRT_SHAPE_TEMPLATE_AVX512_ENABLED)
namespace {

/** @brief 单个特征在源图量化标签中的相对首地址及其方向统计值。 */
struct PackedFeature
{
    const unsigned char *base{nullptr};
    float                expected_response{0.0f};
    int                  label{0};
};

/**
 * @brief 标量精确回退评分。
 *
 * 当累计上界超过 AVX512 16-bit 安全范围，或者调用方选择非连续扫描步长时使用。该路径
 * 直接读取量化标签和响应查表，不依赖 v0/v1 的任何实现。
 */
bool similarityAtLeastFallback(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                               int x, int y, float threshold, float &score)
{
    std::uint64_t sum = 0;
    const std::uint64_t maximum_per_feature = static_cast<std::uint64_t>(response_maps.denominator_per_feature);
    const size_t feature_count = templ.features.size();
    for (size_t index = 0; index < feature_count; ++index)
    {
        const auto &feature = templ.features[index];
        const unsigned char source_label
            = response_maps.quantized_labels.ptr<unsigned char>(y + feature.y)[x + feature.x];
        if (source_label < kShapeTemplateOrientationBins)
        {
            sum += response_maps.response_table[static_cast<size_t>(feature.label)][static_cast<size_t>(source_label)];
        }
        const std::uint64_t remaining = static_cast<std::uint64_t>(feature_count - index - 1);
        const float upper_bound = 100.0f * static_cast<float>(sum + remaining * maximum_per_feature)
                                / static_cast<float>(maximum_per_feature * feature_count);
        if (upper_bound < threshold)
            return false;
    }
    score = 100.0f * static_cast<float>(sum) / static_cast<float>(maximum_per_feature * feature_count);
    return score >= threshold;
}

/** @brief 使用精确标量评分扫描一个模板。 */
std::vector<ShapeTemplateScoredPosition>
scanTemplateFallback(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                     const cv::Mat &search_mask, bool full_search_mask, cv::Size image_size,
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
        const auto *mask_row = full_search_mask ? nullptr : search_mask.ptr<unsigned char>(center_y);
        for (int x = 0; x <= max_x; x += scan_step)
        {
            if (!full_search_mask)
            {
                const int center_x = std::min(image_size.width - 1, x + templ.width / 2);
                if (mask_row[center_x] == 0)
                    continue;
            }

            float score = 0.0f;
            if (similarityAtLeastFallback(response_maps, templ, x, y, threshold, score))
                positions.push_back(ShapeTemplateScoredPosition{x, y, score});
        }
    }
    return positions;
}

/** @brief 构造 16 项方向响应查表；索引 8 至 15 固定为零以屏蔽无效标签。 */
__m128i makeResponseLookup128(int template_label, const ShapeTemplateResponseTable &table)
{
    alignas(16) std::array<unsigned char, 16> values{};
    for (int source_label = 0; source_label < kShapeTemplateOrientationBins; ++source_label)
    {
        values[static_cast<size_t>(source_label)] = table[static_cast<size_t>(template_label)]
                                                         [static_cast<size_t>(source_label)];
    }
    return _mm_load_si128(reinterpret_cast<const __m128i *>(values.data()));
}

/**
 * @brief 使用 64 个 8-bit lane 批量扫描连续滑窗。
 *
 * ``maximum_sum <= 255`` 时每个 lane 的累计值不会溢出。每四个特征以与 v1 相同的理论上界
 * 淘汰整组候选，从而保持最终分数和阈值语义完全一致。
 */
std::vector<ShapeTemplateScoredPosition>
scanTemplateAvx512U8(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                     const std::vector<PackedFeature> &features, const cv::Mat &search_mask,
                     bool full_search_mask, cv::Size image_size, float threshold,
                     std::uint64_t maximum_sum, std::uint64_t conservative_required_sum)
{
    const int max_x = image_size.width - templ.width;
    const int max_y = image_size.height - templ.height;
    const float denominator = static_cast<float>(maximum_sum);
    const size_t response_step = response_maps.quantized_labels.step;
    constexpr size_t kPruneFeatureInterval = 4;

    std::array<__m512i, kShapeTemplateOrientationBins> response_lookups{};
    for (int label = 0; label < kShapeTemplateOrientationBins; ++label)
    {
        response_lookups[static_cast<size_t>(label)] = _mm512_broadcast_i32x4(
            makeResponseLookup128(label, response_maps.response_table));
    }

    const __m512i zero = _mm512_setzero_si512();
    const int last_vector_x = max_x - 63;
    std::vector<ShapeTemplateScoredPosition> positions;
    for (int y = 0; y <= max_y; ++y)
    {
        const int center_y = std::min(image_size.height - 1, y + templ.height / 2);
        const auto *mask_row = full_search_mask ? nullptr : search_mask.ptr<unsigned char>(center_y);
        const size_t row_offset = static_cast<size_t>(y) * response_step;
        int x = 0;
        for (; x <= last_vector_x; x += 64)
        {
            __mmask64 valid_lanes = static_cast<__mmask64>(~std::uint64_t{0});
            if (!full_search_mask)
            {
                const __m512i mask_values = _mm512_loadu_si512(
                    reinterpret_cast<const void *>(mask_row + x + templ.width / 2));
                valid_lanes = ~_mm512_cmpeq_epi8_mask(mask_values, zero);
            }
            if (valid_lanes == 0)
                continue;

            __m512i total = zero;
            bool any_lane_can_match = true;
            for (size_t index = 0; index < features.size(); ++index)
            {
                const auto *source = features[index].base + row_offset + x;
                const __m512i labels = _mm512_loadu_si512(reinterpret_cast<const void *>(source));
                const __m512i responses = _mm512_shuffle_epi8(
                    response_lookups[static_cast<size_t>(features[index].label)], labels);
                total = _mm512_add_epi8(total, responses);

                const bool should_prune = (index + 1) % kPruneFeatureInterval == 0
                                       || index + 1 == features.size();
                if (!should_prune)
                    continue;
                const std::uint64_t remaining = static_cast<std::uint64_t>(features.size() - index - 1);
                const std::uint64_t maximum_remaining = remaining
                                                      * static_cast<std::uint64_t>(response_maps.denominator_per_feature);
                if (conservative_required_sum <= maximum_remaining)
                    continue;
                const auto minimum_current_sum = static_cast<unsigned char>(
                    conservative_required_sum - maximum_remaining - 1);
                const __mmask64 possible = _mm512_cmpgt_epu8_mask(
                    total, _mm512_set1_epi8(static_cast<char>(minimum_current_sum)));
                any_lane_can_match = (possible & valid_lanes) != 0;
                if (!any_lane_can_match)
                    break;
            }
            if (!any_lane_can_match)
                continue;

            alignas(64) std::array<unsigned char, 64> lane_sums{};
            _mm512_storeu_si512(reinterpret_cast<void *>(lane_sums.data()), total);
            for (int lane = 0; lane < 64; ++lane)
            {
                if ((valid_lanes & (std::uint64_t{1} << lane)) == 0)
                    continue;
                const float score = 100.0f * static_cast<float>(lane_sums[static_cast<size_t>(lane)]) / denominator;
                if (score >= threshold)
                    positions.push_back(ShapeTemplateScoredPosition{x + lane, y, score});
            }
        }
        for (; x <= max_x; ++x)
        {
            if (!full_search_mask)
            {
                const int center_x = std::min(image_size.width - 1, x + templ.width / 2);
                if (mask_row[center_x] == 0)
                    continue;
            }
            float score = 0.0f;
            if (similarityAtLeastFallback(response_maps, templ, x, y, threshold, score))
                positions.push_back(ShapeTemplateScoredPosition{x, y, score});
        }
    }
    return positions;
}

/**
 * @brief 使用 32 个 16-bit lane 批量扫描连续滑窗。
 *
 * 用于 8-bit 累加不能安全容纳的精确配置，例如默认 128 特征且方向容差为 1 的情况。
 */
std::vector<ShapeTemplateScoredPosition>
scanTemplateAvx512U16(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                      const std::vector<PackedFeature> &features, const cv::Mat &search_mask,
                      bool full_search_mask, cv::Size image_size, float threshold,
                      std::uint64_t maximum_sum, std::uint64_t conservative_required_sum)
{
    const int max_x = image_size.width - templ.width;
    const int max_y = image_size.height - templ.height;
    const float denominator = static_cast<float>(maximum_sum);
    const size_t response_step = response_maps.quantized_labels.step;
    constexpr size_t kPruneFeatureInterval = 4;

    std::array<__m256i, kShapeTemplateOrientationBins> response_lookups{};
    for (int label = 0; label < kShapeTemplateOrientationBins; ++label)
    {
        response_lookups[static_cast<size_t>(label)] = _mm256_broadcastsi128_si256(
            makeResponseLookup128(label, response_maps.response_table));
    }

    const __m256i zero256 = _mm256_setzero_si256();
    const int last_vector_x = max_x - 31;
    std::vector<ShapeTemplateScoredPosition> positions;
    for (int y = 0; y <= max_y; ++y)
    {
        const int center_y = std::min(image_size.height - 1, y + templ.height / 2);
        const auto *mask_row = full_search_mask ? nullptr : search_mask.ptr<unsigned char>(center_y);
        const size_t row_offset = static_cast<size_t>(y) * response_step;
        int x = 0;
        for (; x <= last_vector_x; x += 32)
        {
            std::uint32_t valid_lanes = std::numeric_limits<std::uint32_t>::max();
            if (!full_search_mask)
            {
                const __m256i mask_values = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(mask_row + x + templ.width / 2));
                valid_lanes = ~static_cast<std::uint32_t>(
                    _mm256_movemask_epi8(_mm256_cmpeq_epi8(mask_values, zero256)));
            }
            if (valid_lanes == 0)
                continue;

            __m512i total = _mm512_setzero_si512();
            bool any_lane_can_match = true;
            for (size_t index = 0; index < features.size(); ++index)
            {
                const auto *source = features[index].base + row_offset + x;
                const __m256i labels = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source));
                const __m256i responses = _mm256_shuffle_epi8(
                    response_lookups[static_cast<size_t>(features[index].label)], labels);
                total = _mm512_add_epi16(total, _mm512_cvtepu8_epi16(responses));

                const bool should_prune = (index + 1) % kPruneFeatureInterval == 0
                                       || index + 1 == features.size();
                if (!should_prune)
                    continue;
                const std::uint64_t remaining = static_cast<std::uint64_t>(features.size() - index - 1);
                const std::uint64_t maximum_remaining = remaining
                                                      * static_cast<std::uint64_t>(response_maps.denominator_per_feature);
                if (conservative_required_sum <= maximum_remaining)
                    continue;
                const auto minimum_current_sum = static_cast<std::int16_t>(
                    conservative_required_sum - maximum_remaining - 1);
                const __mmask32 possible = _mm512_cmpgt_epi16_mask(
                    total, _mm512_set1_epi16(minimum_current_sum));
                any_lane_can_match = (possible & valid_lanes) != 0;
                if (!any_lane_can_match)
                    break;
            }
            if (!any_lane_can_match)
                continue;

            alignas(64) std::array<std::uint16_t, 32> lane_sums{};
            _mm512_storeu_si512(reinterpret_cast<void *>(lane_sums.data()), total);
            for (int lane = 0; lane < 32; ++lane)
            {
                if ((valid_lanes & (std::uint32_t{1} << lane)) == 0)
                    continue;
                const float score = 100.0f * static_cast<float>(lane_sums[static_cast<size_t>(lane)]) / denominator;
                if (score >= threshold)
                    positions.push_back(ShapeTemplateScoredPosition{x + lane, y, score});
            }
        }
        for (; x <= max_x; ++x)
        {
            if (!full_search_mask)
            {
                const int center_x = std::min(image_size.width - 1, x + templ.width / 2);
                if (mask_row[center_x] == 0)
                    continue;
            }
            float score = 0.0f;
            if (similarityAtLeastFallback(response_maps, templ, x, y, threshold, score))
                positions.push_back(ShapeTemplateScoredPosition{x, y, score});
        }
    }
    return positions;
}

/** @brief 在 v2 中按 AVX512 方式扫描模板；非连续扫描保留标量精确回退。 */
std::vector<ShapeTemplateScoredPosition>
scanTemplateAvx512(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                   const cv::Mat &search_mask, bool full_search_mask, cv::Size image_size,
                   int scan_step, float threshold)
{
    if (templ.features.empty() || templ.width > image_size.width || templ.height > image_size.height)
        return {};

    const std::uint64_t maximum_per_feature = static_cast<std::uint64_t>(response_maps.denominator_per_feature);
    const std::uint64_t maximum_sum = maximum_per_feature * static_cast<std::uint64_t>(templ.features.size());
    if (scan_step != 1 || maximum_sum > std::numeric_limits<std::int16_t>::max())
    {
        return scanTemplateFallback(response_maps, templ, search_mask, full_search_mask, image_size,
                                    scan_step, threshold);
    }

    const std::uint64_t conservative_required_sum = static_cast<std::uint64_t>(
        threshold * static_cast<float>(maximum_sum) / 100.0f);
    std::vector<PackedFeature> features;
    features.reserve(templ.features.size());
    for (const auto &feature : templ.features)
    {
        features.push_back(PackedFeature{
            response_maps.quantized_labels.ptr<unsigned char>(feature.y) + feature.x,
            response_maps.average_responses[static_cast<size_t>(feature.label)], feature.label});
    }
    std::stable_sort(features.begin(), features.end(),
                     [](const PackedFeature &a, const PackedFeature &b)
                     { return a.expected_response < b.expected_response; });

    if (maximum_sum <= std::numeric_limits<unsigned char>::max())
    {
        return scanTemplateAvx512U8(response_maps, templ, features, search_mask, full_search_mask, image_size,
                                    threshold, maximum_sum, conservative_required_sum);
    }
    return scanTemplateAvx512U16(response_maps, templ, features, search_mask, full_search_mask, image_size,
                                  threshold, maximum_sum, conservative_required_sum);
}

/** @brief AVX512 批量量化 16 个梯度方向标签，尾部保留标量精确处理。 */
void fillQuantizedLabelsAvx512(const cv::Mat &magnitude, const cv::Mat &angle, const cv::Mat &mask,
                               float threshold, cv::Mat &labels)
{
    const __m512 threshold_vec = _mm512_set1_ps(threshold);
    const std::array<__m512, kShapeTemplateOrientationBins> angle_thresholds{
        _mm512_set1_ps(22.5f),  _mm512_set1_ps(67.5f),  _mm512_set1_ps(112.5f), _mm512_set1_ps(157.5f),
        _mm512_set1_ps(202.5f), _mm512_set1_ps(247.5f), _mm512_set1_ps(292.5f), _mm512_set1_ps(337.5f),
    };

    for (int y = 0; y < labels.rows; ++y)
    {
        const auto *mag_row   = magnitude.ptr<float>(y);
        const auto *angle_row = angle.ptr<float>(y);
        const auto *mask_row  = mask.ptr<unsigned char>(y);
        auto       *label_row = labels.ptr<unsigned char>(y);
        int x = 0;
        for (; x <= labels.cols - 16; x += 16)
        {
            const __m512 magnitude_values = _mm512_loadu_ps(mag_row + x);
            const __mmask16 valid_magnitude = _mm512_cmp_ps_mask(
                magnitude_values, threshold_vec, _CMP_GE_OQ);
            const __m512 angle_values = _mm512_loadu_ps(angle_row + x);
            std::array<__mmask16, kShapeTemplateOrientationBins> angle_masks{};
            for (int index = 0; index < kShapeTemplateOrientationBins; ++index)
            {
                angle_masks[static_cast<size_t>(index)] = _mm512_cmp_ps_mask(
                    angle_values, angle_thresholds[static_cast<size_t>(index)], _CMP_GE_OQ);
            }
            for (int lane = 0; lane < 16; ++lane)
            {
                if ((valid_magnitude & (std::uint16_t{1} << lane)) == 0 || mask_row[x + lane] == 0)
                    continue;
                int label = 0;
                for (int index = 0; index < kShapeTemplateOrientationBins; ++index)
                    label += (angle_masks[static_cast<size_t>(index)] >> lane) & 1;
                label_row[x + lane] = static_cast<unsigned char>(
                    label == kShapeTemplateOrientationBins ? 0 : label);
            }
        }
        for (; x < labels.cols; ++x)
        {
            if (mask_row[x] != 0 && mag_row[x] >= threshold)
                label_row[x] = static_cast<unsigned char>(quantizeShapeTemplateAngle(angle_row[x]));
        }
    }
}

/** @brief AVX512 批量过滤候选点，候选内容和排序规则与 v0/v1 相同。 */
std::vector<ShapeTemplateCandidate>
collectCandidatesAvx512(const ShapeTemplateQuantizedGradient &gradient, const cv::Mat &mask, float threshold)
{
    std::vector<ShapeTemplateCandidate> candidates;
    const __m512 threshold_vec = _mm512_set1_ps(threshold);
    const __m512i invalid_vec = _mm512_set1_epi8(static_cast<char>(kShapeTemplateInvalidLabel));
    const __m512i zero_vec = _mm512_setzero_si512();

    for (int y = 1; y < gradient.labels.rows - 1; ++y)
    {
        const auto *mag_row = gradient.magnitude.ptr<float>(y);
        const auto *angle_row = gradient.angle.ptr<float>(y);
        const auto *mask_row = mask.ptr<unsigned char>(y);
        const auto *label_row = gradient.labels.ptr<unsigned char>(y);
        int x = 1;
        for (; x <= gradient.labels.cols - 65; x += 64)
        {
            const __m512i label_values = _mm512_loadu_si512(reinterpret_cast<const void *>(label_row + x));
            const __m512i mask_values = _mm512_loadu_si512(reinterpret_cast<const void *>(mask_row + x));
            const std::uint64_t label_valid = ~static_cast<std::uint64_t>(
                _mm512_cmpeq_epi8_mask(label_values, invalid_vec));
            const std::uint64_t mask_valid = ~static_cast<std::uint64_t>(
                _mm512_cmpeq_epi8_mask(mask_values, zero_vec));
            std::uint64_t magnitude_valid = 0;
            for (int group = 0; group < 4; ++group)
            {
                const __m512 values = _mm512_loadu_ps(mag_row + x + group * 16);
                magnitude_valid |= static_cast<std::uint64_t>(
                    _mm512_cmp_ps_mask(values, threshold_vec, _CMP_GE_OQ)) << (group * 16);
            }
            std::uint64_t valid = label_valid & mask_valid & magnitude_valid;
            while (valid != 0)
            {
                const int lane = static_cast<int>(std::countr_zero(valid));
                const int col = x + lane;
                candidates.push_back(ShapeTemplateCandidate{col, y, static_cast<int>(label_row[col]), angle_row[col],
                                                            std::max(mag_row[col], 0.0f)});
                valid &= valid - 1;
            }
        }
        for (; x < gradient.labels.cols - 1; ++x)
        {
            if (mask_row[x] == 0 || label_row[x] == kShapeTemplateInvalidLabel || mag_row[x] < threshold)
                continue;
            candidates.push_back(ShapeTemplateCandidate{x, y, static_cast<int>(label_row[x]), angle_row[x],
                                                        std::max(mag_row[x], 0.0f)});
        }
    }
    sortShapeTemplateCandidates(candidates);
    return candidates;
}

/** @brief AVX512 方向响应图构建；v2 默认不物化响应图，但保留完整 kernel 协议实现。 */
void fillResponseMapAvx512(const cv::Mat &labels, cv::Mat &response, int template_label,
                           const ShapeTemplateResponseTable &table)
{
    const __m512i lookup = _mm512_broadcast_i32x4(makeResponseLookup128(template_label, table));
    for (int y = 0; y < labels.rows; ++y)
    {
        const auto *src = labels.ptr<unsigned char>(y);
        auto *dst = response.ptr<unsigned char>(y);
        int x = 0;
        for (; x <= labels.cols - 64; x += 64)
        {
            const __m512i source = _mm512_loadu_si512(reinterpret_cast<const void *>(src + x));
            const __m512i output = _mm512_shuffle_epi8(lookup, source);
            _mm512_storeu_si512(reinterpret_cast<void *>(dst + x), output);
        }
        for (; x < labels.cols; ++x)
            dst[x] = src[x] < kShapeTemplateOrientationBins ? table[template_label][src[x]] : 0;
    }
}

/** @brief v2 AVX512F/BW 热点内核。 */
class ShapeTemplateMatcherAvx512KernelImpl final : public ShapeTemplateMatcherKernel
{
public:
    bool needsMaterializedResponseMaps() const noexcept override
    {
        return false;
    }

    bool usesOptimizedTemplateTraining() const noexcept override
    {
        return true;
    }

    void fillQuantizedLabels(const cv::Mat &magnitude, const cv::Mat &angle, const cv::Mat &mask,
                             float threshold, cv::Mat &labels) const override
    {
        fillQuantizedLabelsAvx512(magnitude, angle, mask, threshold, labels);
    }

    std::vector<ShapeTemplateCandidate>
    collectCandidates(const ShapeTemplateQuantizedGradient &gradient, const cv::Mat &mask, float threshold) const override
    {
        return collectCandidatesAvx512(gradient, mask, threshold);
    }

    void fillResponseMap(const cv::Mat &labels, cv::Mat &response, int template_label,
                         const ShapeTemplateResponseTable &table) const override
    {
        fillResponseMapAvx512(labels, response, template_label, table);
    }

    std::vector<ShapeTemplateScoredPosition>
    scanTemplate(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                 const cv::Mat &search_mask, bool full_search_mask, cv::Size image_size, int scan_step,
                 float threshold) const override
    {
        return scanTemplateAvx512(response_maps, templ, search_mask, full_search_mask, image_size, scan_step,
                                  threshold);
    }
};

/** @brief 创建只属于 v2 的 AVX512F/BW 内核，并在运行时验证 CPU 指令集。 */
static std::unique_ptr<ShapeTemplateMatcherKernel> makeV2Kernel()
{
    if (!cv::checkHardwareSupport(CV_CPU_AVX_512F) || !cv::checkHardwareSupport(CV_CPU_AVX_512BW))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION,
                             "The v2 shape template matcher requires AVX512F and AVX512BW support");
    }
    return std::make_unique<ShapeTemplateMatcherAvx512KernelImpl>();
}

} // namespace
#else
namespace {

/** @brief 未编译 AVX512 时保留明确的 v2 构造错误。 */
static std::unique_ptr<ShapeTemplateMatcherKernel> makeV2Kernel()
{
    throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION,
                         "The v2 shape template matcher was built without AVX512 support");
}

} // namespace
#endif

ShapeTemplateMatcherAvx512Impl::ShapeTemplateMatcherAvx512Impl(ShapeTemplateMatcherConfig config)
    : common::ShapeTemplateMatcherEngine(std::move(config), makeV2Kernel())
{
}

} // namespace irt::features::v2::detail
