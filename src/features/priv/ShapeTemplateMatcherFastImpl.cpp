/**
 * @file ShapeTemplateMatcherFastImpl.cpp
 * @brief v1 快速形状模板匹配器的具体实现与 AVX2 热点内核。
 */

#include "ShapeTemplateMatcherFastImpl.hpp"

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

#if defined(_M_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#endif

// v1 的 AVX2 实现不进入共享 detail 命名空间，避免与 v0 或未来 v2 的热点符号混用。
namespace irt::features::v1::detail {
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

#if defined(_M_AVX2) || defined(__AVX2__)
namespace {

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

/**
 * @brief 在相邻滑窗间批量评分，并保留早停剪枝。
 *
 * 与完整得分图累加不同，此实现每累加四个特征便检查整组窗口是否仍可能达到阈值；
 * 8-bit 累加路径一次处理 32 个候选，16-bit 回退路径一次处理 16 个候选；当整组都不可能
 * 命中时立即停止。这样既利用 AVX2 的连续读取，又保留真实大图中负样本占绝大多数时的关键
 * 剪枝收益。
 */
std::vector<ShapeTemplateScoredPosition>
scanTemplateFast(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                 const cv::Mat &search_mask, bool full_search_mask, cv::Size image_size, int scan_step,
                 float threshold)
{
    if (templ.features.empty() || templ.width > image_size.width || templ.height > image_size.height)
        return {};

    const int max_x = image_size.width - templ.width;
    const int max_y = image_size.height - templ.height;

    const size_t feature_count = templ.features.size();
    const std::uint64_t maximum_per_feature = static_cast<std::uint64_t>(response_maps.denominator_per_feature);
    const std::uint64_t maximum_sum = maximum_per_feature * static_cast<std::uint64_t>(feature_count);
    // 使用 signed 16-bit 比较实现寄存器内早停；较大配置保留原标量路径以避免改变语义。
    if (maximum_sum > std::numeric_limits<std::int16_t>::max())
    {
        return scanTemplateFallback(response_maps, templ, search_mask, full_search_mask, image_size,
                                    scan_step, threshold);
    }

    std::vector<ShapeTemplateScoredPosition> positions;
    const float denominator = static_cast<float>(maximum_sum);
    const std::uint64_t conservative_required_sum = static_cast<std::uint64_t>(
        threshold * static_cast<float>(maximum_sum) / 100.0f);
    constexpr size_t kPruneFeatureInterval = 4;
    const size_t response_step = response_maps.quantized_labels.step;
    struct PackedFeature
    {
        const unsigned char *base{nullptr};
        float expected_response{0.0f};
        int label{0};
    };
    std::vector<PackedFeature> feature_bases;
    feature_bases.reserve(feature_count);
    for (const auto &feature : templ.features)
    {
        feature_bases.push_back(PackedFeature{
            response_maps.quantized_labels.ptr<unsigned char>(feature.y) + feature.x,
            response_maps.average_responses[static_cast<size_t>(feature.label)], feature.label});
    }
    std::stable_sort(feature_bases.begin(), feature_bases.end(),
                     [](const PackedFeature &a, const PackedFeature &b)
                     {
                         return a.expected_response < b.expected_response;
                     });

    std::array<__m128i, kShapeTemplateOrientationBins> response_lookups{};
    for (int template_label = 0; template_label < kShapeTemplateOrientationBins; ++template_label)
    {
        alignas(16) std::array<unsigned char, 16> lookup_values{};
        for (int source_label = 0; source_label < kShapeTemplateOrientationBins; ++source_label)
        {
            lookup_values[static_cast<size_t>(source_label)]
                = response_maps.response_table[static_cast<size_t>(template_label)][static_cast<size_t>(source_label)];
        }
        response_lookups[static_cast<size_t>(template_label)]
            = _mm_loadu_si128(reinterpret_cast<const __m128i *>(lookup_values.data()));
    }

    // step=2 是最常见的空间抽样配置。候选坐标在内存中相隔一个字节，因而可从连续
    // 32-byte load 中用 shuffle 压缩出 16 个偶数位置，明显快于通用 gather。
    if (scan_step == 2)
    {
        alignas(16) static constexpr std::array<unsigned char, 16> kEvenByteIndices{
            0, 2, 4, 6, 8, 10, 12, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
        const __m128i even_byte_indices = _mm_load_si128(
            reinterpret_cast<const __m128i *>(kEvenByteIndices.data()));
        const __m128i zero128 = _mm_setzero_si128();
        int max_feature_x = 0;
        for (const auto &feature : templ.features)
            max_feature_x = std::max(max_feature_x, feature.x);
        const int max_read_offset = std::max(max_feature_x, templ.width / 2);

        // 每组读取 32 连续字节并产生 x, x+2, ..., x+30；末尾改由标量路径处理，避免跨行读取。
        const int last_vector_start = std::min(max_x - 30, image_size.width - max_read_offset - 32);
        const auto select_even_bytes = [&](const unsigned char *source)
        {
            const __m256i loaded = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source));
            const __m128i lower = _mm256_castsi256_si128(loaded);
            const __m128i upper = _mm256_extracti128_si256(loaded, 1);
            return _mm_unpacklo_epi64(_mm_shuffle_epi8(lower, even_byte_indices),
                                      _mm_shuffle_epi8(upper, even_byte_indices));
        };

        for (int y = 0; y <= max_y; y += 2)
        {
            const int center_y = std::min(image_size.height - 1, y + templ.height / 2);
            const auto *mask_row = full_search_mask ? nullptr : search_mask.ptr<unsigned char>(center_y);
            const size_t row_offset = static_cast<size_t>(y) * response_step;
            int x = 0;
            for (; x <= last_vector_start; x += 32)
            {
                int valid_lanes = 0xFFFF;
                if (!full_search_mask)
                {
                    const __m128i mask_values = select_even_bytes(mask_row + x + templ.width / 2);
                    valid_lanes = ~_mm_movemask_epi8(_mm_cmpeq_epi8(mask_values, zero128)) & 0xFFFF;
                }
                if (valid_lanes == 0)
                    continue;

                __m256i total = _mm256_setzero_si256();
                bool any_lane_can_match = true;
                for (size_t index = 0; index < feature_count; ++index)
                {
                    const auto *source = feature_bases[index].base + row_offset + x;
                    const __m128i source_labels = select_even_bytes(source);
                    const __m128i responses = _mm_shuffle_epi8(
                        response_lookups[static_cast<size_t>(feature_bases[index].label)], source_labels);
                    total = _mm256_add_epi16(total, _mm256_cvtepu8_epi16(responses));

                    const bool should_prune = (index + 1) % kPruneFeatureInterval == 0
                                           || index + 1 == feature_count;
                    if (!should_prune)
                        continue;
                    const std::uint64_t remaining = static_cast<std::uint64_t>(feature_count - index - 1);
                    const std::uint64_t maximum_remaining = remaining * maximum_per_feature;
                    if (conservative_required_sum <= maximum_remaining)
                        continue;
                    const auto minimum_current_sum = static_cast<std::int16_t>(
                        conservative_required_sum - maximum_remaining - 1);
                    const __m256i possible = _mm256_cmpgt_epi16(total, _mm256_set1_epi16(minimum_current_sum));
                    any_lane_can_match = _mm256_movemask_epi8(possible) != 0;
                    if (!any_lane_can_match)
                        break;
                }
                if (!any_lane_can_match)
                    continue;

                std::array<std::uint16_t, 16> lane_sums{};
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(lane_sums.data()), total);
                for (int lane = 0; lane < 16; ++lane)
                {
                    if ((valid_lanes & (1 << lane)) == 0)
                        continue;
                    const float score = 100.0f * static_cast<float>(lane_sums[static_cast<size_t>(lane)]) / denominator;
                    if (score >= threshold)
                        positions.push_back(ShapeTemplateScoredPosition{x + lane * 2, y, score});
                }
            }
            for (; x <= max_x; x += 2)
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

    // 对空间步长大于 1 的近似搜索，候选 x 不再连续，原 32-byte load 路径无法直接复用。
    // 这里一次 gather 8 个按步长间隔的位置，随后仍使用相同的查表、16-bit 累加和早停规则。
    // 因此它既保留 scan_step 语义，又避免退回 v0 风格的逐窗口标量评分。
    if (scan_step != 1)
    {
        const __m128i zero128 = _mm_setzero_si128();
        const __m256i low_byte_mask = _mm256_set1_epi32(0xFF);
        const __m256i gather_offsets = _mm256_setr_epi32(0, scan_step, 2 * scan_step, 3 * scan_step,
                                                          4 * scan_step, 5 * scan_step, 6 * scan_step,
                                                          7 * scan_step);
        int max_feature_x = 0;
        for (const auto &feature : templ.features)
            max_feature_x = std::max(max_feature_x, feature.x);
        const int max_read_offset = std::max(max_feature_x, templ.width / 2);

        // i32 gather 读取四个字节；最后几个候选改走标量尾部，确保不会跨越图像行边界。
        if (scan_step > image_size.width / 8)
        {
            return scanTemplateFallback(response_maps, templ, search_mask, full_search_mask, image_size,
                                        scan_step, threshold);
        }
        const int vector_span = 7 * scan_step;
        const int last_vector_start = std::min(max_x - vector_span,
                                               image_size.width - max_read_offset - 4 - vector_span);
        const auto gather_low_bytes = [&](const unsigned char *source)
        {
            const __m256i gathered = _mm256_and_si256(
                _mm256_i32gather_epi32(reinterpret_cast<const int *>(source), gather_offsets, 1), low_byte_mask);
            const __m128i lower = _mm256_castsi256_si128(gathered);
            const __m128i upper = _mm256_extracti128_si256(gathered, 1);
            const __m128i packed16 = _mm_packus_epi32(lower, upper);
            return _mm_packus_epi16(packed16, zero128);
        };

        for (int y = 0; y <= max_y; y += scan_step)
        {
            const int center_y = std::min(image_size.height - 1, y + templ.height / 2);
            const auto *mask_row = full_search_mask ? nullptr : search_mask.ptr<unsigned char>(center_y);
            const size_t row_offset = static_cast<size_t>(y) * response_step;
            int x = 0;
            for (; x <= last_vector_start; x += 8 * scan_step)
            {
                int valid_lanes = 0xFF;
                if (!full_search_mask)
                {
                    const __m128i mask_values = gather_low_bytes(mask_row + x + templ.width / 2);
                    valid_lanes = ~_mm_movemask_epi8(_mm_cmpeq_epi8(mask_values, zero128)) & 0xFF;
                }
                if (valid_lanes == 0)
                    continue;

                __m128i total = zero128;
                bool any_lane_can_match = true;
                for (size_t index = 0; index < feature_count; ++index)
                {
                    const auto *source = feature_bases[index].base + row_offset + x;
                    const __m128i source_labels = gather_low_bytes(source);
                    const __m128i responses = _mm_shuffle_epi8(
                        response_lookups[static_cast<size_t>(feature_bases[index].label)], source_labels);
                    total = _mm_add_epi16(total, _mm_cvtepu8_epi16(responses));

                    const bool should_prune = (index + 1) % kPruneFeatureInterval == 0
                                           || index + 1 == feature_count;
                    if (!should_prune)
                        continue;
                    const std::uint64_t remaining = static_cast<std::uint64_t>(feature_count - index - 1);
                    const std::uint64_t maximum_remaining = remaining * maximum_per_feature;
                    if (conservative_required_sum <= maximum_remaining)
                        continue;
                    const auto minimum_current_sum = static_cast<std::int16_t>(
                        conservative_required_sum - maximum_remaining - 1);
                    const __m128i possible = _mm_cmpgt_epi16(total, _mm_set1_epi16(minimum_current_sum));
                    any_lane_can_match = _mm_movemask_epi8(possible) != 0;
                    if (!any_lane_can_match)
                        break;
                }
                if (!any_lane_can_match)
                    continue;

                std::array<std::uint16_t, 8> lane_sums{};
                _mm_storeu_si128(reinterpret_cast<__m128i *>(lane_sums.data()), total);
                for (int lane = 0; lane < 8; ++lane)
                {
                    if ((valid_lanes & (1 << lane)) == 0)
                        continue;
                    const float score = 100.0f * static_cast<float>(lane_sums[static_cast<size_t>(lane)]) / denominator;
                    if (score >= threshold)
                        positions.push_back(ShapeTemplateScoredPosition{x + lane * scan_step, y, score});
                }
            }
            for (; x <= max_x; x += scan_step)
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

    // 对常见的 96 features、max_label_difference=1，最高分子仅为 192，可安全使用
    // uint8 累加并一次处理 32 个候选；更大配置继续走下方 uint16 路径。
    if (maximum_sum <= std::numeric_limits<std::uint8_t>::max())
    {
        std::array<__m256i, kShapeTemplateOrientationBins> response_lookups256{};
        for (int label = 0; label < kShapeTemplateOrientationBins; ++label)
        {
            response_lookups256[static_cast<size_t>(label)]
                = _mm256_broadcastsi128_si256(response_lookups[static_cast<size_t>(label)]);
        }

        const __m256i zero256 = _mm256_setzero_si256();
        const __m256i sign_bit = _mm256_set1_epi8(static_cast<char>(0x80));
        for (int y = 0; y <= max_y; ++y)
        {
            const int center_y = std::min(image_size.height - 1, y + templ.height / 2);
            const auto *mask_row = full_search_mask ? nullptr : search_mask.ptr<unsigned char>(center_y);
            const size_t row_offset = static_cast<size_t>(y) * response_step;
            int x = 0;
            for (; x <= max_x - 31; x += 32)
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

                __m256i total = zero256;
                bool any_lane_can_match = true;
                for (size_t index = 0; index < feature_count; ++index)
                {
                    const auto *source = feature_bases[index].base + row_offset + x;
                    const __m256i source_labels = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(source));
                    const __m256i responses = _mm256_shuffle_epi8(
                        response_lookups256[static_cast<size_t>(feature_bases[index].label)], source_labels);
                    total = _mm256_add_epi8(total, responses);

                    const bool should_prune = (index + 1) % kPruneFeatureInterval == 0
                                           || index + 1 == feature_count;
                    if (!should_prune)
                        continue;
                    const std::uint64_t remaining = static_cast<std::uint64_t>(feature_count - index - 1);
                    const std::uint64_t maximum_remaining = remaining * maximum_per_feature;
                    if (conservative_required_sum <= maximum_remaining)
                        continue;
                    const auto minimum_current_sum = static_cast<unsigned char>(
                        conservative_required_sum - maximum_remaining - 1);
                    const __m256i signed_total = _mm256_xor_si256(total, sign_bit);
                    const __m256i signed_minimum = _mm256_xor_si256(
                        _mm256_set1_epi8(static_cast<char>(minimum_current_sum)), sign_bit);
                    any_lane_can_match = _mm256_movemask_epi8(
                        _mm256_cmpgt_epi8(signed_total, signed_minimum)) != 0;
                    if (!any_lane_can_match)
                        break;
                }
                if (!any_lane_can_match)
                    continue;

                std::array<unsigned char, 32> lane_sums{};
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(lane_sums.data()), total);
                for (int lane = 0; lane < 32; ++lane)
                {
                    const std::uint32_t lane_bit = std::uint32_t{1} << lane;
                    if ((valid_lanes & lane_bit) == 0)
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

    const __m128i zero128 = _mm_setzero_si128();
    for (int y = 0; y <= max_y; ++y)
    {
        const int center_y = std::min(image_size.height - 1, y + templ.height / 2);
        const auto *mask_row = full_search_mask ? nullptr : search_mask.ptr<unsigned char>(center_y);
        const size_t row_offset = static_cast<size_t>(y) * response_step;
        int x = 0;
        for (; x <= max_x - 15; x += 16)
        {
            int valid_lanes = 0xFFFF;
            if (!full_search_mask)
            {
                const __m128i mask_values = _mm_loadu_si128(
                    reinterpret_cast<const __m128i *>(mask_row + x + templ.width / 2));
                valid_lanes = ~_mm_movemask_epi8(_mm_cmpeq_epi8(mask_values, zero128)) & 0xFFFF;
            }
            if (valid_lanes == 0)
                continue;

            __m256i total = _mm256_setzero_si256();
            bool any_lane_can_match = true;
            for (size_t index = 0; index < feature_count; ++index)
            {
                const auto *source = feature_bases[index].base + row_offset + x;
                const __m128i source_labels = _mm_loadu_si128(reinterpret_cast<const __m128i *>(source));
                const __m128i responses = _mm_shuffle_epi8(
                    response_lookups[static_cast<size_t>(feature_bases[index].label)], source_labels);
                total = _mm256_add_epi16(total, _mm256_cvtepu8_epi16(responses));

                const bool should_prune = (index + 1) % kPruneFeatureInterval == 0
                                       || index + 1 == feature_count;
                if (!should_prune)
                    continue;
                const std::uint64_t remaining = static_cast<std::uint64_t>(feature_count - index - 1);
                const std::uint64_t maximum_remaining = remaining * maximum_per_feature;
                if (conservative_required_sum <= maximum_remaining)
                    continue;
                const auto minimum_current_sum = static_cast<std::int16_t>(
                    conservative_required_sum - maximum_remaining - 1);
                const __m256i possible = _mm256_cmpgt_epi16(total, _mm256_set1_epi16(minimum_current_sum));
                any_lane_can_match = _mm256_movemask_epi8(possible) != 0;
                if (!any_lane_can_match)
                    break;
            }
            if (!any_lane_can_match)
                continue;

            std::array<std::uint16_t, 16> lane_sums{};
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(lane_sums.data()), total);
            for (int lane = 0; lane < 16; ++lane)
            {
                if ((valid_lanes & (1 << lane)) == 0)
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

void fillQuantizedLabelsAvx2(const cv::Mat &magnitude, const cv::Mat &angle, const cv::Mat &mask,
                             float threshold, cv::Mat &labels)
{
    const __m256 threshold_vec = _mm256_set1_ps(threshold);
    const std::array<__m256, kShapeTemplateOrientationBins> angle_thresholds{
        _mm256_set1_ps(22.5f),  _mm256_set1_ps(67.5f),  _mm256_set1_ps(112.5f), _mm256_set1_ps(157.5f),
        _mm256_set1_ps(202.5f), _mm256_set1_ps(247.5f), _mm256_set1_ps(292.5f), _mm256_set1_ps(337.5f),
    };

    for (int y = 0; y < labels.rows; ++y)
    {
        const auto *mag_row   = magnitude.ptr<float>(y);
        const auto *angle_row = angle.ptr<float>(y);
        const auto *mask_row  = mask.ptr<unsigned char>(y);
        auto       *label_row = labels.ptr<unsigned char>(y);

        int x = 0;
        for (; x <= labels.cols - 8; x += 8)
        {
            const __m256 mag_values   = _mm256_loadu_ps(mag_row + x);
            const int valid_mag = _mm256_movemask_ps(_mm256_cmp_ps(mag_values, threshold_vec, _CMP_GE_OQ));
            const __m256 angle_values = _mm256_loadu_ps(angle_row + x);

            int labels8[8]{0, 0, 0, 0, 0, 0, 0, 0};
            for (int threshold_index = 0; threshold_index < kShapeTemplateOrientationBins; ++threshold_index)
            {
                const int ge_mask = _mm256_movemask_ps(
                    _mm256_cmp_ps(angle_values, angle_thresholds[threshold_index], _CMP_GE_OQ));
                for (int lane = 0; lane < 8; ++lane)
                {
                    labels8[lane] += (ge_mask >> lane) & 1;
                }
            }

            for (int lane = 0; lane < 8; ++lane)
            {
                if (((valid_mag >> lane) & 1) != 0 && mask_row[x + lane] != 0)
                {
                    label_row[x + lane] = static_cast<unsigned char>(
                        labels8[lane] == kShapeTemplateOrientationBins ? 0 : labels8[lane]);
                }
            }
        }

        for (; x < labels.cols; ++x)
        {
            if (mask_row[x] != 0 && mag_row[x] >= threshold)
            {
                label_row[x] = static_cast<unsigned char>(quantizeShapeTemplateAngle(angle_row[x]));
            }
        }
    }
}

std::vector<ShapeTemplateCandidate>
collectCandidatesAvx2(const ShapeTemplateQuantizedGradient &gradient, const cv::Mat &mask, float threshold)
{
    std::vector<ShapeTemplateCandidate> candidates;
    const __m256 threshold_vec = _mm256_set1_ps(threshold);
    const __m256i invalid_vec  = _mm256_set1_epi8(static_cast<char>(kShapeTemplateInvalidLabel));
    const __m256i zero_vec     = _mm256_setzero_si256();
    const __m256i all_ones     = _mm256_set1_epi8(static_cast<char>(0xFF));

    for (int y = 1; y < gradient.labels.rows - 1; ++y)
    {
        const auto *mag_row   = gradient.magnitude.ptr<float>(y);
        const auto *angle_row = gradient.angle.ptr<float>(y);
        const auto *mask_row  = mask.ptr<unsigned char>(y);
        const auto *label_row = gradient.labels.ptr<unsigned char>(y);

        int x = 1;
        for (; x <= gradient.labels.cols - 33; x += 32)
        {
            const __m256i label_values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(label_row + x));
            const __m256i mask_values  = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(mask_row + x));
            const __m256i label_valid = _mm256_andnot_si256(_mm256_cmpeq_epi8(label_values, invalid_vec), all_ones);
            const __m256i mask_valid  = _mm256_andnot_si256(_mm256_cmpeq_epi8(mask_values, zero_vec), all_ones);
            const int byte_valid = _mm256_movemask_epi8(_mm256_and_si256(label_valid, mask_valid));
            if (byte_valid == 0)
            {
                continue;
            }

            int magnitude_valid = 0;
            for (int group = 0; group < 4; ++group)
            {
                const __m256 values = _mm256_loadu_ps(mag_row + x + group * 8);
                magnitude_valid |= _mm256_movemask_ps(_mm256_cmp_ps(values, threshold_vec, _CMP_GE_OQ))
                                << (group * 8);
            }

            uint32_t valid = static_cast<uint32_t>(byte_valid) & static_cast<uint32_t>(magnitude_valid);
            while (valid != 0)
            {
                const int lane = static_cast<int>(std::countr_zero(valid));
                const int col  = x + lane;
                candidates.push_back(ShapeTemplateCandidate{col, y, static_cast<int>(label_row[col]), angle_row[col],
                                                            std::max(mag_row[col], 0.0f)});
                valid &= valid - 1;
            }
        }

        for (; x < gradient.labels.cols - 1; ++x)
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

void fillResponseMapAvx2(const cv::Mat &labels, cv::Mat &response, int template_label,
                         const ShapeTemplateResponseTable &table)
{
    alignas(32) std::array<unsigned char, 32> lookup_values{};
    for (int label = 0; label < kShapeTemplateOrientationBins; ++label)
    {
        lookup_values[static_cast<size_t>(label)]      = table[template_label][label];
        lookup_values[static_cast<size_t>(label) + 16] = table[template_label][label];
    }
    const __m256i lookup = _mm256_load_si256(reinterpret_cast<const __m256i *>(lookup_values.data()));

    for (int y = 0; y < labels.rows; ++y)
    {
        const auto *src = labels.ptr<unsigned char>(y);
        auto       *dst = response.ptr<unsigned char>(y);

        int x = 0;
        for (; x <= labels.cols - 32; x += 32)
        {
            const __m256i source = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + x));
            const __m256i out    = _mm256_shuffle_epi8(lookup, source);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + x), out);
        }
        for (; x < labels.cols; ++x)
        {
            dst[x] = src[x] < kShapeTemplateOrientationBins ? table[template_label][src[x]] : 0;
        }
    }
}

/** @brief v1 快速热点内核，使用 AVX2 批量量化、筛选和评分。 */
class ShapeTemplateMatcherFastKernelImpl final : public ShapeTemplateMatcherKernel
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
        fillQuantizedLabelsAvx2(magnitude, angle, mask, threshold, labels);
    }

    std::vector<ShapeTemplateCandidate>
    collectCandidates(const ShapeTemplateQuantizedGradient &gradient, const cv::Mat &mask, float threshold) const override
    {
        return collectCandidatesAvx2(gradient, mask, threshold);
    }

    void fillResponseMap(const cv::Mat &labels, cv::Mat &response, int template_label,
                         const ShapeTemplateResponseTable &table) const override
    {
        fillResponseMapAvx2(labels, response, template_label, table);
    }

    std::vector<ShapeTemplateScoredPosition>
    scanTemplate(const ShapeTemplateResponseMaps &response_maps, const ShapeTemplateInfo &templ,
                 const cv::Mat &search_mask, bool full_search_mask, cv::Size image_size, int scan_step,
                 float threshold) const override
    {
        return scanTemplateFast(response_maps, templ, search_mask, full_search_mask, image_size, scan_step,
                                threshold);
    }
};

} // namespace

/** @brief 创建只属于 v1 的 AVX2 热点内核，并在运行时验证 CPU 指令集。 */
static std::unique_ptr<ShapeTemplateMatcherKernel> makeV1Kernel()
{
    if (!cv::checkHardwareSupport(CV_CPU_AVX2))
    {
        throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION,
                             "The v1 shape template matcher requires an AVX2-capable CPU");
    }
    return std::make_unique<ShapeTemplateMatcherFastKernelImpl>();
}

#else

/** @brief 未编译 AVX2 时保留明确的 v1 构造错误。 */
static std::unique_ptr<ShapeTemplateMatcherKernel> makeV1Kernel()
{
    throw irt::Exception(irt::Status::ERROR_INVALID_OPERATION,
                         "The v1 shape template matcher was built without AVX2 support");
}

#endif

ShapeTemplateMatcherFastImpl::ShapeTemplateMatcherFastImpl(ShapeTemplateMatcherConfig config)
    : common::ShapeTemplateMatcherEngine(std::move(config), makeV1Kernel())
{
}

} // namespace irt::features::v1::detail
