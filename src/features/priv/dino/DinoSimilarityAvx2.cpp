/**
 * @file DinoSimilarityAvx2.cpp
 * @brief 相似度归约内核的 AVX2 实现。
 *
 * 该编译单元单独启用 AVX2，与特征维无关的公共流程保持标量目标；
 * 运行期是否调用由 ``DinoSimilarity.cpp`` 按 CPU 支持决定。
 */

#include "DinoSimilarity.hpp"

#include <inferrt/core/Exception.hpp>

#include <cstddef>

#if defined(INFERRT_DINO_SIMD_AVX2)
#include <immintrin.h>
#endif

namespace irt::features::priv {

#if defined(INFERRT_DINO_SIMD_AVX2)
namespace {

/** @brief 8 路累加器的水平求和。 */
inline float horizontalSum(const __m256 value)
{
    const __m128 low  = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128       sum  = _mm_add_ps(low, high);
    sum               = _mm_hadd_ps(sum, sum);
    sum               = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

/** @brief 单个描述子与单个查询 token 的点积；尾部不足一个向量时按标量收尾。 */
inline float dotAvx2(const float *descriptor, const float *query, const int dimension)
{
    __m256 accumulator = _mm256_setzero_ps();
    int    channel     = 0;
    for (; channel + 8 <= dimension; channel += 8)
    {
        accumulator = _mm256_fmadd_ps(_mm256_loadu_ps(descriptor + channel), _mm256_loadu_ps(query + channel),
                                      accumulator);
    }
    float total = horizontalSum(accumulator);
    for (; channel < dimension; ++channel)
    {
        total += descriptor[channel] * query[channel];
    }
    return total;
}

/** @brief 一次处理 4 个描述子，让同一条查询 token 的载入被 4 个点积复用。 */
constexpr std::size_t kDescriptorUnroll = 4;

} // namespace
#endif

void dinoAccumulateSimilarityMaximaFast(const float *descriptors, const std::size_t descriptor_count,
                                        const float *tokens, const int token_count, const int dimension,
                                        const int *slot_of_token, float *maxima)
{
#if defined(INFERRT_DINO_SIMD_AVX2)
    std::size_t index = 0;
    for (; index + kDescriptorUnroll <= descriptor_count; index += kDescriptorUnroll)
    {
        const float *group[kDescriptorUnroll];
        for (std::size_t slot = 0; slot < kDescriptorUnroll; ++slot)
        {
            group[slot] = descriptors + (index + slot) * static_cast<std::size_t>(dimension);
        }
        for (int token = 0; token < token_count; ++token)
        {
            const float *query = tokens + static_cast<std::size_t>(token) * static_cast<std::size_t>(dimension);
            const int    slot  = slot_of_token[token];
            for (std::size_t member = 0; member < kDescriptorUnroll; ++member)
            {
                const float score = dotAvx2(group[member], query, dimension);
                if (score > maxima[slot])
                {
                    maxima[slot] = score;
                }
            }
        }
    }
    for (; index < descriptor_count; ++index)
    {
        const float *descriptor = descriptors + index * static_cast<std::size_t>(dimension);
        for (int token = 0; token < token_count; ++token)
        {
            const float *query = tokens + static_cast<std::size_t>(token) * static_cast<std::size_t>(dimension);
            const int    slot  = slot_of_token[token];
            const float  score = dotAvx2(descriptor, query, dimension);
            if (score > maxima[slot])
            {
                maxima[slot] = score;
            }
        }
    }
#else
    (void)descriptors;
    (void)descriptor_count;
    (void)tokens;
    (void)token_count;
    (void)dimension;
    (void)slot_of_token;
    (void)maxima;
    throw irt::Exception(irt::Status::INVALID_OPERATION,
                         "The DINO similarity kernel was built without AVX2 support");
#endif
}

} // namespace irt::features::priv
