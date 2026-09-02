#include "ClusteringDistance.hpp"

#include <inferrt/core/Exception.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#if defined(__AVX2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <immintrin.h>
#define IRT_CLUSTERING_DISTANCE_HAS_AVX2 1
#else
#define IRT_CLUSTERING_DISTANCE_HAS_AVX2 0
#endif

namespace irt::ops::detail {
namespace {

#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
[[nodiscard]] double squaredEuclideanDistanceAvx2(const float *lhs_ptr, const float *rhs_ptr, int64_t num_features);
#endif

[[nodiscard]] double sampleValue(const float *samples, int64_t sample, int64_t feature, int64_t num_features)
{
    return static_cast<double>(samples[sample * num_features + feature]);
}

[[nodiscard]] double squaredEuclideanDistanceUnchecked(const float *samples, int64_t lhs, int64_t rhs,
                                                       int64_t num_features)
{
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;
#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
    if (num_features >= 4)
    {
        return squaredEuclideanDistanceAvx2(lhs_ptr, rhs_ptr, num_features);
    }
#endif

    double sum = 0.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        const double diff = static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature]);
        sum += diff * diff;
    }
    return sum;
}

[[nodiscard]] bool isMinkowskiP3(double minkowski_p)
{
    return std::abs(minkowski_p - 3.0) <= 1e-12;
}

[[nodiscard]] double minkowskiPower(double value, double minkowski_p)
{
    if (isMinkowskiP3(minkowski_p))
    {
        return value * value * value;
    }
    if (minkowski_p == 2.0)
    {
        return value * value;
    }
    if (minkowski_p == 1.0)
    {
        return value;
    }
    return std::pow(value, minkowski_p);
}

[[nodiscard]] double minkowskiRoot(double value, double minkowski_p)
{
    if (isMinkowskiP3(minkowski_p))
    {
        return std::cbrt(value);
    }
    if (minkowski_p == 2.0)
    {
        return std::sqrt(value);
    }
    if (minkowski_p == 1.0)
    {
        return value;
    }
    return std::pow(value, 1.0 / minkowski_p);
}

#if IRT_CLUSTERING_DISTANCE_HAS_AVX2

[[nodiscard]] __m256d absPd256(__m256d value)
{
    return _mm256_andnot_pd(_mm256_set1_pd(-0.0), value);
}

[[nodiscard]] __m256d loadFloat4AsDouble(const float *values)
{
    return _mm256_cvtps_pd(_mm_loadu_ps(values));
}

[[nodiscard]] double sumPd256(__m256d values)
{
    double lanes[4];
    _mm256_storeu_pd(lanes, values);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

[[nodiscard]] double maxPd256(__m256d values)
{
    double lanes[4];
    _mm256_storeu_pd(lanes, values);
    return std::max(std::max(lanes[0], lanes[1]), std::max(lanes[2], lanes[3]));
}

[[nodiscard]] bool canUseAvx2FeatureDistance(int64_t num_features, ClusteringMetric metric, double minkowski_p)
{
    if (num_features < 4)
    {
        return false;
    }
    return metric == ClusteringMetric::Euclidean || metric == ClusteringMetric::Manhattan
           || metric == ClusteringMetric::Chebyshev || metric == ClusteringMetric::Cosine
           || (metric == ClusteringMetric::Minkowski
               && (isMinkowskiP3(minkowski_p) || minkowski_p == 1.0 || minkowski_p == 2.0));
}

[[nodiscard]] double squaredEuclideanDistanceAvx2(const float *lhs_ptr, const float *rhs_ptr, int64_t num_features)
{
    __m256d sum     = _mm256_setzero_pd();
    int64_t feature = 0;
    for (; feature + 3 < num_features; feature += 4)
    {
        const __m256d lhs  = loadFloat4AsDouble(lhs_ptr + feature);
        const __m256d rhs  = loadFloat4AsDouble(rhs_ptr + feature);
        const __m256d diff = _mm256_sub_pd(lhs, rhs);
        sum                = _mm256_add_pd(sum, _mm256_mul_pd(diff, diff));
    }
    double result = sumPd256(sum);
    for (; feature < num_features; ++feature)
    {
        const double diff = static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature]);
        result += diff * diff;
    }
    return result;
}

[[nodiscard]] double manhattanDistanceAvx2(const float *lhs_ptr, const float *rhs_ptr, int64_t num_features)
{
    __m256d sum     = _mm256_setzero_pd();
    int64_t feature = 0;
    for (; feature + 3 < num_features; feature += 4)
    {
        const __m256d lhs  = loadFloat4AsDouble(lhs_ptr + feature);
        const __m256d rhs  = loadFloat4AsDouble(rhs_ptr + feature);
        const __m256d diff = absPd256(_mm256_sub_pd(lhs, rhs));
        sum                = _mm256_add_pd(sum, diff);
    }
    double result = sumPd256(sum);
    for (; feature < num_features; ++feature)
    {
        result += std::abs(static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature]));
    }
    return result;
}

[[nodiscard]] double chebyshevDistanceAvx2(const float *lhs_ptr, const float *rhs_ptr, int64_t num_features)
{
    __m256d max_diff = _mm256_setzero_pd();
    int64_t feature  = 0;
    for (; feature + 3 < num_features; feature += 4)
    {
        const __m256d lhs  = loadFloat4AsDouble(lhs_ptr + feature);
        const __m256d rhs  = loadFloat4AsDouble(rhs_ptr + feature);
        const __m256d diff = absPd256(_mm256_sub_pd(lhs, rhs));
        max_diff           = _mm256_max_pd(max_diff, diff);
    }
    double result = maxPd256(max_diff);
    for (; feature < num_features; ++feature)
    {
        result = std::max(result,
                          std::abs(static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature])));
    }
    return result;
}

[[nodiscard]] double minkowskiP3DistanceAvx2(const float *lhs_ptr, const float *rhs_ptr, int64_t num_features)
{
    __m256d sum     = _mm256_setzero_pd();
    int64_t feature = 0;
    for (; feature + 3 < num_features; feature += 4)
    {
        const __m256d lhs    = loadFloat4AsDouble(lhs_ptr + feature);
        const __m256d rhs    = loadFloat4AsDouble(rhs_ptr + feature);
        const __m256d diff   = absPd256(_mm256_sub_pd(lhs, rhs));
        const __m256d diff_2 = _mm256_mul_pd(diff, diff);
        sum                  = _mm256_add_pd(sum, _mm256_mul_pd(diff_2, diff));
    }
    double result = sumPd256(sum);
    for (; feature < num_features; ++feature)
    {
        const double diff = std::abs(static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature]));
        result += diff * diff * diff;
    }
    return result;
}

[[nodiscard]] double dotProductAvx2(const float *lhs_ptr, const float *rhs_ptr, int64_t num_features)
{
    __m256d sum     = _mm256_setzero_pd();
    int64_t feature = 0;
    for (; feature + 3 < num_features; feature += 4)
    {
        const __m256d lhs = loadFloat4AsDouble(lhs_ptr + feature);
        const __m256d rhs = loadFloat4AsDouble(rhs_ptr + feature);
        sum               = _mm256_add_pd(sum, _mm256_mul_pd(lhs, rhs));
    }
    double result = sumPd256(sum);
    for (; feature < num_features; ++feature)
    {
        result += static_cast<double>(lhs_ptr[feature]) * static_cast<double>(rhs_ptr[feature]);
    }
    return result;
}

[[nodiscard]] __m256d loadContiguousFeature4(const float *samples, int64_t num_features, int64_t first_sample,
                                             int64_t feature)
{
    return _mm256_set_pd(static_cast<double>(samples[(first_sample + 3) * num_features + feature]),
                         static_cast<double>(samples[(first_sample + 2) * num_features + feature]),
                         static_cast<double>(samples[(first_sample + 1) * num_features + feature]),
                         static_cast<double>(samples[first_sample * num_features + feature]));
}

[[nodiscard]] __m256d loadIndexedFeature4(const float *samples, int64_t num_features, const int64_t *indices,
                                          int64_t feature)
{
    return _mm256_set_pd(static_cast<double>(samples[indices[3] * num_features + feature]),
                         static_cast<double>(samples[indices[2] * num_features + feature]),
                         static_cast<double>(samples[indices[1] * num_features + feature]),
                         static_cast<double>(samples[indices[0] * num_features + feature]));
}

[[nodiscard]] DistanceBlock4 storeDistanceBlock4(__m256d values)
{
    double lanes[4];
    _mm256_storeu_pd(lanes, values);
    return {lanes[0], lanes[1], lanes[2], lanes[3]};
}

[[nodiscard]] DistanceBlock4 highDimSquaredEuclideanBlock4(const float *lhs_ptr, const float *rhs0_ptr,
                                                           const float *rhs1_ptr, const float *rhs2_ptr,
                                                           const float *rhs3_ptr, int64_t num_features)
{
    __m256d sum0    = _mm256_setzero_pd();
    __m256d sum1    = _mm256_setzero_pd();
    __m256d sum2    = _mm256_setzero_pd();
    __m256d sum3    = _mm256_setzero_pd();
    int64_t feature = 0;
    for (; feature + 3 < num_features; feature += 4)
    {
        const __m256d lhs = loadFloat4AsDouble(lhs_ptr + feature);
        const auto accumulate = [&](const float *rhs_ptr, __m256d &sum)
        {
            const __m256d diff = _mm256_sub_pd(lhs, loadFloat4AsDouble(rhs_ptr + feature));
            sum                = _mm256_add_pd(sum, _mm256_mul_pd(diff, diff));
        };
        accumulate(rhs0_ptr, sum0);
        accumulate(rhs1_ptr, sum1);
        accumulate(rhs2_ptr, sum2);
        accumulate(rhs3_ptr, sum3);
    }

    DistanceBlock4 result{sumPd256(sum0), sumPd256(sum1), sumPd256(sum2), sumPd256(sum3)};
    for (; feature < num_features; ++feature)
    {
        const double lhs = static_cast<double>(lhs_ptr[feature]);
        const auto add = [&](const float *rhs_ptr, double &sum)
        {
            const double diff = lhs - static_cast<double>(rhs_ptr[feature]);
            sum += diff * diff;
        };
        add(rhs0_ptr, result.first);
        add(rhs1_ptr, result.second);
        add(rhs2_ptr, result.third);
        add(rhs3_ptr, result.fourth);
    }
    return result;
}

[[nodiscard]] DistanceBlock4 highDimManhattanBlock4(const float *lhs_ptr, const float *rhs0_ptr,
                                                    const float *rhs1_ptr, const float *rhs2_ptr,
                                                    const float *rhs3_ptr, int64_t num_features)
{
    __m256d sum0    = _mm256_setzero_pd();
    __m256d sum1    = _mm256_setzero_pd();
    __m256d sum2    = _mm256_setzero_pd();
    __m256d sum3    = _mm256_setzero_pd();
    int64_t feature = 0;
    for (; feature + 3 < num_features; feature += 4)
    {
        const __m256d lhs = loadFloat4AsDouble(lhs_ptr + feature);
        const auto accumulate = [&](const float *rhs_ptr, __m256d &sum)
        {
            const __m256d diff = absPd256(_mm256_sub_pd(lhs, loadFloat4AsDouble(rhs_ptr + feature)));
            sum                = _mm256_add_pd(sum, diff);
        };
        accumulate(rhs0_ptr, sum0);
        accumulate(rhs1_ptr, sum1);
        accumulate(rhs2_ptr, sum2);
        accumulate(rhs3_ptr, sum3);
    }

    DistanceBlock4 result{sumPd256(sum0), sumPd256(sum1), sumPd256(sum2), sumPd256(sum3)};
    for (; feature < num_features; ++feature)
    {
        const double lhs = static_cast<double>(lhs_ptr[feature]);
        const auto add = [&](const float *rhs_ptr, double &sum)
        { sum += std::abs(lhs - static_cast<double>(rhs_ptr[feature])); };
        add(rhs0_ptr, result.first);
        add(rhs1_ptr, result.second);
        add(rhs2_ptr, result.third);
        add(rhs3_ptr, result.fourth);
    }
    return result;
}

[[nodiscard]] DistanceBlock4 highDimChebyshevBlock4(const float *lhs_ptr, const float *rhs0_ptr,
                                                    const float *rhs1_ptr, const float *rhs2_ptr,
                                                    const float *rhs3_ptr, int64_t num_features)
{
    __m256d max0    = _mm256_setzero_pd();
    __m256d max1    = _mm256_setzero_pd();
    __m256d max2    = _mm256_setzero_pd();
    __m256d max3    = _mm256_setzero_pd();
    int64_t feature = 0;
    for (; feature + 3 < num_features; feature += 4)
    {
        const __m256d lhs = loadFloat4AsDouble(lhs_ptr + feature);
        const auto accumulate = [&](const float *rhs_ptr, __m256d &max_value)
        {
            const __m256d diff = absPd256(_mm256_sub_pd(lhs, loadFloat4AsDouble(rhs_ptr + feature)));
            max_value          = _mm256_max_pd(max_value, diff);
        };
        accumulate(rhs0_ptr, max0);
        accumulate(rhs1_ptr, max1);
        accumulate(rhs2_ptr, max2);
        accumulate(rhs3_ptr, max3);
    }

    DistanceBlock4 result{maxPd256(max0), maxPd256(max1), maxPd256(max2), maxPd256(max3)};
    for (; feature < num_features; ++feature)
    {
        const double lhs = static_cast<double>(lhs_ptr[feature]);
        const auto update = [&](const float *rhs_ptr, double &max_value)
        { max_value = std::max(max_value, std::abs(lhs - static_cast<double>(rhs_ptr[feature]))); };
        update(rhs0_ptr, result.first);
        update(rhs1_ptr, result.second);
        update(rhs2_ptr, result.third);
        update(rhs3_ptr, result.fourth);
    }
    return result;
}

[[nodiscard]] DistanceBlock4 highDimMinkowskiP3Block4(const float *lhs_ptr, const float *rhs0_ptr,
                                                      const float *rhs1_ptr, const float *rhs2_ptr,
                                                      const float *rhs3_ptr, int64_t num_features)
{
    __m256d sum0    = _mm256_setzero_pd();
    __m256d sum1    = _mm256_setzero_pd();
    __m256d sum2    = _mm256_setzero_pd();
    __m256d sum3    = _mm256_setzero_pd();
    int64_t feature = 0;
    for (; feature + 3 < num_features; feature += 4)
    {
        const __m256d lhs = loadFloat4AsDouble(lhs_ptr + feature);
        const auto accumulate = [&](const float *rhs_ptr, __m256d &sum)
        {
            const __m256d diff   = absPd256(_mm256_sub_pd(lhs, loadFloat4AsDouble(rhs_ptr + feature)));
            const __m256d diff_2 = _mm256_mul_pd(diff, diff);
            sum                  = _mm256_add_pd(sum, _mm256_mul_pd(diff_2, diff));
        };
        accumulate(rhs0_ptr, sum0);
        accumulate(rhs1_ptr, sum1);
        accumulate(rhs2_ptr, sum2);
        accumulate(rhs3_ptr, sum3);
    }

    DistanceBlock4 result{sumPd256(sum0), sumPd256(sum1), sumPd256(sum2), sumPd256(sum3)};
    for (; feature < num_features; ++feature)
    {
        const double lhs = static_cast<double>(lhs_ptr[feature]);
        const auto add = [&](const float *rhs_ptr, double &sum)
        {
            const double diff = std::abs(lhs - static_cast<double>(rhs_ptr[feature]));
            sum += diff * diff * diff;
        };
        add(rhs0_ptr, result.first);
        add(rhs1_ptr, result.second);
        add(rhs2_ptr, result.third);
        add(rhs3_ptr, result.fourth);
    }
    return result;
}

[[nodiscard]] DistanceBlock4 highDimDotBlock4(const float *lhs_ptr, const float *rhs0_ptr, const float *rhs1_ptr,
                                              const float *rhs2_ptr, const float *rhs3_ptr, int64_t num_features)
{
    __m256d sum0    = _mm256_setzero_pd();
    __m256d sum1    = _mm256_setzero_pd();
    __m256d sum2    = _mm256_setzero_pd();
    __m256d sum3    = _mm256_setzero_pd();
    int64_t feature = 0;
    for (; feature + 3 < num_features; feature += 4)
    {
        const __m256d lhs = loadFloat4AsDouble(lhs_ptr + feature);
        const auto accumulate = [&](const float *rhs_ptr, __m256d &sum)
        { sum = _mm256_add_pd(sum, _mm256_mul_pd(lhs, loadFloat4AsDouble(rhs_ptr + feature))); };
        accumulate(rhs0_ptr, sum0);
        accumulate(rhs1_ptr, sum1);
        accumulate(rhs2_ptr, sum2);
        accumulate(rhs3_ptr, sum3);
    }

    DistanceBlock4 result{sumPd256(sum0), sumPd256(sum1), sumPd256(sum2), sumPd256(sum3)};
    for (; feature < num_features; ++feature)
    {
        const double lhs = static_cast<double>(lhs_ptr[feature]);
        const auto add = [&](const float *rhs_ptr, double &sum)
        { sum += lhs * static_cast<double>(rhs_ptr[feature]); };
        add(rhs0_ptr, result.first);
        add(rhs1_ptr, result.second);
        add(rhs2_ptr, result.third);
        add(rhs3_ptr, result.fourth);
    }
    return result;
}

[[nodiscard]] DistanceBlock4 highDimSearchDistanceBlock4(const float *lhs_ptr, const float *rhs0_ptr,
                                                         const float *rhs1_ptr, const float *rhs2_ptr,
                                                         const float *rhs3_ptr, int64_t num_features,
                                                         ClusteringMetric metric, double minkowski_p)
{
    if (metric == ClusteringMetric::Euclidean || (metric == ClusteringMetric::Minkowski && minkowski_p == 2.0))
    {
        return highDimSquaredEuclideanBlock4(lhs_ptr, rhs0_ptr, rhs1_ptr, rhs2_ptr, rhs3_ptr, num_features);
    }
    if (metric == ClusteringMetric::Manhattan || (metric == ClusteringMetric::Minkowski && minkowski_p == 1.0))
    {
        return highDimManhattanBlock4(lhs_ptr, rhs0_ptr, rhs1_ptr, rhs2_ptr, rhs3_ptr, num_features);
    }
    if (metric == ClusteringMetric::Chebyshev)
    {
        return highDimChebyshevBlock4(lhs_ptr, rhs0_ptr, rhs1_ptr, rhs2_ptr, rhs3_ptr, num_features);
    }
    return highDimMinkowskiP3Block4(lhs_ptr, rhs0_ptr, rhs1_ptr, rhs2_ptr, rhs3_ptr, num_features);
}

[[nodiscard]] double cosineDistanceFromDot(double dot, double lhs_inv_norm, double rhs_inv_norm)
{
    if (lhs_inv_norm == 0.0 && rhs_inv_norm == 0.0)
    {
        return 0.0;
    }
    if (lhs_inv_norm == 0.0 || rhs_inv_norm == 0.0)
    {
        return 1.0;
    }
    const double similarity = std::clamp(dot * lhs_inv_norm * rhs_inv_norm, -1.0, 1.0);
    return 1.0 - similarity;
}

[[nodiscard]] DistanceBlock4 highDimCosineDistanceBlock4(const float *lhs_ptr, const float *rhs0_ptr,
                                                         const float *rhs1_ptr, const float *rhs2_ptr,
                                                         const float *rhs3_ptr, int64_t num_features,
                                                         double lhs_inv_norm, double rhs_inv0, double rhs_inv1,
                                                         double rhs_inv2, double rhs_inv3)
{
    DistanceBlock4 dots = highDimDotBlock4(lhs_ptr, rhs0_ptr, rhs1_ptr, rhs2_ptr, rhs3_ptr, num_features);
    return {cosineDistanceFromDot(dots.first, lhs_inv_norm, rhs_inv0),
            cosineDistanceFromDot(dots.second, lhs_inv_norm, rhs_inv1),
            cosineDistanceFromDot(dots.third, lhs_inv_norm, rhs_inv2),
            cosineDistanceFromDot(dots.fourth, lhs_inv_norm, rhs_inv3)};
}

template <typename LoadRhsFeature>
[[nodiscard]] __m256d lowDimSearchDistance4(const float *samples, int64_t num_features, int64_t lhs,
                                            LoadRhsFeature load_rhs_feature, ClusteringMetric metric,
                                            double minkowski_p)
{
    __m256d result = metric == ClusteringMetric::Chebyshev ? _mm256_setzero_pd() : _mm256_setzero_pd();
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        const __m256d lhs_value = _mm256_set1_pd(sampleValue(samples, lhs, feature, num_features));
        const __m256d rhs_value = load_rhs_feature(feature);
        const __m256d diff      = absPd256(_mm256_sub_pd(lhs_value, rhs_value));

        if (metric == ClusteringMetric::Chebyshev)
        {
            result = _mm256_max_pd(result, diff);
        }
        else if (metric == ClusteringMetric::Minkowski && isMinkowskiP3(minkowski_p))
        {
            const __m256d diff_2 = _mm256_mul_pd(diff, diff);
            result               = _mm256_add_pd(result, _mm256_mul_pd(diff_2, diff));
        }
        else if (metric == ClusteringMetric::Euclidean
                 || (metric == ClusteringMetric::Minkowski && minkowski_p == 2.0))
        {
            result = _mm256_add_pd(result, _mm256_mul_pd(diff, diff));
        }
        else
        {
            result = _mm256_add_pd(result, diff);
        }
    }
    return result;
}

template <typename LoadRhsFeature>
[[nodiscard]] __m256d lowDimCosineDistance4(const float *samples, int64_t num_features, int64_t lhs,
                                            LoadRhsFeature load_rhs_feature, double lhs_inv_norm,
                                            __m256d rhs_inv_norms)
{
    __m256d dot = _mm256_setzero_pd();
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        const __m256d lhs_value = _mm256_set1_pd(sampleValue(samples, lhs, feature, num_features));
        dot                     = _mm256_add_pd(dot, _mm256_mul_pd(lhs_value, load_rhs_feature(feature)));
    }
    __m256d similarity = _mm256_mul_pd(_mm256_mul_pd(dot, _mm256_set1_pd(lhs_inv_norm)), rhs_inv_norms);
    similarity         = _mm256_min_pd(_mm256_max_pd(similarity, _mm256_set1_pd(-1.0)), _mm256_set1_pd(1.0));
    return _mm256_sub_pd(_mm256_set1_pd(1.0), similarity);
}

#endif

[[nodiscard]] int maskFromDistanceBlock4(const DistanceBlock4 &distances, double search_radius)
{
    int mask = 0;
    if (distances.first <= search_radius)
    {
        mask |= 1;
    }
    if (distances.second <= search_radius)
    {
        mask |= 2;
    }
    if (distances.third <= search_radius)
    {
        mask |= 4;
    }
    if (distances.fourth <= search_radius)
    {
        mask |= 8;
    }
    return mask;
}

[[nodiscard]] double manhattanDistanceUnchecked(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features)
{
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;
#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
    if (num_features >= 4)
    {
        return manhattanDistanceAvx2(lhs_ptr, rhs_ptr, num_features);
    }
#endif
    if (num_features == 3)
    {
        return std::abs(static_cast<double>(lhs_ptr[0]) - static_cast<double>(rhs_ptr[0]))
               + std::abs(static_cast<double>(lhs_ptr[1]) - static_cast<double>(rhs_ptr[1]))
               + std::abs(static_cast<double>(lhs_ptr[2]) - static_cast<double>(rhs_ptr[2]));
    }

    double sum = 0.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        sum += std::abs(static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature]));
    }
    return sum;
}

[[nodiscard]] double chebyshevDistanceUnchecked(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features)
{
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;
#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
    if (num_features >= 4)
    {
        return chebyshevDistanceAvx2(lhs_ptr, rhs_ptr, num_features);
    }
#endif
    if (num_features == 3)
    {
        const double diff0 = std::abs(static_cast<double>(lhs_ptr[0]) - static_cast<double>(rhs_ptr[0]));
        const double diff1 = std::abs(static_cast<double>(lhs_ptr[1]) - static_cast<double>(rhs_ptr[1]));
        const double diff2 = std::abs(static_cast<double>(lhs_ptr[2]) - static_cast<double>(rhs_ptr[2]));
        return std::max(diff0, std::max(diff1, diff2));
    }

    double max_diff = 0.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        max_diff = std::max(
            max_diff, std::abs(static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature])));
    }
    return max_diff;
}

[[nodiscard]] double minkowskiPoweredDistanceUnchecked(const float *samples, int64_t lhs, int64_t rhs,
                                                       int64_t num_features, double minkowski_p)
{
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;
#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
    if (num_features >= 4 && isMinkowskiP3(minkowski_p))
    {
        return minkowskiP3DistanceAvx2(lhs_ptr, rhs_ptr, num_features);
    }
#endif
    if (isMinkowskiP3(minkowski_p) && num_features == 3)
    {
        const double diff0 = std::abs(static_cast<double>(lhs_ptr[0]) - static_cast<double>(rhs_ptr[0]));
        const double diff1 = std::abs(static_cast<double>(lhs_ptr[1]) - static_cast<double>(rhs_ptr[1]));
        const double diff2 = std::abs(static_cast<double>(lhs_ptr[2]) - static_cast<double>(rhs_ptr[2]));
        return diff0 * diff0 * diff0 + diff1 * diff1 * diff1 + diff2 * diff2 * diff2;
    }
    if (minkowski_p == 2.0)
    {
        return squaredEuclideanDistanceUnchecked(samples, lhs, rhs, num_features);
    }
    if (minkowski_p == 1.0)
    {
        return manhattanDistanceUnchecked(samples, lhs, rhs, num_features);
    }

    double sum = 0.0;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        const double diff = std::abs(static_cast<double>(lhs_ptr[feature]) - static_cast<double>(rhs_ptr[feature]));
        sum += std::pow(diff, minkowski_p);
    }
    return sum;
}

[[nodiscard]] double cosineDistanceUnchecked(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features)
{
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;

    double dot      = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
    if (num_features >= 4)
    {
        dot      = dotProductAvx2(lhs_ptr, rhs_ptr, num_features);
        lhs_norm = dotProductAvx2(lhs_ptr, lhs_ptr, num_features);
        rhs_norm = dotProductAvx2(rhs_ptr, rhs_ptr, num_features);
    }
    else
#endif
    if (num_features == 3)
    {
        const double lhs0 = static_cast<double>(lhs_ptr[0]);
        const double lhs1 = static_cast<double>(lhs_ptr[1]);
        const double lhs2 = static_cast<double>(lhs_ptr[2]);
        const double rhs0 = static_cast<double>(rhs_ptr[0]);
        const double rhs1 = static_cast<double>(rhs_ptr[1]);
        const double rhs2 = static_cast<double>(rhs_ptr[2]);
        dot               = lhs0 * rhs0 + lhs1 * rhs1 + lhs2 * rhs2;
        lhs_norm          = lhs0 * lhs0 + lhs1 * lhs1 + lhs2 * lhs2;
        rhs_norm          = rhs0 * rhs0 + rhs1 * rhs1 + rhs2 * rhs2;
    }
    else
    {
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const double lhs_value = static_cast<double>(lhs_ptr[feature]);
            const double rhs_value = static_cast<double>(rhs_ptr[feature]);
            dot += lhs_value * rhs_value;
            lhs_norm += lhs_value * lhs_value;
            rhs_norm += rhs_value * rhs_value;
        }
    }

    if (lhs_norm == 0.0 && rhs_norm == 0.0)
    {
        return 0.0;
    }
    if (lhs_norm == 0.0 || rhs_norm == 0.0)
    {
        return 1.0;
    }

    const double similarity = std::clamp(dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm)), -1.0, 1.0);
    return 1.0 - similarity;
}

} // namespace

std::vector<double> cosineInverseNorms(const float *samples, int64_t num_samples, int64_t num_features)
{
    std::vector<double> result(static_cast<size_t>(num_samples), 0.0);
    for (int64_t sample = 0; sample < num_samples; ++sample)
    {
        const float *sample_ptr = samples + sample * num_features;
#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
        const double norm = num_features >= 4 ? dotProductAvx2(sample_ptr, sample_ptr, num_features) : [&]()
        {
            double value = 0.0;
            for (int64_t feature = 0; feature < num_features; ++feature)
            {
                const double sample_value = static_cast<double>(sample_ptr[feature]);
                value += sample_value * sample_value;
            }
            return value;
        }();
#else
        double norm = 0.0;
        for (int64_t feature = 0; feature < num_features; ++feature)
        {
            const double sample_value = static_cast<double>(sample_ptr[feature]);
            norm += sample_value * sample_value;
        }
#endif
        if (norm > 0.0)
        {
            result[static_cast<size_t>(sample)] = 1.0 / std::sqrt(norm);
        }
    }
    return result;
}

SearchDistanceCalculator::SearchDistanceCalculator(const float *samples, int64_t num_features, ClusteringMetric metric,
                                                   double minkowski_p,
                                                   const std::vector<double> *inverse_norms)
    : samples_(samples)
    , num_features_(num_features)
    , metric_(metric)
    , minkowski_p_(minkowski_p)
    , inverse_norms_(inverse_norms)
{
}

double SearchDistanceCalculator::operator()(int64_t lhs, int64_t rhs) const
{
    if (metric_ == ClusteringMetric::Cosine && inverse_norms_ != nullptr)
    {
        const double lhs_inv_norm = (*inverse_norms_)[static_cast<size_t>(lhs)];
        const double rhs_inv_norm = (*inverse_norms_)[static_cast<size_t>(rhs)];
        if (lhs_inv_norm == 0.0 && rhs_inv_norm == 0.0)
        {
            return 0.0;
        }
        if (lhs_inv_norm == 0.0 || rhs_inv_norm == 0.0)
        {
            return 1.0;
        }

        const float *lhs_ptr = samples_ + lhs * num_features_;
        const float *rhs_ptr = samples_ + rhs * num_features_;
#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
        const double dot = num_features_ >= 4 ? dotProductAvx2(lhs_ptr, rhs_ptr, num_features_)
                                              : [&]()
        {
            double result = 0.0;
            for (int64_t feature = 0; feature < num_features_; ++feature)
            {
                result += static_cast<double>(lhs_ptr[feature]) * static_cast<double>(rhs_ptr[feature]);
            }
            return result;
        }();
#else
        double dot = 0.0;
        for (int64_t feature = 0; feature < num_features_; ++feature)
        {
            dot += static_cast<double>(lhs_ptr[feature]) * static_cast<double>(rhs_ptr[feature]);
        }
#endif
        const double similarity = std::clamp(dot * lhs_inv_norm * rhs_inv_norm, -1.0, 1.0);
        return 1.0 - similarity;
    }

    return clusteringSearchDistance(samples_, lhs, rhs, num_features_, metric_, minkowski_p_);
}

bool SearchDistanceCalculator::canUseBlock4() const
{
#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
    if (num_features_ >= 4)
    {
        return canUseAvx2FeatureDistance(num_features_, metric_, minkowski_p_);
    }
    if (num_features_ < 1)
    {
        return false;
    }
    if (metric_ == ClusteringMetric::Cosine)
    {
        return inverse_norms_ != nullptr;
    }
    return metric_ == ClusteringMetric::Euclidean || metric_ == ClusteringMetric::Manhattan
           || metric_ == ClusteringMetric::Chebyshev
           || (metric_ == ClusteringMetric::Minkowski
               && (isMinkowskiP3(minkowski_p_) || minkowski_p_ == 1.0 || minkowski_p_ == 2.0));
#else
    return false;
#endif
}

DistanceBlock4 SearchDistanceCalculator::block4(int64_t lhs, int64_t first_rhs) const
{
#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
    if (canUseBlock4())
    {
        if (num_features_ < 4)
        {
            const auto load_rhs_feature = [&](int64_t feature)
            { return loadContiguousFeature4(samples_, num_features_, first_rhs, feature); };
            if (metric_ == ClusteringMetric::Cosine)
            {
                const double lhs_inv_norm = (*inverse_norms_)[static_cast<size_t>(lhs)];
                const double rhs_inv0     = (*inverse_norms_)[static_cast<size_t>(first_rhs)];
                const double rhs_inv1     = (*inverse_norms_)[static_cast<size_t>(first_rhs + 1)];
                const double rhs_inv2     = (*inverse_norms_)[static_cast<size_t>(first_rhs + 2)];
                const double rhs_inv3     = (*inverse_norms_)[static_cast<size_t>(first_rhs + 3)];
                if (lhs_inv_norm != 0.0 && rhs_inv0 != 0.0 && rhs_inv1 != 0.0 && rhs_inv2 != 0.0
                    && rhs_inv3 != 0.0)
                {
                    return storeDistanceBlock4(
                        lowDimCosineDistance4(samples_, num_features_, lhs, load_rhs_feature, lhs_inv_norm,
                                              _mm256_set_pd(rhs_inv3, rhs_inv2, rhs_inv1, rhs_inv0)));
                }
                return {(*this)(lhs, first_rhs), (*this)(lhs, first_rhs + 1), (*this)(lhs, first_rhs + 2),
                        (*this)(lhs, first_rhs + 3)};
            }
            return storeDistanceBlock4(
                lowDimSearchDistance4(samples_, num_features_, lhs, load_rhs_feature, metric_, minkowski_p_));
        }

        const float *lhs_ptr  = samples_ + lhs * num_features_;
        const float *rhs0_ptr = samples_ + first_rhs * num_features_;
        const float *rhs1_ptr = rhs0_ptr + num_features_;
        const float *rhs2_ptr = rhs1_ptr + num_features_;
        const float *rhs3_ptr = rhs2_ptr + num_features_;
        if (metric_ == ClusteringMetric::Cosine)
        {
            const double lhs_inv_norm = (*inverse_norms_)[static_cast<size_t>(lhs)];
            const double rhs_inv0     = (*inverse_norms_)[static_cast<size_t>(first_rhs)];
            const double rhs_inv1     = (*inverse_norms_)[static_cast<size_t>(first_rhs + 1)];
            const double rhs_inv2     = (*inverse_norms_)[static_cast<size_t>(first_rhs + 2)];
            const double rhs_inv3     = (*inverse_norms_)[static_cast<size_t>(first_rhs + 3)];
            return highDimCosineDistanceBlock4(lhs_ptr, rhs0_ptr, rhs1_ptr, rhs2_ptr, rhs3_ptr, num_features_,
                                               lhs_inv_norm, rhs_inv0, rhs_inv1, rhs_inv2, rhs_inv3);
        }
        return highDimSearchDistanceBlock4(lhs_ptr, rhs0_ptr, rhs1_ptr, rhs2_ptr, rhs3_ptr, num_features_, metric_,
                                           minkowski_p_);
    }
#endif
    return {(*this)(lhs, first_rhs), (*this)(lhs, first_rhs + 1), (*this)(lhs, first_rhs + 2),
            (*this)(lhs, first_rhs + 3)};
}

DistanceBlock4 SearchDistanceCalculator::indexedBlock4(int64_t lhs, const int64_t *rhs_indices) const
{
#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
    if (canUseBlock4())
    {
        if (num_features_ < 4)
        {
            const auto load_rhs_feature = [&](int64_t feature)
            { return loadIndexedFeature4(samples_, num_features_, rhs_indices, feature); };
            if (metric_ == ClusteringMetric::Cosine)
            {
                const double lhs_inv_norm = (*inverse_norms_)[static_cast<size_t>(lhs)];
                const double rhs_inv0     = (*inverse_norms_)[static_cast<size_t>(rhs_indices[0])];
                const double rhs_inv1     = (*inverse_norms_)[static_cast<size_t>(rhs_indices[1])];
                const double rhs_inv2     = (*inverse_norms_)[static_cast<size_t>(rhs_indices[2])];
                const double rhs_inv3     = (*inverse_norms_)[static_cast<size_t>(rhs_indices[3])];
                if (lhs_inv_norm != 0.0 && rhs_inv0 != 0.0 && rhs_inv1 != 0.0 && rhs_inv2 != 0.0
                    && rhs_inv3 != 0.0)
                {
                    return storeDistanceBlock4(
                        lowDimCosineDistance4(samples_, num_features_, lhs, load_rhs_feature, lhs_inv_norm,
                                              _mm256_set_pd(rhs_inv3, rhs_inv2, rhs_inv1, rhs_inv0)));
                }
                return {(*this)(lhs, rhs_indices[0]), (*this)(lhs, rhs_indices[1]), (*this)(lhs, rhs_indices[2]),
                        (*this)(lhs, rhs_indices[3])};
            }
            return storeDistanceBlock4(
                lowDimSearchDistance4(samples_, num_features_, lhs, load_rhs_feature, metric_, minkowski_p_));
        }

        const float *lhs_ptr  = samples_ + lhs * num_features_;
        const float *rhs0_ptr = samples_ + rhs_indices[0] * num_features_;
        const float *rhs1_ptr = samples_ + rhs_indices[1] * num_features_;
        const float *rhs2_ptr = samples_ + rhs_indices[2] * num_features_;
        const float *rhs3_ptr = samples_ + rhs_indices[3] * num_features_;
        if (metric_ == ClusteringMetric::Cosine)
        {
            const double lhs_inv_norm = (*inverse_norms_)[static_cast<size_t>(lhs)];
            const double rhs_inv0     = (*inverse_norms_)[static_cast<size_t>(rhs_indices[0])];
            const double rhs_inv1     = (*inverse_norms_)[static_cast<size_t>(rhs_indices[1])];
            const double rhs_inv2     = (*inverse_norms_)[static_cast<size_t>(rhs_indices[2])];
            const double rhs_inv3     = (*inverse_norms_)[static_cast<size_t>(rhs_indices[3])];
            return highDimCosineDistanceBlock4(lhs_ptr, rhs0_ptr, rhs1_ptr, rhs2_ptr, rhs3_ptr, num_features_,
                                               lhs_inv_norm, rhs_inv0, rhs_inv1, rhs_inv2, rhs_inv3);
        }
        return highDimSearchDistanceBlock4(lhs_ptr, rhs0_ptr, rhs1_ptr, rhs2_ptr, rhs3_ptr, num_features_, metric_,
                                           minkowski_p_);
    }
#endif
    return {(*this)(lhs, rhs_indices[0]), (*this)(lhs, rhs_indices[1]), (*this)(lhs, rhs_indices[2]),
            (*this)(lhs, rhs_indices[3])};
}

int SearchDistanceCalculator::withinRadiusMask4(int64_t lhs, int64_t first_rhs, double search_radius) const
{
#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
    if (num_features_ >= 4 && canUseBlock4())
    {
        if (metric_ == ClusteringMetric::Euclidean
            || (metric_ == ClusteringMetric::Minkowski && minkowski_p_ == 2.0))
        {
            return maskFromDistanceBlock4(block4(lhs, first_rhs), search_radius);
        }
        return maskFromDistanceBlock4({(*this)(lhs, first_rhs), (*this)(lhs, first_rhs + 1),
                                       (*this)(lhs, first_rhs + 2), (*this)(lhs, first_rhs + 3)},
                                      search_radius);
    }
#endif
    const DistanceBlock4 distances = block4(lhs, first_rhs);
    return maskFromDistanceBlock4(distances, search_radius);
}

int SearchDistanceCalculator::indexedWithinRadiusMask4(int64_t lhs, const int64_t *rhs_indices,
                                                       double search_radius) const
{
#if IRT_CLUSTERING_DISTANCE_HAS_AVX2
    if (num_features_ >= 4 && canUseBlock4())
    {
        if (metric_ == ClusteringMetric::Euclidean
            || (metric_ == ClusteringMetric::Minkowski && minkowski_p_ == 2.0))
        {
            return maskFromDistanceBlock4(indexedBlock4(lhs, rhs_indices), search_radius);
        }
        return maskFromDistanceBlock4({(*this)(lhs, rhs_indices[0]), (*this)(lhs, rhs_indices[1]),
                                       (*this)(lhs, rhs_indices[2]), (*this)(lhs, rhs_indices[3])},
                                      search_radius);
    }
#endif
    const DistanceBlock4 distances = indexedBlock4(lhs, rhs_indices);
    return maskFromDistanceBlock4(distances, search_radius);
}

namespace {

[[nodiscard]] double searchRadiusForMetric(double radius, ClusteringMetric metric, double minkowski_p)
{
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
        return radius * radius;
    case ClusteringMetric::Minkowski:
        return minkowskiPower(radius, minkowski_p);
    case ClusteringMetric::Manhattan:
    case ClusteringMetric::Chebyshev:
    case ClusteringMetric::Cosine:
        return radius;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

[[nodiscard]] double outputDistanceForMetric(double search_distance, ClusteringMetric metric, double minkowski_p)
{
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
        return std::sqrt(search_distance);
    case ClusteringMetric::Minkowski:
        return minkowskiRoot(search_distance, minkowski_p);
    case ClusteringMetric::Manhattan:
    case ClusteringMetric::Chebyshev:
    case ClusteringMetric::Cosine:
        return search_distance;
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

} // namespace

double squaredEuclideanDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features)
{
    double       sum     = 0.0;
    const float *lhs_ptr = samples + lhs * num_features;
    const float *rhs_ptr = samples + rhs * num_features;
    for (int64_t feature = 0; feature < num_features; ++feature)
    {
        const float lhs_value = lhs_ptr[feature];
        const float rhs_value = rhs_ptr[feature];
        if (!std::isfinite(lhs_value) || !std::isfinite(rhs_value))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "samples must be finite");
        }
        const double diff = static_cast<double>(lhs_value) - static_cast<double>(rhs_value);
        sum += diff * diff;
    }
    return sum;
}

double euclideanDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features)
{
    return std::sqrt(squaredEuclideanDistance(samples, lhs, rhs, num_features));
}

double clusteringDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features, ClusteringMetric metric,
                          double minkowski_p)
{
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
        return euclideanDistance(samples, lhs, rhs, num_features);
    case ClusteringMetric::Manhattan:
        return manhattanDistanceUnchecked(samples, lhs, rhs, num_features);
    case ClusteringMetric::Chebyshev:
        return chebyshevDistanceUnchecked(samples, lhs, rhs, num_features);
    case ClusteringMetric::Minkowski:
    {
        if (!std::isfinite(minkowski_p) || minkowski_p <= 0.0)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "minkowski_p must be positive and finite");
        }
        return minkowskiRoot(minkowskiPoweredDistanceUnchecked(samples, lhs, rhs, num_features, minkowski_p),
                             minkowski_p);
    }
    case ClusteringMetric::Cosine:
        return cosineDistanceUnchecked(samples, lhs, rhs, num_features);
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

double clusteringSearchDistance(const float *samples, int64_t lhs, int64_t rhs, int64_t num_features,
                                ClusteringMetric metric, double minkowski_p)
{
    switch (metric)
    {
    case ClusteringMetric::Euclidean:
        return squaredEuclideanDistanceUnchecked(samples, lhs, rhs, num_features);
    case ClusteringMetric::Manhattan:
        return manhattanDistanceUnchecked(samples, lhs, rhs, num_features);
    case ClusteringMetric::Chebyshev:
        return chebyshevDistanceUnchecked(samples, lhs, rhs, num_features);
    case ClusteringMetric::Minkowski:
    {
        if (!std::isfinite(minkowski_p) || minkowski_p <= 0.0)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "minkowski_p must be positive and finite");
        }
        return minkowskiPoweredDistanceUnchecked(samples, lhs, rhs, num_features, minkowski_p);
    }
    case ClusteringMetric::Cosine:
        return cosineDistanceUnchecked(samples, lhs, rhs, num_features);
    default:
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "unsupported clustering metric");
    }
}

double clusteringMinkowskiPower(double value, double minkowski_p)
{
    return minkowskiPower(value, minkowski_p);
}

double clusteringMinkowskiRoot(double value, double minkowski_p)
{
    return minkowskiRoot(value, minkowski_p);
}

double clusteringSearchRadius(double radius, ClusteringMetric metric, double minkowski_p)
{
    return searchRadiusForMetric(radius, metric, minkowski_p);
}

double clusteringOutputDistance(double search_distance, ClusteringMetric metric, double minkowski_p)
{
    return outputDistanceForMetric(search_distance, metric, minkowski_p);
}

} // namespace irt::ops::detail
