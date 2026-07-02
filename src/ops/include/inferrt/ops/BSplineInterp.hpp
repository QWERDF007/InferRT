#pragma once

#include <inferrt/ops/Export.h>

#include <cstdint>
#include <vector>

namespace irt::ops {

struct INFERRT_OPS_API BSplineInterpResult
{
    int64_t            degree{0};
    int64_t            dimensions{0};
    std::vector<float> knots;        ///< Shape: [num_coefficients + degree + 1].
    std::vector<float> coefficients; ///< Shape: [num_coefficients, dimensions], row-major.
};

struct INFERRT_OPS_API SplPrepResult
{
    int64_t            degree{0};
    int64_t            dimensions{0};
    float              smoothing{0.0F};
    double             residual_sum_squares{0.0};
    std::vector<float> knots;        ///< Shape: [num_coefficients + degree + 1].
    std::vector<float> coefficients; ///< Shape: [num_coefficients, dimensions], row-major.
    std::vector<float> parameters;   ///< Shape: [num_points], normalized to [0, 1].
};

[[nodiscard]] INFERRT_OPS_API BSplineInterpResult makeInterpSpline(const float *x, const float *y,
                                                                   int64_t num_points, int64_t num_dims,
                                                                   int degree = 3);

[[nodiscard]] INFERRT_OPS_API SplPrepResult splPrep(const float *points, int64_t num_points, int64_t num_dims,
                                                    float smoothing = 0.0F, int degree = 3,
                                                    const float *parameters = nullptr);

INFERRT_OPS_API void evaluateBSpline(const float *knots, int64_t num_knots, const float *coefficients,
                                     int64_t num_coefficients, int64_t num_dims, int degree, const float *x_eval,
                                     int64_t num_eval, float *output);

[[nodiscard]] INFERRT_OPS_API std::vector<float> evaluateBSpline(const float *knots, int64_t num_knots,
                                                                 const float *coefficients,
                                                                 int64_t num_coefficients, int64_t num_dims,
                                                                 int degree, const float *x_eval,
                                                                 int64_t num_eval);

} // namespace irt::ops
