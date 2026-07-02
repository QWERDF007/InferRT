#pragma once

#include <inferrt/ops/Export.h>

#include <cstdint>
#include <vector>

namespace irt::ops {

struct INFERRT_OPS_API BezierFitResult
{
    int64_t            degree{0};
    int64_t            dimensions{0};
    std::vector<float> control_points;       ///< Shape: [degree + 1, dimensions], row-major.
    std::vector<float> parameters;           ///< Shape: [num_points], normalized to [0, 1].
    double             residual_sum_squares{0.0};
};

[[nodiscard]] INFERRT_OPS_API std::vector<float> chordLengthParameters(const float *points, int64_t num_points,
                                                                       int64_t num_dims);

[[nodiscard]] INFERRT_OPS_API BezierFitResult fitBezierCurve(const float *points, int64_t num_points,
                                                             int64_t num_dims, int degree,
                                                             const float *parameters = nullptr);

INFERRT_OPS_API void evaluateBezierCurve(const float *control_points, int64_t num_control_points, int64_t num_dims,
                                         const float *parameters, int64_t num_parameters, float *output);

[[nodiscard]] INFERRT_OPS_API std::vector<float> evaluateBezierCurve(const float *control_points,
                                                                     int64_t num_control_points, int64_t num_dims,
                                                                     const float *parameters,
                                                                     int64_t num_parameters);

} // namespace irt::ops
