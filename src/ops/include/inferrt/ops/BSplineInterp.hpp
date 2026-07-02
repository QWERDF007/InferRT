#pragma once

#include <inferrt/ops/Export.h>

#include <cstdint>
#include <vector>

namespace irt::ops {

/**
 * @brief B-spline 插值结果。
 */
struct INFERRT_OPS_API BSplineInterpResult
{
    int64_t            degree{0};     ///< B-spline 次数。
    int64_t            dimensions{0}; ///< 点坐标或输出值维度。
    std::vector<float> knots;         ///< 节点向量，形状为 [num_coefficients + degree + 1]。
    std::vector<float> coefficients;  ///< 系数矩阵，形状为 [num_coefficients, dimensions]，按行主序存储。
};

/**
 * @brief 参数曲线 B-spline 拟合结果。
 */
struct INFERRT_OPS_API SplPrepResult
{
    int64_t            degree{0};                 ///< B-spline 次数。
    int64_t            dimensions{0};             ///< 曲线点坐标维度。
    float              smoothing{0.0F};           ///< 平滑残差预算，0 表示精确插值。
    double             residual_sum_squares{0.0}; ///< 拟合残差平方和。
    std::vector<float> knots;                     ///< 节点向量，形状为 [num_coefficients + degree + 1]。
    std::vector<float> coefficients;              ///< 系数矩阵，形状为 [num_coefficients, dimensions]，按行主序存储。
    std::vector<float> parameters;                ///< 每个输入点对应的参数，形状为 [num_points]，范围归一化到 [0, 1]。
};

/**
 * @brief 构造一维自变量上的插值 B-spline。
 * @param x 严格递增的自变量数组，形状为 [num_points]。
 * @param y 因变量矩阵，形状为 [num_points, num_dims]，按行主序存储。
 * @param num_points 样本点数量。
 * @param num_dims 因变量维度。
 * @param degree B-spline 次数，目前支持 1 和 3。
 * @return B-spline 插值结果。
 */
[[nodiscard]] INFERRT_OPS_API BSplineInterpResult makeInterpSpline(const float *x, const float *y, int64_t num_points,
                                                                   int64_t num_dims, int degree = 3);

/**
 * @brief 构造参数曲线的 B-spline 拟合。
 * @param points 输入曲线点矩阵，形状为 [num_points, num_dims]，按行主序存储。
 * @param num_points 曲线点数量。
 * @param num_dims 曲线点坐标维度，例如二维点为 2。
 * @param smoothing 平滑残差预算；0 表示精确插值，大于 0 时允许残差平方和不超过该值。
 * @param degree B-spline 次数，目前支持 1 和 3。
 * @param parameters 可选参数数组，形状为 [num_points]；为空时使用弦长参数化。
 * @return 参数曲线 B-spline 拟合结果。
 */
[[nodiscard]] INFERRT_OPS_API SplPrepResult splPrep(const float *points, int64_t num_points, int64_t num_dims,
                                                    float smoothing = 0.0F, int degree = 3,
                                                    const float *parameters = nullptr);

/**
 * @brief 在给定自变量处评估 B-spline 并写入外部输出缓冲区。
 * @param knots 节点向量，形状为 [num_coefficients + degree + 1]。
 * @param num_knots 节点数量。
 * @param coefficients 系数矩阵，形状为 [num_coefficients, num_dims]，按行主序存储。
 * @param num_coefficients 系数数量。
 * @param num_dims 输出值维度。
 * @param degree B-spline 次数。
 * @param x_eval 评估自变量数组，形状为 [num_eval]。
 * @param num_eval 评估点数量。
 * @param output 输出矩阵，形状为 [num_eval, num_dims]，按行主序存储。
 * @return 无。
 */
INFERRT_OPS_API void evaluateBSpline(const float *knots, int64_t num_knots, const float *coefficients,
                                     int64_t num_coefficients, int64_t num_dims, int degree, const float *x_eval,
                                     int64_t num_eval, float *output);

/**
 * @brief 在给定自变量处评估 B-spline 并返回输出向量。
 * @param knots 节点向量，形状为 [num_coefficients + degree + 1]。
 * @param num_knots 节点数量。
 * @param coefficients 系数矩阵，形状为 [num_coefficients, num_dims]，按行主序存储。
 * @param num_coefficients 系数数量。
 * @param num_dims 输出值维度。
 * @param degree B-spline 次数。
 * @param x_eval 评估自变量数组，形状为 [num_eval]。
 * @param num_eval 评估点数量。
 * @return 输出矩阵，形状为 [num_eval, num_dims]，按行主序存储。
 */
[[nodiscard]] INFERRT_OPS_API std::vector<float> evaluateBSpline(const float *knots, int64_t num_knots,
                                                                 const float *coefficients, int64_t num_coefficients,
                                                                 int64_t num_dims, int degree, const float *x_eval,
                                                                 int64_t num_eval);

} // namespace irt::ops
