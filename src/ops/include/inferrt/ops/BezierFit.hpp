#pragma once

#include <inferrt/ops/Export.h>

#include <cstdint>
#include <vector>

namespace irt::ops {

/**
 * @brief Bezier 曲线拟合结果。
 */
struct INFERRT_OPS_API BezierFitResult
{
    int64_t            degree{0};                 ///< Bezier 曲线次数。
    int64_t            dimensions{0};             ///< 点坐标维度。
    std::vector<float> control_points;            ///< 控制点矩阵，形状为 [degree + 1, dimensions]，按行主序存储。
    std::vector<float> parameters;                ///< 每个输入点对应的参数，形状为 [num_points]，范围归一化到 [0, 1]。
    double             residual_sum_squares{0.0}; ///< 拟合残差平方和。
};

/**
 * @brief 根据点间弦长生成归一化曲线参数。
 * @param points 输入点矩阵，形状为 [num_points, num_dims]，按行主序存储。
 * @param num_points 点数量。
 * @param num_dims 点坐标维度。
 * @return 归一化参数数组，形状为 [num_points]。
 */
[[nodiscard]] INFERRT_OPS_API std::vector<float> chordLengthParameters(const float *points, int64_t num_points,
                                                                       int64_t num_dims);

/**
 * @brief 使用最小二乘拟合单段 Bezier 曲线。
 * @param points 输入点矩阵，形状为 [num_points, num_dims]，按行主序存储。
 * @param num_points 点数量。
 * @param num_dims 点坐标维度。
 * @param degree Bezier 曲线次数。
 * @param parameters 可选参数数组，形状为 [num_points]；为空时使用弦长参数化。
 * @return Bezier 曲线拟合结果。
 */
[[nodiscard]] INFERRT_OPS_API BezierFitResult fitBezierCurve(const float *points, int64_t num_points, int64_t num_dims,
                                                             int degree, const float *parameters = nullptr);

/**
 * @brief 在给定参数处评估 Bezier 曲线并写入外部输出缓冲区。
 * @param control_points 控制点矩阵，形状为 [num_control_points, num_dims]，按行主序存储。
 * @param num_control_points 控制点数量，曲线次数为 num_control_points - 1。
 * @param num_dims 点坐标维度。
 * @param parameters 评估参数数组，形状为 [num_parameters]。
 * @param num_parameters 评估参数数量。
 * @param output 输出点矩阵，形状为 [num_parameters, num_dims]，按行主序存储。
 * @return 无。
 */
INFERRT_OPS_API void evaluateBezierCurve(const float *control_points, int64_t num_control_points, int64_t num_dims,
                                         const float *parameters, int64_t num_parameters, float *output);

/**
 * @brief 在给定参数处评估 Bezier 曲线并返回输出向量。
 * @param control_points 控制点矩阵，形状为 [num_control_points, num_dims]，按行主序存储。
 * @param num_control_points 控制点数量，曲线次数为 num_control_points - 1。
 * @param num_dims 点坐标维度。
 * @param parameters 评估参数数组，形状为 [num_parameters]。
 * @param num_parameters 评估参数数量。
 * @return 输出点矩阵，形状为 [num_parameters, num_dims]，按行主序存储。
 */
[[nodiscard]] INFERRT_OPS_API std::vector<float> evaluateBezierCurve(const float *control_points,
                                                                     int64_t num_control_points, int64_t num_dims,
                                                                     const float *parameters, int64_t num_parameters);

} // namespace irt::ops
