#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/BezierFit.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace irt::ops {
namespace {

void validatePointMatrix(const float *points, int64_t num_points, int64_t num_dims)
{
    if (num_points < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_points must be non-negative, got %lld",
                        static_cast<long long>(num_points));
    }
    if (num_dims <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_dims must be positive, got %lld",
                        static_cast<long long>(num_dims));
    }
    if (num_points > 0 && points == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "points must not be null");
    }
    for (int64_t i = 0; i < num_points * num_dims; ++i)
    {
        if (!std::isfinite(points[i]))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "points contains non-finite value at flat index %lld",
                            static_cast<long long>(i));
        }
    }
}

void validateParameters(const float *parameters, int64_t num_parameters)
{
    if (num_parameters < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_parameters must be non-negative, got %lld",
                        static_cast<long long>(num_parameters));
    }
    if (num_parameters > 0 && parameters == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "parameters must not be null");
    }
    for (int64_t i = 0; i < num_parameters; ++i)
    {
        const float value = parameters[i];
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "parameters[%lld] must be finite and in [0, 1]",
                            static_cast<long long>(i));
        }
    }
}

double binomial(int n, int k)
{
    if (k < 0 || k > n)
    {
        return 0.0;
    }
    k             = std::min(k, n - k);
    double result = 1.0;
    for (int i = 1; i <= k; ++i)
    {
        result *= static_cast<double>(n - k + i);
        result /= static_cast<double>(i);
    }
    return result;
}

std::vector<double> bernsteinBasis(int degree, double u)
{
    u = std::clamp(u, 0.0, 1.0);
    std::vector<double> basis(static_cast<size_t>(degree) + 1U);
    if (u == 0.0)
    {
        basis.front() = 1.0;
        return basis;
    }
    if (u == 1.0)
    {
        basis.back() = 1.0;
        return basis;
    }

    const double one_minus_u = 1.0 - u;
    for (int j = 0; j <= degree; ++j)
    {
        basis[static_cast<size_t>(j)]
            = binomial(degree, j) * std::pow(one_minus_u, degree - j) * std::pow(u, j);
    }
    return basis;
}

std::vector<double> solveLinearSystem(std::vector<double> matrix, std::vector<double> rhs, int size)
{
    for (int pivot = 0; pivot < size; ++pivot)
    {
        int    best_row = pivot;
        double best_abs = std::abs(matrix[static_cast<size_t>(pivot) * size + pivot]);
        for (int row = pivot + 1; row < size; ++row)
        {
            const double value_abs = std::abs(matrix[static_cast<size_t>(row) * size + pivot]);
            if (value_abs > best_abs)
            {
                best_abs = value_abs;
                best_row = row;
            }
        }
        if (best_abs <= std::numeric_limits<double>::epsilon())
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT,
                            "Bezier least-squares system is singular; provide more distinct samples or lower degree");
        }
        if (best_row != pivot)
        {
            for (int col = pivot; col < size; ++col)
            {
                std::swap(matrix[static_cast<size_t>(pivot) * size + col],
                          matrix[static_cast<size_t>(best_row) * size + col]);
            }
            std::swap(rhs[static_cast<size_t>(pivot)], rhs[static_cast<size_t>(best_row)]);
        }

        const double pivot_value = matrix[static_cast<size_t>(pivot) * size + pivot];
        for (int row = pivot + 1; row < size; ++row)
        {
            const double factor = matrix[static_cast<size_t>(row) * size + pivot] / pivot_value;
            if (factor == 0.0)
            {
                continue;
            }
            matrix[static_cast<size_t>(row) * size + pivot] = 0.0;
            for (int col = pivot + 1; col < size; ++col)
            {
                matrix[static_cast<size_t>(row) * size + col]
                    -= factor * matrix[static_cast<size_t>(pivot) * size + col];
            }
            rhs[static_cast<size_t>(row)] -= factor * rhs[static_cast<size_t>(pivot)];
        }
    }

    std::vector<double> solution(static_cast<size_t>(size));
    for (int row = size - 1; row >= 0; --row)
    {
        double value = rhs[static_cast<size_t>(row)];
        for (int col = row + 1; col < size; ++col)
        {
            value -= matrix[static_cast<size_t>(row) * size + col] * solution[static_cast<size_t>(col)];
        }
        solution[static_cast<size_t>(row)] = value / matrix[static_cast<size_t>(row) * size + row];
    }
    return solution;
}

} // namespace

std::vector<float> chordLengthParameters(const float *points, int64_t num_points, int64_t num_dims)
{
    validatePointMatrix(points, num_points, num_dims);
    std::vector<float> parameters(static_cast<size_t>(num_points), 0.0f);
    if (num_points <= 1)
    {
        return parameters;
    }

    double total = 0.0;
    for (int64_t i = 1; i < num_points; ++i)
    {
        double squared = 0.0;
        for (int64_t dim = 0; dim < num_dims; ++dim)
        {
            const double diff = static_cast<double>(points[i * num_dims + dim])
                              - static_cast<double>(points[(i - 1) * num_dims + dim]);
            squared += diff * diff;
        }
        total += std::sqrt(squared);
        parameters[static_cast<size_t>(i)] = static_cast<float>(total);
    }

    if (total <= std::numeric_limits<double>::epsilon())
    {
        for (int64_t i = 0; i < num_points; ++i)
        {
            parameters[static_cast<size_t>(i)] = static_cast<float>(static_cast<double>(i) / (num_points - 1));
        }
        return parameters;
    }

    for (auto &value : parameters)
    {
        value = static_cast<float>(static_cast<double>(value) / total);
    }
    parameters.front() = 0.0f;
    parameters.back()  = 1.0f;
    return parameters;
}

BezierFitResult fitBezierCurve(const float *points, int64_t num_points, int64_t num_dims, int degree,
                               const float *parameters)
{
    validatePointMatrix(points, num_points, num_dims);
    if (degree < 1)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Bezier degree must be at least 1, got %d", degree);
    }
    if (num_points < static_cast<int64_t>(degree) + 1)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT,
                        "Bezier fit requires at least degree + 1 samples, got samples=%lld degree=%d",
                        static_cast<long long>(num_points), degree);
    }

    std::vector<float> owned_parameters;
    if (parameters == nullptr)
    {
        owned_parameters = chordLengthParameters(points, num_points, num_dims);
        parameters       = owned_parameters.data();
    }
    else
    {
        validateParameters(parameters, num_points);
    }

    const int control_count = degree + 1;
    std::vector<double> ata(static_cast<size_t>(control_count) * control_count, 0.0);
    std::vector<double> basis_values(static_cast<size_t>(num_points) * control_count);
    for (int64_t sample = 0; sample < num_points; ++sample)
    {
        const auto basis = bernsteinBasis(degree, static_cast<double>(parameters[sample]));
        for (int row = 0; row < control_count; ++row)
        {
            basis_values[static_cast<size_t>(sample) * control_count + row] = basis[static_cast<size_t>(row)];
            for (int col = 0; col < control_count; ++col)
            {
                ata[static_cast<size_t>(row) * control_count + col]
                    += basis[static_cast<size_t>(row)] * basis[static_cast<size_t>(col)];
            }
        }
    }

    BezierFitResult result;
    result.degree       = degree;
    result.dimensions   = num_dims;
    result.parameters.assign(parameters, parameters + num_points);
    result.control_points.resize(static_cast<size_t>(control_count) * num_dims);

    for (int64_t dim = 0; dim < num_dims; ++dim)
    {
        std::vector<double> aty(static_cast<size_t>(control_count), 0.0);
        for (int64_t sample = 0; sample < num_points; ++sample)
        {
            const double y = static_cast<double>(points[sample * num_dims + dim]);
            for (int row = 0; row < control_count; ++row)
            {
                aty[static_cast<size_t>(row)]
                    += basis_values[static_cast<size_t>(sample) * control_count + row] * y;
            }
        }

        const auto solution = solveLinearSystem(ata, aty, control_count);
        for (int row = 0; row < control_count; ++row)
        {
            result.control_points[static_cast<size_t>(row) * num_dims + dim]
                = static_cast<float>(solution[static_cast<size_t>(row)]);
        }
    }

    const auto fitted = evaluateBezierCurve(result.control_points.data(), control_count, num_dims, parameters, num_points);
    for (int64_t i = 0; i < num_points * num_dims; ++i)
    {
        const double diff = static_cast<double>(fitted[static_cast<size_t>(i)]) - static_cast<double>(points[i]);
        result.residual_sum_squares += diff * diff;
    }
    return result;
}

void evaluateBezierCurve(const float *control_points, int64_t num_control_points, int64_t num_dims,
                         const float *parameters, int64_t num_parameters, float *output)
{
    validatePointMatrix(control_points, num_control_points, num_dims);
    validateParameters(parameters, num_parameters);
    if (num_control_points < 2)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_control_points must be at least 2");
    }
    if (num_parameters > 0 && output == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "output must not be null");
    }

    const int degree = static_cast<int>(num_control_points - 1);
    for (int64_t sample = 0; sample < num_parameters; ++sample)
    {
        const auto basis = bernsteinBasis(degree, static_cast<double>(parameters[sample]));
        for (int64_t dim = 0; dim < num_dims; ++dim)
        {
            double value = 0.0;
            for (int64_t control = 0; control < num_control_points; ++control)
            {
                value += basis[static_cast<size_t>(control)] * control_points[control * num_dims + dim];
            }
            output[sample * num_dims + dim] = static_cast<float>(value);
        }
    }
}

std::vector<float> evaluateBezierCurve(const float *control_points, int64_t num_control_points, int64_t num_dims,
                                       const float *parameters, int64_t num_parameters)
{
    std::vector<float> output(static_cast<size_t>(num_parameters) * num_dims);
    evaluateBezierCurve(control_points, num_control_points, num_dims, parameters, num_parameters, output.data());
    return output;
}

} // namespace irt::ops
