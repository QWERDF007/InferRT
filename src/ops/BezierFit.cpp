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

void fillBernsteinBasis(int degree, double u, double *basis)
{
    u = std::clamp(u, 0.0, 1.0);
    std::fill(basis, basis + degree + 1, 0.0);
    if (u == 0.0)
    {
        basis[0] = 1.0;
        return;
    }
    if (u == 1.0)
    {
        basis[degree] = 1.0;
        return;
    }

    const double one_minus_u = 1.0 - u;
    basis[0]                 = 1.0;
    for (int j = 1; j <= degree; ++j)
    {
        double saved = 0.0;
        for (int k = 0; k < j; ++k)
        {
            const double current = basis[k];
            basis[k]             = saved + one_minus_u * current;
            saved                = u * current;
        }
        basis[j] = saved;
    }
}

std::vector<int> factorLinearSystem(std::vector<double> &matrix, int size, const char *singular_message)
{
    std::vector<int> pivots(static_cast<size_t>(size));
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
            throw Exception(Status::ERROR_INVALID_ARGUMENT, singular_message);
        }
        pivots[static_cast<size_t>(pivot)] = best_row;
        if (best_row != pivot)
        {
            for (int col = 0; col < size; ++col)
            {
                std::swap(matrix[static_cast<size_t>(pivot) * size + col],
                          matrix[static_cast<size_t>(best_row) * size + col]);
            }
        }

        const double pivot_value = matrix[static_cast<size_t>(pivot) * size + pivot];
        for (int row = pivot + 1; row < size; ++row)
        {
            const double factor = matrix[static_cast<size_t>(row) * size + pivot] / pivot_value;
            if (factor == 0.0)
            {
                continue;
            }
            matrix[static_cast<size_t>(row) * size + pivot] = factor;
            for (int col = pivot + 1; col < size; ++col)
            {
                matrix[static_cast<size_t>(row) * size + col]
                    -= factor * matrix[static_cast<size_t>(pivot) * size + col];
            }
        }
    }
    return pivots;
}

void solveFactoredLinearSystem(const std::vector<double> &matrix, const std::vector<int> &pivots, double *rhs,
                               int size)
{
    for (int pivot = 0; pivot < size; ++pivot)
    {
        const int best_row = pivots[static_cast<size_t>(pivot)];
        if (best_row != pivot)
        {
            std::swap(rhs[pivot], rhs[best_row]);
        }
    }

    for (int row = 0; row < size; ++row)
    {
        double value = rhs[row];
        for (int col = 0; col < row; ++col)
        {
            value -= matrix[static_cast<size_t>(row) * size + col] * rhs[col];
        }
        rhs[row] = value;
    }

    for (int row = size - 1; row >= 0; --row)
    {
        double value = rhs[row];
        for (int col = row + 1; col < size; ++col)
        {
            value -= matrix[static_cast<size_t>(row) * size + col] * rhs[col];
        }
        rhs[row] = value / matrix[static_cast<size_t>(row) * size + row];
    }
}

std::vector<double> solveLinearSystem(std::vector<double> matrix, std::vector<double> rhs, int size)
{
    const auto pivots = factorLinearSystem(
        matrix, size, "Bezier least-squares system is singular; provide more distinct samples or lower degree");
    solveFactoredLinearSystem(matrix, pivots, rhs.data(), size);
    return rhs;
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

    const int           control_count = degree + 1;
    std::vector<double> ata(static_cast<size_t>(control_count) * control_count, 0.0);
    std::vector<double> aty_matrix(static_cast<size_t>(num_dims) * control_count, 0.0);
    std::vector<double> basis_values(static_cast<size_t>(num_points) * control_count);
    std::vector<double> basis(static_cast<size_t>(control_count));
    for (int64_t sample = 0; sample < num_points; ++sample)
    {
        fillBernsteinBasis(degree, static_cast<double>(parameters[sample]), basis.data());
        for (int row = 0; row < control_count; ++row)
        {
            basis_values[static_cast<size_t>(sample) * control_count + row] = basis[static_cast<size_t>(row)];
            for (int col = 0; col < control_count; ++col)
            {
                ata[static_cast<size_t>(row) * control_count + col]
                    += basis[static_cast<size_t>(row)] * basis[static_cast<size_t>(col)];
            }
            for (int64_t dim = 0; dim < num_dims; ++dim)
            {
                aty_matrix[static_cast<size_t>(dim) * control_count + row]
                    += basis[static_cast<size_t>(row)]
                     * static_cast<double>(points[static_cast<size_t>(sample) * num_dims + dim]);
            }
        }
    }

    BezierFitResult result;
    result.degree     = degree;
    result.dimensions = num_dims;
    result.parameters.assign(parameters, parameters + num_points);
    result.control_points.resize(static_cast<size_t>(control_count) * num_dims);

    auto       lu     = ata;
    const auto pivots = factorLinearSystem(
        lu, control_count, "Bezier least-squares system is singular; provide more distinct samples or lower degree");
    std::vector<double> aty(static_cast<size_t>(control_count));
    for (int64_t dim = 0; dim < num_dims; ++dim)
    {
        std::copy_n(aty_matrix.data() + static_cast<size_t>(dim) * control_count,
                    static_cast<size_t>(control_count), aty.data());
        solveFactoredLinearSystem(lu, pivots, aty.data(), control_count);
        for (int row = 0; row < control_count; ++row)
        {
            result.control_points[static_cast<size_t>(row) * num_dims + dim]
                = static_cast<float>(aty[static_cast<size_t>(row)]);
        }
    }

    for (int64_t sample = 0; sample < num_points; ++sample)
    {
        const auto *sample_basis = basis_values.data() + static_cast<size_t>(sample) * control_count;
        for (int64_t dim = 0; dim < num_dims; ++dim)
        {
            double fitted = 0.0;
            for (int control = 0; control < control_count; ++control)
            {
                fitted += sample_basis[control]
                        * result.control_points[static_cast<size_t>(control) * num_dims + dim];
            }
            const double diff = fitted - static_cast<double>(points[static_cast<size_t>(sample) * num_dims + dim]);
            result.residual_sum_squares += diff * diff;
        }
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
    std::vector<double> basis(static_cast<size_t>(num_control_points));
    for (int64_t sample = 0; sample < num_parameters; ++sample)
    {
        fillBernsteinBasis(degree, static_cast<double>(parameters[sample]), basis.data());
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
