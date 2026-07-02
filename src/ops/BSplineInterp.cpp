#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/BSplineInterp.hpp>
#include <inferrt/ops/BezierFit.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace irt::ops {
namespace {

void validateX(const float *x, int64_t num_points)
{
    if (num_points < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_points must be non-negative, got %lld",
                        static_cast<long long>(num_points));
    }
    if (num_points > 0 && x == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "x must not be null");
    }
    for (int64_t i = 0; i < num_points; ++i)
    {
        if (!std::isfinite(x[i]))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "x[%lld] must be finite", static_cast<long long>(i));
        }
        if (i > 0 && x[i] <= x[i - 1])
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "x must be strictly increasing");
        }
    }
}

void validateY(const float *y, int64_t num_points, int64_t num_dims)
{
    if (num_dims <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_dims must be positive, got %lld",
                        static_cast<long long>(num_dims));
    }
    if (num_points > 0 && y == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "y must not be null");
    }
    for (int64_t i = 0; i < num_points * num_dims; ++i)
    {
        if (!std::isfinite(y[i]))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "y contains non-finite value at flat index %lld",
                            static_cast<long long>(i));
        }
    }
}

void validateEvalX(const float *x_eval, int64_t num_eval)
{
    if (num_eval < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_eval must be non-negative, got %lld",
                        static_cast<long long>(num_eval));
    }
    if (num_eval > 0 && x_eval == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "x_eval must not be null");
    }
    for (int64_t i = 0; i < num_eval; ++i)
    {
        if (!std::isfinite(x_eval[i]))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "x_eval[%lld] must be finite", static_cast<long long>(i));
        }
    }
}

void validateSplineInputs(const float *knots, int64_t num_knots, const float *coefficients, int64_t num_coefficients,
                          int64_t num_dims, int degree)
{
    if (degree < 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "degree must be non-negative, got %d", degree);
    }
    if (num_coefficients < degree + 1)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_coefficients must be at least degree + 1");
    }
    if (num_knots != num_coefficients + degree + 1)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT,
                        "num_knots must equal num_coefficients + degree + 1, got knots=%lld coeffs=%lld degree=%d",
                        static_cast<long long>(num_knots), static_cast<long long>(num_coefficients), degree);
    }
    if (num_dims <= 0)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "num_dims must be positive");
    }
    if (knots == nullptr || coefficients == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "knots and coefficients must not be null");
    }
    for (int64_t i = 0; i < num_knots; ++i)
    {
        if (!std::isfinite(knots[i]))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "knots[%lld] must be finite", static_cast<long long>(i));
        }
        if (i > 0 && knots[i] < knots[i - 1])
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "knots must be sorted");
        }
    }
    for (int64_t i = 0; i < num_coefficients * num_dims; ++i)
    {
        if (!std::isfinite(coefficients[i]))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "coefficients contains non-finite value");
        }
    }
}

std::vector<float> notAKnotKnots(const float *x, int64_t num_points, int degree)
{
    std::vector<float> knots;
    if (degree == 1)
    {
        knots.reserve(static_cast<size_t>(num_points) + 2U);
        knots.push_back(x[0]);
        knots.insert(knots.end(), x, x + num_points);
        knots.push_back(x[num_points - 1]);
        return knots;
    }
    if (degree == 3)
    {
        knots.reserve(static_cast<size_t>(num_points) + 4U);
        for (int i = 0; i < degree + 1; ++i)
        {
            knots.push_back(x[0]);
        }
        for (int64_t i = 2; i <= num_points - 3; ++i)
        {
            knots.push_back(x[i]);
        }
        for (int i = 0; i < degree + 1; ++i)
        {
            knots.push_back(x[num_points - 1]);
        }
        return knots;
    }
    throw Exception(Status::ERROR_NOT_IMPLEMENTED, "makeInterpSpline currently supports degree 1 and 3");
}

double bsplineBasis(int basis_index, int degree, double x_value, const std::vector<float> &knots)
{
    if (degree == 0)
    {
        const double left  = knots[static_cast<size_t>(basis_index)];
        const double right = knots[static_cast<size_t>(basis_index + 1)];
        const double last  = knots.back();
        if ((left <= x_value && x_value < right) || (x_value == last && left <= x_value && x_value <= right))
        {
            return 1.0;
        }
        return 0.0;
    }

    double       value    = 0.0;
    const double left_den = knots[static_cast<size_t>(basis_index + degree)] - knots[static_cast<size_t>(basis_index)];
    if (left_den != 0.0)
    {
        value += (x_value - knots[static_cast<size_t>(basis_index)]) / left_den
               * bsplineBasis(basis_index, degree - 1, x_value, knots);
    }

    const double right_den
        = knots[static_cast<size_t>(basis_index + degree + 1)] - knots[static_cast<size_t>(basis_index + 1)];
    if (right_den != 0.0)
    {
        value += (knots[static_cast<size_t>(basis_index + degree + 1)] - x_value) / right_den
               * bsplineBasis(basis_index + 1, degree - 1, x_value, knots);
    }
    return value;
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
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "B-spline collocation matrix is singular");
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

struct PenalizedSplineSolution
{
    std::vector<float> coefficients;
    double             residual_sum_squares{0.0};
};

std::vector<double> buildBasisMatrix(const std::vector<float> &knots, const float *x, int64_t num_points,
                                     int64_t num_coefficients, int degree)
{
    std::vector<double> basis(static_cast<size_t>(num_points) * num_coefficients, 0.0);
    for (int64_t row = 0; row < num_points; ++row)
    {
        for (int64_t col = 0; col < num_coefficients; ++col)
        {
            basis[static_cast<size_t>(row) * num_coefficients + col]
                = bsplineBasis(static_cast<int>(col), degree, static_cast<double>(x[row]), knots);
        }
    }
    return basis;
}

std::vector<double> secondDifferencePenalty(int64_t num_coefficients)
{
    std::vector<double> penalty(static_cast<size_t>(num_coefficients) * num_coefficients, 0.0);
    if (num_coefficients < 3)
    {
        return penalty;
    }

    for (int64_t row = 0; row < num_coefficients - 2; ++row)
    {
        const int64_t indices[3] = {row, row + 1, row + 2};
        const double  values[3]  = {1.0, -2.0, 1.0};
        for (int lhs = 0; lhs < 3; ++lhs)
        {
            for (int rhs = 0; rhs < 3; ++rhs)
            {
                penalty[static_cast<size_t>(indices[lhs]) * num_coefficients + indices[rhs]]
                    += values[lhs] * values[rhs];
            }
        }
    }
    return penalty;
}

double computeSplineResidual(const std::vector<double> &basis, const float *y, int64_t num_points, int64_t num_dims,
                             int64_t num_coefficients, const std::vector<float> &coefficients)
{
    double residual = 0.0;
    for (int64_t row = 0; row < num_points; ++row)
    {
        for (int64_t dim = 0; dim < num_dims; ++dim)
        {
            double predicted = 0.0;
            for (int64_t col = 0; col < num_coefficients; ++col)
            {
                predicted += basis[static_cast<size_t>(row) * num_coefficients + col]
                           * coefficients[static_cast<size_t>(col) * num_dims + dim];
            }
            const double diff = static_cast<double>(y[static_cast<size_t>(row) * num_dims + dim]) - predicted;
            residual += diff * diff;
        }
    }
    return residual;
}

PenalizedSplineSolution solvePenalizedSpline(const std::vector<double> &basis, const std::vector<double> &penalty,
                                             const float *y, int64_t num_points, int64_t num_dims,
                                             int64_t num_coefficients, double lambda)
{
    std::vector<double> normal(static_cast<size_t>(num_coefficients) * num_coefficients, 0.0);
    for (int64_t row = 0; row < num_coefficients; ++row)
    {
        for (int64_t col = 0; col < num_coefficients; ++col)
        {
            double value = lambda * penalty[static_cast<size_t>(row) * num_coefficients + col];
            for (int64_t sample = 0; sample < num_points; ++sample)
            {
                value += basis[static_cast<size_t>(sample) * num_coefficients + row]
                       * basis[static_cast<size_t>(sample) * num_coefficients + col];
            }
            normal[static_cast<size_t>(row) * num_coefficients + col] = value;
        }
    }

    PenalizedSplineSolution solution;
    solution.coefficients.resize(static_cast<size_t>(num_coefficients) * num_dims);
    for (int64_t dim = 0; dim < num_dims; ++dim)
    {
        std::vector<double> rhs(static_cast<size_t>(num_coefficients), 0.0);
        for (int64_t col = 0; col < num_coefficients; ++col)
        {
            for (int64_t sample = 0; sample < num_points; ++sample)
            {
                rhs[static_cast<size_t>(col)] += basis[static_cast<size_t>(sample) * num_coefficients + col]
                                               * static_cast<double>(y[static_cast<size_t>(sample) * num_dims + dim]);
            }
        }

        const auto coeff = solveLinearSystem(normal, rhs, static_cast<int>(num_coefficients));
        for (int64_t col = 0; col < num_coefficients; ++col)
        {
            solution.coefficients[static_cast<size_t>(col) * num_dims + dim]
                = static_cast<float>(coeff[static_cast<size_t>(col)]);
        }
    }

    solution.residual_sum_squares
        = computeSplineResidual(basis, y, num_points, num_dims, num_coefficients, solution.coefficients);
    return solution;
}

PenalizedSplineSolution fitSmoothedSpline(const std::vector<float> &knots, const float *x, const float *y,
                                          int64_t num_points, int64_t num_dims, int degree, double smoothing)
{
    const int64_t num_coefficients = static_cast<int64_t>(knots.size()) - degree - 1;
    const auto    basis            = buildBasisMatrix(knots, x, num_points, num_coefficients, degree);
    const auto    penalty          = secondDifferencePenalty(num_coefficients);

    auto best = solvePenalizedSpline(basis, penalty, y, num_points, num_dims, num_coefficients, 0.0);
    if (best.residual_sum_squares >= smoothing)
    {
        return best;
    }

    double lambda_low  = 0.0;
    double lambda_high = 1.0e-12;
    bool   bracketed   = false;
    for (int iter = 0; iter < 80; ++iter)
    {
        const auto candidate
            = solvePenalizedSpline(basis, penalty, y, num_points, num_dims, num_coefficients, lambda_high);
        if (candidate.residual_sum_squares <= smoothing)
        {
            best       = candidate;
            lambda_low = lambda_high;
            lambda_high *= 10.0;
            continue;
        }
        bracketed = true;
        break;
    }

    if (!bracketed)
    {
        return best;
    }

    for (int iter = 0; iter < 80; ++iter)
    {
        const double lambda_mid = lambda_low == 0.0 ? lambda_high * 0.5 : std::sqrt(lambda_low * lambda_high);
        const auto   candidate
            = solvePenalizedSpline(basis, penalty, y, num_points, num_dims, num_coefficients, lambda_mid);
        if (candidate.residual_sum_squares <= smoothing)
        {
            best       = candidate;
            lambda_low = lambda_mid;
        }
        else
        {
            lambda_high = lambda_mid;
        }
    }
    return best;
}

int findKnotInterval(const float *knots, int64_t num_coefficients, int degree, double x_value)
{
    const int last_interval = static_cast<int>(num_coefficients - 1);
    if (x_value <= knots[static_cast<size_t>(degree)])
    {
        return degree;
    }
    if (x_value >= knots[static_cast<size_t>(num_coefficients)])
    {
        return last_interval;
    }
    auto begin = knots + degree;
    auto end   = knots + num_coefficients + 1;
    auto upper = std::upper_bound(begin, end, static_cast<float>(x_value));
    return std::max(degree, static_cast<int>((upper - knots) - 1));
}

} // namespace

BSplineInterpResult makeInterpSpline(const float *x, const float *y, int64_t num_points, int64_t num_dims, int degree)
{
    validateX(x, num_points);
    validateY(y, num_points, num_dims);
    if (degree != 1 && degree != 3)
    {
        throw Exception(Status::ERROR_NOT_IMPLEMENTED, "makeInterpSpline currently supports degree 1 and 3");
    }
    if (num_points < degree + 1)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "makeInterpSpline requires at least degree + 1 points");
    }

    BSplineInterpResult result;
    result.degree                  = degree;
    result.dimensions              = num_dims;
    result.knots                   = notAKnotKnots(x, num_points, degree);
    const int64_t num_coefficients = static_cast<int64_t>(result.knots.size()) - degree - 1;
    if (num_coefficients != num_points)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT,
                        "internal knot construction produced unexpected coefficient count");
    }

    std::vector<double> collocation(static_cast<size_t>(num_points) * num_points, 0.0);
    for (int64_t row = 0; row < num_points; ++row)
    {
        for (int64_t col = 0; col < num_points; ++col)
        {
            collocation[static_cast<size_t>(row) * num_points + col]
                = bsplineBasis(static_cast<int>(col), degree, static_cast<double>(x[row]), result.knots);
        }
    }

    result.coefficients.resize(static_cast<size_t>(num_coefficients) * num_dims);
    for (int64_t dim = 0; dim < num_dims; ++dim)
    {
        std::vector<double> rhs(static_cast<size_t>(num_points));
        for (int64_t row = 0; row < num_points; ++row)
        {
            rhs[static_cast<size_t>(row)] = y[row * num_dims + dim];
        }
        const auto solution = solveLinearSystem(collocation, rhs, static_cast<int>(num_points));
        for (int64_t row = 0; row < num_coefficients; ++row)
        {
            result.coefficients[static_cast<size_t>(row) * num_dims + dim]
                = static_cast<float>(solution[static_cast<size_t>(row)]);
        }
    }
    return result;
}

SplPrepResult splPrep(const float *points, int64_t num_points, int64_t num_dims, float smoothing, int degree,
                      const float *parameters)
{
    validateY(points, num_points, num_dims);
    if (!std::isfinite(smoothing) || smoothing < 0.0F)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "smoothing must be finite and non-negative");
    }
    if (degree != 1 && degree != 3)
    {
        throw Exception(Status::ERROR_NOT_IMPLEMENTED, "splPrep currently supports degree 1 and 3");
    }
    if (num_points < degree + 1)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "splPrep requires at least degree + 1 points");
    }

    std::vector<float> owned_parameters;
    if (parameters == nullptr)
    {
        owned_parameters = chordLengthParameters(points, num_points, num_dims);
        parameters       = owned_parameters.data();
    }
    validateX(parameters, num_points);

    SplPrepResult result;
    result.degree     = degree;
    result.dimensions = num_dims;
    result.smoothing  = smoothing;
    result.parameters.assign(parameters, parameters + num_points);

    if (smoothing == 0.0F)
    {
        const auto spline   = makeInterpSpline(parameters, points, num_points, num_dims, degree);
        result.knots        = spline.knots;
        result.coefficients = spline.coefficients;
        return result;
    }

    result.knots        = notAKnotKnots(parameters, num_points, degree);
    const auto solution = fitSmoothedSpline(result.knots, parameters, points, num_points, num_dims, degree, smoothing);
    result.coefficients = solution.coefficients;
    result.residual_sum_squares = solution.residual_sum_squares;
    return result;
}

void evaluateBSpline(const float *knots, int64_t num_knots, const float *coefficients, int64_t num_coefficients,
                     int64_t num_dims, int degree, const float *x_eval, int64_t num_eval, float *output)
{
    validateSplineInputs(knots, num_knots, coefficients, num_coefficients, num_dims, degree);
    validateEvalX(x_eval, num_eval);
    if (num_eval > 0 && output == nullptr)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "output must not be null");
    }

    std::vector<double> work(static_cast<size_t>(degree + 1) * num_dims);
    for (int64_t sample = 0; sample < num_eval; ++sample)
    {
        const int interval = findKnotInterval(knots, num_coefficients, degree, x_eval[sample]);
        for (int j = 0; j <= degree; ++j)
        {
            const int coeff_index = interval - degree + j;
            for (int64_t dim = 0; dim < num_dims; ++dim)
            {
                work[static_cast<size_t>(j) * num_dims + dim]
                    = coefficients[static_cast<size_t>(coeff_index) * num_dims + dim];
            }
        }

        for (int r = 1; r <= degree; ++r)
        {
            for (int j = degree; j >= r; --j)
            {
                const int    knot_index = interval - degree + j;
                const double left       = knots[static_cast<size_t>(knot_index)];
                const double right      = knots[static_cast<size_t>(knot_index + degree + 1 - r)];
                const double denom      = right - left;
                const double alpha      = denom == 0.0 ? 0.0 : (static_cast<double>(x_eval[sample]) - left) / denom;
                for (int64_t dim = 0; dim < num_dims; ++dim)
                {
                    auto &value = work[static_cast<size_t>(j) * num_dims + dim];
                    value       = (1.0 - alpha) * work[static_cast<size_t>(j - 1) * num_dims + dim] + alpha * value;
                }
            }
        }

        for (int64_t dim = 0; dim < num_dims; ++dim)
        {
            output[static_cast<size_t>(sample) * num_dims + dim]
                = static_cast<float>(work[static_cast<size_t>(degree) * num_dims + dim]);
        }
    }
}

std::vector<float> evaluateBSpline(const float *knots, int64_t num_knots, const float *coefficients,
                                   int64_t num_coefficients, int64_t num_dims, int degree, const float *x_eval,
                                   int64_t num_eval)
{
    std::vector<float> output(static_cast<size_t>(num_eval) * num_dims);
    evaluateBSpline(knots, num_knots, coefficients, num_coefficients, num_dims, degree, x_eval, num_eval,
                    output.data());
    return output;
}

} // namespace irt::ops
