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

void bsplineBasisValues(int span, int degree, double x_value, const float *knots, double *basis, double *left,
                        double *right)
{
    basis[0] = 1.0;
    for (int j = 1; j <= degree; ++j)
    {
        left[j]       = x_value - static_cast<double>(knots[span + 1 - j]);
        right[j]      = static_cast<double>(knots[span + j]) - x_value;
        double saved  = 0.0;
        for (int r = 0; r < j; ++r)
        {
            const double denominator = right[r + 1] + left[j - r];
            const double temp        = denominator == 0.0 ? 0.0 : basis[r] / denominator;
            basis[r]                 = saved + right[r + 1] * temp;
            saved                    = left[j - r] * temp;
        }
        basis[j] = saved;
    }
}

int findKnotInterval(const float *knots, int64_t num_coefficients, int degree, double x_value);

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
    const auto pivots = factorLinearSystem(matrix, size, "B-spline collocation matrix is singular");
    solveFactoredLinearSystem(matrix, pivots, rhs.data(), size);
    return rhs;
}

std::vector<double> buildNormalMatrix(const std::vector<double> &basis, int64_t num_points, int64_t num_coefficients)
{
    std::vector<double> normal(static_cast<size_t>(num_coefficients) * num_coefficients, 0.0);
    for (int64_t sample = 0; sample < num_points; ++sample)
    {
        const double *basis_row = basis.data() + static_cast<size_t>(sample) * num_coefficients;
        for (int64_t row = 0; row < num_coefficients; ++row)
        {
            const double row_value = basis_row[row];
            if (row_value == 0.0)
            {
                continue;
            }
            for (int64_t col = row; col < num_coefficients; ++col)
            {
                const double col_value = basis_row[col];
                if (col_value != 0.0)
                {
                    normal[static_cast<size_t>(row) * num_coefficients + col] += row_value * col_value;
                }
            }
        }
    }
    for (int64_t row = 1; row < num_coefficients; ++row)
    {
        for (int64_t col = 0; col < row; ++col)
        {
            normal[static_cast<size_t>(row) * num_coefficients + col]
                = normal[static_cast<size_t>(col) * num_coefficients + row];
        }
    }
    return normal;
}

std::vector<double> buildRhsMatrix(const std::vector<double> &basis, const float *y, int64_t num_points,
                                   int64_t num_dims, int64_t num_coefficients)
{
    std::vector<double> rhs(static_cast<size_t>(num_dims) * num_coefficients, 0.0);
    for (int64_t sample = 0; sample < num_points; ++sample)
    {
        const double *basis_row = basis.data() + static_cast<size_t>(sample) * num_coefficients;
        for (int64_t col = 0; col < num_coefficients; ++col)
        {
            const double basis_value = basis_row[col];
            if (basis_value == 0.0)
            {
                continue;
            }
            for (int64_t dim = 0; dim < num_dims; ++dim)
            {
                rhs[static_cast<size_t>(dim) * num_coefficients + col]
                    += basis_value * static_cast<double>(y[static_cast<size_t>(sample) * num_dims + dim]);
            }
        }
    }
    return rhs;
}

class BandedMatrix
{
public:
    BandedMatrix(int64_t size, int lower_bandwidth, int upper_bandwidth)
        : size_(size),
          lower_bandwidth_(lower_bandwidth),
          upper_bandwidth_(upper_bandwidth),
          stride_(lower_bandwidth + upper_bandwidth + 1),
          values_(static_cast<size_t>(size) * stride_, 0.0)
    {}

    [[nodiscard]] int64_t size() const
    {
        return size_;
    }

    [[nodiscard]] int lowerBandwidth() const
    {
        return lower_bandwidth_;
    }

    [[nodiscard]] int upperBandwidth() const
    {
        return upper_bandwidth_;
    }

    [[nodiscard]] int stride() const
    {
        return stride_;
    }

    [[nodiscard]] const double *rowData(int64_t row) const
    {
        return values_.data() + static_cast<size_t>(row) * stride_;
    }

    [[nodiscard]] double *rowData(int64_t row)
    {
        return values_.data() + static_cast<size_t>(row) * stride_;
    }

    [[nodiscard]] bool contains(int64_t row, int64_t col) const
    {
        const int64_t offset = col - row + lower_bandwidth_;
        return offset >= 0 && offset < stride_;
    }

    [[nodiscard]] double get(int64_t row, int64_t col) const
    {
        if (!contains(row, col))
        {
            return 0.0;
        }
        return values_[static_cast<size_t>(row) * stride_ + static_cast<size_t>(col - row + lower_bandwidth_)];
    }

    void set(int64_t row, int64_t col, double value)
    {
        if (!contains(row, col))
        {
            if (value != 0.0)
            {
                throw Exception(Status::ERROR_INVALID_ARGUMENT, "B-spline collocation exceeded banded storage");
            }
            return;
        }
        values_[static_cast<size_t>(row) * stride_ + static_cast<size_t>(col - row + lower_bandwidth_)] = value;
    }

    void add(int64_t row, int64_t col, double value)
    {
        if (value == 0.0)
        {
            return;
        }
        set(row, col, get(row, col) + value);
    }

private:
    int64_t             size_{0};
    int                 lower_bandwidth_{0};
    int                 upper_bandwidth_{0};
    int                 stride_{0};
    std::vector<double> values_;
};

void factorBandedSystem(BandedMatrix &matrix, const char *singular_message)
{
    const int64_t size  = matrix.size();
    const int     lower = matrix.lowerBandwidth();
    const int     upper = matrix.upperBandwidth();
    for (int64_t pivot = 0; pivot < size; ++pivot)
    {
        double      *pivot_row   = matrix.rowData(pivot);
        const double pivot_value = pivot_row[lower];
        if (std::abs(pivot_value) <= std::numeric_limits<double>::epsilon())
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, singular_message);
        }

        const int64_t last_row = std::min<int64_t>(size - 1, pivot + lower);
        const int64_t last_col = std::min<int64_t>(size - 1, pivot + upper);
        for (int64_t row = pivot + 1; row <= last_row; ++row)
        {
            double      *target_row    = matrix.rowData(row);
            const int    target_pivot  = lower - static_cast<int>(row - pivot);
            const double factor        = target_row[target_pivot] / pivot_value;
            if (factor == 0.0)
            {
                continue;
            }
            target_row[target_pivot] = factor;
            for (int64_t col = pivot + 1; col <= last_col; ++col)
            {
                target_row[lower + static_cast<int>(col - row)]
                    -= factor * pivot_row[lower + static_cast<int>(col - pivot)];
            }
        }
    }
}

void solveFactoredBandedSystem(const BandedMatrix &matrix, double *rhs)
{
    const int64_t size  = matrix.size();
    const int     lower = matrix.lowerBandwidth();
    const int     upper = matrix.upperBandwidth();

    for (int64_t pivot = 0; pivot < size; ++pivot)
    {
        const int64_t last_row = std::min<int64_t>(size - 1, pivot + lower);
        const double  rhs_pivot = rhs[pivot];
        for (int64_t row = pivot + 1; row <= last_row; ++row)
        {
            rhs[row] -= matrix.rowData(row)[lower - static_cast<int>(row - pivot)] * rhs_pivot;
        }
    }

    for (int64_t row = size - 1; row >= 0; --row)
    {
        const double *matrix_row = matrix.rowData(row);
        double        value    = rhs[row];
        const int64_t last_col = std::min<int64_t>(size - 1, row + upper);
        for (int64_t col = row + 1; col <= last_col; ++col)
        {
            value -= matrix_row[lower + static_cast<int>(col - row)] * rhs[col];
        }
        rhs[row] = value / matrix_row[lower];
    }
}

struct PenalizedSplineSolution
{
    std::vector<float> coefficients;
    double             residual_sum_squares{0.0};
};

struct LocalBasisRows
{
    int                 degree{0};
    std::vector<int>    first_columns;
    std::vector<double> values;
};

LocalBasisRows buildLocalBasisRows(const std::vector<float> &knots, const float *x, int64_t num_points,
                                   int64_t num_coefficients, int degree)
{
    LocalBasisRows rows;
    rows.degree = degree;
    rows.first_columns.resize(static_cast<size_t>(num_points));
    rows.values.resize(static_cast<size_t>(num_points) * static_cast<size_t>(degree + 1));

    std::vector<double> left(static_cast<size_t>(degree + 1));
    std::vector<double> right(static_cast<size_t>(degree + 1));
    int                 span = degree;
    for (int64_t sample = 0; sample < num_points; ++sample)
    {
        const double x_value = static_cast<double>(x[sample]);
        if (x_value <= knots[static_cast<size_t>(degree)])
        {
            span = degree;
        }
        else if (x_value >= knots[static_cast<size_t>(num_coefficients)])
        {
            span = static_cast<int>(num_coefficients - 1);
        }
        else
        {
            while (span + 1 <= num_coefficients && x_value >= knots[static_cast<size_t>(span + 1)])
            {
                ++span;
            }
            span = std::max(span, degree);
        }

        rows.first_columns[static_cast<size_t>(sample)] = span - degree;
        bsplineBasisValues(span, degree, x_value, knots.data(),
                           rows.values.data() + static_cast<size_t>(sample) * static_cast<size_t>(degree + 1),
                           left.data(), right.data());
    }
    return rows;
}

struct BandedSplineSystem
{
    BandedMatrix       normal;
    std::vector<double> rhs;
};

BandedSplineSystem buildBandedNormalSystem(const LocalBasisRows &basis_rows, const float *y, int64_t num_points,
                                           int64_t num_dims, int64_t num_coefficients)
{
    const int degree = basis_rows.degree;
    const int bandwidth = std::max(degree, 2);
    BandedSplineSystem system{BandedMatrix(num_coefficients, bandwidth, bandwidth),
                              std::vector<double>(static_cast<size_t>(num_dims) * num_coefficients, 0.0)};

    for (int64_t sample = 0; sample < num_points; ++sample)
    {
        const int     first_col   = basis_rows.first_columns[static_cast<size_t>(sample)];
        const double *local_basis = basis_rows.values.data()
                                  + static_cast<size_t>(sample) * static_cast<size_t>(degree + 1);
        for (int lhs = 0; lhs <= degree; ++lhs)
        {
            const int64_t lhs_col = static_cast<int64_t>(first_col + lhs);
            if (lhs_col < 0 || lhs_col >= num_coefficients)
            {
                continue;
            }
            const double lhs_value = local_basis[lhs];
            for (int64_t dim = 0; dim < num_dims; ++dim)
            {
                system.rhs[static_cast<size_t>(dim) * num_coefficients + lhs_col]
                    += lhs_value * static_cast<double>(y[static_cast<size_t>(sample) * num_dims + dim]);
            }
            for (int rhs = lhs; rhs <= degree; ++rhs)
            {
                const int64_t rhs_col = static_cast<int64_t>(first_col + rhs);
                if (rhs_col < 0 || rhs_col >= num_coefficients)
                {
                    continue;
                }
                const double value = lhs_value * local_basis[rhs];
                system.normal.add(lhs_col, rhs_col, value);
                if (rhs_col != lhs_col)
                {
                    system.normal.add(rhs_col, lhs_col, value);
                }
            }
        }
    }
    return system;
}

void addSecondDifferencePenalty(BandedMatrix &matrix, int64_t num_coefficients, double lambda)
{
    if (lambda == 0.0 || num_coefficients < 3)
    {
        return;
    }
    for (int64_t row = 0; row < num_coefficients - 2; ++row)
    {
        const int64_t indices[3] = {row, row + 1, row + 2};
        const double  values[3]  = {1.0, -2.0, 1.0};
        for (int lhs = 0; lhs < 3; ++lhs)
        {
            double *matrix_row = matrix.rowData(indices[lhs]);
            for (int rhs = 0; rhs < 3; ++rhs)
            {
                matrix_row[matrix.lowerBandwidth() + static_cast<int>(indices[rhs] - indices[lhs])]
                    += lambda * values[lhs] * values[rhs];
            }
        }
    }
}

double computeSplineResidualLocal(const LocalBasisRows &basis_rows, const float *y, int64_t num_points,
                                  int64_t num_dims, int64_t num_coefficients,
                                  const std::vector<float> &coefficients)
{
    const int degree   = basis_rows.degree;
    double    residual = 0.0;
    for (int64_t sample = 0; sample < num_points; ++sample)
    {
        const int     first_col   = basis_rows.first_columns[static_cast<size_t>(sample)];
        const double *local_basis = basis_rows.values.data()
                                  + static_cast<size_t>(sample) * static_cast<size_t>(degree + 1);
        for (int64_t dim = 0; dim < num_dims; ++dim)
        {
            double predicted = 0.0;
            for (int j = 0; j <= degree; ++j)
            {
                const int64_t col = static_cast<int64_t>(first_col + j);
                if (col >= 0 && col < num_coefficients)
                {
                    predicted += local_basis[j] * coefficients[static_cast<size_t>(col) * num_dims + dim];
                }
            }
            const double diff = static_cast<double>(y[static_cast<size_t>(sample) * num_dims + dim]) - predicted;
            residual += diff * diff;
        }
    }
    return residual;
}

PenalizedSplineSolution solveAffineCoefficientSpline(const LocalBasisRows &basis_rows, const float *y,
                                                     int64_t num_points, int64_t num_dims,
                                                     int64_t num_coefficients)
{
    const int           degree = basis_rows.degree;
    std::vector<double> normal(4, 0.0);
    std::vector<double> rhs(static_cast<size_t>(num_dims) * 2U, 0.0);

    for (int64_t sample = 0; sample < num_points; ++sample)
    {
        const int     first_col   = basis_rows.first_columns[static_cast<size_t>(sample)];
        const double *local_basis = basis_rows.values.data()
                                  + static_cast<size_t>(sample) * static_cast<size_t>(degree + 1);

        double basis_sum               = 0.0;
        double weighted_coordinate_sum = 0.0;
        for (int j = 0; j <= degree; ++j)
        {
            const int64_t col = static_cast<int64_t>(first_col + j);
            if (col >= 0 && col < num_coefficients)
            {
                const double basis_value = local_basis[j];
                const double coordinate
                    = num_coefficients <= 1 ? 0.0 : 2.0 * static_cast<double>(col) / static_cast<double>(num_coefficients - 1) - 1.0;
                basis_sum += basis_value;
                weighted_coordinate_sum += basis_value * coordinate;
            }
        }

        normal[0] += basis_sum * basis_sum;
        normal[1] += basis_sum * weighted_coordinate_sum;
        normal[2] += weighted_coordinate_sum * basis_sum;
        normal[3] += weighted_coordinate_sum * weighted_coordinate_sum;
        for (int64_t dim = 0; dim < num_dims; ++dim)
        {
            const double y_value = static_cast<double>(y[static_cast<size_t>(sample) * num_dims + dim]);
            rhs[static_cast<size_t>(dim) * 2U] += basis_sum * y_value;
            rhs[static_cast<size_t>(dim) * 2U + 1U] += weighted_coordinate_sum * y_value;
        }
    }

    const auto pivots = factorLinearSystem(normal, 2, "B-spline smoothing affine limit is singular");

    PenalizedSplineSolution solution;
    solution.coefficients.resize(static_cast<size_t>(num_coefficients) * num_dims);
    double affine[2] = {0.0, 0.0};
    for (int64_t dim = 0; dim < num_dims; ++dim)
    {
        affine[0] = rhs[static_cast<size_t>(dim) * 2U];
        affine[1] = rhs[static_cast<size_t>(dim) * 2U + 1U];
        solveFactoredLinearSystem(normal, pivots, affine, 2);
        for (int64_t col = 0; col < num_coefficients; ++col)
        {
            const double coordinate
                = num_coefficients <= 1 ? 0.0 : 2.0 * static_cast<double>(col) / static_cast<double>(num_coefficients - 1) - 1.0;
            solution.coefficients[static_cast<size_t>(col) * num_dims + dim]
                = static_cast<float>(affine[0] + affine[1] * coordinate);
        }
    }

    solution.residual_sum_squares
        = computeSplineResidualLocal(basis_rows, y, num_points, num_dims, num_coefficients, solution.coefficients);
    return solution;
}

PenalizedSplineSolution solvePenalizedSplineBanded(const BandedSplineSystem &system,
                                                   const LocalBasisRows &basis_rows, const float *y,
                                                   int64_t num_points, int64_t num_dims,
                                                   int64_t num_coefficients, double lambda)
{
    auto normal = system.normal;
    addSecondDifferencePenalty(normal, num_coefficients, lambda);
    factorBandedSystem(normal, "B-spline smoothing normal matrix is singular");

    PenalizedSplineSolution solution;
    solution.coefficients.resize(static_cast<size_t>(num_coefficients) * num_dims);
    std::vector<double> rhs(static_cast<size_t>(num_coefficients));
    for (int64_t dim = 0; dim < num_dims; ++dim)
    {
        std::copy_n(system.rhs.data() + static_cast<size_t>(dim) * num_coefficients,
                    static_cast<size_t>(num_coefficients), rhs.data());
        solveFactoredBandedSystem(normal, rhs.data());
        for (int64_t col = 0; col < num_coefficients; ++col)
        {
            solution.coefficients[static_cast<size_t>(col) * num_dims + dim]
                = static_cast<float>(rhs[static_cast<size_t>(col)]);
        }
    }

    solution.residual_sum_squares
        = computeSplineResidualLocal(basis_rows, y, num_points, num_dims, num_coefficients, solution.coefficients);
    return solution;
}

std::vector<double> buildBasisMatrix(const std::vector<float> &knots, const float *x, int64_t num_points,
                                     int64_t num_coefficients, int degree)
{
    std::vector<double> basis(static_cast<size_t>(num_points) * num_coefficients, 0.0);
    std::vector<double> local_basis(static_cast<size_t>(degree + 1));
    std::vector<double> left(static_cast<size_t>(degree + 1));
    std::vector<double> right(static_cast<size_t>(degree + 1));
    for (int64_t row = 0; row < num_points; ++row)
    {
        const int span = findKnotInterval(knots.data(), num_coefficients, degree, static_cast<double>(x[row]));
        bsplineBasisValues(span, degree, static_cast<double>(x[row]), knots.data(), local_basis.data(), left.data(),
                           right.data());
        const int first_col = span - degree;
        for (int j = 0; j <= degree; ++j)
        {
            const int64_t col = static_cast<int64_t>(first_col + j);
            if (col >= 0 && col < num_coefficients)
            {
                basis[static_cast<size_t>(row) * num_coefficients + col] = local_basis[static_cast<size_t>(j)];
            }
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

PenalizedSplineSolution solvePenalizedSpline(const std::vector<double> &basis, const std::vector<double> &normal_base,
                                             const std::vector<double> &penalty, const std::vector<double> &rhs_base,
                                             const float *y, int64_t num_points, int64_t num_dims,
                                             int64_t num_coefficients, double lambda)
{
    auto normal = normal_base;
    for (size_t i = 0; i < normal.size(); ++i)
    {
        normal[i] += lambda * penalty[i];
    }
    const auto pivots = factorLinearSystem(normal, static_cast<int>(num_coefficients),
                                           "B-spline smoothing normal matrix is singular");

    PenalizedSplineSolution solution;
    solution.coefficients.resize(static_cast<size_t>(num_coefficients) * num_dims);
    std::vector<double> rhs(static_cast<size_t>(num_coefficients));
    for (int64_t dim = 0; dim < num_dims; ++dim)
    {
        std::copy_n(rhs_base.data() + static_cast<size_t>(dim) * num_coefficients,
                    static_cast<size_t>(num_coefficients), rhs.data());
        solveFactoredLinearSystem(normal, pivots, rhs.data(), static_cast<int>(num_coefficients));
        for (int64_t col = 0; col < num_coefficients; ++col)
        {
            solution.coefficients[static_cast<size_t>(col) * num_dims + dim]
                = static_cast<float>(rhs[static_cast<size_t>(col)]);
        }
    }

    solution.residual_sum_squares
        = computeSplineResidual(basis, y, num_points, num_dims, num_coefficients, solution.coefficients);
    return solution;
}

PenalizedSplineSolution fitSmoothedSpline(const std::vector<float> &knots, const float *x, const float *y,
                                          int64_t num_points, int64_t num_dims, int degree, double smoothing)
{
    constexpr int    kMaxSearchIterations  = 48;
    constexpr int    kMaxRefineIterations  = 12;
    constexpr double kLambdaGrowth         = 3.1622776601683793319988935444327;
    constexpr double kRelativeTolerance    = 2.0e-3;

    const int64_t num_coefficients = static_cast<int64_t>(knots.size()) - degree - 1;
    const auto    basis_rows       = buildLocalBasisRows(knots, x, num_points, num_coefficients, degree);

    if (num_coefficients < 3)
    {
        const auto system = buildBandedNormalSystem(basis_rows, y, num_points, num_dims, num_coefficients);
        return solvePenalizedSplineBanded(system, basis_rows, y, num_points, num_dims, num_coefficients, 0.0);
    }

    const auto affine_limit = solveAffineCoefficientSpline(basis_rows, y, num_points, num_dims, num_coefficients);
    if (affine_limit.residual_sum_squares <= smoothing)
    {
        return affine_limit;
    }

    const auto              system = buildBandedNormalSystem(basis_rows, y, num_points, num_dims, num_coefficients);
    PenalizedSplineSolution best;
    bool                    has_best         = false;
    double                  best_lambda      = 0.0;
    double                  upper_after_best = 0.0;
    bool                    has_upper_after_best = false;
    const double            coefficient_count = std::max(1.0, static_cast<double>(num_coefficients));
    double                  lambda = std::max(1.0e-12, coefficient_count * coefficient_count * coefficient_count
                                                            * coefficient_count * 1.0e-10);
    const double            residual_tolerance = std::max(1.0e-8, smoothing * kRelativeTolerance);
    int                     over_budget_after_best = 0;
    for (int iter = 0; iter < kMaxSearchIterations; ++iter)
    {
        const auto candidate = solvePenalizedSplineBanded(system, basis_rows, y, num_points, num_dims,
                                                          num_coefficients, lambda);
        if (candidate.residual_sum_squares <= smoothing)
        {
            if (!has_best || candidate.residual_sum_squares > best.residual_sum_squares)
            {
                best                   = candidate;
                has_best               = true;
                best_lambda            = lambda;
                has_upper_after_best   = false;
                over_budget_after_best = 0;
                if (smoothing - candidate.residual_sum_squares <= residual_tolerance)
                {
                    break;
                }
            }
            else if (lambda > best_lambda)
            {
                over_budget_after_best = 0;
            }
        }
        else if (!has_best)
        {
            lambda *= 0.1;
            continue;
        }
        else if (lambda > best_lambda)
        {
            if (!has_upper_after_best)
            {
                upper_after_best     = lambda;
                has_upper_after_best = true;
            }
            ++over_budget_after_best;
            if (over_budget_after_best >= 2)
            {
                break;
            }
        }

        lambda *= kLambdaGrowth;
    }

    if (!has_best)
    {
        return solvePenalizedSplineBanded(system, basis_rows, y, num_points, num_dims, num_coefficients, 0.0);
    }

    if (has_upper_after_best && best.residual_sum_squares < smoothing * 0.8)
    {
        double lambda_low  = best_lambda;
        double lambda_high = upper_after_best;
        for (int iter = 0; iter < kMaxRefineIterations; ++iter)
        {
            const double lambda_mid = std::sqrt(lambda_low * lambda_high);
            const auto   candidate = solvePenalizedSplineBanded(system, basis_rows, y, num_points, num_dims,
                                                                num_coefficients, lambda_mid);
            if (candidate.residual_sum_squares <= smoothing)
            {
                lambda_low = lambda_mid;
                if (candidate.residual_sum_squares > best.residual_sum_squares)
                {
                    best        = candidate;
                    best_lambda = lambda_mid;
                    if (smoothing - candidate.residual_sum_squares <= residual_tolerance)
                    {
                        break;
                    }
                }
            }
            else
            {
                lambda_high = lambda_mid;
            }
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

    result.coefficients.resize(static_cast<size_t>(num_coefficients) * num_dims);
    if (degree == 1)
    {
        std::copy(y, y + static_cast<size_t>(num_points) * num_dims, result.coefficients.begin());
        return result;
    }

    BandedMatrix       collocation(num_points, degree, degree);
    std::vector<double> basis(static_cast<size_t>(degree + 1));
    std::vector<double> left(static_cast<size_t>(degree + 1));
    std::vector<double> right(static_cast<size_t>(degree + 1));
    for (int64_t row = 0; row < num_points; ++row)
    {
        const int span = findKnotInterval(result.knots.data(), num_coefficients, degree, static_cast<double>(x[row]));
        bsplineBasisValues(span, degree, static_cast<double>(x[row]), result.knots.data(), basis.data(), left.data(),
                           right.data());
        const int first_col = span - degree;
        for (int j = 0; j <= degree; ++j)
        {
            const int64_t col = static_cast<int64_t>(first_col + j);
            if (col >= 0 && col < num_coefficients)
            {
                collocation.set(row, col, basis[static_cast<size_t>(j)]);
            }
        }
    }
    factorBandedSystem(collocation, "B-spline collocation matrix is singular");

    std::vector<double> rhs(static_cast<size_t>(num_points));
    for (int64_t dim = 0; dim < num_dims; ++dim)
    {
        for (int64_t row = 0; row < num_points; ++row)
        {
            rhs[static_cast<size_t>(row)] = y[row * num_dims + dim];
        }
        solveFactoredBandedSystem(collocation, rhs.data());
        for (int64_t row = 0; row < num_coefficients; ++row)
        {
            result.coefficients[static_cast<size_t>(row) * num_dims + dim]
                = static_cast<float>(rhs[static_cast<size_t>(row)]);
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
