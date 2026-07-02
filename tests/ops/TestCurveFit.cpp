#include <gtest/gtest.h>
#include <inferrt/core/Exception.hpp>
#include <inferrt/ops/BSplineInterp.hpp>
#include <inferrt/ops/BezierFit.hpp>

#include <cmath>
#include <vector>

namespace {

void expectNearVector(const std::vector<float> &actual, const std::vector<float> &expected, float tol)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], tol) << "index=" << i;
    }
}

} // namespace

TEST(BezierFitTest, RecoversQuadraticBezierWithExplicitParameters)
{
    const std::vector<float> control_points{
        0.0F, 0.0F,
        1.0F, 2.0F,
        3.0F, 0.0F,
    };
    const std::vector<float> parameters{0.0F, 0.25F, 0.5F, 0.75F, 1.0F};
    const auto samples = irt::ops::evaluateBezierCurve(control_points.data(), 3, 2, parameters.data(),
                                                       static_cast<int64_t>(parameters.size()));

    const auto fit = irt::ops::fitBezierCurve(samples.data(), static_cast<int64_t>(parameters.size()), 2, 2,
                                              parameters.data());

    EXPECT_EQ(fit.degree, 2);
    EXPECT_EQ(fit.dimensions, 2);
    expectNearVector(fit.control_points, control_points, 1.0e-5F);
    EXPECT_NEAR(fit.residual_sum_squares, 0.0, 1.0e-9);
}

TEST(BezierFitTest, UsesChordLengthParametersByDefault)
{
    const std::vector<float> points{
        0.0F, 0.0F,
        3.0F, 4.0F,
        6.0F, 4.0F,
    };

    const auto params = irt::ops::chordLengthParameters(points.data(), 3, 2);

    expectNearVector(params, {0.0F, 0.625F, 1.0F}, 1.0e-6F);
}

TEST(BSplineInterpTest, LinearSplineInterpolatesSamples)
{
    const std::vector<float> x{0.0F, 1.0F, 2.0F, 4.0F};
    const std::vector<float> y{
        0.0F, 0.0F,
        1.0F, 2.0F,
        2.0F, 0.0F,
        4.0F, 4.0F,
    };

    const auto spline = irt::ops::makeInterpSpline(x.data(), y.data(), 4, 2, 1);
    const auto actual = irt::ops::evaluateBSpline(spline.knots.data(), static_cast<int64_t>(spline.knots.size()),
                                                  spline.coefficients.data(), 4, 2, 1, x.data(), 4);

    expectNearVector(actual, y, 1.0e-6F);
}

TEST(BSplineInterpTest, CubicNotAKnotSplineInterpolatesSamples)
{
    const std::vector<float> x{0.0F, 0.5F, 1.5F, 2.5F, 4.0F, 5.0F};
    const std::vector<float> y{
        0.0F, 1.0F,
        0.25F, 0.8F,
        2.25F, -0.2F,
        6.25F, 0.1F,
        16.0F, 0.5F,
        25.0F, 1.0F,
    };

    const auto spline = irt::ops::makeInterpSpline(x.data(), y.data(), 6, 2, 3);
    const auto actual = irt::ops::evaluateBSpline(spline.knots.data(), static_cast<int64_t>(spline.knots.size()),
                                                  spline.coefficients.data(), 6, 2, 3, x.data(), 6);

    expectNearVector(actual, y, 2.0e-5F);
}

TEST(SplPrepTest, GeneratesChordLengthParametersAndInterpolatesSamples)
{
    const std::vector<float> points{
        0.0F, 0.0F,
        3.0F, 4.0F,
        6.0F, 4.0F,
        10.0F, 4.0F,
    };

    const auto spline = irt::ops::splPrep(points.data(), 4, 2, 0.0F, 1);
    const auto actual = irt::ops::evaluateBSpline(spline.knots.data(), static_cast<int64_t>(spline.knots.size()),
                                                  spline.coefficients.data(), 4, 2, 1, spline.parameters.data(), 4);

    expectNearVector(spline.parameters, {0.0F, 5.0F / 12.0F, 8.0F / 12.0F, 1.0F}, 1.0e-6F);
    expectNearVector(actual, points, 1.0e-6F);
}

TEST(SplPrepTest, SmoothingHonorsResidualBudget)
{
    const std::vector<float> points{
        0.0F, 0.0F,
        0.2F, 0.3F,
        0.5F, -0.2F,
        0.8F, 0.4F,
        1.1F, -0.1F,
        1.4F, 0.5F,
        1.7F, 0.0F,
        2.0F, 0.6F,
    };
    constexpr float smoothing = 0.08F;

    const auto spline = irt::ops::splPrep(points.data(), 8, 2, smoothing, 3);
    const auto actual = irt::ops::evaluateBSpline(spline.knots.data(), static_cast<int64_t>(spline.knots.size()),
                                                  spline.coefficients.data(), 8, 2, 3, spline.parameters.data(), 8);

    double residual = 0.0;
    for (size_t i = 0; i < points.size(); ++i)
    {
        const double diff = static_cast<double>(points[i]) - actual[i];
        residual += diff * diff;
    }

    EXPECT_EQ(spline.degree, 3);
    EXPECT_EQ(spline.dimensions, 2);
    EXPECT_NEAR(spline.smoothing, smoothing, 0.0F);
    EXPECT_GT(spline.residual_sum_squares, 1.0e-5);
    EXPECT_LE(spline.residual_sum_squares, static_cast<double>(smoothing) * 1.001 + 1.0e-6);
    EXPECT_NEAR(spline.residual_sum_squares, residual, 1.0e-5);
}

TEST(CurveFitTest, RejectsInvalidArguments)
{
    const std::vector<float> x{0.0F, 0.0F, 1.0F, 2.0F};
    const std::vector<float> y{0.0F, 1.0F, 2.0F, 3.0F};

    EXPECT_THROW((void)irt::ops::makeInterpSpline(x.data(), y.data(), 4, 1, 3), irt::Exception);
    EXPECT_THROW((void)irt::ops::fitBezierCurve(y.data(), 2, 1, 3), irt::Exception);
    EXPECT_THROW((void)irt::ops::splPrep(y.data(), 2, 1, 0.0F, 3), irt::Exception);
    EXPECT_THROW((void)irt::ops::splPrep(y.data(), 4, 1, -1.0F, 1), irt::Exception);
}
