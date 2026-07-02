"""Curve fitting/interpolation ops compared with SciPy."""

from __future__ import annotations

import numpy as np
import pytest


def _as_matrix(values: list[float], dims: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, dims)


def test_fit_bezier_curve_matches_scipy_bpoly_evaluation(ops_module) -> None:
    scipy_interpolate = pytest.importorskip("scipy.interpolate")

    control_points = _as_matrix(
        [
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            0.0,
            4.0,
            3.0,
        ],
        2,
    )
    sample_u = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    samples = ops_module.evaluate_bezier_curve(control_points, sample_u)

    fit = ops_module.fit_bezier_curve(samples, 3, sample_u)
    fitted_control = np.asarray(fit.control_points, dtype=np.float32).reshape(4, 2)
    eval_u = np.linspace(0.0, 1.0, 33, dtype=np.float32)

    actual = ops_module.evaluate_bezier_curve(fitted_control, eval_u)
    scipy_curve = scipy_interpolate.BPoly(fitted_control[:, None, :], [0.0, 1.0], extrapolate=True)
    expected = np.asarray(scipy_curve(eval_u), dtype=np.float32)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fitted_control, control_points, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("degree", [1, 3])
def test_make_interp_spline_matches_scipy(ops_module, degree: int) -> None:
    scipy_interpolate = pytest.importorskip("scipy.interpolate")

    x = np.asarray([0.0, 0.4, 1.2, 2.0, 3.5, 4.0], dtype=np.float32)
    y = np.stack(
        [
            np.sin(x * 0.7),
            np.cos(x * 0.5),
        ],
        axis=1,
    ).astype(np.float32)
    x_eval = np.linspace(float(x[0]), float(x[-1]), 41, dtype=np.float32)

    spline = ops_module.make_interp_spline(x, y, degree=degree)
    knots = np.asarray(spline.knots, dtype=np.float32)
    coefficients = np.asarray(spline.coefficients, dtype=np.float32).reshape(len(x), y.shape[1])
    actual = ops_module.evaluate_b_spline(knots, coefficients, int(spline.degree), x_eval)

    scipy_spline = scipy_interpolate.make_interp_spline(x.astype(np.float64), y.astype(np.float64), k=degree)
    expected = np.asarray(scipy_spline(x_eval.astype(np.float64)), dtype=np.float32)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("degree", [1, 3])
def test_splprep_matches_scipy_splprep_s0(ops_module, degree: int) -> None:
    scipy_interpolate = pytest.importorskip("scipy.interpolate")

    points = _as_matrix(
        [
            0.0,
            0.0,
            0.4,
            0.2,
            1.2,
            0.7,
            2.0,
            0.1,
            3.5,
            1.0,
            4.0,
            0.0,
        ],
        2,
    )
    spl = ops_module.splprep(points, smoothing=0.0, degree=degree)
    knots = np.asarray(spl.knots, dtype=np.float32)
    coefficients = np.asarray(spl.coefficients, dtype=np.float32).reshape(points.shape)
    u = np.asarray(spl.parameters, dtype=np.float32)
    u_eval = np.linspace(0.0, 1.0, 41, dtype=np.float32)

    actual = ops_module.evaluate_b_spline(knots, coefficients, int(spl.degree), u_eval)
    scipy_tck, scipy_u = scipy_interpolate.splprep(points.T.astype(np.float64), u=u.astype(np.float64), s=0, k=degree)
    expected = np.asarray(scipy_interpolate.splev(u_eval.astype(np.float64), scipy_tck), dtype=np.float32).T

    np.testing.assert_allclose(u, scipy_u.astype(np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_splprep_smoothing_matches_scipy_residual_condition(ops_module) -> None:
    scipy_interpolate = pytest.importorskip("scipy.interpolate")

    points = _as_matrix(
        [
            0.0,
            0.0,
            0.2,
            0.3,
            0.5,
            -0.2,
            0.8,
            0.4,
            1.1,
            -0.1,
            1.4,
            0.5,
            1.7,
            0.0,
            2.0,
            0.6,
            2.3,
            0.1,
            2.6,
            0.7,
        ],
        2,
    )
    smoothing = 0.12

    spl = ops_module.splprep(points, smoothing=smoothing, degree=3)
    knots = np.asarray(spl.knots, dtype=np.float32)
    coefficients = np.asarray(spl.coefficients, dtype=np.float32).reshape(points.shape)
    u = np.asarray(spl.parameters, dtype=np.float32)

    actual_at_samples = ops_module.evaluate_b_spline(knots, coefficients, int(spl.degree), u)
    actual_residual = float(np.sum((points - actual_at_samples) ** 2))

    scipy_tck, scipy_u = scipy_interpolate.splprep(
        points.T.astype(np.float64),
        u=u.astype(np.float64),
        s=smoothing,
        k=3,
    )
    scipy_at_samples = np.asarray(scipy_interpolate.splev(scipy_u, scipy_tck), dtype=np.float32).T
    scipy_residual = float(np.sum((points - scipy_at_samples) ** 2))

    assert actual_residual > 1.0e-5
    assert actual_residual <= smoothing * 1.001 + 1.0e-6
    assert scipy_residual <= smoothing * 1.001 + 1.0e-6
    np.testing.assert_allclose(actual_residual, spl.residual_sum_squares, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual_residual, scipy_residual, rtol=5e-2, atol=5e-3)


def test_curve_ops_reject_invalid_inputs(ops_module) -> None:
    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.make_interp_spline(
            np.asarray([0.0, 0.0, 1.0, 2.0], dtype=np.float32),
            np.zeros((4, 1), dtype=np.float32),
            degree=3,
        )

    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.fit_bezier_curve(np.zeros((2, 2), dtype=np.float32), 3)

    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.splprep(np.zeros((2, 2), dtype=np.float32), degree=3)

    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.splprep(np.zeros((4, 2), dtype=np.float32), smoothing=-1.0, degree=1)
