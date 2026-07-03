"""Bezier、B-spline、splprep pybind11 与 Python reference 的 google-benchmark 基准测试。

运行示例：
    D:\\Software\\anaconda3\\envs\\py312\\python.exe benchmark\\python\\benchmark_curve.py --benchmark_min_time=0.05s

可选参数：
    --build-dir <path>  指定 CMake 构建目录，默认使用 INFERRT_BUILD_DIR 或 <repo>/build。

其余 ``--benchmark_*`` 参数会原样传给 google-benchmark。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import google_benchmark as benchmark
import numpy as np
from scipy import interpolate


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_DIR = Path(os.environ.get("INFERRT_BUILD_DIR", REPO_ROOT / "build")).resolve()
CURVE_SIZES = (16, 32, 64)
BEZIER_DEGREE = 3

_DLL_DIRECTORY_HANDLES: list[object] = []
_SINK: Any = None


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    return parser.parse_known_args(argv)


ARGS, BENCHMARK_ARGV_TAIL = _parse_args(sys.argv[1:])
BENCHMARK_ARGV = [sys.argv[0], *BENCHMARK_ARGV_TAIL]


def _read_cmake_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("//") or line.startswith("#") or "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0].strip()
        values[key] = value.strip()
    return values


def _add_runtime_dir(path: Path, path_entries: list[str]) -> None:
    if not path.exists():
        return

    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    if path_text not in path_entries:
        path_entries.insert(0, path_text)
    if hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(path_text))


def _configure_inferrt_runtime(build_dir: Path) -> None:
    build_dir = build_dir.resolve()
    if not build_dir.exists():
        raise FileNotFoundError(f"Build directory not found: {build_dir}")

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for path in (build_dir / "bin", build_dir / "lib"):
        _add_runtime_dir(path, path_entries)

    cache = _read_cmake_cache(build_dir / "CMakeCache.txt")
    trt_root = cache.get("TRT_ROOT", "")
    if trt_root:
        _add_runtime_dir(Path(trt_root) / "bin", path_entries)
        _add_runtime_dir(Path(trt_root) / "lib", path_entries)

    opencv_dir = cache.get("OpenCV_DIR", "")
    if not opencv_dir:
        details = cache.get("FIND_PACKAGE_MESSAGE_DETAILS_OpenCV", "")
        match = re.search(r"\[([A-Za-z]:/.+?)\]", details)
        if match:
            opencv_dir = match.group(1)
    if opencv_dir:
        opencv_root = Path(opencv_dir)
        _add_runtime_dir(opencv_root / "bin", path_entries)
        _add_runtime_dir(opencv_root / "x64" / "vc16" / "bin", path_entries)
        _add_runtime_dir(opencv_root.parent / "bin", path_entries)

    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        _add_runtime_dir(Path(cuda_path) / "bin", path_entries)

    extra_dirs = os.environ.get("INFERRT_DLL_DIRS", "")
    for raw in extra_dirs.split(";"):
        raw = raw.strip()
        if raw:
            _add_runtime_dir(Path(raw), path_entries)

    os.environ["PATH"] = os.pathsep.join(path_entries)


_configure_inferrt_runtime(ARGS.build_dir)
import inferrt_ops_py as ops  # noqa: E402


def _store(value: Any) -> None:
    global _SINK
    _SINK = value


def _bernstein_basis(parameters: np.ndarray, degree: int) -> np.ndarray:
    columns = []
    one_minus = 1.0 - parameters
    for index in range(degree + 1):
        coeff = math.comb(degree, index)
        columns.append(coeff * np.power(parameters, index) * np.power(one_minus, degree - index))
    return np.stack(columns, axis=1)


def _python_fit_bezier(points: np.ndarray, degree: int, parameters: np.ndarray) -> np.ndarray:
    basis = _bernstein_basis(parameters.astype(np.float64), degree)
    control_points, *_ = np.linalg.lstsq(basis, points.astype(np.float64), rcond=None)
    return control_points.astype(np.float32)


def _spline_data(num_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    y = np.stack(
        [
            np.sin(2.0 * np.pi * x),
            np.cos(3.0 * np.pi * x) * 0.5,
        ],
        axis=1,
    ).astype(np.float32)
    return x, y, x.astype(np.float64), y.astype(np.float64)


def _curve_points(num_points: int) -> tuple[np.ndarray, list[np.ndarray]]:
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    points = np.stack(
        [
            t,
            np.sin(2.0 * np.pi * t) * 0.35 + t * 0.2,
        ],
        axis=1,
    ).astype(np.float32)
    return points, [points[:, 0].astype(np.float64), points[:, 1].astype(np.float64)]


def _bezier_samples(num_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, interpolate.BPoly]:
    control_points = np.asarray(
        [
            [0.0, 0.0],
            [0.25, 0.75],
            [0.75, -0.25],
            [1.0, 0.2],
        ],
        dtype=np.float32,
    )
    parameters = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    samples = ops.evaluate_bezier_curve(control_points, parameters)
    bpoly = interpolate.BPoly(control_points.astype(np.float64)[:, None, :], [0.0, 1.0], extrapolate=True)
    return control_points, parameters, np.asarray(samples, dtype=np.float32), bpoly


SPLINE_DATA = {size: _spline_data(size) for size in CURVE_SIZES}
CURVE_DATA = {size: _curve_points(size) for size in CURVE_SIZES}
BEZIER_DATA = {size: _bezier_samples(size) for size in CURVE_SIZES}


@benchmark.register(name="InferRTPybind11/fit_bezier_curve")
@benchmark.option.use_real_time()
@benchmark.option.arg_name("N")
@benchmark.option.arg(CURVE_SIZES[0])
@benchmark.option.arg(CURVE_SIZES[1])
@benchmark.option.arg(CURVE_SIZES[2])
@benchmark.option.unit(benchmark.kMillisecond)
def bench_inferrt_fit_bezier_curve(state: benchmark.State) -> None:
    num_points = state.range(0)
    _, parameters, samples, _ = BEZIER_DATA[num_points]
    while state:
        _store(ops.fit_bezier_curve(samples, BEZIER_DEGREE, parameters))
    state.items_processed = state.iterations * num_points


@benchmark.register(name="Python/numpy_fit_bezier_curve")
@benchmark.option.use_real_time()
@benchmark.option.arg_name("N")
@benchmark.option.arg(CURVE_SIZES[0])
@benchmark.option.arg(CURVE_SIZES[1])
@benchmark.option.arg(CURVE_SIZES[2])
@benchmark.option.unit(benchmark.kMillisecond)
def bench_python_fit_bezier_curve(state: benchmark.State) -> None:
    num_points = state.range(0)
    _, parameters, samples, _ = BEZIER_DATA[num_points]
    while state:
        _store(_python_fit_bezier(samples, BEZIER_DEGREE, parameters))
    state.items_processed = state.iterations * num_points


@benchmark.register(name="InferRTPybind11/evaluate_bezier_curve")
@benchmark.option.use_real_time()
@benchmark.option.arg_name("N")
@benchmark.option.arg(CURVE_SIZES[0])
@benchmark.option.arg(CURVE_SIZES[1])
@benchmark.option.arg(CURVE_SIZES[2])
@benchmark.option.unit(benchmark.kMillisecond)
def bench_inferrt_evaluate_bezier_curve(state: benchmark.State) -> None:
    num_points = state.range(0)
    control_points, parameters, _, _ = BEZIER_DATA[num_points]
    while state:
        _store(ops.evaluate_bezier_curve(control_points, parameters))
    state.items_processed = state.iterations * num_points


@benchmark.register(name="Python/scipy.BPoly_evaluate")
@benchmark.option.use_real_time()
@benchmark.option.arg_name("N")
@benchmark.option.arg(CURVE_SIZES[0])
@benchmark.option.arg(CURVE_SIZES[1])
@benchmark.option.arg(CURVE_SIZES[2])
@benchmark.option.unit(benchmark.kMillisecond)
def bench_scipy_bpoly_evaluate(state: benchmark.State) -> None:
    num_points = state.range(0)
    _, parameters, _, bpoly = BEZIER_DATA[num_points]
    parameters64 = parameters.astype(np.float64)
    while state:
        _store(bpoly(parameters64))
    state.items_processed = state.iterations * num_points


@benchmark.register(name="InferRTPybind11/make_interp_spline")
@benchmark.option.use_real_time()
@benchmark.option.arg_name("N")
@benchmark.option.arg(CURVE_SIZES[0])
@benchmark.option.arg(CURVE_SIZES[1])
@benchmark.option.arg(CURVE_SIZES[2])
@benchmark.option.unit(benchmark.kMillisecond)
def bench_inferrt_make_interp_spline(state: benchmark.State) -> None:
    num_points = state.range(0)
    x, y, _, _ = SPLINE_DATA[num_points]
    while state:
        _store(ops.make_interp_spline(x, y, degree=3))
    state.items_processed = state.iterations * num_points


@benchmark.register(name="Python/scipy.make_interp_spline")
@benchmark.option.use_real_time()
@benchmark.option.arg_name("N")
@benchmark.option.arg(CURVE_SIZES[0])
@benchmark.option.arg(CURVE_SIZES[1])
@benchmark.option.arg(CURVE_SIZES[2])
@benchmark.option.unit(benchmark.kMillisecond)
def bench_scipy_make_interp_spline(state: benchmark.State) -> None:
    num_points = state.range(0)
    _, _, x64, y64 = SPLINE_DATA[num_points]
    while state:
        _store(interpolate.make_interp_spline(x64, y64, k=3))
    state.items_processed = state.iterations * num_points


@benchmark.register(name="InferRTPybind11/splprep_s0")
@benchmark.option.use_real_time()
@benchmark.option.arg_name("N")
@benchmark.option.arg(CURVE_SIZES[0])
@benchmark.option.arg(CURVE_SIZES[1])
@benchmark.option.arg(CURVE_SIZES[2])
@benchmark.option.unit(benchmark.kMillisecond)
def bench_inferrt_splprep_s0(state: benchmark.State) -> None:
    num_points = state.range(0)
    points, _ = CURVE_DATA[num_points]
    while state:
        _store(ops.splprep(points, smoothing=0.0, degree=3))
    state.items_processed = state.iterations * num_points


@benchmark.register(name="Python/scipy.splprep_s0")
@benchmark.option.use_real_time()
@benchmark.option.arg_name("N")
@benchmark.option.arg(CURVE_SIZES[0])
@benchmark.option.arg(CURVE_SIZES[1])
@benchmark.option.arg(CURVE_SIZES[2])
@benchmark.option.unit(benchmark.kMillisecond)
def bench_scipy_splprep_s0(state: benchmark.State) -> None:
    num_points = state.range(0)
    _, scipy_points = CURVE_DATA[num_points]
    while state:
        _store(interpolate.splprep(scipy_points, s=0.0, k=3))
    state.items_processed = state.iterations * num_points


@benchmark.register(name="InferRTPybind11/splprep_smooth")
@benchmark.option.use_real_time()
@benchmark.option.arg_name("N")
@benchmark.option.arg(CURVE_SIZES[0])
@benchmark.option.arg(CURVE_SIZES[1])
@benchmark.option.arg(CURVE_SIZES[2])
@benchmark.option.unit(benchmark.kMillisecond)
def bench_inferrt_splprep_smooth(state: benchmark.State) -> None:
    num_points = state.range(0)
    points, _ = CURVE_DATA[num_points]
    smoothing = num_points * 1.0e-5
    while state:
        _store(ops.splprep(points, smoothing=smoothing, degree=3))
    state.items_processed = state.iterations * num_points


@benchmark.register(name="Python/scipy.splprep_smooth")
@benchmark.option.use_real_time()
@benchmark.option.arg_name("N")
@benchmark.option.arg(CURVE_SIZES[0])
@benchmark.option.arg(CURVE_SIZES[1])
@benchmark.option.arg(CURVE_SIZES[2])
@benchmark.option.unit(benchmark.kMillisecond)
def bench_scipy_splprep_smooth(state: benchmark.State) -> None:
    num_points = state.range(0)
    _, scipy_points = CURVE_DATA[num_points]
    smoothing = num_points * 1.0e-5
    while state:
        _store(interpolate.splprep(scipy_points, s=smoothing, k=3))
    state.items_processed = state.iterations * num_points


if __name__ == "__main__":
    benchmark.add_custom_context("build_dir", str(ARGS.build_dir.resolve()))
    benchmark.add_custom_context("python", sys.version.split()[0])
    benchmark.add_custom_context("numpy", np.__version__)
    benchmark.main(BENCHMARK_ARGV)
