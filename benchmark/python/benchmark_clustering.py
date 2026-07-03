"""DBSCAN/HDBSCAN pybind11 与 scikit-learn 的 google-benchmark 基准测试。

每个聚类算子都会分别测试 brute、kd_tree、ball_tree 三种邻域搜索算法。

运行示例：
    D:\\Software\\anaconda3\\envs\\py312\\python.exe benchmark\\python\\benchmark_clustering.py --benchmark_min_time=0.05s

可选参数：
    --build-dir <path>  指定 CMake 构建目录，默认使用 INFERRT_BUILD_DIR 或 <repo>/build。

其余 ``--benchmark_*`` 参数会原样传给 google-benchmark。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import google_benchmark as benchmark
import numpy as np
from sklearn import cluster as sklearn_cluster


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_DIR = Path(os.environ.get("INFERRT_BUILD_DIR", REPO_ROOT / "build")).resolve()
CLUSTER_SIZES = (64, 128, 256, 512)

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


def _cluster_samples(num_samples: int) -> np.ndarray:
    rng = np.random.default_rng(12345 + num_samples)
    centers = np.asarray(
        [
            [-1.5, -1.5, 0.0],
            [1.5, -1.5, 0.5],
            [-1.5, 1.5, -0.5],
            [1.5, 1.5, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.arange(num_samples, dtype=np.int64) % len(centers)
    samples = centers[labels] + rng.normal(0.0, 0.16, size=(num_samples, centers.shape[1])).astype(np.float32)
    rng.shuffle(samples, axis=0)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _dbscan_config(algorithm: Any) -> Any:
    config = ops.DBSCANConfig()
    config.eps = 0.45
    config.min_samples = 5
    config.algorithm = algorithm
    config.leaf_size = 30
    config.metric = ops.ClusteringMetric.Euclidean
    return config


def _hdbscan_config(algorithm: Any) -> Any:
    config = ops.HDBSCANConfig()
    config.min_cluster_size = 10
    config.min_samples = 5
    config.algorithm = algorithm
    config.leaf_size = 30
    config.metric = ops.ClusteringMetric.Euclidean
    config.cluster_selection_method = ops.HDBSCANClusterSelectionMethod.Eom
    return config


ALGORITHM_CASES = (
    ("brute", ops.ClusteringAlgorithm.Brute, "brute"),
    ("kd_tree", ops.ClusteringAlgorithm.KDTree, "kd_tree"),
    ("ball_tree", ops.ClusteringAlgorithm.BallTree, "ball_tree"),
)
CLUSTER_DATA = {size: _cluster_samples(size) for size in CLUSTER_SIZES}


def _register_cluster_benchmark(name: str, func: Any) -> None:
    options = func
    for size in reversed(CLUSTER_SIZES):
        options = benchmark.option.arg(size)(options)
    options = benchmark.option.arg_name("N")(options)
    options = benchmark.option.unit(benchmark.kMillisecond)(options)
    options = benchmark.option.use_real_time()(options)
    benchmark.register(options, name=name)


def _register_dbscan_benchmarks() -> None:
    for algorithm_name, irt_algorithm, sklearn_algorithm in ALGORITHM_CASES:
        config = _dbscan_config(irt_algorithm)

        def bench_inferrt_dbscan(state: benchmark.State, *, config: Any = config) -> None:
            num_samples = state.range(0)
            samples = CLUSTER_DATA[num_samples]
            while state:
                _store(ops.dbscan(samples, config))
            state.items_processed = state.iterations * num_samples

        _register_cluster_benchmark(f"InferRTPybind11/DBSCAN/{algorithm_name}", bench_inferrt_dbscan)

        def bench_sklearn_dbscan(
            state: benchmark.State,
            *,
            config: Any = config,
            sklearn_algorithm: str = sklearn_algorithm,
        ) -> None:
            num_samples = state.range(0)
            samples = CLUSTER_DATA[num_samples]
            while state:
                _store(
                    sklearn_cluster.DBSCAN(
                        eps=config.eps,
                        min_samples=config.min_samples,
                        algorithm=sklearn_algorithm,
                        leaf_size=config.leaf_size,
                        metric="euclidean",
                    ).fit_predict(samples)
                )
            state.items_processed = state.iterations * num_samples

        _register_cluster_benchmark(f"Python/sklearn.DBSCAN/{algorithm_name}", bench_sklearn_dbscan)


def _register_hdbscan_benchmarks() -> None:
    for algorithm_name, irt_algorithm, sklearn_algorithm in ALGORITHM_CASES:
        config = _hdbscan_config(irt_algorithm)

        def bench_inferrt_hdbscan(state: benchmark.State, *, config: Any = config) -> None:
            num_samples = state.range(0)
            samples = CLUSTER_DATA[num_samples]
            while state:
                _store(ops.hdbscan(samples, config))
            state.items_processed = state.iterations * num_samples

        _register_cluster_benchmark(f"InferRTPybind11/HDBSCAN/{algorithm_name}", bench_inferrt_hdbscan)

        def bench_sklearn_hdbscan(
            state: benchmark.State,
            *,
            config: Any = config,
            sklearn_algorithm: str = sklearn_algorithm,
        ) -> None:
            num_samples = state.range(0)
            samples = CLUSTER_DATA[num_samples]
            while state:
                _store(
                    sklearn_cluster.HDBSCAN(
                        min_cluster_size=config.min_cluster_size,
                        min_samples=config.min_samples,
                        algorithm=sklearn_algorithm,
                        leaf_size=config.leaf_size,
                        metric="euclidean",
                        cluster_selection_method="eom",
                        copy=False,
                    ).fit_predict(samples)
                )
            state.items_processed = state.iterations * num_samples

        _register_cluster_benchmark(f"Python/sklearn.HDBSCAN/{algorithm_name}", bench_sklearn_hdbscan)


_register_dbscan_benchmarks()
_register_hdbscan_benchmarks()


if __name__ == "__main__":
    benchmark.add_custom_context("build_dir", str(ARGS.build_dir.resolve()))
    benchmark.add_custom_context("python", sys.version.split()[0])
    benchmark.add_custom_context("numpy", np.__version__)
    benchmark.main(BENCHMARK_ARGV)
