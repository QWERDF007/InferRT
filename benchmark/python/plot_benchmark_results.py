"""Plot google-benchmark JSON results produced by the Python benchmarks.

Example:
    D:\\Software\\anaconda3\\envs\\py312\\python.exe benchmark\\python\\benchmark_clustering.py ^
        --benchmark_min_time=0.05s ^
        --benchmark_format=json ^
        --benchmark_out=benchmark\\python\\results\\clustering.json ^
        --benchmark_out_format=json

    D:\\Software\\anaconda3\\envs\\py312\\python.exe benchmark\\python\\benchmark_curve.py ^
        --benchmark_min_time=0.05s ^
        --benchmark_format=json ^
        --benchmark_out=benchmark\\python\\results\\curve.json ^
        --benchmark_out_format=json

    D:\\Software\\anaconda3\\envs\\py312\\python.exe benchmark\\python\\plot_benchmark_results.py ^
        benchmark\\python\\results\\clustering.json benchmark\\python\\results\\curve.json ^
        --output-dir benchmark\\python\\plots
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmark" / "python" / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark" / "python" / "plots"
TIME_FIELD_CHOICES = ("real_time", "cpu_time")
TIME_UNIT_TO_MS = {
    "ns": 1.0e-6,
    "us": 1.0e-3,
    "ms": 1.0,
    "s": 1.0e3,
}
AGGREGATE_SUFFIX_RE = re.compile(r"_(mean|median|stddev|cv)$")
SIZE_RE = re.compile(r"(?:^|/)N:(\d+)(?:/|$)")
DIM_RE = re.compile(r"(?:^|/)D:(\d+)(?:/|$)")
CURVE_SUITE_BY_REFERENCE = {
    "numpy_fit_bezier_curve": "fit_bezier_curve",
    "scipy.BPoly_evaluate": "evaluate_bezier_curve",
    "scipy.make_interp_spline": "make_interp_spline",
    "scipy.BSpline_evaluate": "evaluate_b_spline",
    "scipy.splprep_s0": "splprep_s0",
    "scipy.splprep_smooth": "splprep_smooth",
}


@dataclass(frozen=True)
class ParsedName:
    suite: str
    label: str
    size: int
    dimension: int | None = None


@dataclass(frozen=True)
class BenchmarkRecord:
    suite: str
    label: str
    size: int
    dimension: int | None
    time_ms: float
    run_type: str
    aggregate_name: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot InferRT Python benchmark JSON results.")
    parser.add_argument(
        "json_files",
        nargs="*",
        type=Path,
        help=f"google-benchmark JSON files. Defaults to {DEFAULT_RESULTS_DIR}\\*.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saved figures. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--time-field",
        choices=TIME_FIELD_CHOICES,
        default="real_time",
        help="Benchmark time field to plot.",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "svg", "pdf"),
        default="png",
        help="Saved figure format.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="DPI used for raster images.")
    parser.add_argument("--linear-x", action="store_true", help="Use a linear X axis instead of the default log scale.")
    parser.add_argument("--log-y", action="store_true", help="Use a logarithmic Y axis.")
    return parser.parse_args()


def _discover_json_files(paths: list[Path]) -> list[Path]:
    if paths:
        return [path.resolve() for path in paths]

    return sorted(DEFAULT_RESULTS_DIR.glob("*.json"))


def _implementation_label(raw_impl: str, raw_op: str) -> str:
    if raw_impl == "InferRTPybind11":
        return "InferRT pybind11"
    if raw_impl == "Python" and raw_op.startswith("sklearn."):
        return "scikit-learn"
    if raw_impl == "Python" and raw_op.startswith("scipy."):
        return "SciPy"
    if raw_impl == "Python" and raw_op.startswith("numpy_"):
        return "NumPy"
    return raw_impl


def _strip_aggregate_suffix(name: str) -> str:
    return AGGREGATE_SUFFIX_RE.sub("", name)


def _parse_benchmark_name(name: str) -> ParsedName | None:
    clean_name = _strip_aggregate_suffix(name)
    size_match = SIZE_RE.search(clean_name)
    if not size_match:
        return None

    size = int(size_match.group(1))
    dim_match = DIM_RE.search(clean_name)
    dimension = int(dim_match.group(1)) if dim_match else None
    prefix = clean_name[: size_match.start()].rstrip("/")
    parts = prefix.split("/")
    if len(parts) < 2:
        return None

    raw_impl = parts[0]
    raw_op = parts[1]
    impl = _implementation_label(raw_impl, raw_op)

    if raw_op in {"DBSCAN", "HDBSCAN"} and len(parts) >= 3:
        label = f"{impl}/{parts[2]}"
        if len(parts) >= 4:
            label = f"{label}/{parts[3]}"
        return ParsedName(suite=raw_op, label=label, size=size, dimension=dimension)

    if raw_op in {"sklearn.DBSCAN", "sklearn.HDBSCAN"} and len(parts) >= 3:
        suite = raw_op.split(".", 1)[1]
        label = f"{impl}/{parts[2]}"
        if len(parts) >= 4:
            label = f"{label}/{parts[3]}"
        return ParsedName(suite=suite, label=label, size=size, dimension=dimension)

    suite = CURVE_SUITE_BY_REFERENCE.get(raw_op, raw_op.removeprefix("scipy."))
    return ParsedName(suite=suite, label=impl, size=size)


def _read_records(json_files: list[Path], time_field: str) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        for item in payload.get("benchmarks", []):
            parsed = _parse_benchmark_name(str(item.get("name", "")))
            if parsed is None:
                continue

            raw_time = item.get(time_field)
            if raw_time is None:
                continue

            unit = str(item.get("time_unit", "ms"))
            scale = TIME_UNIT_TO_MS.get(unit)
            if scale is None:
                raise ValueError(f"Unsupported time unit '{unit}' in {json_file}")

            time_ms = float(raw_time) * scale
            if not math.isfinite(time_ms):
                continue

            records.append(
                BenchmarkRecord(
                    suite=parsed.suite,
                    label=parsed.label,
                    size=parsed.size,
                    dimension=parsed.dimension,
                    time_ms=time_ms,
                    run_type=str(item.get("run_type", "iteration")),
                    aggregate_name=str(item.get("aggregate_name", "")),
                )
            )

    return records


def _summarize_records(records: list[BenchmarkRecord]) -> dict[tuple[str, int | None], dict[str, dict[int, float]]]:
    grouped: dict[tuple[str, int | None, str, int], list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.suite, record.dimension, record.label, record.size)].append(record)

    summary: dict[tuple[str, int | None], dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for (suite, dimension, label, size), values in grouped.items():
        mean_aggregates = [record.time_ms for record in values if record.aggregate_name == "mean"]
        iteration_values = [record.time_ms for record in values if record.run_type == "iteration"]
        selected = mean_aggregates or iteration_values or [record.time_ms for record in values]
        summary[(suite, dimension)][label][size] = mean(selected)

    return summary


def _label_sort_key(label: str) -> tuple[int, str]:
    if label.startswith("InferRT"):
        return (0, label)
    return (1, label)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return slug or "benchmark"


def _plot_suite(
    suite: str,
    dimension: int | None,
    series_by_label: dict[str, dict[int, float]],
    output_dir: Path,
    image_format: str,
    dpi: int,
    log_x: bool,
    log_y: bool,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    all_sizes = sorted({size for series in series_by_label.values() for size in series})

    for label in sorted(series_by_label, key=_label_sort_key):
        series = series_by_label[label]
        sizes = [size for size in all_sizes if size in series]
        values = [series[size] for size in sizes]
        ax.plot(sizes, values, marker="o", linewidth=2.0, markersize=5.5, label=label)

    title = suite if dimension is None else f"{suite} D={dimension}"
    ax.set_title(title)
    ax.set_xlabel("N")
    ax.set_ylabel("Time (ms)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
    if log_y:
        ax.set_yscale("log")
    if all_sizes:
        ax.set_xticks(all_sizes)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if dimension is None else f"_D{dimension}"
    output_path = output_dir / f"{_slugify(suite + suffix)}.{image_format}"
    fig.savefig(output_path, dpi=dpi if image_format == "png" else None)
    plt.close(fig)
    return output_path


def main() -> int:
    args = _parse_args()
    json_files = _discover_json_files(args.json_files)
    if not json_files:
        raise FileNotFoundError(
            f"No benchmark JSON files were provided and none were found in {DEFAULT_RESULTS_DIR}."
        )

    records = _read_records(json_files, args.time_field)
    if not records:
        raise RuntimeError("No plottable benchmark records were found.")

    summary = _summarize_records(records)
    output_paths = []
    for suite, dimension in sorted(summary, key=lambda key: (key[0], -1 if key[1] is None else key[1])):
        output_paths.append(
            _plot_suite(
                suite=suite,
                dimension=dimension,
                series_by_label=summary[(suite, dimension)],
                output_dir=args.output_dir,
                image_format=args.image_format,
                dpi=args.dpi,
                log_x=not args.linear_x,
                log_y=args.log_y,
            )
        )

    for output_path in output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
