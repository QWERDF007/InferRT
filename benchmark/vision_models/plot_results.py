"""Plot the collected vision benchmark results as PNG figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from collect_results import (
    BATCH_ORDER,
    MODEL_LABELS,
    MODEL_NAMES,
    PRECISION_ORDER,
    RUNTIME_ORDER,
    _all_runtimes,
    _base_for,
    _result_map,
)


COLORS = plt.get_cmap("tab10").colors
MARKERS = ("o", "s", "^", "D", "P", "X")


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.axisbelow": True,
        }
    )


def _plot_n100(results, output: Path, runtimes: list[str]) -> None:
    figure, axes = plt.subplots(
        1,
        len(runtimes),
        figsize=(7.0 * len(runtimes), 5.8),
        squeeze=False,
    )
    for column, runtime in enumerate(runtimes):
        axis = axes[0][column]
        for model_index, model in enumerate(MODEL_NAMES):
            for precision in PRECISION_ORDER:
                points = [results.get((runtime, model, precision, batch)) for batch in BATCH_ORDER]
                if not all(points):
                    continue
                axis.plot(
                    BATCH_ORDER,
                    [point.total_100_ms for point in points],
                    color=COLORS[model_index],
                    marker=MARKERS[model_index],
                    linestyle="-" if precision == "fp32" else "--",
                    linewidth=1.8,
                    markersize=5,
                )
        axis.set_title(runtime)
        axis.set_xlabel("Batch")
        axis.set_ylabel("N=100 total latency (ms)")
        axis.set_xticks(BATCH_ORDER)
        axis.set_yscale("log")
        axis.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
    model_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[index],
            marker=MARKERS[index],
            linestyle="-",
            linewidth=1.8,
            markersize=5,
            label=MODEL_LABELS[model],
        )
        for index, model in enumerate(MODEL_NAMES)
    ]
    precision_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=1.8, label="FP32"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="FP16"),
    ]
    figure.suptitle("Vision model throughput: N=100", fontsize=16, y=0.995)
    figure.legend(
        handles=model_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.035),
        frameon=False,
    )
    figure.legend(
        handles=precision_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.095),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.15, 1, 0.96))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_n1(results, output: Path, runtimes: list[str]) -> None:
    figure, axes = plt.subplots(
        1,
        len(runtimes),
        figsize=(7.0 * len(runtimes), 5.8),
        squeeze=False,
    )
    width = 0.36
    x = np.arange(len(MODEL_NAMES))
    for column, runtime in enumerate(runtimes):
        axis = axes[0][column]
        fp32_values = []
        fp16_values = []
        for model in MODEL_NAMES:
            fp32 = results.get((runtime, model, "fp32", 1))
            fp16 = results.get((runtime, model, "fp16", 1))
            fp32_values.append(fp32.iteration_ms if fp32 else np.nan)
            fp16_values.append(fp16.iteration_ms if fp16 else np.nan)
        axis.bar(x - width / 2, fp32_values, width, label="FP32", color=COLORS[0], alpha=0.9)
        axis.bar(x + width / 2, fp16_values, width, label="FP16", color=COLORS[1], alpha=0.9)
        axis.set_title(runtime)
        axis.set_ylabel("N=1 latency (ms)")
        axis.set_xticks(x)
        axis.set_xticklabels([MODEL_LABELS[model].replace(" ", "\n", 1) for model in MODEL_NAMES], rotation=0)
        axis.set_yscale("log")
        axis.grid(True, axis="y", which="major", linestyle="--", linewidth=0.6, alpha=0.45)
        axis.legend(frameon=False)
    figure.suptitle("Vision model precision comparison: N=1", fontsize=16, y=0.995)
    figure.tight_layout(rect=(0, 0.02, 1, 0.95))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-json", type=Path, nargs="+", required=True)
    parser.add_argument("--python-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("plots"))
    args = parser.parse_args()

    _configure_style()
    results = _result_map(args.cpp_json, args.python_json)
    runtimes = _all_runtimes(results)
    if not runtimes:
        raise RuntimeError("No benchmark results found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_n100(results, args.output_dir / "n100_batch_latency.png", runtimes)
    _plot_n1(results, args.output_dir / "n1_precision_latency.png", runtimes)
    print(f"wrote plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
