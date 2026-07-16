"""汇总视觉模型 benchmark JSON，并生成 benchmark/vision_models/README.md。

该脚本不重新运行 benchmark；它把 C++ 和 Python google-benchmark 的 JSON
统一成相同的表格。N=100 的总耗时按吞吐量换算，N=1 使用 batch=1 的单次
real_time。相对倍率始终在相同 runtime/precision 下以 DINOv2 ViT-S/14 为基准。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable


MODEL_NAMES = (
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov3_vits16",
    "dinov3_vitb16",
    "lingbot_vision_vits16",
    "lingbot_vision_vitb16",
)
MODEL_LABELS = {
    "dinov2_vits14": "DINOv2 ViT-S/14",
    "dinov2_vitb14": "DINOv2 ViT-B/14",
    "dinov3_vits16": "DINOv3 ViT-S/16",
    "dinov3_vitb16": "DINOv3 ViT-B/16",
    "lingbot_vision_vits16": "LingBot-Vision ViT-S/16",
    "lingbot_vision_vitb16": "LingBot-Vision ViT-B/16",
}
RUNTIME_ORDER = ("InferRT C++", "PyTorch Python", "InferRT Python")
PRECISION_ORDER = ("fp32", "fp16")
BATCH_ORDER = (1, 2, 4, 8)
BASE_MODEL = "dinov2_vits14"
BENCHMARK_RE = re.compile(r"^(?P<runtime>[^/]+)/(?P<model>[^/]+)/(?P<precision>fp32|fp16)/batch_(?P<batch>\d+)")


@dataclass(frozen=True)
class Result:
    runtime: str
    model: str
    precision: str
    batch: int
    iteration_ms: float
    items_per_second: float

    @property
    def total_100_ms(self) -> float:
        if self.items_per_second > 0:
            return 100_000.0 / self.items_per_second
        return self.iteration_ms * 100.0 / self.batch


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _time_in_ms(value: Any, unit: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    normalized = str(unit or "ms").lower()
    scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1000.0}.get(normalized)
    if scale is None:
        raise ValueError(f"Unsupported benchmark time unit: {unit}")
    return number * scale


def _runtime_label(runtime: str, source: str) -> str:
    if source == "cpp":
        return "InferRT C++"
    if runtime == "PyTorch":
        return "PyTorch Python"
    if runtime == "InferRT":
        return "InferRT Python"
    return f"{runtime} Python"


def _load_results(path: Path, source: str) -> list[Result]:
    document = _read_json(path)
    results: list[Result] = []
    for item in document.get("benchmarks", []):
        name = str(item.get("name", ""))
        match = BENCHMARK_RE.match(name)
        if match is None:
            continue
        model = match.group("model")
        if model not in MODEL_LABELS:
            continue
        # Ignore aggregate rows and keep the measured iteration row. The
        # collection command uses --benchmark_repetitions=1, but this also
        # behaves sensibly for JSON files containing aggregate entries.
        if item.get("run_type") not in (None, "iteration"):
            continue
        batch = int(match.group("batch"))
        iteration_ms = _time_in_ms(item.get("real_time"), item.get("time_unit"))
        if iteration_ms is None or iteration_ms <= 0:
            continue
        try:
            items_per_second = float(item.get("items_per_second", 0.0))
        except (TypeError, ValueError):
            items_per_second = 0.0
        results.append(
            Result(
                runtime=_runtime_label(match.group("runtime"), source),
                model=model,
                precision=match.group("precision"),
                batch=batch,
                iteration_ms=iteration_ms,
                items_per_second=items_per_second,
            )
        )
    return results


def _deduplicate(results: Iterable[Result]) -> dict[tuple[str, str, str, int], Result]:
    # Prefer the last row. This supports concatenated JSON collections while
    # still allowing a benchmark file with a single row per case.
    unique: dict[tuple[str, str, str, int], Result] = {}
    for result in results:
        unique[(result.runtime, result.model, result.precision, result.batch)] = result
    return unique


def _format(value: float | None, ratio: float | None) -> str:
    if value is None:
        return "—"
    if ratio is None:
        return f"{value:.2f} ms"
    return f"{value:.2f} ms ({ratio:.2f}x)"


def _ratio(result: Result, base: Result | None, n100: bool) -> float | None:
    if base is None:
        return None
    denominator = base.total_100_ms if n100 else base.iteration_ms
    numerator = result.total_100_ms if n100 else result.iteration_ms
    return numerator / denominator if denominator > 0 else None


def _result_map(cpp_json: Iterable[Path], python_json: Iterable[Path]) -> dict[tuple[str, str, str, int], Result]:
    results: list[Result] = []
    for path in cpp_json:
        results.extend(_load_results(path, "cpp"))
    for path in python_json:
        results.extend(_load_results(path, "python"))
    return _deduplicate(results)


def _base_for(results: dict[tuple[str, str, str, int], Result], runtime: str, precision: str, batch: int) -> Result | None:
    return results.get((runtime, BASE_MODEL, precision, batch))


def _all_runtimes(results: dict[tuple[str, str, str, int], Result]) -> list[str]:
    present = {key[0] for key in results}
    return [runtime for runtime in RUNTIME_ORDER if runtime in present]


def _n100_table(results: dict[tuple[str, str, str, int], Result]) -> str:
    lines = [
        "| Runtime | Precision | Model | batch=1 | batch=2 | batch=4 | batch=8 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for runtime in _all_runtimes(results):
        for precision in PRECISION_ORDER:
            for model in MODEL_NAMES:
                cells = []
                for batch in BATCH_ORDER:
                    result = results.get((runtime, model, precision, batch))
                    base = _base_for(results, runtime, precision, batch)
                    cells.append(_format(result.total_100_ms if result else None, _ratio(result, base, True) if result else None))
                if any(cell != "—" for cell in cells):
                    lines.append(f"| {runtime} | {precision.upper()} | {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _n1_table(results: dict[tuple[str, str, str, int], Result]) -> str:
    lines = [
        "| Runtime | Model | FP32（batch=1） | FP16（batch=1） |",
        "|---|---|---:|---:|",
    ]
    for runtime in _all_runtimes(results):
        for model in MODEL_NAMES:
            cells = []
            for precision in PRECISION_ORDER:
                result = results.get((runtime, model, precision, 1))
                base = _base_for(results, runtime, precision, 1)
                cells.append(_format(result.iteration_ms if result else None, _ratio(result, base, False) if result else None))
            if any(cell != "—" for cell in cells):
                lines.append(f"| {runtime} | {MODEL_LABELS[model]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _context_lines(paths: Iterable[Path], label: str) -> str:
    lines = []
    for path in paths:
        lines.append(_context_line(path, label))
    return "  \n".join(lines)


def _context_line(path: Path, label: str) -> str:
    try:
        context = _read_json(path).get("context", {})
    except (OSError, json.JSONDecodeError):
        return f"{label}: `{path}`"
    date = context.get("date")
    device = context.get("device")
    torch_version = context.get("torch")
    details = [f"{label} `{path}`"]
    if date:
        details.append(f"时间 {date}")
    if device:
        details.append(f"device `{device}`")
    if torch_version:
        details.append(f"PyTorch `{torch_version}`")
    return "，".join(details)


def _render(cpp_json: list[Path], python_json: list[Path], results: dict[tuple[str, str, str, int], Result]) -> str:
    return f"""# Vision model benchmark

本文记录 DINOv2、DINOv3 和 LingBot-Vision small/base 模型的 GPU 前向性能。C++
测试使用 InferRT CUDA event；Python 测试分别记录 PyTorch 和 InferRT Python
绑定的 GPU 常驻输入前向时间。

## 测试口径

- 输入图像数量为 N=100 时，表格单元格为顺序处理 100 张图的估算总耗时：`100 / items_per_second * 1000 ms`。
- N=1 时使用 `batch=1` 的单次前向 `real_time`。
- 每个单元格格式为 `耗时（相对倍率）`；相对倍率 = 当前耗时 / 同一 runtime、precision、batch 下的 DINOv2 ViT-S/14 耗时，DINOv2 ViT-S/14 固定为 `1.00x`。
- batch 测试：`1, 2, 4, 8`；精度测试：FP32、FP16；benchmark 参数：`--benchmark_min_time=0.2s --benchmark_repetitions=1`。
- C++ 和 Python 的 InferRT 结果分别列出，二者的计时边界不同，不应直接混作同一条曲线比较。

## 性能图

### N=100：batch 对总耗时的影响

![N=100 batch latency](plots/n100_batch_latency.png)

### N=1：FP32/FP16 延迟对比

![N=1 precision latency](plots/n1_precision_latency.png)

## N=100：不同 batch

{_n100_table(results)}

## N=1：不同精度

{_n1_table(results)}

## 原始结果

{_context_lines(cpp_json, "C++ JSON")}  
{_context_lines(python_json, "Python JSON")}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-json", type=Path, nargs="+", required=True)
    parser.add_argument("--python-json", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("README.md"),
    )
    args = parser.parse_args()
    results = _result_map(args.cpp_json, args.python_json)
    expected = len(RUNTIME_ORDER) * len(MODEL_NAMES) * len(PRECISION_ORDER) * len(BATCH_ORDER)
    if len(results) < expected:
        print(f"warning: collected {len(results)} of {expected} expected benchmark rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render(args.cpp_json, args.python_json, results), encoding="utf-8")
    print(f"wrote {args.output} ({len(results)} benchmark rows)")


if __name__ == "__main__":
    main()
