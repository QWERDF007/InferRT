from __future__ import annotations

import argparse
from pathlib import Path
import time

from yolo_model_zoo import list_supported_models, load_ultralytics_model, resolve_model_weights, write_wts


def elapsed_ms(start: float, end: float) -> float:
    """将 ``perf_counter`` 时间差转换为毫秒。"""

    return (end - start) * 1000.0


def parse_args() -> argparse.Namespace:
    """解析 YOLO ``.wts`` 导出命令行参数。"""

    parser = argparse.ArgumentParser(description="Generate YOLOv5/YOLOv8 detection or segmentation weights for InferRT")
    parser.add_argument("-m", "--model", default="yolov8n", choices=list_supported_models(), help="InferRT YOLO key")
    parser.add_argument("-w", "--weights", default=None, help="Ultralytics .pt weights path or model name")
    parser.add_argument("-o", "--output", default=None, help="Output .wts path")
    parser.add_argument(
        "--ultralytics-repo",
        default=None,
        help="Optional local ultralytics repository path, e.g. D:/Github/ultralytics",
    )
    parser.add_argument(
        "--yolov5-repo",
        default=None,
        help="Optional local YOLOv5 repository path for legacy yolov5*.pt checkpoints",
    )
    parser.add_argument("-l", "--list-model", action="store_true", help="List supported YOLO model keys")
    parser.add_argument("--quiet", action="store_true", help="Do not print every exported weight key")
    return parser.parse_args()


def main() -> None:
    """加载 Ultralytics YOLO 权重并写出 InferRT ``.wts``。"""

    args = parse_args()
    if args.list_model:
        models = list_supported_models()
        print(f"YOLO models: {len(models)}")
        print(models)
        return

    source = resolve_model_weights(args.model, args.weights)
    output_path = Path(args.output) if args.output else Path(args.model).with_suffix(".wts")

    print(f"Loading Ultralytics model: {source}")
    load_start = time.perf_counter()
    model = load_ultralytics_model(
        args.model,
        weights=args.weights,
        repo=args.ultralytics_repo,
        yolov5_repo=args.yolov5_repo,
    )
    load_end = time.perf_counter()

    print(f"Writing weights: {output_path}")
    export_start = time.perf_counter()
    write_wts(model, output_path, verbose=not args.quiet)
    export_end = time.perf_counter()

    print(
        "Timing: "
        f"load={elapsed_ms(load_start, load_end):.3f} ms, "
        f"export={elapsed_ms(export_start, export_end):.3f} ms"
    )


if __name__ == "__main__":
    main()
