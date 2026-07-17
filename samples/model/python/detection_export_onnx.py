"""通过 Ultralytics 将 YOLO 检测模型导出为 ONNX。"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from yolo_model_zoo import (
    add_ultralytics_repo,
    ensure_ultralytics_config_dir,
    list_supported_models,
    resolve_model_weights,
)


def parse_image_size(value: str) -> int | tuple[int, int]:
    """解析 YOLO 导出尺寸。"""

    tokens = [token for token in value.lower().replace(",", "x").split("x") if token]
    if len(tokens) == 1:
        size = int(tokens[0])
        if size <= 0:
            raise ValueError("image size must be positive")
        return size
    if len(tokens) == 2:
        height, width = (int(token) for token in tokens)
        if height <= 0 or width <= 0:
            raise ValueError("image size must be positive")
        return height, width
    raise ValueError(f"invalid image size: {value!r}")


def parse_cli_image_size(value: str) -> int | tuple[int, int]:
    try:
        return parse_image_size(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def export_onnx(
    model_name: str,
    *,
    weights: str | Path | None = None,
    output: str | Path | None = None,
    ultralytics_repo: str | Path | None = None,
    image_size: int | tuple[int, int] | None = None,
    batch_size: int = 1,
    dynamic_batch: bool = False,
    opset: int | None = 17,
    half: bool = False,
) -> Path:
    """导出 YOLO ONNX，并将结果复制到用户指定路径。"""

    source = resolve_model_weights(model_name, weights)
    ensure_ultralytics_config_dir()
    add_ultralytics_repo(ultralytics_repo)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLO ONNX export requires Ultralytics. Install it or pass --ultralytics-repo."
        ) from exc

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    export_kwargs: dict[str, object] = {
        "format": "onnx",
        "batch": batch_size,
        "dynamic": dynamic_batch,
        "half": half,
    }
    if image_size is not None:
        export_kwargs["imgsz"] = image_size
    if opset is not None:
        export_kwargs["opset"] = opset

    exported = Path(YOLO(source).export(**export_kwargs))
    output_path = Path(output) if output is not None else exported
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output_path.resolve():
        shutil.copy2(exported, output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export YOLO detection models to ONNX")
    parser.add_argument("-m", "--model", default="yolov8n", choices=list_supported_models())
    parser.add_argument("-w", "--weights", default=None, help="Ultralytics .pt checkpoint or model name")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .onnx file")
    parser.add_argument("--ultralytics-repo", default=None, help="Local Ultralytics repository root")
    parser.add_argument("--input-size", type=parse_cli_image_size, default=None, metavar="SIZE")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch", action="store_true", help="Export a dynamic batch dimension")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--half", action="store_true", help="Export FP16 ONNX weights")
    parser.add_argument("-l", "--list-model", action="store_true", help="List supported YOLO model keys")
    return parser


def main(args: argparse.Namespace) -> None:
    if args.list_model:
        print(list_supported_models())
        return
    output = export_onnx(
        args.model,
        weights=args.weights,
        output=args.output,
        ultralytics_repo=args.ultralytics_repo,
        image_size=args.input_size,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
        opset=args.opset,
        half=args.half,
    )
    print(f"Exported {args.model} ONNX to {output}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
