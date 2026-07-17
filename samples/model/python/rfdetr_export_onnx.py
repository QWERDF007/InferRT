"""通过 RF-DETR 官方导出接口生成 ONNX。"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from rfdetr_gen_wts import (
    DEFAULT_MODEL_DIR,
    DEFAULT_RFDETR_ROOT,
    MODEL_CHECKPOINTS,
    infer_checkpoint_num_classes,
    load_rfdetr_wrapper,
    resolve_checkpoint,
)


def parse_shape(value: str) -> tuple[int, int]:
    tokens = [token for token in value.lower().replace(",", "x").split("x") if token]
    if len(tokens) == 1:
        height = width = int(tokens[0])
    elif len(tokens) == 2:
        height, width = (int(token) for token in tokens)
    else:
        raise ValueError(f"invalid RF-DETR shape: {value!r}")
    if height <= 0 or width <= 0:
        raise ValueError("RF-DETR shape must be positive")
    return height, width


def parse_cli_shape(value: str) -> tuple[int, int]:
    try:
        return parse_shape(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def export_onnx(
    model_name: str,
    *,
    checkpoint: str | Path | None = None,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    rfdetr_root: str | Path | None = DEFAULT_RFDETR_ROOT,
    output: str | Path | None = None,
    device: str = "cpu",
    num_classes: int | None = None,
    shape: tuple[int, int] | None = None,
    batch_size: int = 1,
    dynamic_batch: bool = False,
    opset: int = 17,
    backbone_only: bool = False,
    verbose: bool = True,
) -> Path:
    """导出 RF-DETR ONNX，并可复制为用户指定文件名。"""

    checkpoint_path = resolve_checkpoint(
        model_name,
        Path(checkpoint) if checkpoint is not None else None,
        Path(model_dir),
    )
    classes = num_classes if num_classes is not None else infer_checkpoint_num_classes(checkpoint_path)
    wrapper = load_rfdetr_wrapper(
        checkpoint_path,
        model_name=model_name,
        rfdetr_root=Path(rfdetr_root) if rfdetr_root is not None else None,
        device=device,
        num_classes=classes,
    )

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    output_path = Path(output) if output is not None else None
    output_dir = output_path.parent if output_path is not None else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = Path(
        wrapper.export(
            output_dir=str(output_dir),
            backbone_only=backbone_only,
            opset_version=opset,
            shape=shape,
            batch_size=batch_size,
            dynamic_batch=dynamic_batch,
            verbose=verbose,
        )
    )
    if output_path is None:
        return exported
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != output_path.resolve():
        shutil.copy2(exported, output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export RF-DETR models to ONNX")
    parser.add_argument("-m", "--model", default="rfdetr_nano", choices=sorted(MODEL_CHECKPOINTS))
    parser.add_argument("-c", "--checkpoint", type=Path, default=None, help="RF-DETR checkpoint path")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--rfdetr-root", type=Path, default=DEFAULT_RFDETR_ROOT)
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .onnx file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--shape", type=parse_cli_shape, default=None, metavar="SIZE")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--backbone-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-l", "--list-model", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    if args.list_model:
        print(sorted(MODEL_CHECKPOINTS))
        return
    output = export_onnx(
        args.model,
        checkpoint=args.checkpoint,
        model_dir=args.model_dir,
        rfdetr_root=args.rfdetr_root,
        output=args.output,
        device=args.device,
        num_classes=args.num_classes,
        shape=args.shape,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
        opset=args.opset,
        backbone_only=args.backbone_only,
        verbose=not args.quiet,
    )
    print(f"Exported {args.model} ONNX to {output}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
