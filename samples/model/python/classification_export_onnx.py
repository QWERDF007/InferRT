from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import sys
import warnings

import torch

from classification_model_zoo import create_model, list_supported_models, parse_image_size, resolve_input_size


MODEL_BACKENDS = ("torchvision", "timm")


def list_supported_classification_models(backend: str) -> list[str]:
    models = list_supported_models(backend)
    if backend == "timm":
        return [name for name in models if "dino" not in name.lower()]
    return models


def configure_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            continue
        except (AttributeError, ValueError):
            pass

        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            setattr(sys, stream_name, io.TextIOWrapper(buffer, encoding="utf-8", errors="replace"))


def parse_cli_image_size(value: str) -> tuple[int, int]:
    try:
        return parse_image_size(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def resolve_export_image_size(model: torch.nn.Module, args: argparse.Namespace) -> tuple[tuple[int, int], str]:
    if args.input_size is not None:
        return args.input_size, "manual"

    if args.height or args.width:
        if args.height <= 0 or args.width <= 0:
            raise ValueError("--height and --width must both be positive when either one is specified")
        return (args.height, args.width), "manual"

    return resolve_input_size(model), "model"


def make_dummy_input(args: argparse.Namespace, image_size: tuple[int, int]) -> torch.Tensor:
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.channels <= 0:
        raise ValueError(f"--channels must be positive, got {args.channels}")

    height, width = image_size
    return torch.randn(args.batch_size, args.channels, height, width)


def export_with_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Path,
    args: argparse.Namespace,
    *,
    use_dynamo: bool,
) -> None:
    export_kwargs: dict[str, object] = {
        "export_params": True,
        "opset_version": args.opset,
        "do_constant_folding": True,
        "input_names": [args.input_name],
        "output_names": [args.output_name],
    }
    if args.dynamic_batch:
        export_kwargs["dynamic_axes"] = {
            args.input_name: {0: "batch"},
            args.output_name: {0: "batch"},
        }
    if use_dynamo:
        export_kwargs["dynamo"] = True

    torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)


def main(args: argparse.Namespace) -> None:
    configure_stdio()

    if args.list_model:
        models = list_supported_classification_models(args.backend)
        print(f"{args.backend} models:", len(models))
        print(models)
        return

    model = create_model(
        args.model,
        args.backend,
        pretrained=args.pretrained,
    )
    model.eval()

    image_size, size_source = resolve_export_image_size(model, args)
    dummy_input = make_dummy_input(args, image_size)
    output_path = Path(args.output) if args.output else Path(f"{args.model}.onnx")
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Exporting model '{args.model}' to {output_path} "
        f"with input shape {tuple(dummy_input.shape)} ({size_source}) ..."
    )

    if args.exporter == "legacy":
        export_with_onnx(model, dummy_input, output_path, args, use_dynamo=False)
        print(f"ONNX file '{output_path}' exported successfully with legacy exporter.")
        return

    if args.exporter == "dynamo":
        export_with_onnx(model, dummy_input, output_path, args, use_dynamo=True)
        print(f"ONNX file '{output_path}' exported successfully with dynamo exporter.")
        return

    try:
        export_with_onnx(model, dummy_input, output_path, args, use_dynamo=True)
        print(f"ONNX file '{output_path}' exported successfully with dynamo exporter.")
    except Exception as exc:
        warnings.warn(
            f"Falling back to legacy ONNX exporter because dynamo export failed: {exc}",
            RuntimeWarning,
            stacklevel=1,
        )
        export_with_onnx(model, dummy_input, output_path, args, use_dynamo=False)
        print(f"ONNX file '{output_path}' exported successfully with legacy exporter.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export classification models to ONNX")
    parser.add_argument("-m", "--model", type=str, default="alexnet", help="Model name")
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=MODEL_BACKENDS,
        default="torchvision",
        help="Model provider backend",
    )
    parser.add_argument("-o", "--output", type=str, default="", help="Output ONNX file path")
    parser.add_argument("--input-name", type=str, default="input", help="Input tensor name")
    parser.add_argument("--output-name", type=str, default="output", help="Output tensor name")
    parser.add_argument("--batch-size", type=int, default=1, help="Input batch size")
    parser.add_argument("--channels", type=int, default=3, help="Input channels")
    parser.add_argument(
        "--input-size",
        "--image-size",
        dest="input_size",
        type=parse_cli_image_size,
        default=None,
        metavar="SIZE",
        help="Override input image size, e.g. 384, 384x384, 3x384x384, or 1x3x384x384",
    )
    parser.add_argument("--height", type=int, default=0, help="Deprecated alias for input height")
    parser.add_argument("--width", type=int, default=0, help="Deprecated alias for input width")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--exporter",
        type=str,
        choices=("auto", "dynamo", "legacy"),
        default="auto",
        help="Choose ONNX exporter backend",
    )
    parser.add_argument("--dynamic-batch", action="store_true", help="Export dynamic batch axes")
    parser.add_argument("-l", "--list-model", "--list_model", dest="list_model", action="store_true")
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Create model without pretrained weights, useful for offline structural checks",
    )
    parser.set_defaults(pretrained=True)
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
