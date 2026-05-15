import argparse
import io
import os
import sys
import warnings

import torch
from torchvision.models import (
    AlexNet_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    VGG11_Weights,
    VGG13_Weights,
    VGG16_Weights,
    VGG19_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    alexnet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    vgg11,
    vgg13,
    vgg16,
    vgg19,
    wide_resnet50_2,
    wide_resnet101_2,
)


TORCHVISION_MODEL_ZOO = {
    "alexnet": (alexnet, AlexNet_Weights.IMAGENET1K_V1),
    "vgg11": (vgg11, VGG11_Weights.IMAGENET1K_V1),
    "vgg13": (vgg13, VGG13_Weights.IMAGENET1K_V1),
    "vgg16": (vgg16, VGG16_Weights.IMAGENET1K_V1),
    "vgg19": (vgg19, VGG19_Weights.IMAGENET1K_V1),
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2),
    "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V2),
    "wide_resnet50_2": (wide_resnet50_2, Wide_ResNet50_2_Weights.IMAGENET1K_V2),
    "wide_resnet101_2": (wide_resnet101_2, Wide_ResNet101_2_Weights.IMAGENET1K_V2),
}


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


def list_supported_models(backend: str) -> list[str]:
    if backend == "torchvision":
        return list(TORCHVISION_MODEL_ZOO.keys())
    if backend == "timm":
        from timm import list_models

        return list_models("resnet*") + list_models("wide_resnet*")
    raise ValueError(f"Unsupported backend: {backend}")


def create_model(model_name: str, backend: str) -> torch.nn.Module:
    if backend == "torchvision":
        if model_name not in TORCHVISION_MODEL_ZOO:
            available = ", ".join(TORCHVISION_MODEL_ZOO.keys())
            raise ValueError(f"Unsupported torchvision model: {model_name}. Available: {available}")

        model_fn, weights = TORCHVISION_MODEL_ZOO[model_name]
        return model_fn(weights=weights)

    if backend == "timm":
        from timm import create_model

        return create_model(model_name, pretrained=True)

    raise ValueError(f"Unsupported backend: {backend}")


def export_with_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    args: argparse.Namespace,
    use_dynamo: bool,
) -> None:
    export_kwargs = {
        "export_params": True,
        "opset_version": args.opset,
        "do_constant_folding": True,
        "input_names": [args.input_name],
        "output_names": [args.output_name],
    }
    if use_dynamo:
        export_kwargs["dynamo"] = True

    torch.onnx.export(model, dummy_input, output_path, **export_kwargs)


def main(args: argparse.Namespace) -> None:
    configure_stdio()

    if args.list_model:
        models = list_supported_models(args.backend)
        print(f"{args.backend} models:", len(models))
        print(models)
        return

    model = create_model(args.model, args.backend)
    model.eval()

    dummy_input = torch.randn(1, args.channels, args.height, args.width)
    output_path = args.output if args.output else f"{args.model}.onnx"

    print(f"Exporting model '{args.model}' to {output_path} ...")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export classification models to ONNX")
    parser.add_argument("-m", "--model", type=str, default="alexnet", help="Model name")
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=("timm", "torchvision"),
        default="torchvision",
        help="Model provider backend",
    )
    parser.add_argument("-o", "--output", type=str, default="", help="Output ONNX file path")
    parser.add_argument("--input-name", type=str, default="input", help="Input tensor name")
    parser.add_argument("--output-name", type=str, default="output", help="Output tensor name")
    parser.add_argument("--channels", type=int, default=3, help="Input channels")
    parser.add_argument("--height", type=int, default=224, help="Input height")
    parser.add_argument("--width", type=int, default=224, help="Input width")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--exporter",
        type=str,
        choices=("auto", "dynamo", "legacy"),
        default="auto",
        help="Choose ONNX exporter backend",
    )
    parser.add_argument("-l", "--list-model", action="store_true", help="List supported models")
    main(parser.parse_args())
