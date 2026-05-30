from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import re
import sys
import warnings

import torch

from export_onnx import (
    MODEL_BACKENDS,
    configure_stdio,
    make_dummy_input,
    parse_cli_image_size,
    resolve_export_image_size,
)

from classification_model_zoo import create_model, list_supported_models


DEFAULT_OPENVINO_ROOT = os.environ.get("OPENVINO_ROOT", "D:/Software/openvino_toolkit")
_OPENVINO_DLL_HANDLES: list[object] = []


def parse_feature_names(value: str) -> list[str]:
    features = [item.strip() for item in value.split(",") if item.strip()]
    if not features:
        raise argparse.ArgumentTypeError("feature list must not be empty")

    duplicates = sorted({name for name in features if features.count(name) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(f"duplicate feature names: {', '.join(duplicates)}")
    return features


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_") or "features"


def default_onnx_output_path(model_name: str, feature_names: list[str]) -> Path:
    feature_suffix = "_".join(_safe_name(name) for name in feature_names)
    return Path(f"{_safe_name(model_name)}_features_{feature_suffix}.onnx")


class FeatureOutputWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, feature_names: list[str]):
        super().__init__()
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        self.model = model
        self.feature_names = list(feature_names)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        forward_features = getattr(self.model, "forward_features", None)
        if not callable(forward_features):
            raise TypeError(f"{type(self.model).__name__} does not provide forward_features()")

        outputs = forward_features(x)
        if not isinstance(outputs, Mapping):
            raise TypeError(
                "forward_features() must return a mapping from feature names to tensors; "
                f"got {type(outputs)!r}"
            )

        missing = [name for name in self.feature_names if name not in outputs]
        if missing:
            available = ", ".join(str(name) for name in outputs.keys())
            raise KeyError(f"Missing feature outputs: {', '.join(missing)}. Available: {available}")

        selected: list[torch.Tensor] = []
        for name in self.feature_names:
            value = outputs[name]
            if not torch.is_tensor(value):
                raise TypeError(f"Feature '{name}' must be a torch.Tensor, got {type(value)!r}")
            selected.append(value)
        return tuple(selected)


def export_features_with_onnx(
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
        "output_names": list(args.features),
        'external_data': False, # 不要拆分成两个文件, .data 和 .onnx
    }
    if args.dynamic_batch:
        export_kwargs["dynamic_axes"] = {
            args.input_name: {0: "batch"},
            **{name: {0: "batch"} for name in args.features},
        }
    if use_dynamo:
        export_kwargs["dynamo"] = True

    torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)


def add_openvino_toolkit_paths(openvino_root: str | None) -> None:
    if not openvino_root:
        return

    root = Path(openvino_root).expanduser()
    if not root.exists():
        return

    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    py_major = f"python{sys.version_info.major}"
    python_dirs = [
        root / "python" / py_version,
        root / "python" / py_major,
        root / "python",
        root / "runtime" / "python" / py_version,
        root / "runtime" / "python" / py_major,
        root / "runtime" / "python",
    ]
    for path in reversed(python_dirs):
        if path.exists():
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        dll_dirs = [
            root / "runtime" / "bin" / "intel64",
            root / "runtime" / "bin" / "intel64" / "Release",
            root / "runtime" / "bin" / "intel64" / "Debug",
            root / "runtime" / "3rdparty" / "tbb" / "bin",
        ]
        existing_dll_dirs = [path for path in dll_dirs if path.exists()]
        if existing_dll_dirs:
            current_lib_paths = [path for path in os.environ.get("OPENVINO_LIB_PATHS", "").split(";") if path]
            for path in existing_dll_dirs:
                path_text = str(path)
                if path_text not in current_lib_paths:
                    current_lib_paths.append(path_text)
            os.environ["OPENVINO_LIB_PATHS"] = ";".join(current_lib_paths)

        for path in existing_dll_dirs:
            _OPENVINO_DLL_HANDLES.append(os.add_dll_directory(str(path)))


def import_openvino(openvino_root: str | None):
    add_openvino_toolkit_paths(openvino_root)
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError(
            "OpenVINO Python API is required for IR export. "
            "Install openvino in the active Python environment or pass --openvino-root "
            "pointing to an OpenVINO toolkit root."
        ) from exc
    except SystemExit as exc:
        raise RuntimeError(f"OpenVINO Python API failed to initialize: {exc}") from exc
    return ov


def resolve_openvino_output_path(onnx_path: Path, output: str) -> Path:
    if not output:
        return onnx_path.with_suffix(".xml")

    path = Path(output)
    if path.suffix.lower() == ".xml":
        return path
    return path / onnx_path.with_suffix(".xml").name


def convert_to_openvino_ir(onnx_path: Path, xml_path: Path, openvino_root: str | None) -> None:
    ov = import_openvino(openvino_root)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(xml_path))


def main(args: argparse.Namespace) -> None:
    configure_stdio()

    if args.list_model:
        models = list_supported_models(args.backend)
        print(f"{args.backend} models:", len(models))
        print(models)
        return

    if not args.features:
        raise ValueError("--features is required unless --list-model is used")

    model = create_model(
        args.model,
        args.backend,
        hub_repo=args.hub_repo,
        hub_source=args.hub_source,
        pretrained=args.pretrained,
        hub_weights=args.hub_weights,
        hf_model_id=args.hf_model_id,
        local_files_only=args.local_files_only,
    )
    model.eval()

    image_size, size_source = resolve_export_image_size(model, args)
    dummy_input = make_dummy_input(args, image_size)
    wrapped_model = FeatureOutputWrapper(model, args.features)
    wrapped_model.eval()

    output_path = Path(args.output) if args.output else default_onnx_output_path(args.model, args.features)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Exporting feature model '{args.model}' to {output_path} "
        f"with input shape {tuple(dummy_input.shape)} ({size_source}) "
        f"and outputs {args.features} ..."
    )

    if args.exporter == "legacy":
        export_features_with_onnx(wrapped_model, dummy_input, output_path, args, use_dynamo=False)
        print(f"ONNX feature model '{output_path}' exported successfully with legacy exporter.")
    elif args.exporter == "dynamo":
        export_features_with_onnx(wrapped_model, dummy_input, output_path, args, use_dynamo=True)
        print(f"ONNX feature model '{output_path}' exported successfully with dynamo exporter.")
    else:
        try:
            export_features_with_onnx(wrapped_model, dummy_input, output_path, args, use_dynamo=True)
            print(f"ONNX feature model '{output_path}' exported successfully with dynamo exporter.")
        except Exception as exc:
            warnings.warn(
                f"Falling back to legacy ONNX exporter because dynamo export failed: {exc}",
                RuntimeWarning,
                stacklevel=1,
            )
            export_features_with_onnx(wrapped_model, dummy_input, output_path, args, use_dynamo=False)
            print(f"ONNX feature model '{output_path}' exported successfully with legacy exporter.")

    if args.emit_openvino or args.openvino_output:
        xml_path = resolve_openvino_output_path(output_path, args.openvino_output)
        convert_to_openvino_ir(output_path, xml_path, args.openvino_root)
        print(f"OpenVINO IR '{xml_path}' exported successfully.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export forward_features() outputs to ONNX or OpenVINO IR")
    parser.add_argument("-m", "--model", type=str, default="dinov2_vits14", help="Model name")
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=MODEL_BACKENDS,
        default="torchhub",
        help="Model provider backend",
    )
    parser.add_argument(
        "-f",
        "--features",
        type=parse_feature_names,
        default=None,
        metavar="NAMES",
        help="Comma-separated forward_features() keys to export as graph outputs",
    )
    parser.add_argument("-o", "--output", type=str, default="", help="Output ONNX file path")
    parser.add_argument("--input-name", type=str, default="input", help="Input tensor name")
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
    parser.add_argument(
        "--emit-openvino",
        action="store_true",
        help="Convert the exported ONNX file to OpenVINO IR beside the ONNX file",
    )
    parser.add_argument(
        "--openvino-output",
        type=str,
        default="",
        help="OpenVINO IR output .xml path or directory; also enables OpenVINO conversion",
    )
    parser.add_argument(
        "--openvino-root",
        type=str,
        default=DEFAULT_OPENVINO_ROOT,
        help="OpenVINO toolkit root used to locate Python package and runtime DLLs",
    )
    parser.add_argument("-l", "--list-model", "--list_model", dest="list_model", action="store_true")
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="torch.hub repo or local directory; default is inferred from the DINO model name",
    )
    parser.add_argument(
        "--hub-source",
        type=str,
        choices=("github", "local"),
        default="github",
        help="torch.hub source, used when --backend torchhub",
    )
    parser.add_argument(
        "--hub-weights",
        type=str,
        default=None,
        help="Optional DINO torch.hub weights path or URL, used when --backend torchhub",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=None,
        help="Optional Hugging Face model id, used when --backend transformers",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Hugging Face models from the local cache only, used when --backend transformers",
    )
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
