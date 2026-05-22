"""InferRT Python binding feature extraction sample."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from util import (
    dims_to_csv,
    ensure_module_path,
    numpy_dtype_name,
    preprocess_image,
    resolve_project_root,
    sanitize_file_stem,
    split_csv_names,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Dump intermediate features from InferRT Python bindings")
    parser.add_argument("-m", "--model", default="resnet18", help="Built-in model name")
    parser.add_argument("-w", "--weights", default="samples/model/classification/resnet18.wts", help="Path to .wts file")
    parser.add_argument("-f", "--features", required=True, help="Comma-separated feature names, e.g. layer1,layer4")
    parser.add_argument("-i", "--image", default="assets/pics/dog.jpg", help="Input image path")
    parser.add_argument("-o", "--output-dir", default="feature_dump_py", help="Output directory for dumped features")
    parser.add_argument("-b", "--build-dir", default="build", help="CMake build directory used to locate .pyd and DLLs")
    return parser.parse_args()


def write_manifest(
    manifest_path: Path,
    model_name: str,
    weights_path: Path,
    image_path: Path,
    feature_names: list[str],
    input_tensor: np.ndarray,
    outputs: dict[str, np.ndarray],
) -> None:
    """Write a manifest aligned with the C++ feature dump sample."""

    lines = [
        "version=1",
        "backend=inferrt_python",
        f"model_name={model_name}",
        f"weights_file={weights_path.resolve().as_posix()}",
        f"image_path={image_path.resolve().as_posix()}",
        "feature_only=true",
        f"feature_tensor_names={','.join(feature_names)}",
        f"tensor|input|{numpy_dtype_name(input_tensor)}|{dims_to_csv(input_tensor.shape)}|input.bin",
    ]

    for tensor_name, tensor in outputs.items():
        file_name = f"{sanitize_file_stem(tensor_name)}.bin"
        lines.append(f"tensor|{tensor_name}|{numpy_dtype_name(tensor)}|{dims_to_csv(tensor.shape)}|{file_name}")

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_stats(name: str, tensor: np.ndarray) -> None:
    """Print basic statistics for one feature tensor."""

    flat = tensor.reshape(-1)
    print(
        f"{name} dims=[{dims_to_csv(tensor.shape)}] "
        f"min={float(flat.min()):.6f} max={float(flat.max()):.6f} mean={float(flat.mean()):.6f}"
    )


def main() -> int:
    """Run the feature extraction sample."""

    args = parse_args()
    feature_names = split_csv_names(args.features)
    if not feature_names:
        raise ValueError("At least one feature name is required")

    project_root = resolve_project_root()
    build_dir = project_root / args.build_dir
    ensure_module_path(build_dir)

    import inferrt_model_py as irt

    model_name = args.model
    weights_path = project_root / args.weights
    image_path = project_root / args.image
    output_dir = project_root / args.output_dir

    config = irt.ModelConfig()
    config.feature_tensor_names = feature_names
    config.output_tensor_names = feature_names
    config.feature_only = True

    print(f"Creating feature-only model: {model_name}")
    model = irt.create_model(model_name, config)
    model.log_level = irt.LogLevel.INFO

    print(f"Building or loading engine from: {weights_path}")
    model.build_or_load(str(weights_path))

    input_tensor_names = model.input_tensor_names()
    output_tensor_names = model.output_tensor_names()
    print(f"Input tensors: {input_tensor_names}")
    print(f"Output tensors: {output_tensor_names}")

    if not input_tensor_names:
        raise RuntimeError("Model has no configured input tensors")

    preprocess_start = time.perf_counter()
    input_tensor = preprocess_image(image_path)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
    print(f"Running feature forward for image: {image_path}")
    if len(input_tensor_names) != 1:
        raise RuntimeError(
            f"This feature extraction sample expects exactly one input tensor, got: {input_tensor_names}"
        )

    input_tensors = {name: input_tensor for name in input_tensor_names}
    infer_start = time.perf_counter()
    output = model.forward_features(input_tensors)
    infer_ms = (time.perf_counter() - infer_start) * 1000.0

    postprocess_start = time.perf_counter()
    if isinstance(output, np.ndarray):
        output_name = output_tensor_names[0] if output_tensor_names else feature_names[0]
        output_dict = {output_name: output}
    else:
        output_dict = dict(output)

    output_dir.mkdir(parents=True, exist_ok=True)
    input_tensor.astype(np.float32, copy=False).tofile(output_dir / "input.bin")

    for tensor_name, tensor in output_dict.items():
        file_name = output_dir / f"{sanitize_file_stem(tensor_name)}.bin"
        np.ascontiguousarray(tensor).tofile(file_name)

    write_manifest(output_dir / "manifest.txt", model_name, weights_path, image_path, feature_names, input_tensor, output_dict)
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0

    print(f"Saved feature dump to: {output_dir.resolve()}")
    print(
        f"Timing: preprocess={preprocess_ms:.3f} ms, "
        f"inference={infer_ms:.3f} ms, postprocess={postprocess_ms:.3f} ms"
    )
    print(f"Input dims=[{dims_to_csv(input_tensor.shape)}]")
    for tensor_name, tensor in output_dict.items():
        print_stats(tensor_name, np.asarray(tensor))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
