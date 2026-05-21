"""Runtime helpers for locating InferRT binaries and running Python bindings."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from util import allocate_output_tensors, preprocess_image


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_build_dir(root: Path) -> Path:
    return Path(os.environ.get("INFERRT_BUILD_DIR", root / "build")).resolve()


def sample_executable(build_dir: Path, sample_name: str) -> Path:
    exe_name = f"inferrt_sample_{sample_name}"
    if os.name == "nt":
        exe_name += ".exe"
    candidates = [
        build_dir / "bin" / exe_name,
        build_dir / "bin" / "Debug" / exe_name,
        build_dir / "bin" / "Release" / exe_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Sample executable '{exe_name}' not found under {build_dir}. "
        "Build targets inferrt_sample_classification and inferrt_sample_feature_extract first."
    )


def require_weights(root: Path, relative_weights: str) -> Path:
    weights_path = (root / relative_weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file missing: {weights_path}. "
            "Generate it with samples/model/classification/gen_wts.py."
        )
    return weights_path


def run_process(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  cmd: {' '.join(command)}\n"
            f"  exit: {completed.returncode}\n"
            f"  stdout:\n{completed.stdout}\n"
            f"  stderr:\n{completed.stderr}"
        )


def run_cpp_classification_dump(
    *,
    build_dir: Path,
    model_name: str,
    weights_path: Path,
    image_path: Path,
    dump_dir: Path,
    cwd: Path,
) -> None:
    executable = sample_executable(build_dir, "classification")
    command = [
        str(executable),
        "--model",
        model_name,
        "--weights-file",
        str(weights_path),
        "--image-path",
        str(image_path),
        "--dump-dir",
        str(dump_dir),
    ]
    run_process(command, cwd=cwd)


def run_cpp_feature_dump(
    *,
    build_dir: Path,
    model_name: str,
    weights_path: Path,
    image_path: Path,
    feature_names: list[str],
    dump_dir: Path,
    cwd: Path,
) -> None:
    executable = sample_executable(build_dir, "feature_extract")
    command = [
        str(executable),
        "--model",
        model_name,
        "--weights-file",
        str(weights_path),
        "--features",
        ",".join(feature_names),
        "--image-path",
        str(image_path),
        "--output-dir",
        str(dump_dir),
    ]
    run_process(command, cwd=cwd)


def run_python_classification(
    irt_module: object,
    *,
    model_name: str,
    weights_path: Path,
    input_tensor: np.ndarray,
) -> dict[str, np.ndarray]:
    model = irt_module.create_model(model_name)
    model.build_or_load(str(weights_path))

    input_names = model.input_tensor_names()
    output_names = model.output_tensor_names()
    if len(input_names) != 1:
        raise RuntimeError(f"Expected one input tensor, got: {input_names}")

    inputs = {input_names[0]: input_tensor}
    outputs = allocate_output_tensors(model, output_names)
    model.infer(inputs, outputs)
    return {"input": input_tensor, **outputs}


def run_python_features(
    irt_module: object,
    *,
    model_name: str,
    weights_path: Path,
    input_tensor: np.ndarray,
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    config = irt_module.ModelConfig()
    config.feature_tensor_names = feature_names
    config.feature_only = True

    model = irt_module.create_model(model_name, config)
    model.build_or_load(str(weights_path))

    input_names = model.input_tensor_names()
    if len(input_names) != 1:
        raise RuntimeError(f"Expected one input tensor, got: {input_names}")

    outputs = model.forward_features({input_names[0]: input_tensor})
    if isinstance(outputs, np.ndarray):
        feature_output_names = model.feature_output_tensor_names()
        output_name = feature_output_names[0] if feature_output_names else feature_names[0]
        output_dict = {output_name: outputs}
    else:
        output_dict = dict(outputs)

    return {"input": input_tensor, **output_dict}
