"""DINO：PyTorch 参考模型与 InferRT pybind11 后端输出一致性测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers.manifest import assert_tensors_close
from test_model_dir_parity import (
    ModelDirCase,
    _assert_feature_outputs,
    _case_artifact_dir,
    _ensure_onnx_artifacts,
    _ensure_wts,
    _feature_output_tolerances,
    _input_tensor_for_model,
    _load_case_model,
    _primary_output_tolerances,
    _run_inferrt_features,
    _run_inferrt_primary,
    _runtime_device_pairs,
    _runtime_label,
    _torch_features,
    _torch_primary,
    discover_model_dir_cases,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

DINO_FAMILIES = ["dinov2", "dinov3"]


def _discover_dino_cases(model_root: Path, pytestconfig: pytest.Config) -> list[ModelDirCase]:
    """从模型根目录发现 DINOv2/DINOv3 测试用例。

    Args:
        model_root: 用户通过 ``--inferrt-model-root`` 指定的真实模型根目录。
        pytestconfig: pytest 配置对象，用于读取最大用例数。

    Returns:
        已发现的 DINO 模型目录用例列表。
    """

    max_cases = int(pytestconfig.getoption("--inferrt-model-dir-max-cases"))
    cases = discover_model_dir_cases(model_root, DINO_FAMILIES, max_cases=max_cases)
    if not cases:
        pytest.skip(f"No supported DINO checkpoints found under {model_root}/{{dinov2,dinov3}}")
    return cases


def test_dino_pybind_matches_pytorch_forward_and_features(
    pytestconfig: pytest.Config,
    compare_runtimes: list[str],
    compare_devices: list[str],
    irt_module: Any,
    model_root: Path,
    tolerances: tuple[float, float],
    feature_tolerances: tuple[float, float],
) -> None:
    """逐项比较 DINO PyTorch 参考输出与 pybind11 后端输出。

    Args:
        pytestconfig: pytest 配置对象。
        compare_runtimes: 用户选择的后端列表。
        compare_devices: 用户选择的设备列表。
        irt_module: 已导入的 ``inferrt_model_py`` 模块。
        model_root: 真实模型根目录。
        tolerances: 主输出默认容差。
        feature_tolerances: 特征输出默认容差。
    """

    if not compare_runtimes:
        pytest.skip("No runtimes selected; pass --inferrt-compare-runtime=TensorRT,onnx,openvino")
    runtime_device_pairs = _runtime_device_pairs(compare_runtimes, compare_devices)
    if not runtime_device_pairs:
        pytest.skip("No compatible runtime/device pairs selected; TensorRT requires --inferrt-compare-devices=gpu")

    for case in _discover_dino_cases(model_root, pytestconfig):
        model = _load_case_model(case, pytestconfig)
        input_tensor = _input_tensor_for_model(model)
        torch_primary = _torch_primary(model, input_tensor)
        torch_features = _torch_features(case, model, input_tensor)

        output_dir = _case_artifact_dir(model_root, case)
        wts_path = _ensure_wts(case, model, output_dir)
        primary_onnx, feature_onnx = _ensure_onnx_artifacts(case, model, input_tensor, output_dir)

        for backend_attr, device in runtime_device_pairs:
            device_attr = device.upper()
            label = f"{case.model_name}.{_runtime_label(backend_attr)}.{device}"
            primary_model_file = wts_path if backend_attr == "TENSORRT" else primary_onnx
            feature_model_file = wts_path if backend_attr == "TENSORRT" else feature_onnx

            backend_primary = _run_inferrt_primary(
                irt_module,
                model_name=case.model_name,
                model_file=primary_model_file,
                input_tensor=input_tensor,
                backend_attr=backend_attr,
                device_attr=device_attr,
            )
            primary_rtol, primary_atol = _primary_output_tolerances(case, backend_attr, tolerances, feature_tolerances)
            assert_tensors_close(
                torch_primary,
                np.asarray(backend_primary),
                rtol=primary_rtol,
                atol=primary_atol,
                name=f"{label}.primary",
            )

            backend_features = _run_inferrt_features(
                irt_module,
                case=case,
                model_file=feature_model_file,
                input_tensor=input_tensor,
                backend_attr=backend_attr,
                device_attr=device_attr,
            )
            feature_rtol, feature_atol = _feature_output_tolerances(case, backend_attr, feature_tolerances)
            _assert_feature_outputs(
                torch_features,
                backend_features,
                rtol=feature_rtol,
                atol=feature_atol,
                label=f"{label}.features",
            )
