"""YOLOv8-Seg：Ultralytics PyTorch 与 InferRT TensorRT 输出一致性测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers.manifest import assert_tensors_close
from helpers.model_integration import ensure_yolo_wts
from test_yolo_parity import (
    YOLO_INPUT_SIZE,
    YOLO_OUTPUT_NAMES,
    YOLO_STRIDES,
    _build_model_or_skip,
    _deterministic_yolo_input,
    _load_torch_yolo_model,
    _sigmoid,
)
from helpers.vision import allocate_output_tensors

pytestmark = [pytest.mark.integration, pytest.mark.slow]

YOLO_SEG_OUTPUT_NAMES = (*YOLO_OUTPUT_NAMES, "proto")
YOLO_SEG_NUM_CLASSES = 80
YOLO_SEG_MASK_CHANNELS = 32
YOLO_SEG_CHANNELS = 4 + YOLO_SEG_NUM_CLASSES + YOLO_SEG_MASK_CHANNELS


def _as_numpy_tensor(value: Any, *, name: str) -> np.ndarray:
    """将 PyTorch/NumPy 张量转换为连续的 ``float32`` NumPy 数组。

    Args:
        value: 待转换的张量对象。
        name: 错误消息中使用的张量名称。

    Returns:
        ``float32`` NumPy 数组。
    """

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise AssertionError(f"{name} output is empty")
    return np.ascontiguousarray(array)


def _normalize_seg_prediction(values: np.ndarray) -> np.ndarray:
    """将 YOLOv8-Seg 预测张量规整为 ``N x 116 x A`` 排列。

    Args:
        values: Ultralytics 输出的预测张量。

    Returns:
        通道维位于第 1 维的预测张量。
    """

    if values.ndim != 3:
        raise AssertionError(f"Unexpected YOLOv8-Seg prediction shape: {values.shape}")
    if values.shape[1] == YOLO_SEG_CHANNELS:
        return values
    if values.shape[2] == YOLO_SEG_CHANNELS:
        return np.transpose(values, (0, 2, 1))
    raise AssertionError(f"Unexpected YOLOv8-Seg prediction shape: {values.shape}")


def _extract_torch_yolo_seg_outputs(output: Any) -> tuple[np.ndarray, np.ndarray]:
    """从 Ultralytics YOLOv8-Seg 前向结果中提取 ``prediction`` 与 ``proto``。

    Args:
        output: PyTorch 模型前向返回值。

    Returns:
        ``(prediction, proto)``，其中 prediction 为 ``N x 116 x A``。
    """

    candidate = output
    if isinstance(candidate, (tuple, list)) and candidate:
        candidate = candidate[0]
    if not isinstance(candidate, (tuple, list)) or len(candidate) < 2:
        raise AssertionError(f"Unexpected YOLOv8-Seg output type: {type(output)!r}")

    prediction = _normalize_seg_prediction(_as_numpy_tensor(candidate[0], name="prediction"))
    proto = _as_numpy_tensor(candidate[1], name="proto")
    if proto.ndim != 4 or proto.shape[1] != YOLO_SEG_MASK_CHANNELS:
        raise AssertionError(f"Unexpected YOLOv8-Seg proto shape: {proto.shape}")
    return prediction, proto


def _run_torch_yolo_seg(
    *,
    repo_root: Path,
    build_dir: Path,
    checkpoint: Path,
    ultralytics_repo: Path,
    yolov5_repo: Path,
    input_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """运行 Ultralytics YOLOv8-Seg PyTorch 前向，返回预测张量和 proto mask。

    Args:
        repo_root: InferRT 仓库根目录。
        build_dir: CMake 构建目录。
        checkpoint: YOLOv8-Seg checkpoint 路径。
        ultralytics_repo: 本地 Ultralytics 仓库路径。
        yolov5_repo: 本地 YOLOv5 仓库路径；保持加载 helper 参数一致。
        input_tensor: ``NCHW`` 输入张量。

    Returns:
        ``(prediction, proto)`` 两个 NumPy 张量。
    """

    torch = pytest.importorskip("torch")
    model = _load_torch_yolo_model(
        repo_root=repo_root,
        build_dir=build_dir,
        model_name="yolov8n_seg",
        checkpoint=checkpoint,
        ultralytics_repo=ultralytics_repo,
        yolov5_repo=yolov5_repo,
    )
    batch = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    with torch.inference_mode():
        output = model(batch)
    return _extract_torch_yolo_seg_outputs(output)


def _run_inferrt_yolo_seg(
    irt_module: Any,
    *,
    weights_path: Path,
    input_tensor: np.ndarray,
) -> dict[str, np.ndarray]:
    """通过 InferRT pybind11 运行 YOLOv8-Seg TensorRT engine。

    Args:
        irt_module: 已导入的 ``inferrt_model_py`` 模块。
        weights_path: InferRT ``.wts`` 权重路径。
        input_tensor: ``NCHW`` 输入张量。

    Returns:
        四个输出张量，键为 ``output0/output1/output2/proto``。
    """

    config = irt_module.ModelConfig()
    config.runtime = "tensorrt:0"

    model = irt_module.create_model("yolov8n_seg", config=config)
    _build_model_or_skip(model, weights_path, label="TENSORRT/GPU/yolov8n_seg")

    input_names = list(model.input_tensor_names())
    output_names = list(model.output_tensor_names())
    assert input_names == ["input"]
    assert output_names == list(YOLO_SEG_OUTPUT_NAMES)

    outputs = allocate_output_tensors(model, output_names)
    model.infer({input_names[0]: np.ascontiguousarray(input_tensor, dtype=np.float32)}, outputs)
    return {name: np.asarray(outputs[name], dtype=np.float32) for name in output_names}


def _decode_inferrt_yolo_seg_outputs(outputs: dict[str, np.ndarray], image_size: int) -> np.ndarray:
    """将 InferRT YOLOv8-Seg 分支解码为 Ultralytics ``xywh + class + mask`` 排列。

    Args:
        outputs: InferRT 三个检测分支输出。
        image_size: 输入图像边长。

    Returns:
        形状为 ``N x 116 x A`` 的解码预测张量。
    """

    decoded_parts: list[np.ndarray] = []
    for output_name, stride in zip(YOLO_OUTPUT_NAMES, YOLO_STRIDES, strict=True):
        branch = np.asarray(outputs[output_name], dtype=np.float32)
        batch, channels, anchors = branch.shape
        if channels != YOLO_SEG_CHANNELS:
            raise AssertionError(f"{output_name} should have {YOLO_SEG_CHANNELS} channels, got {channels}")

        grid_h = image_size // stride
        grid_w = image_size // stride
        if anchors != grid_h * grid_w:
            raise AssertionError(f"{output_name} anchor count mismatch: {anchors} vs {grid_h * grid_w}")

        grid_y, grid_x = np.meshgrid(
            np.arange(grid_h, dtype=np.float32),
            np.arange(grid_w, dtype=np.float32),
            indexing="ij",
        )
        anchor_xy = np.stack((grid_x.reshape(-1) + 0.5, grid_y.reshape(-1) + 0.5), axis=0)

        distances = branch[:, :4, :]
        x1 = (anchor_xy[0][None, :] - distances[:, 0, :]) * stride
        y1 = (anchor_xy[1][None, :] - distances[:, 1, :]) * stride
        x2 = (anchor_xy[0][None, :] + distances[:, 2, :]) * stride
        y2 = (anchor_xy[1][None, :] + distances[:, 3, :]) * stride
        xywh = np.stack(((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1), axis=1)

        class_scores = _sigmoid(branch[:, 4 : 4 + YOLO_SEG_NUM_CLASSES, :])
        mask_coefficients = branch[:, 4 + YOLO_SEG_NUM_CLASSES :, :]
        decoded_parts.append(
            np.concatenate([xywh, class_scores, mask_coefficients], axis=1).reshape(
                batch, YOLO_SEG_CHANNELS, anchors
            )
        )

    return np.concatenate(decoded_parts, axis=2)


def _assert_yolo_seg_outputs_close(
    reference_prediction: np.ndarray,
    reference_proto: np.ndarray,
    actual_outputs: dict[str, np.ndarray],
) -> None:
    """比较 YOLOv8-Seg 的框、分类分数、mask 系数与 proto 输出。

    Args:
        reference_prediction: PyTorch 参考预测张量。
        reference_proto: PyTorch 参考 proto 张量。
        actual_outputs: InferRT TensorRT 输出张量。
    """

    actual_prediction = _decode_inferrt_yolo_seg_outputs(actual_outputs, YOLO_INPUT_SIZE)
    assert actual_prediction.shape == reference_prediction.shape
    assert_tensors_close(
        reference_prediction[:, :4, :],
        actual_prediction[:, :4, :],
        rtol=1e-3,
        atol=1.5,
        name="yolov8n_seg.boxes",
    )
    assert_tensors_close(
        reference_prediction[:, 4 : 4 + YOLO_SEG_NUM_CLASSES, :],
        actual_prediction[:, 4 : 4 + YOLO_SEG_NUM_CLASSES, :],
        rtol=1e-3,
        atol=5e-4,
        name="yolov8n_seg.classes",
    )
    assert_tensors_close(
        reference_prediction[:, 4 + YOLO_SEG_NUM_CLASSES :, :],
        actual_prediction[:, 4 + YOLO_SEG_NUM_CLASSES :, :],
        rtol=1e-3,
        atol=8e-3,
        name="yolov8n_seg.mask_coefficients",
    )
    assert_tensors_close(
        reference_proto,
        actual_outputs["proto"],
        rtol=1e-3,
        atol=8e-3,
        name="yolov8n_seg.proto",
    )


def test_yolov8_seg_tensorrt_matches_ultralytics_forward(
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    ultralytics_repo: Path,
    yolov5_repo: Path,
    irt_module: Any,
    compare_runtimes: list[str],
    compare_devices: list[str],
) -> None:
    """同一输入下，InferRT YOLOv8-Seg TensorRT 输出应接近 Ultralytics PyTorch 前向。

    Args:
        repo_root: InferRT 仓库根目录。
        build_dir: CMake 构建目录。
        model_root: 真实模型根目录，包含 ``yolov8/yolov8n-seg.pt``。
        ultralytics_repo: 本地 Ultralytics 仓库路径。
        yolov5_repo: 本地 YOLOv5 仓库路径；保持 helper 参数一致。
        irt_module: 已导入的 ``inferrt_model_py`` 模块。
        compare_runtimes: 用户选择的后端列表。
        compare_devices: 用户选择的设备列表。
    """

    if "TENSORRT" not in compare_runtimes:
        pytest.skip("YOLOv8-Seg parity requires --inferrt-compare-runtime=tensorrt")
    if "gpu" not in compare_devices:
        pytest.skip("YOLOv8-Seg TensorRT parity requires --inferrt-compare-devices=gpu")

    checkpoint = model_root / "yolov8" / "yolov8n-seg.pt"
    weights_path = ensure_yolo_wts(
        repo_root=repo_root,
        build_dir=build_dir,
        model_root=model_root,
        model_name="yolov8n_seg",
        checkpoint=checkpoint,
        ultralytics_repo=ultralytics_repo,
        yolov5_repo=yolov5_repo,
    )
    input_tensor = _deterministic_yolo_input()

    torch_prediction, torch_proto = _run_torch_yolo_seg(
        repo_root=repo_root,
        build_dir=build_dir,
        checkpoint=checkpoint,
        ultralytics_repo=ultralytics_repo,
        yolov5_repo=yolov5_repo,
        input_tensor=input_tensor,
    )
    inferrt_outputs = _run_inferrt_yolo_seg(irt_module, weights_path=weights_path, input_tensor=input_tensor)

    _assert_yolo_seg_outputs_close(torch_prediction, torch_proto, inferrt_outputs)
