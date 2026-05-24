"""YOLO：官方 PyTorch 前向与 InferRT pybind11 输出一致性测试。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers.manifest import assert_tensors_close
from helpers.model_integration import ensure_yolo_wts
from util import allocate_output_tensors

pytestmark = [pytest.mark.integration, pytest.mark.slow]

YOLO_INPUT_SIZE = 640
YOLO_STRIDES = (8, 16, 32)
YOLO_OUTPUT_NAMES = ("output0", "output1", "output2")


def _configure_ultralytics(repo_root: Path, build_dir: Path, ultralytics_repo: Path) -> None:
    """配置 Ultralytics 导入与运行环境，避免测试过程中触发自动安装依赖。"""

    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(build_dir / "ultralytics_config"))

    detection_samples = repo_root / "samples" / "model" / "detection"
    for path in (detection_samples, ultralytics_repo):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _deterministic_yolo_input() -> np.ndarray:
    """生成稳定的 ``0..1`` NCHW 输入，便于重复比较数值误差。"""

    rng = np.random.default_rng(20240524)
    return rng.random((1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), dtype=np.float32)


def _run_torch_yolo(
    *,
    repo_root: Path,
    build_dir: Path,
    model_name: str,
    checkpoint: Path,
    ultralytics_repo: Path,
    input_tensor: np.ndarray,
) -> np.ndarray:
    """直接运行 Ultralytics PyTorch 模型，返回已解码的 ``1x84x8400`` 输出。"""

    torch = pytest.importorskip("torch")
    _configure_ultralytics(repo_root, build_dir, ultralytics_repo)

    from yolo_model_zoo import load_ultralytics_model

    model = load_ultralytics_model(model_name, weights=checkpoint, repo=ultralytics_repo)
    batch = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    with torch.inference_mode():
        output = model(batch)

    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise AssertionError(f"Unexpected Ultralytics output type: {type(output)!r}")
    return output.detach().cpu().numpy().astype(np.float32, copy=False)


def _run_inferrt_yolo(
    irt_module: Any,
    *,
    model_name: str,
    weights_path: Path,
    input_tensor: np.ndarray,
) -> dict[str, np.ndarray]:
    """通过 InferRT pybind11 运行 YOLO TensorRT engine，返回三个尺度分支输出。"""

    model = irt_module.create_model(model_name)
    model.build_or_load(str(weights_path))

    input_names = list(model.input_tensor_names())
    output_names = list(model.output_tensor_names())
    assert input_names == ["input"]
    assert output_names == list(YOLO_OUTPUT_NAMES)

    outputs = allocate_output_tensors(model, output_names)
    model.infer({input_names[0]: input_tensor}, outputs)
    return {name: np.asarray(outputs[name], dtype=np.float32) for name in output_names}


def _decode_inferrt_yolo_outputs(outputs: dict[str, np.ndarray], image_size: int) -> np.ndarray:
    """将 InferRT 三个 YOLO 分支解码为 Ultralytics ``xywh + class`` 排列。"""

    decoded_parts: list[np.ndarray] = []
    for output_name, stride in zip(YOLO_OUTPUT_NAMES, YOLO_STRIDES, strict=True):
        branch = np.asarray(outputs[output_name], dtype=np.float32)
        batch, channels, anchors = branch.shape
        if channels != 84:
            raise AssertionError(f"{output_name} should have 84 channels, got {channels}")

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

        class_scores = 1.0 / (1.0 + np.exp(-branch[:, 4:, :]))
        decoded_parts.append(np.concatenate([xywh, class_scores], axis=1).reshape(batch, 84, anchors))

    return np.concatenate(decoded_parts, axis=2)


@pytest.mark.parametrize(
    "model_name,relative_checkpoint",
    [
        pytest.param("yolov8n", Path("yolov8") / "yolov8n.pt", id="yolov8n"),
        pytest.param("yolov5n", Path("yolov5") / "yolov5nu.pt", id="yolov5n"),
    ],
)
def test_yolo_pybind_matches_ultralytics_forward(
    irt_module: Any,
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    ultralytics_repo: Path,
    model_name: str,
    relative_checkpoint: Path,
) -> None:
    """同一输入下，InferRT YOLO 解码结果应与 Ultralytics PyTorch 前向保持一致。

    Args:
        irt_module: 已导入的 ``inferrt_model_py`` 模块。
        repo_root: 仓库根目录。
        build_dir: CMake 构建目录。
        model_root: 真实模型根目录，默认 ``D:/Models``。
        ultralytics_repo: 本地 ultralytics 仓库路径。
        model_name: InferRT YOLO 模型 key。
        relative_checkpoint: 相对模型根目录的 checkpoint 路径。
    """

    checkpoint = model_root / relative_checkpoint
    input_tensor = _deterministic_yolo_input()
    weights = ensure_yolo_wts(
        repo_root=repo_root,
        build_dir=build_dir,
        model_name=model_name,
        checkpoint=checkpoint,
        ultralytics_repo=ultralytics_repo,
    )

    torch_output = _run_torch_yolo(
        repo_root=repo_root,
        build_dir=build_dir,
        model_name=model_name,
        checkpoint=checkpoint,
        ultralytics_repo=ultralytics_repo,
        input_tensor=input_tensor,
    )
    inferrt_outputs = _run_inferrt_yolo(
        irt_module,
        model_name=model_name,
        weights_path=weights,
        input_tensor=input_tensor,
    )
    decoded_output = _decode_inferrt_yolo_outputs(inferrt_outputs, YOLO_INPUT_SIZE)

    assert decoded_output.shape == torch_output.shape == (1, 84, 8400)
    assert_tensors_close(torch_output[:, :4, :], decoded_output[:, :4, :], rtol=1e-3, atol=1.1, name="boxes")
    assert_tensors_close(torch_output[:, 4:, :], decoded_output[:, 4:, :], rtol=1e-3, atol=5e-4, name="classes")
