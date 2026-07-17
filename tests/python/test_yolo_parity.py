"""YOLO：官方 PyTorch 前向与 InferRT pybind11 输出一致性测试。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers.manifest import assert_tensors_close
from helpers.model_integration import conversion_artifact_dir, ensure_yolo_wts, is_fresh_against_all
from helpers.vision import allocate_output_tensors

pytestmark = [pytest.mark.integration, pytest.mark.slow]

YOLO_INPUT_SIZE = 640
YOLO_STRIDES = (8, 16, 32)
YOLO_OUTPUT_NAMES = ("output0", "output1", "output2")
YOLOV5_ANCHORS = (
    ((10.0, 13.0), (16.0, 30.0), (33.0, 23.0)),
    ((30.0, 61.0), (62.0, 45.0), (59.0, 119.0)),
    ((116.0, 90.0), (156.0, 198.0), (373.0, 326.0)),
)


def _runtime_label(runtime_attr: str) -> str:
    """将后端枚举名转换为断言标签中的短名称。

    Args:
        runtime_attr: 后端枚举属性名。

    Returns:
        用于断言名称的短标签。
    """

    return {
        "TENSORRT": "tensorrt",
        "ONNXRUNTIME": "onnx",
        "OPENVINO": "openvino",
    }[runtime_attr]


def _runtime_spec(runtime_attr: str, device_attr: str) -> str:
    """将测试后端和设备组合为统一 runtime 字符串。"""

    device = "0" if device_attr.upper() == "GPU" else "cpu"
    return f"{_runtime_label(runtime_attr)}:{device}"


def _runtime_device_pairs(compare_runtimes: list[str], compare_devices: list[str]) -> list[tuple[str, str]]:
    """根据 runtime/device 参数生成 YOLO 可执行的后端组合。

    Args:
        compare_runtimes: 用户选择的后端列表。
        compare_devices: 用户选择的设备列表。

    Returns:
        ``(后端枚举名, 设备名)`` 组合列表。
    """

    pairs: list[tuple[str, str]] = []
    for runtime_attr in compare_runtimes:
        if runtime_attr == "TENSORRT":
            if "gpu" in compare_devices:
                pairs.append((runtime_attr, "gpu"))
            continue
        pairs.extend((runtime_attr, device) for device in compare_devices)
    return pairs


def _configure_ultralytics(repo_root: Path, build_dir: Path, ultralytics_repo: Path) -> None:
    """配置 Ultralytics 导入与运行环境，避免测试过程中触发自动安装依赖。"""

    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(build_dir / "ultralytics_config"))

    detection_samples = repo_root / "samples" / "model" / "detection"
    for path in (detection_samples, ultralytics_repo):
        if path.exists() and str(path) not in sys.path:
            sys.path.append(str(path))


def _deterministic_yolo_input() -> np.ndarray:
    """生成稳定的 ``0..1`` NCHW 输入，便于重复比较数值误差。"""

    rng = np.random.default_rng(20240524)
    return rng.random((1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), dtype=np.float32)


def _load_torch_yolo_model(
    *,
    repo_root: Path,
    build_dir: Path,
    model_name: str,
    checkpoint: Path,
    ultralytics_repo: Path,
    yolov5_repo: Path,
) -> Any:
    """加载 Ultralytics/YOLOv5 PyTorch 参考模型。

    Args:
        repo_root: InferRT 仓库根目录。
        build_dir: CMake 构建目录，用于配置可写缓存。
        model_name: InferRT YOLO 模型 key。
        checkpoint: checkpoint 路径。
        ultralytics_repo: 本地 Ultralytics 仓库路径。
        yolov5_repo: 本地 YOLOv5 仓库路径。

    Returns:
        已加载并切到 eval 模式的 PyTorch 模型。
    """

    _configure_ultralytics(repo_root, build_dir, ultralytics_repo)

    from yolo_model_zoo import load_ultralytics_model

    return load_ultralytics_model(
        model_name,
        weights=checkpoint,
        repo=ultralytics_repo,
        yolov5_repo=yolov5_repo,
    )


def _run_torch_yolo(
    *,
    repo_root: Path,
    build_dir: Path,
    model_name: str,
    checkpoint: Path,
    ultralytics_repo: Path,
    yolov5_repo: Path,
    input_tensor: np.ndarray,
) -> np.ndarray:
    """直接运行 Ultralytics PyTorch 模型，返回已解码的 ``1x84x8400`` 输出。"""

    torch = pytest.importorskip("torch")
    model = _load_torch_yolo_model(
        repo_root=repo_root,
        build_dir=build_dir,
        model_name=model_name,
        checkpoint=checkpoint,
        ultralytics_repo=ultralytics_repo,
        yolov5_repo=yolov5_repo,
    )
    batch = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    with torch.inference_mode():
        output = model(batch)

    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise AssertionError(f"Unexpected Ultralytics output type: {type(output)!r}")
    values = output.detach().cpu().numpy().astype(np.float32, copy=False)
    if values.ndim != 3:
        raise AssertionError(f"Unexpected YOLO output shape: {values.shape}")
    if values.shape[1] in (84, 85):
        return values
    if values.shape[2] in (84, 85):
        return np.transpose(values, (0, 2, 1))
    raise AssertionError(f"Unexpected YOLO output shape: {values.shape}")


def _normalize_yolo_graph_output(output: np.ndarray) -> np.ndarray:
    """将图后端 YOLO 单输出归一化为 ``N x C x A`` 排列。

    Args:
        output: ONNX/OpenVINO 后端输出数组。

    Returns:
        通道维在第 1 维的 YOLO 输出数组。
    """

    values = np.asarray(output, dtype=np.float32)
    if values.ndim != 3:
        raise AssertionError(f"Unexpected YOLO graph output shape: {values.shape}")
    if values.shape[1] in (84, 85):
        return values
    if values.shape[2] in (84, 85):
        return np.transpose(values, (0, 2, 1))
    raise AssertionError(f"Unexpected YOLO graph output shape: {values.shape}")


def _export_yolo_onnx(model: Any, input_tensor: np.ndarray, output_path: Path) -> None:
    """导出用于 ONNX Runtime/OpenVINO 对比的 YOLO ONNX 图。

    Args:
        model: PyTorch YOLO 模型。
        input_tensor: 导出示例输入。
        output_path: ONNX 输出路径。
    """

    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")

    class _YOLOOutputWrapper(torch.nn.Module):
        """将 YOLO 模型可能返回的 tuple/list 规整为单输出张量。"""

        def __init__(self, wrapped: Any) -> None:
            """保存待导出的 YOLO 模型。

            Args:
                wrapped: PyTorch YOLO 模型。
            """

            super().__init__()
            self.wrapped = wrapped

        def forward(self, x):
            """执行 YOLO 前向并返回第一个张量输出。

            Args:
                x: 形状为 ``NCHW`` 的输入张量。

            Returns:
                YOLO 原始预测张量。
            """

            output = self.wrapped(x)
            if isinstance(output, (tuple, list)):
                output = output[0]
            return output

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _YOLOOutputWrapper(model).eval()
    dummy_input = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


def _ensure_yolo_onnx(
    *,
    repo_root: Path,
    build_dir: Path,
    model_name: str,
    checkpoint: Path,
    ultralytics_repo: Path,
    yolov5_repo: Path,
    input_tensor: np.ndarray,
    output_path: Path,
) -> Path:
    """确保 YOLO ONNX 产物存在且不早于 checkpoint。

    Args:
        repo_root: InferRT 仓库根目录。
        build_dir: CMake 构建目录。
        model_name: InferRT YOLO 模型 key。
        checkpoint: checkpoint 路径。
        ultralytics_repo: 本地 Ultralytics 仓库路径。
        yolov5_repo: 本地 YOLOv5 仓库路径。
        input_tensor: 导出示例输入。
        output_path: ONNX 输出路径。

    Returns:
        ONNX 模型路径。
    """

    if is_fresh_against_all(output_path, [checkpoint]):
        return output_path

    model = _load_torch_yolo_model(
        repo_root=repo_root,
        build_dir=build_dir,
        model_name=model_name,
        checkpoint=checkpoint,
        ultralytics_repo=ultralytics_repo,
        yolov5_repo=yolov5_repo,
    )
    _export_yolo_onnx(model, input_tensor, output_path)
    return output_path


def _build_model_or_skip(model: Any, model_file: Path, *, label: str) -> None:
    """构建/加载 YOLO 后端模型，后端不可用时跳过。

    Args:
        model: InferRT Python 模型对象。
        model_file: ``.wts`` 或 ONNX 模型路径。
        label: skip 消息中的后端/模型标签。
    """

    try:
        model.build_or_load(str(model_file))
    except Exception as exc:
        message = str(exc)
        unavailable_markers = (
            "backend is not enabled",
            "Failed to load ONNX Runtime DLL",
            "CUDA driver",
            "CUDA failure",
            "CUDA error",
            "CUDA provider",
            "Failed to initialize CUDA",
            "Device with \"GPU\" name is not registered",
            "Cannot get DEVICE_PROPERTIES",
        )
        if any(marker in message for marker in unavailable_markers):
            pytest.skip(f"{label} backend unavailable: {message}")
        raise


def _run_inferrt_yolo(
    irt_module: Any,
    *,
    model_name: str,
    weights_path: Path,
    input_tensor: np.ndarray,
) -> dict[str, np.ndarray]:
    """通过 InferRT pybind11 运行 YOLO TensorRT engine，返回三个尺度分支输出。"""

    config = irt_module.ModelConfig()
    config.runtime = "tensorrt:0"
    model = irt_module.create_model(model_name, config=config)
    _build_model_or_skip(model, weights_path, label=f"TENSORRT/GPU/{model_name}")

    input_names = list(model.input_tensor_names())
    output_names = list(model.output_tensor_names())
    assert input_names == ["input"]
    assert output_names == list(YOLO_OUTPUT_NAMES)

    outputs = allocate_output_tensors(model, output_names)
    model.infer({input_names[0]: input_tensor}, outputs)
    return {name: np.asarray(outputs[name], dtype=np.float32) for name in output_names}


def _run_graph_yolo(
    irt_module: Any,
    *,
    onnx_path: Path,
    input_tensor: np.ndarray,
    backend_attr: str,
    device_attr: str,
    model_name: str,
) -> np.ndarray:
    """使用 ONNX Runtime 或 OpenVINO 后端执行 YOLO ONNX 图。

    Args:
        irt_module: 已导入的 ``inferrt_model_py`` 模块。
        onnx_path: YOLO ONNX 模型路径。
        input_tensor: NumPy 输入张量。
        backend_attr: 后端枚举属性名。
        device_attr: 设备枚举属性名。
        model_name: 当前 YOLO 模型 key。

    Returns:
        已归一化排列的 YOLO 图后端输出。
    """

    config = irt_module.ModelConfig()
    config.runtime = _runtime_spec(backend_attr, device_attr)
    config.output_tensor_names = ["output"]

    model = irt_module.create_model("onnx", config=config)
    _build_model_or_skip(model, onnx_path, label=f"{backend_attr}/{device_attr}/{model_name}")

    outputs = model.infer({"input": np.ascontiguousarray(input_tensor, dtype=np.float32)}, None)
    if isinstance(outputs, np.ndarray):
        return _normalize_yolo_graph_output(outputs)
    output_dict = {str(name): np.asarray(value) for name, value in dict(outputs).items()}
    if "output" not in output_dict:
        raise RuntimeError(f"Expected YOLO graph output named 'output', got {sorted(output_dict)}")
    return _normalize_yolo_graph_output(output_dict["output"])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """计算 NumPy 数组的 sigmoid。

    Args:
        values: 输入数组。

    Returns:
        sigmoid 后的数组。
    """

    return 1.0 / (1.0 + np.exp(-values))


def _decode_dfl_inferrt_yolo_outputs(outputs: dict[str, np.ndarray], image_size: int) -> np.ndarray:
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

        class_scores = _sigmoid(branch[:, 4:, :])
        decoded_parts.append(np.concatenate([xywh, class_scores], axis=1).reshape(batch, 84, anchors))

    return np.concatenate(decoded_parts, axis=2)


def _decode_legacy_inferrt_yolo_outputs(outputs: dict[str, np.ndarray]) -> np.ndarray:
    """将旧版 YOLOv5 raw anchor head 解码为 ``xywh + obj + class`` 排列。"""

    decoded_parts: list[np.ndarray] = []
    for output_name, stride, anchors in zip(YOLO_OUTPUT_NAMES, YOLO_STRIDES, YOLOV5_ANCHORS, strict=True):
        branch = np.asarray(outputs[output_name], dtype=np.float32)
        batch, channels, grid_h, grid_w = branch.shape
        if channels != 255:
            raise AssertionError(f"{output_name} should have 255 channels, got {channels}")

        raw = branch.reshape(batch, 3, 85, grid_h, grid_w).transpose(0, 1, 3, 4, 2)
        grid_y, grid_x = np.meshgrid(
            np.arange(grid_h, dtype=np.float32),
            np.arange(grid_w, dtype=np.float32),
            indexing="ij",
        )
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, 1, grid_h, grid_w, 2)
        anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(1, 3, 1, 1, 2)

        xy = (_sigmoid(raw[..., 0:2]) * 2.0 - 0.5 + grid) * float(stride)
        wh = np.square(_sigmoid(raw[..., 2:4]) * 2.0) * anchor_grid
        scores = _sigmoid(raw[..., 4:])
        decoded = np.concatenate((xy, wh, scores), axis=-1)
        decoded_parts.append(decoded.reshape(batch, -1, 85).transpose(0, 2, 1))

    return np.concatenate(decoded_parts, axis=2)


def _decode_inferrt_yolo_outputs(outputs: dict[str, np.ndarray], image_size: int) -> np.ndarray:
    """根据 InferRT 输出形状自动选择 YOLOv5u/YOLOv8 DFL 或旧版 YOLOv5 anchor 解码。"""

    first = np.asarray(outputs[YOLO_OUTPUT_NAMES[0]])
    if first.ndim == 3:
        return _decode_dfl_inferrt_yolo_outputs(outputs, image_size)
    if first.ndim == 4:
        return _decode_legacy_inferrt_yolo_outputs(outputs)
    raise AssertionError(f"Unsupported YOLO output rank: {first.shape}")


def _assert_yolo_outputs_close(reference: np.ndarray, actual: np.ndarray, *, label: str) -> None:
    """比较 YOLO 输出的框、置信度和分类分数。

    Args:
        reference: PyTorch 参考输出。
        actual: InferRT 后端输出。
        label: 断言失败时使用的标签前缀。
    """

    assert actual.shape == reference.shape
    assert_tensors_close(reference[:, :4, :], actual[:, :4, :], rtol=1e-3, atol=1.1, name=f"{label}.boxes")
    if reference.shape[1] == 85:
        assert_tensors_close(
            reference[:, 4:5, :],
            actual[:, 4:5, :],
            rtol=1e-3,
            atol=5e-4,
            name=f"{label}.objectness",
        )
        class_start = 5
    else:
        class_start = 4
    class_atol = 2e-3 if reference.shape[1] == 85 else 5e-4
    assert_tensors_close(
        reference[:, class_start:, :],
        actual[:, class_start:, :],
        rtol=1e-3,
        atol=class_atol,
        name=f"{label}.classes",
    )


@pytest.mark.parametrize(
    "model_name,relative_checkpoint",
    [
        pytest.param("yolov8n", Path("yolov8") / "yolov8n.pt", id="yolov8n"),
        pytest.param("yolov5n", Path("yolov5") / "yolov5n.pt", id="yolov5n_legacy"),
        pytest.param("yolov5s", Path("yolov5") / "yolov5s.pt", id="yolov5s_legacy"),
    ],
)
def test_yolo_pybind_matches_ultralytics_forward(
    compare_runtimes: list[str],
    compare_devices: list[str],
    irt_module: Any,
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    ultralytics_repo: Path,
    yolov5_repo: Path,
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

    if not compare_runtimes:
        pytest.skip("No runtimes selected; pass --inferrt-compare-runtime=TensorRT,onnx,openvino")
    runtime_device_pairs = _runtime_device_pairs(compare_runtimes, compare_devices)
    if not runtime_device_pairs:
        pytest.skip("No compatible runtime/device pairs selected; TensorRT requires --inferrt-compare-devices=gpu")

    checkpoint = model_root / relative_checkpoint
    if not checkpoint.exists():
        pytest.skip(f"{model_name} checkpoint not found: {checkpoint}")
    input_tensor = _deterministic_yolo_input()
    torch_output = _run_torch_yolo(
        repo_root=repo_root,
        build_dir=build_dir,
        model_name=model_name,
        checkpoint=checkpoint,
        ultralytics_repo=ultralytics_repo,
        yolov5_repo=yolov5_repo,
        input_tensor=input_tensor,
    )

    output_dir = conversion_artifact_dir(model_root, checkpoint)
    weights: Path | None = None
    onnx_path: Path | None = None

    for backend_attr, device in runtime_device_pairs:
        label = f"{model_name}.{_runtime_label(backend_attr)}.{device}"
        if backend_attr == "TENSORRT":
            if weights is None:
                weights = ensure_yolo_wts(
                    repo_root=repo_root,
                    build_dir=build_dir,
                    model_root=model_root,
                    model_name=model_name,
                    checkpoint=checkpoint,
                    ultralytics_repo=ultralytics_repo,
                    yolov5_repo=yolov5_repo,
                )
            inferrt_outputs = _run_inferrt_yolo(
                irt_module,
                model_name=model_name,
                weights_path=weights,
                input_tensor=input_tensor,
            )
            decoded_output = _decode_inferrt_yolo_outputs(inferrt_outputs, YOLO_INPUT_SIZE)
            _assert_yolo_outputs_close(torch_output, decoded_output, label=label)
            continue

        if onnx_path is None:
            onnx_path = _ensure_yolo_onnx(
                repo_root=repo_root,
                build_dir=build_dir,
                model_name=model_name,
                checkpoint=checkpoint,
                ultralytics_repo=ultralytics_repo,
                yolov5_repo=yolov5_repo,
                input_tensor=input_tensor,
                output_path=output_dir / f"{model_name}.onnx",
            )
        graph_output = _run_graph_yolo(
            irt_module,
            onnx_path=onnx_path,
            input_tensor=input_tensor,
            backend_attr=backend_attr,
            device_attr=device.upper(),
            model_name=model_name,
        )
        _assert_yolo_outputs_close(torch_output, graph_output, label=label)
