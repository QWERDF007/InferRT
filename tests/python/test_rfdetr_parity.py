"""RF-DETR：上游 PyTorch 前向与 InferRT 原生 TensorRT 图输出一致性测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers.manifest import assert_tensors_close
from helpers.model_integration import ensure_rfdetr_wts
from util import allocate_output_tensors, preprocess_image

pytestmark = [pytest.mark.integration, pytest.mark.slow]

RFDETR_CASES = (
    ("rfdetr_nano", "rf-detr-nano.pth", 384),
    ("rfdetr_small", "rf-detr-small.pth", 512),
)

RFDETR_STABLE_PREFIX = 40
RFDETR_PREFIX_LOGITS_MAX_ATOL = 0.25
RFDETR_PREFIX_LOGITS_MEAN_ATOL = 0.02
RFDETR_FULL_DETS_MEAN_ATOL = 0.08
RFDETR_FULL_LOGITS_MEAN_ATOL = 0.30


def _assert_mean_abs_below(reference: np.ndarray, actual: np.ndarray, *, atol: float, name: str) -> None:
    """断言两路张量的平均绝对误差低于阈值，并在失败时输出最大误差。"""

    ref = np.asarray(reference, dtype=np.float32)
    act = np.asarray(actual, dtype=np.float32)
    if ref.shape != act.shape:
        raise AssertionError(f"Shape mismatch for {name}: reference={ref.shape}, actual={act.shape}")
    diff = np.abs(act - ref)
    mean_abs = float(diff.mean())
    if mean_abs > atol:
        raise AssertionError(f"Tensor '{name}' mean_abs={mean_abs:.6g} exceeds {atol:.6g}; max_abs={diff.max():.6g}")


def _assert_max_mean_abs_below(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    max_atol: float,
    mean_atol: float,
    name: str,
) -> None:
    """断言两路张量同时满足最大绝对误差和平均绝对误差阈值。"""

    ref = np.asarray(reference, dtype=np.float32)
    act = np.asarray(actual, dtype=np.float32)
    if ref.shape != act.shape:
        raise AssertionError(f"Shape mismatch for {name}: reference={ref.shape}, actual={act.shape}")
    diff = np.abs(act - ref)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    if max_abs > max_atol or mean_abs > mean_atol:
        raise AssertionError(
            f"Tensor '{name}' mismatch: max_abs={max_abs:.6g} (limit {max_atol:.6g}), "
            f"mean_abs={mean_abs:.6g} (limit {mean_atol:.6g})"
        )


def _build_model_or_skip(model: Any, model_file: Path, *, label: str) -> None:
    """构建或加载 InferRT 模型；后端不可用时跳过当前集成测试。"""

    try:
        model.build_or_load(str(model_file))
    except Exception as exc:
        message = str(exc)
        unavailable_markers = (
            "backend is not enabled",
            "CUDA driver",
            "CUDA failure",
            "CUDA error",
            "Failed to initialize CUDA",
            "Cannot get DEVICE_PROPERTIES",
        )
        if any(marker in message for marker in unavailable_markers):
            pytest.skip(f"{label} backend unavailable: {message}")
        raise


def _torch_device() -> Any:
    """选择 PyTorch reference 设备，优先使用 CUDA。"""

    torch = pytest.importorskip("torch")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_torch_rfdetr(*, rfdetr_root: Path, checkpoint: Path, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
    """运行上游 RF-DETR PyTorch export 前向，返回原始 boxes/logits 输出。"""

    torch = pytest.importorskip("torch")
    from rfdetr_compat import configure_rfdetr_import
    from rfdetr_gen_wts import infer_checkpoint_num_classes

    configure_rfdetr_import(rfdetr_root)
    try:
        from rfdetr import from_checkpoint  # type: ignore
    except ImportError as exc:
        pytest.skip(f"RF-DETR import failed: {exc}")

    device = _torch_device()
    kwargs: dict[str, object] = {"device": str(device)}
    num_classes = infer_checkpoint_num_classes(checkpoint)
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    wrapper = from_checkpoint(checkpoint, **kwargs)
    model = wrapper.model.model.eval().to(device)
    model.export()

    batch = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32)).to(device=device)
    with torch.inference_mode():
        outputs = model(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise AssertionError(f"Unexpected RF-DETR PyTorch output type: {type(outputs)!r}")

    boxes, logits = outputs
    return {
        "dets": boxes.detach().cpu().numpy().astype(np.float32, copy=False),
        "labels": logits.detach().cpu().numpy().astype(np.float32, copy=False),
    }


def _run_inferrt_rfdetr(
    irt_module: Any,
    *,
    model_name: str,
    weights_path: Path,
    input_tensor: np.ndarray,
    resolution: int,
) -> dict[str, np.ndarray]:
    """通过 InferRT pybind11 运行 RF-DETR 原生 TensorRT engine。"""

    config = irt_module.ModelConfig()
    config.backend = irt_module.ModelBackend.TENSORRT
    config.device = irt_module.ModelDevice.GPU
    config.input_shape = [1, 3, resolution, resolution]

    model = irt_module.create_model(model_name, config=config)
    _build_model_or_skip(model, weights_path, label=f"TENSORRT/GPU/{model_name}")

    input_names = list(model.input_tensor_names())
    output_names = list(model.output_tensor_names())
    assert input_names == ["input"]
    assert output_names == ["dets", "labels"]

    outputs = allocate_output_tensors(model, output_names)
    model.infer({input_names[0]: np.ascontiguousarray(input_tensor, dtype=np.float32)}, outputs)
    return {name: np.asarray(outputs[name], dtype=np.float32) for name in output_names}


@pytest.mark.parametrize(("model_name", "checkpoint_name", "resolution"), RFDETR_CASES)
def test_rfdetr_native_tensorrt_matches_pytorch_export(
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    rfdetr_root: Path,
    default_image: Path,
    irt_module: Any,
    compare_runtimes: list[str],
    tolerances: tuple[float, float],
    model_name: str,
    checkpoint_name: str,
    resolution: int,
) -> None:
    """同一输入下，InferRT RF-DETR 原生 TensorRT 输出应接近上游 PyTorch export 输出。"""

    if "TENSORRT" not in compare_runtimes:
        pytest.skip("RF-DETR native parity requires --inferrt-compare-runtime=tensorrt")

    checkpoint = model_root / "rfdetr" / checkpoint_name
    weights_path = ensure_rfdetr_wts(
        repo_root=repo_root,
        build_dir=build_dir,
        model_name=model_name,
        checkpoint=checkpoint,
        rfdetr_root=rfdetr_root,
    )
    input_tensor = preprocess_image(default_image, image_size=(resolution, resolution))

    torch_outputs = _run_torch_rfdetr(rfdetr_root=rfdetr_root, checkpoint=checkpoint, input_tensor=input_tensor)
    inferrt_outputs = _run_inferrt_rfdetr(
        irt_module,
        model_name=model_name,
        weights_path=weights_path,
        input_tensor=input_tensor,
        resolution=resolution,
    )

    rtol, atol = tolerances
    # RF-DETR 的 DINO backbone 在 PyTorch SDPA 与 TensorRT 展开图之间存在逐层数值累积；
    # 低置信候选的 two-stage TopK 顺序会因此漂移。检测语义更关注排序稳定的前缀候选，
    # 同时用全量平均误差约束尾部候选没有整体跑偏。
    assert_tensors_close(
        torch_outputs["dets"][:, :RFDETR_STABLE_PREFIX],
        inferrt_outputs["dets"][:, :RFDETR_STABLE_PREFIX],
        rtol=rtol,
        atol=atol,
        name=f"{model_name}.dets.top{RFDETR_STABLE_PREFIX}",
    )
    _assert_max_mean_abs_below(
        torch_outputs["labels"][:, :RFDETR_STABLE_PREFIX],
        inferrt_outputs["labels"][:, :RFDETR_STABLE_PREFIX],
        max_atol=RFDETR_PREFIX_LOGITS_MAX_ATOL,
        mean_atol=RFDETR_PREFIX_LOGITS_MEAN_ATOL,
        name=f"{model_name}.labels.top{RFDETR_STABLE_PREFIX}",
    )
    _assert_mean_abs_below(
        torch_outputs["dets"],
        inferrt_outputs["dets"],
        atol=RFDETR_FULL_DETS_MEAN_ATOL,
        name=f"{model_name}.dets.full",
    )
    _assert_mean_abs_below(
        torch_outputs["labels"],
        inferrt_outputs["labels"],
        atol=RFDETR_FULL_LOGITS_MEAN_ATOL,
        name=f"{model_name}.labels.full",
    )
