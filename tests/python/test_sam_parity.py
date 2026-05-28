"""SAM：官方 PyTorch 前向与 InferRT pybind11 输出一致性测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers.manifest import assert_tensors_close
from helpers.model_integration import artifact_dir, ensure_sam2_wts, ensure_sam_v1_wts, is_fresh_against_all
from util import allocate_output_tensors

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SAM_V1_MODEL_NAME = "sam_vit_b"
SAM2_MODEL_NAME = "sam2_1_hiera_tiny"
SAM2_INPUT_SIZE = 1024
SAM2_MASK_SIZE = 256
SAM2_MAX_POINTS = 16
SAM_V1_MASK_RTOL = 1e-2
SAM_V1_MASK_ATOL = 3e-1
SAM_MASK_RTOL = 1e-3
SAM_MASK_ATOL = 1e-2
SAM_V1_IOU_RTOL = 5e-3
SAM_V1_IOU_ATOL = 3e-3
SAM_IOU_RTOL = 1e-3
SAM_IOU_ATOL = 1e-3
SAM_INPUT_NAMES = ["image", "point_coords", "point_labels", "mask_input", "has_mask_input"]
SAM_OUTPUT_NAMES = ["masks", "iou_predictions", "low_res_masks"]


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


def _runtime_device_pairs(compare_runtimes: list[str], compare_devices: list[str]) -> list[tuple[str, str]]:
    """根据 runtime/device 参数生成 SAM 可执行的后端组合。

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


def _torch_device() -> Any:
    """选择 PyTorch 参考前向使用的设备。

    Returns:
        CUDA 可用时返回 ``cuda``，否则返回 ``cpu``。
    """

    torch = pytest.importorskip("torch")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _import_sam_v1_export_helpers(repo_root: Path):
    """导入 SAM v1 导出脚本中的官方构建与预处理 helper。"""

    segmentation_samples = repo_root / "samples" / "model" / "segmentation"
    if str(segmentation_samples) not in sys.path:
        sys.path.insert(0, str(segmentation_samples))

    from gen_sam_wts import SAM_IMAGE_SIZE, import_segment_anything, preprocess_image

    return SAM_IMAGE_SIZE, import_segment_anything, preprocess_image


def _import_sam2_export_helpers(repo_root: Path):
    """导入 SAM2 导出脚本中的官方构建与预处理 helper。"""

    segmentation_samples = repo_root / "samples" / "model" / "segmentation"
    if str(segmentation_samples) not in sys.path:
        sys.path.insert(0, str(segmentation_samples))

    from gen_sam2_wts import SAM2_IMAGE_SIZE, build_sam2_without_hydra, preprocess_image

    return SAM2_IMAGE_SIZE, build_sam2_without_hydra, preprocess_image


def _make_point_prompt(x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
    """构造与 segmentation sample 输入契约一致的单个正样本点提示。"""

    point_coords = np.zeros((1, SAM2_MAX_POINTS, 2, 1), dtype=np.float32)
    point_labels = np.full((1, SAM2_MAX_POINTS, 1, 1), -1.0, dtype=np.float32)
    point_coords[0, 0, :, 0] = (x, y)
    point_labels[0, 0, 0, 0] = 1.0
    return point_coords, point_labels


def _make_center_point_prompt() -> tuple[np.ndarray, np.ndarray]:
    """构造与 SAM2 fixed-square preprocessing 一致的中心正样本点提示。"""

    return _make_point_prompt(SAM2_INPUT_SIZE / 2.0, SAM2_INPUT_SIZE / 2.0)


def _load_torch_sam_v1_model(
    *,
    repo_root: Path,
    checkpoint: Path,
    sam_root: Path,
    device: Any,
) -> Any:
    """加载官方 SAM v1 PyTorch 模型。

    Args:
        repo_root: InferRT 仓库根目录，用于导入 sample helper。
        checkpoint: SAM v1 checkpoint 路径。
        sam_root: 官方 segment-anything 仓库路径。
        device: PyTorch 设备。

    Returns:
        已加载并切到 eval 模式的 SAM v1 模型。
    """

    _sam_image_size, import_segment_anything, _preprocess_image = _import_sam_v1_export_helpers(repo_root)
    registry = import_segment_anything(str(sam_root) if sam_root.exists() else None)
    model = registry["vit_b"](checkpoint=str(checkpoint)).to(device)
    model.eval()
    return model


def _load_torch_sam2_model(
    *,
    repo_root: Path,
    checkpoint: Path,
    sam2_root: Path,
    device: Any,
) -> Any:
    """加载官方 SAM2 PyTorch 模型。

    Args:
        repo_root: InferRT 仓库根目录，用于导入 sample helper。
        checkpoint: SAM2 checkpoint 路径。
        sam2_root: 官方 SAM2 仓库路径。
        device: PyTorch 设备。

    Returns:
        已加载并切到 eval 模式的 SAM2 模型。
    """

    _sam2_image_size, build_sam2_without_hydra, _preprocess_image = _import_sam2_export_helpers(repo_root)
    return build_sam2_without_hydra(SAM2_MODEL_NAME, checkpoint, str(sam2_root), device)


def _run_torch_sam_v1_reference(
    *,
    repo_root: Path,
    checkpoint: Path,
    sam_root: Path,
    image_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """运行官方 SAM v1 image encoder / prompt encoder / mask decoder 全链路。"""

    torch = pytest.importorskip("torch")
    device = _torch_device()

    sam_image_size, _import_segment_anything, preprocess_image = _import_sam_v1_export_helpers(repo_root)
    if sam_image_size != SAM2_INPUT_SIZE:
        raise AssertionError(f"Unexpected SAM image size: {sam_image_size}")

    model = _load_torch_sam_v1_model(repo_root=repo_root, checkpoint=checkpoint, sam_root=sam_root, device=device)

    image, _original_size, resized_size = preprocess_image(image_path, device)
    resized_h, resized_w = resized_size
    point_coords_np, point_labels_np = _make_point_prompt(resized_w / 2.0, resized_h / 2.0)
    point_coords = torch.from_numpy(point_coords_np.reshape(1, SAM2_MAX_POINTS, 2)).to(device=device)
    point_labels = torch.from_numpy(point_labels_np.reshape(1, SAM2_MAX_POINTS)).to(device=device, dtype=torch.int64)

    with torch.inference_mode():
        image_embeddings = model.image_encoder(image)
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    inputs = {
        "image": image.detach().cpu().numpy().astype(np.float32, copy=False),
        "point_coords": point_coords_np,
        "point_labels": point_labels_np,
        "mask_input": np.zeros((1, 1, SAM2_MASK_SIZE, SAM2_MASK_SIZE), dtype=np.float32),
        "has_mask_input": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }
    outputs = {
        "masks": low_res_masks.detach().cpu().numpy().astype(np.float32, copy=False),
        "low_res_masks": low_res_masks.detach().cpu().numpy().astype(np.float32, copy=False),
        "iou_predictions": iou_predictions.detach().cpu().numpy().astype(np.float32, copy=False).reshape(1, 3, 1, 1),
    }

    del model, image, point_coords, point_labels, image_embeddings, low_res_masks, iou_predictions
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return inputs, outputs


def _run_torch_sam2_reference(
    *,
    repo_root: Path,
    checkpoint: Path,
    sam2_root: Path,
    image_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """运行官方 SAM2 image encoder / prompt encoder / mask decoder 全链路。"""

    torch = pytest.importorskip("torch")
    device = _torch_device()

    sam2_image_size, _build_sam2_without_hydra, preprocess_image = _import_sam2_export_helpers(repo_root)
    if sam2_image_size != SAM2_INPUT_SIZE:
        raise AssertionError(f"Unexpected SAM2 image size: {sam2_image_size}")

    model = _load_torch_sam2_model(repo_root=repo_root, checkpoint=checkpoint, sam2_root=sam2_root, device=device)
    image, _original_size = preprocess_image(image_path, device)
    point_coords_np, point_labels_np = _make_center_point_prompt()
    point_coords = torch.from_numpy(point_coords_np.reshape(1, SAM2_MAX_POINTS, 2)).to(device=device)
    point_labels = torch.from_numpy(point_labels_np.reshape(1, SAM2_MAX_POINTS)).to(device=device, dtype=torch.int64)

    with torch.inference_mode():
        backbone_out = model.forward_image(image)
        sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
            image_embeddings=backbone_out["vision_features"],
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=[backbone_out["backbone_fpn"][0], backbone_out["backbone_fpn"][1]],
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    inputs = {
        "image": image.detach().cpu().numpy().astype(np.float32, copy=False),
        "point_coords": point_coords_np,
        "point_labels": point_labels_np,
        "mask_input": np.zeros((1, 1, SAM2_MASK_SIZE, SAM2_MASK_SIZE), dtype=np.float32),
        "has_mask_input": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }
    outputs = {
        "masks": low_res_masks.detach().cpu().numpy().astype(np.float32, copy=False),
        "low_res_masks": low_res_masks.detach().cpu().numpy().astype(np.float32, copy=False),
        "iou_predictions": iou_predictions.detach().cpu().numpy().astype(np.float32, copy=False).reshape(1, 3, 1, 1),
    }

    del model, image, point_coords, point_labels, low_res_masks, iou_predictions
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return inputs, outputs


def _run_inferrt_sam(
    irt_module: Any,
    *,
    model_name: str,
    weights_path: Path,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """通过 InferRT pybind11 运行 SAM/SAM2 全链路。"""

    config = irt_module.ModelConfig()
    config.backend = irt_module.ModelBackend.TENSORRT
    config.device = irt_module.ModelDevice.GPU
    model = irt_module.create_model(model_name, config=config)
    # Rebuild the engine for parity so stale TensorRT caches cannot hide graph changes.
    _build_sam_model_or_skip(model, weights_path, label=f"TENSORRT/GPU/{model_name}")

    input_names = list(model.input_tensor_names())
    output_names = list(model.output_tensor_names())
    assert input_names == SAM_INPUT_NAMES
    assert output_names == SAM_OUTPUT_NAMES

    output_tensors = allocate_output_tensors(model, output_names)
    model_inputs = {name: np.ascontiguousarray(inputs[name], dtype=np.float32) for name in input_names}
    model.infer(model_inputs, output_tensors)
    return {name: np.asarray(output_tensors[name], dtype=np.float32) for name in output_names}


def _build_sam_model_or_skip(model: Any, model_file: Path, *, label: str) -> None:
    """构建 TensorRT 原生 SAM 模型，后端不可用时跳过。

    Args:
        model: InferRT Python 模型对象。
        model_file: ``.wts`` 权重路径。
        label: skip 消息中的后端/模型标签。
    """

    try:
        model.build(str(model_file))
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


def _build_graph_model_or_skip(model: Any, model_file: Path, *, label: str) -> None:
    """构建 ONNX/OpenVINO 图模型，后端不可用时跳过。

    Args:
        model: InferRT Python 模型对象。
        model_file: ONNX 或 OpenVINO IR 路径。
        label: skip 消息中的后端/模型标签。
    """

    try:
        model.build_or_load(str(model_file))
    except Exception as exc:
        message = str(exc)
        unavailable_markers = (
            "backend is not enabled",
            "Failed to load ONNX Runtime DLL",
            "OpenVINO backend is not enabled",
            "OpenVINO error while loading",
            "CUDA provider",
            "Device with \"GPU\" name is not registered",
            "Cannot get DEVICE_PROPERTIES",
        )
        if any(marker in message for marker in unavailable_markers):
            pytest.skip(f"{label} backend unavailable: {message}")
        raise


def _sam_input_tensors(inputs: dict[str, np.ndarray], device: Any) -> tuple[Any, Any, Any, Any, Any]:
    """按 SAM 输入契约将 NumPy 输入转换为 PyTorch 张量元组。

    Args:
        inputs: 以 SAM 输入名为 key 的 NumPy 输入字典。
        device: PyTorch 设备。

    Returns:
        按 ``SAM_INPUT_NAMES`` 排列的张量元组。
    """

    torch = pytest.importorskip("torch")
    return tuple(
        torch.from_numpy(np.ascontiguousarray(inputs[name], dtype=np.float32)).to(device=device)
        for name in SAM_INPUT_NAMES
    )


def _export_sam_v1_onnx(
    *,
    repo_root: Path,
    checkpoint: Path,
    sam_root: Path,
    inputs: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """导出保持 InferRT SAM v1 输入/输出契约的 ONNX 图。

    Args:
        repo_root: InferRT 仓库根目录。
        checkpoint: SAM v1 checkpoint 路径。
        sam_root: 官方 segment-anything 仓库路径。
        inputs: 导出使用的示例输入。
        output_path: ONNX 输出路径。
    """

    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    device = torch.device("cpu")
    model = _load_torch_sam_v1_model(repo_root=repo_root, checkpoint=checkpoint, sam_root=sam_root, device=device)

    class _SAMV1Wrapper(torch.nn.Module):
        """把官方 SAM v1 拆分前向包装为 5 输入、3 输出的 ONNX 模块。"""

        def __init__(self, wrapped: Any) -> None:
            """保存官方 SAM v1 模型实例。

            Args:
                wrapped: 官方 SAM v1 PyTorch 模型。
            """

            super().__init__()
            self.wrapped = wrapped

        def forward(self, image, point_coords, point_labels, mask_input, has_mask_input):
            """执行 SAM v1 图导出的完整前向。

            Args:
                image: 预处理后的图像张量。
                point_coords: 点提示坐标张量。
                point_labels: 点提示标签张量。
                mask_input: mask 输入占位张量。
                has_mask_input: mask 标志占位张量。

            Returns:
                ``masks``、``iou_predictions``、``low_res_masks`` 三个输出。
            """

            point_coords = point_coords.reshape(point_coords.shape[0], SAM2_MAX_POINTS, 2)
            point_labels = point_labels.reshape(point_labels.shape[0], SAM2_MAX_POINTS).to(torch.int64)
            image_embeddings = self.wrapped.image_encoder(image)
            sparse_embeddings, dense_embeddings = self.wrapped.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            low_res_masks, iou_predictions = self.wrapped.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.wrapped.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            keep_inputs = (mask_input.sum() + has_mask_input.sum()) * 0.0
            low_res_masks = low_res_masks + keep_inputs
            iou_predictions = iou_predictions.reshape(iou_predictions.shape[0], 3, 1, 1) + keep_inputs
            return low_res_masks, iou_predictions, low_res_masks

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _SAMV1Wrapper(model).eval()
    torch.onnx.export(
        wrapper,
        _sam_input_tensors(inputs, device),
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=SAM_INPUT_NAMES,
        output_names=SAM_OUTPUT_NAMES,
    )


def _export_sam2_onnx(
    *,
    repo_root: Path,
    checkpoint: Path,
    sam2_root: Path,
    inputs: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """导出保持 InferRT SAM2 输入/输出契约的 ONNX 图。

    Args:
        repo_root: InferRT 仓库根目录。
        checkpoint: SAM2 checkpoint 路径。
        sam2_root: 官方 SAM2 仓库路径。
        inputs: 导出使用的示例输入。
        output_path: ONNX 输出路径。
    """

    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    device = torch.device("cpu")
    model = _load_torch_sam2_model(repo_root=repo_root, checkpoint=checkpoint, sam2_root=sam2_root, device=device)

    class _SAM2Wrapper(torch.nn.Module):
        """把官方 SAM2 拆分前向包装为 5 输入、3 输出的 ONNX 模块。"""

        def __init__(self, wrapped: Any) -> None:
            """保存官方 SAM2 模型实例。

            Args:
                wrapped: 官方 SAM2 PyTorch 模型。
            """

            super().__init__()
            self.wrapped = wrapped

        def forward(self, image, point_coords, point_labels, mask_input, has_mask_input):
            """执行 SAM2 图导出的完整前向。

            Args:
                image: 预处理后的图像张量。
                point_coords: 点提示坐标张量。
                point_labels: 点提示标签张量。
                mask_input: mask 输入占位张量。
                has_mask_input: mask 标志占位张量。

            Returns:
                ``masks``、``iou_predictions``、``low_res_masks`` 三个输出。
            """

            point_coords = point_coords.reshape(point_coords.shape[0], SAM2_MAX_POINTS, 2)
            point_labels = point_labels.reshape(point_labels.shape[0], SAM2_MAX_POINTS).to(torch.int64)
            backbone_out = self.wrapped.forward_image(image)
            sparse_embeddings, dense_embeddings = self.wrapped.sam_prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            low_res_masks, iou_predictions, _, _ = self.wrapped.sam_mask_decoder(
                image_embeddings=backbone_out["vision_features"],
                image_pe=self.wrapped.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
                high_res_features=[backbone_out["backbone_fpn"][0], backbone_out["backbone_fpn"][1]],
            )
            keep_inputs = (mask_input.sum() + has_mask_input.sum()) * 0.0
            low_res_masks = low_res_masks + keep_inputs
            iou_predictions = iou_predictions.reshape(iou_predictions.shape[0], 3, 1, 1) + keep_inputs
            return low_res_masks, iou_predictions, low_res_masks

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _SAM2Wrapper(model).eval()
    torch.onnx.export(
        wrapper,
        _sam_input_tensors(inputs, device),
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=SAM_INPUT_NAMES,
        output_names=SAM_OUTPUT_NAMES,
    )


def _ensure_sam_onnx(
    *,
    output_path: Path,
    checkpoint: Path,
    export_fn: Any,
) -> Path:
    """确保 SAM ONNX 产物存在且不早于 checkpoint。

    Args:
        output_path: ONNX 输出路径。
        checkpoint: 原始 checkpoint 路径。
        export_fn: 执行实际导出的回调。

    Returns:
        ONNX 模型路径。
    """

    if is_fresh_against_all(output_path, [checkpoint]):
        return output_path
    export_fn(output_path)
    return output_path


def _run_graph_sam(
    irt_module: Any,
    *,
    model_path: Path,
    inputs: dict[str, np.ndarray],
    backend_attr: str,
    device_attr: str,
    label: str,
) -> dict[str, np.ndarray]:
    """使用 ONNX Runtime 或 OpenVINO 后端执行 SAM 图模型。

    Args:
        irt_module: 已导入的 ``inferrt_model_py`` 模块。
        model_path: ONNX 或 OpenVINO IR 模型路径。
        inputs: SAM 输入字典。
        backend_attr: 后端枚举属性名。
        device_attr: 设备枚举属性名。
        label: skip 消息中的后端/模型标签。

    Returns:
        输出名到 NumPy 数组的映射。
    """

    config = irt_module.ModelConfig()
    config.backend = getattr(irt_module.ModelBackend, backend_attr)
    config.device = getattr(irt_module.ModelDevice, device_attr)
    config.output_tensor_names = SAM_OUTPUT_NAMES

    model = irt_module.create_model("onnx", config=config)
    _build_graph_model_or_skip(model, model_path, label=label)

    input_names = list(model.input_tensor_names())
    model_inputs = {name: np.ascontiguousarray(inputs[name], dtype=np.float32) for name in input_names}
    outputs = model.infer(model_inputs, None)
    if isinstance(outputs, np.ndarray):
        return {SAM_OUTPUT_NAMES[0]: np.asarray(outputs, dtype=np.float32)}
    return {str(name): np.asarray(value, dtype=np.float32) for name, value in dict(outputs).items()}


def _assert_sam_outputs_close(
    reference: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    mask_rtol: float,
    mask_atol: float,
    iou_rtol: float,
    iou_atol: float,
    label: str,
) -> None:
    """比较 SAM 三个输出与 PyTorch 参考是否一致。

    Args:
        reference: PyTorch 参考输出。
        actual: InferRT 后端输出。
        mask_rtol: mask 输出相对误差容差。
        mask_atol: mask 输出绝对误差容差。
        iou_rtol: IoU 输出相对误差容差。
        iou_atol: IoU 输出绝对误差容差。
        label: 断言失败时使用的标签前缀。
    """

    assert sorted(actual) == sorted(SAM_OUTPUT_NAMES), f"{label} output names mismatch"
    assert_tensors_close(
        reference["masks"],
        actual["masks"],
        rtol=mask_rtol,
        atol=mask_atol,
        name=f"{label}.masks",
    )
    assert_tensors_close(
        reference["low_res_masks"],
        actual["low_res_masks"],
        rtol=mask_rtol,
        atol=mask_atol,
        name=f"{label}.low_res_masks",
    )
    assert_tensors_close(
        reference["iou_predictions"],
        actual["iou_predictions"],
        rtol=iou_rtol,
        atol=iou_atol,
        name=f"{label}.iou_predictions",
    )


def test_sam_v1_pybind_matches_official_pytorch_forward(
    compare_runtimes: list[str],
    compare_devices: list[str],
    irt_module: Any,
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    default_image: Path,
    sam_root: Path,
) -> None:
    """按所选后端运行 SAM v1，并与官方 PyTorch 输出逐项比较。"""

    if not compare_runtimes:
        pytest.skip("No runtimes selected; pass --inferrt-compare-runtime=TensorRT,onnx,openvino")
    runtime_device_pairs = _runtime_device_pairs(compare_runtimes, compare_devices)
    if not runtime_device_pairs:
        pytest.skip("No compatible runtime/device pairs selected; TensorRT requires --inferrt-compare-devices=gpu")

    checkpoint = model_root / "sam" / "sam_vit_b_01ec64.pth"
    if not checkpoint.exists():
        pytest.skip(f"SAM ViT-B checkpoint not found: {checkpoint}")
    inputs, torch_outputs = _run_torch_sam_v1_reference(
        repo_root=repo_root,
        checkpoint=checkpoint,
        sam_root=sam_root,
        image_path=default_image,
    )

    output_dir = artifact_dir(build_dir, "sam_parity") / SAM_V1_MODEL_NAME
    weights: Path | None = None
    onnx_path: Path | None = None

    for backend_attr, device in runtime_device_pairs:
        label = f"{SAM_V1_MODEL_NAME}.{_runtime_label(backend_attr)}.{device}"
        if backend_attr == "TENSORRT":
            if weights is None:
                weights = ensure_sam_v1_wts(
                    repo_root=repo_root,
                    build_dir=build_dir,
                    checkpoint=checkpoint,
                    sam_root=sam_root,
                )
            outputs = _run_inferrt_sam(
                irt_module,
                model_name=SAM_V1_MODEL_NAME,
                weights_path=weights,
                inputs=inputs,
            )
        else:
            if onnx_path is None:
                onnx_path = _ensure_sam_onnx(
                    output_path=output_dir / f"{SAM_V1_MODEL_NAME}.onnx",
                    checkpoint=checkpoint,
                    export_fn=lambda output_path: _export_sam_v1_onnx(
                        repo_root=repo_root,
                        checkpoint=checkpoint,
                        sam_root=sam_root,
                        inputs=inputs,
                        output_path=output_path,
                    ),
                )
            outputs = _run_graph_sam(
                irt_module,
                model_path=onnx_path,
                inputs=inputs,
                backend_attr=backend_attr,
                device_attr=device.upper(),
                label=label,
            )
        _assert_sam_outputs_close(
            torch_outputs,
            outputs,
            mask_rtol=SAM_V1_MASK_RTOL,
            mask_atol=SAM_V1_MASK_ATOL,
            iou_rtol=SAM_V1_IOU_RTOL,
            iou_atol=SAM_V1_IOU_ATOL,
            label=label,
        )

def test_sam2_pybind_matches_official_pytorch_forward(
    compare_runtimes: list[str],
    compare_devices: list[str],
    irt_module: Any,
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    default_image: Path,
    sam2_root: Path,
) -> None:
    """按所选后端运行 SAM2，并与官方 PyTorch 输出逐项比较。"""

    if not compare_runtimes:
        pytest.skip("No runtimes selected; pass --inferrt-compare-runtime=TensorRT,onnx,openvino")
    runtime_device_pairs = _runtime_device_pairs(compare_runtimes, compare_devices)
    if not runtime_device_pairs:
        pytest.skip("No compatible runtime/device pairs selected; TensorRT requires --inferrt-compare-devices=gpu")

    checkpoint = model_root / "sam" / "sam2.1_hiera_tiny.pt"
    if not checkpoint.exists():
        pytest.skip(f"SAM2.1 Hiera-Tiny checkpoint not found: {checkpoint}")
    inputs, torch_outputs = _run_torch_sam2_reference(
        repo_root=repo_root,
        checkpoint=checkpoint,
        sam2_root=sam2_root,
        image_path=default_image,
    )

    output_dir = artifact_dir(build_dir, "sam_parity") / SAM2_MODEL_NAME
    weights: Path | None = None
    onnx_path: Path | None = None

    for backend_attr, device in runtime_device_pairs:
        label = f"{SAM2_MODEL_NAME}.{_runtime_label(backend_attr)}.{device}"
        if backend_attr == "TENSORRT":
            if weights is None:
                weights = ensure_sam2_wts(
                    repo_root=repo_root,
                    build_dir=build_dir,
                    checkpoint=checkpoint,
                    sam2_root=sam2_root,
                )
            outputs = _run_inferrt_sam(
                irt_module,
                model_name=SAM2_MODEL_NAME,
                weights_path=weights,
                inputs=inputs,
            )
        else:
            if onnx_path is None:
                onnx_path = _ensure_sam_onnx(
                    output_path=output_dir / f"{SAM2_MODEL_NAME}.onnx",
                    checkpoint=checkpoint,
                    export_fn=lambda output_path: _export_sam2_onnx(
                        repo_root=repo_root,
                        checkpoint=checkpoint,
                        sam2_root=sam2_root,
                        inputs=inputs,
                        output_path=output_path,
                    ),
                )
            outputs = _run_graph_sam(
                irt_module,
                model_path=onnx_path,
                inputs=inputs,
                backend_attr=backend_attr,
                device_attr=device.upper(),
                label=label,
            )
        _assert_sam_outputs_close(
            torch_outputs,
            outputs,
            mask_rtol=SAM_MASK_RTOL,
            mask_atol=SAM_MASK_ATOL,
            iou_rtol=SAM_IOU_RTOL,
            iou_atol=SAM_IOU_ATOL,
            label=label,
        )
