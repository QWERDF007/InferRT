"""SAM：官方 PyTorch 前向与 InferRT pybind11 输出一致性测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers.manifest import assert_tensors_close
from helpers.model_integration import ensure_sam2_wts
from util import allocate_output_tensors

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SAM2_MODEL_NAME = "sam2_1_hiera_tiny"
SAM2_INPUT_SIZE = 1024
SAM2_MASK_SIZE = 256
SAM2_MAX_POINTS = 16


def _import_sam2_export_helpers(repo_root: Path):
    """导入 SAM2 导出脚本中的官方构建与预处理 helper。"""

    segmentation_samples = repo_root / "samples" / "model" / "segmentation"
    if str(segmentation_samples) not in sys.path:
        sys.path.insert(0, str(segmentation_samples))

    from gen_sam2_wts import SAM2_IMAGE_SIZE, build_sam2_without_hydra, preprocess_image

    return SAM2_IMAGE_SIZE, build_sam2_without_hydra, preprocess_image


def _make_center_point_prompt() -> tuple[np.ndarray, np.ndarray]:
    """构造与 segmentation sample 一致的中心正样本点提示。"""

    point_coords = np.zeros((1, SAM2_MAX_POINTS, 2, 1), dtype=np.float32)
    point_labels = np.full((1, SAM2_MAX_POINTS, 1, 1), -1.0, dtype=np.float32)
    point_coords[0, 0, :, 0] = SAM2_INPUT_SIZE / 2.0
    point_labels[0, 0, 0, 0] = 1.0
    return point_coords, point_labels


def _run_torch_sam2_reference(
    *,
    repo_root: Path,
    checkpoint: Path,
    sam2_root: Path,
    image_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """运行官方 SAM2 image encoder / prompt encoder / mask decoder 全链路。"""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SAM2 TensorRT parity requires CUDA")
    device = torch.device("cuda")

    sam2_image_size, build_sam2_without_hydra, preprocess_image = _import_sam2_export_helpers(repo_root)
    if sam2_image_size != SAM2_INPUT_SIZE:
        raise AssertionError(f"Unexpected SAM2 image size: {sam2_image_size}")

    model = build_sam2_without_hydra(SAM2_MODEL_NAME, checkpoint, str(sam2_root), device)
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
    torch.cuda.empty_cache()
    return inputs, outputs


def _run_inferrt_sam2(
    irt_module: Any,
    *,
    weights_path: Path,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """通过 InferRT pybind11 运行 SAM2.1 tiny 全链路。"""

    model = irt_module.create_model(SAM2_MODEL_NAME)
    # 数值一致性测试需要直接重建 engine，避免旧缓存掩盖网络定义变更。
    model.build(str(weights_path))

    input_names = list(model.input_tensor_names())
    output_names = list(model.output_tensor_names())
    assert input_names == ["image", "point_coords", "point_labels", "mask_input", "has_mask_input"]
    assert output_names == ["masks", "iou_predictions", "low_res_masks"]

    output_tensors = allocate_output_tensors(model, output_names)
    model_inputs = {name: np.ascontiguousarray(inputs[name], dtype=np.float32) for name in input_names}
    model.infer(model_inputs, output_tensors)
    return {name: np.asarray(output_tensors[name], dtype=np.float32) for name in output_names}


def test_sam2_pybind_matches_official_pytorch_forward(
    irt_module: Any,
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    default_image: Path,
    sam2_root: Path,
) -> None:
    """同一图片和点提示下，InferRT SAM2 输出应与官方 PyTorch 前向保持一致。

    Args:
        irt_module: 已导入的 ``inferrt_model_py`` 模块。
        repo_root: 仓库根目录。
        build_dir: CMake 构建目录。
        model_root: 真实模型根目录，默认 ``D:/Models``。
        default_image: 默认 dog 测试图。
        sam2_root: 本地 SAM2 仓库路径。
    """

    checkpoint = model_root / "sam" / "sam2.1_hiera_tiny.pt"
    weights = ensure_sam2_wts(
        repo_root=repo_root,
        build_dir=build_dir,
        checkpoint=checkpoint,
        sam2_root=sam2_root,
    )

    inputs, torch_outputs = _run_torch_sam2_reference(
        repo_root=repo_root,
        checkpoint=checkpoint,
        sam2_root=sam2_root,
        image_path=default_image,
    )
    inferrt_outputs = _run_inferrt_sam2(irt_module, weights_path=weights, inputs=inputs)

    assert_tensors_close(
        torch_outputs["masks"],
        inferrt_outputs["masks"],
        rtol=1e-3,
        atol=1e-2,
        name="masks",
    )
    assert_tensors_close(
        torch_outputs["low_res_masks"],
        inferrt_outputs["low_res_masks"],
        rtol=1e-3,
        atol=1e-2,
        name="low_res_masks",
    )
    assert_tensors_close(
        torch_outputs["iou_predictions"],
        inferrt_outputs["iou_predictions"],
        rtol=1e-3,
        atol=1e-3,
        name="iou_predictions",
    )
