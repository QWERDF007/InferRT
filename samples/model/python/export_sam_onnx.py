from __future__ import annotations

import argparse
import io
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np
import torch

from gen_sam_wts import (
    SAM2_IMAGE_SIZE,
    SAM2_MODEL_CONFIGS,
    SAM_IMAGE_SIZE,
    build_sam2_without_hydra,
    import_segment_anything,
)


SAM_MASK_SIZE = 256
SAM_MAX_POINTS = 16
SAM_INPUT_NAMES = ["image", "point_coords", "point_labels", "mask_input", "has_mask_input"]
SAM_OUTPUT_NAMES = ["masks", "iou_predictions", "low_res_masks"]
SAM_V1_ALIASES = {
    "sam": "default",
    "sam_vit_b": "vit_b",
    "sam_vit_l": "vit_l",
    "sam_vit_h": "vit_h",
    "default": "default",
    "vit_b": "vit_b",
    "vit_l": "vit_l",
    "vit_h": "vit_h",
}
SAM2_MODEL_NAMES = tuple(SAM2_MODEL_CONFIGS)
SAM3_MODEL_NAMES = ("sam3", "sam3_image")
SAM_MODEL_NAMES = tuple(SAM_V1_ALIASES) + SAM2_MODEL_NAMES + SAM3_MODEL_NAMES


def configure_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            continue
        except (AttributeError, ValueError):
            pass

        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            setattr(sys, stream_name, io.TextIOWrapper(buffer, encoding="utf-8", errors="replace"))


class SAMV1OnnxWrapper(torch.nn.Module):
    """Wrap official SAM v1 modules into the InferRT SAM graph contract."""

    def __init__(self, wrapped: Any) -> None:
        super().__init__()
        self.wrapped = wrapped

    def forward(self, image, point_coords, point_labels, mask_input, has_mask_input):
        point_coords = point_coords.reshape(point_coords.shape[0], SAM_MAX_POINTS, 2)
        point_labels = point_labels.reshape(point_labels.shape[0], SAM_MAX_POINTS).to(torch.int64)
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


class SAM2OnnxWrapper(torch.nn.Module):
    """Wrap official SAM2 modules into the InferRT SAM graph contract."""

    def __init__(self, wrapped: Any) -> None:
        super().__init__()
        self.wrapped = wrapped

    def forward(self, image, point_coords, point_labels, mask_input, has_mask_input):
        point_coords = point_coords.reshape(point_coords.shape[0], SAM_MAX_POINTS, 2)
        point_labels = point_labels.reshape(point_labels.shape[0], SAM_MAX_POINTS).to(torch.int64)

        backbone_out = self.wrapped.forward_image(image)
        sparse_embeddings, dense_embeddings = self.wrapped.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions, _, _ = self.wrapped.sam_mask_decoder(
            image_embeddings=self._image_embedding(backbone_out),
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

    def _image_embedding(self, backbone_out: dict[str, torch.Tensor]) -> torch.Tensor:
        image_embedding = backbone_out["vision_features"]
        if getattr(self.wrapped, "directly_add_no_mem_embed", False):
            no_mem_embed = self.wrapped.no_mem_embed.reshape(1, 1, -1).permute(0, 2, 1).reshape(1, -1, 1, 1)
            image_embedding = image_embedding + no_mem_embed
        return image_embedding


class SAM3OnnxWrapper(torch.nn.Module):
    """Wrap Transformers SAM3 into the same five-input/three-output graph shape."""

    def __init__(self, wrapped: Any, text_embeds: torch.Tensor, image_size: int) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.image_size = float(image_size)
        self.register_buffer("text_embeds", text_embeds.detach())

    @staticmethod
    def _first_three_masks(masks: torch.Tensor) -> torch.Tensor:
        if masks.dim() == 5:
            masks = masks[:, 0]
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        masks = masks[:, :3]
        if masks.shape[1] == 1:
            masks = masks.repeat(1, 3, 1, 1)
        elif masks.shape[1] == 2:
            masks = torch.cat([masks, masks[:, -1:]], dim=1)
        return masks

    @staticmethod
    def _first_three_scores(scores: torch.Tensor | None, masks: torch.Tensor) -> torch.Tensor:
        if scores is None:
            return masks.new_zeros((masks.shape[0], 3, 1, 1))
        if scores.dim() == 3:
            scores = scores.max(dim=-1).values
        if scores.dim() == 1:
            scores = scores.reshape(scores.shape[0], 1)
        scores = scores[:, :3]
        if scores.shape[1] == 1:
            scores = scores.repeat(1, 3)
        elif scores.shape[1] == 2:
            scores = torch.cat([scores, scores[:, -1:]], dim=1)
        return scores.reshape(scores.shape[0], 3, 1, 1)

    def _box_prompt(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        batch = point_coords.shape[0]
        box_xy0 = point_coords[:, 1, :] / self.image_size
        box_xy1 = point_coords[:, 2, :] / self.image_size
        min_xy = torch.minimum(box_xy0, box_xy1).clamp(0.0, 1.0)
        max_xy = torch.maximum(box_xy0, box_xy1).clamp(0.0, 1.0)
        box_cxcy = (min_xy + max_xy) * 0.5
        box_wh = (max_xy - min_xy).clamp_min(1.0 / self.image_size)
        box = torch.cat([box_cxcy, box_wh], dim=1)

        has_box = ((point_labels[:, 1] == 2) & (point_labels[:, 2] == 3)).to(point_coords.dtype).reshape(batch, 1)
        full_image = point_coords.new_tensor([[0.5, 0.5, 1.0, 1.0]]).repeat(batch, 1)
        return (has_box * box + (1.0 - has_box) * full_image).reshape(batch, 1, 4)

    def forward(self, image, point_coords, point_labels, mask_input, has_mask_input):
        batch = image.shape[0]
        point_coords = point_coords.reshape(batch, SAM_MAX_POINTS, 2)
        point_labels = point_labels.reshape(batch, SAM_MAX_POINTS).to(torch.int64)
        input_boxes = self._box_prompt(point_coords, point_labels)
        input_boxes_labels = torch.ones((batch, 1), dtype=torch.int64, device=image.device)
        text_embeds = self.text_embeds.to(device=image.device, dtype=image.dtype)
        if text_embeds.shape[0] == 1 and batch != 1:
            text_embeds = text_embeds.repeat(batch, 1, 1)

        outputs = self.wrapped(
            pixel_values=image,
            text_embeds=text_embeds,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_dict=True,
        )
        masks = self._first_three_masks(outputs.pred_masks.float())
        iou_predictions = self._first_three_scores(outputs.pred_logits, masks)
        keep_inputs = (mask_input.sum() + has_mask_input.sum()) * 0.0
        masks = masks + keep_inputs
        iou_predictions = iou_predictions + keep_inputs
        return masks, iou_predictions, masks


def require_onnx() -> None:
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError("ONNX export requires the 'onnx' Python package in the active environment")


def make_default_inputs(image_size: int) -> dict[str, np.ndarray]:
    point_coords = np.zeros((1, SAM_MAX_POINTS, 2, 1), dtype=np.float32)
    point_labels = np.full((1, SAM_MAX_POINTS, 1, 1), -1.0, dtype=np.float32)
    point_coords[0, 0, :, 0] = (image_size / 2.0, image_size / 2.0)
    point_labels[0, 0, 0, 0] = 1.0

    return {
        "image": np.zeros((1, 3, image_size, image_size), dtype=np.float32),
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": np.zeros((1, 1, SAM_MASK_SIZE, SAM_MASK_SIZE), dtype=np.float32),
        "has_mask_input": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }


def sam_input_tensors(inputs: dict[str, np.ndarray], device: torch.device) -> tuple[torch.Tensor, ...]:
    missing = [name for name in SAM_INPUT_NAMES if name not in inputs]
    if missing:
        raise KeyError(f"Missing SAM input(s): {', '.join(missing)}")

    return tuple(
        torch.from_numpy(np.ascontiguousarray(inputs[name], dtype=np.float32)).to(device=device)
        for name in SAM_INPUT_NAMES
    )


def export_graph(
    wrapper: torch.nn.Module,
    inputs: dict[str, np.ndarray],
    output_path: Path,
    device: torch.device,
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": SAM_INPUT_NAMES,
        "output_names": SAM_OUTPUT_NAMES,
    }
    export_inputs = sam_input_tensors(inputs, device)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using the legacy TorchScript-based ONNX export.*",
                category=DeprecationWarning,
            )
            torch.onnx.export(wrapper, export_inputs, str(output_path), dynamo=False, **export_kwargs)
    except TypeError as exc:
        if "dynamo" not in str(exc):
            raise
        torch.onnx.export(wrapper, export_inputs, str(output_path), **export_kwargs)


def export_sam_v1_onnx(
    *,
    model_name: str,
    checkpoint: Path,
    output_path: Path,
    sam_root: str | Path | None = None,
    inputs: dict[str, np.ndarray] | None = None,
    device: str | torch.device = "cpu",
    opset: int = 17,
) -> None:
    require_onnx()
    model_type = SAM_V1_ALIASES.get(model_name)
    if model_type is None:
        raise ValueError(f"Unsupported SAM v1 model '{model_name}'")

    torch_device = torch.device(device)
    sam_root_arg = None if sam_root is None or str(sam_root) == "" else str(sam_root)
    registry = import_segment_anything(sam_root_arg)
    model = registry[model_type](checkpoint=str(checkpoint)).to(torch_device).eval()
    wrapper = SAMV1OnnxWrapper(model).eval()
    export_graph(wrapper, inputs or make_default_inputs(SAM_IMAGE_SIZE), output_path, torch_device, opset)


def export_sam2_onnx(
    *,
    model_name: str,
    checkpoint: Path,
    output_path: Path,
    sam2_root: str | Path | None = None,
    inputs: dict[str, np.ndarray] | None = None,
    device: str | torch.device = "cpu",
    opset: int = 17,
) -> None:
    require_onnx()
    if model_name not in SAM2_MODEL_NAMES:
        raise ValueError(f"Unsupported SAM2 model '{model_name}'. Supported: {', '.join(SAM2_MODEL_NAMES)}")
    checkpoint_text = str(checkpoint).replace("\\", "/").lower()
    if ("sam2.1" in checkpoint_text or "sam2_1" in checkpoint_text) and not model_name.startswith("sam2_1"):
        suggested = model_name.replace("sam2_", "sam2_1_", 1)
        raise ValueError(f"Checkpoint looks like SAM2.1; use -m {suggested} instead of -m {model_name}")

    torch_device = torch.device(device)
    sam2_root_arg = None if sam2_root is None or str(sam2_root) == "" else str(sam2_root)
    model = build_sam2_without_hydra(model_name, checkpoint, sam2_root_arg, torch_device)
    wrapper = SAM2OnnxWrapper(model).eval()
    export_graph(wrapper, inputs or make_default_inputs(SAM2_IMAGE_SIZE), output_path, torch_device, opset)


def resolve_sam3_image_size(model: Any) -> int:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    backbone_config = getattr(vision_config, "backbone_config", None)
    image_size = getattr(backbone_config, "image_size", None) or getattr(vision_config, "image_size", None)
    if isinstance(image_size, (tuple, list)):
        return int(image_size[0])
    return int(image_size or 1008)


def make_sam3_text_embeds(model: Any, device: torch.device, text_length: int) -> torch.Tensor:
    text_config = getattr(model.config, "text_config", None)
    bos_token_id = int(getattr(text_config, "bos_token_id", 49406))
    eos_token_id = int(getattr(text_config, "eos_token_id", 49407))
    pad_token_id = int(getattr(text_config, "pad_token_id", 1))
    max_position_embeddings = int(getattr(text_config, "max_position_embeddings", text_length))
    text_length = max(2, min(text_length, max_position_embeddings))

    input_ids = torch.full((1, text_length), pad_token_id, dtype=torch.int64, device=device)
    attention_mask = torch.zeros((1, text_length), dtype=torch.int64, device=device)
    input_ids[0, 0] = bos_token_id
    input_ids[0, 1] = eos_token_id
    attention_mask[0, :2] = 1
    with torch.no_grad():
        text_features = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    text_embeds = text_features.pooler_output
    if text_embeds.dim() == 2:
        text_embeds = text_embeds.unsqueeze(1)
    return text_embeds.detach()


def export_sam3_onnx(
    *,
    model_name: str,
    checkpoint: str | Path,
    output_path: Path,
    inputs: dict[str, np.ndarray] | None = None,
    device: str | torch.device = "cpu",
    opset: int = 17,
    sam3_text_length: int = 32,
) -> None:
    require_onnx()
    if model_name not in SAM3_MODEL_NAMES:
        raise ValueError(f"Unsupported SAM3 model '{model_name}'")
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("SAM3 export requires the 'transformers' Python package")

    from transformers import Sam3Model

    torch_device = torch.device(device)
    model = Sam3Model.from_pretrained(str(checkpoint)).to(torch_device).eval()
    image_size = resolve_sam3_image_size(model)
    text_embeds = make_sam3_text_embeds(model, torch_device, sam3_text_length)
    wrapper = SAM3OnnxWrapper(model, text_embeds, image_size).eval()
    export_graph(wrapper, inputs or make_default_inputs(image_size), output_path, torch_device, opset)


def export_sam_onnx(
    *,
    model_name: str,
    checkpoint: str | Path,
    output_path: Path,
    sam_root: str | Path | None = None,
    sam2_root: str | Path | None = None,
    inputs: dict[str, np.ndarray] | None = None,
    device: str | torch.device = "cpu",
    opset: int = 17,
    sam3_text_length: int = 32,
) -> None:
    if model_name in SAM_V1_ALIASES:
        export_sam_v1_onnx(
            model_name=model_name,
            checkpoint=Path(checkpoint),
            output_path=output_path,
            sam_root=sam_root,
            inputs=inputs,
            device=device,
            opset=opset,
        )
        return
    if model_name in SAM2_MODEL_NAMES:
        export_sam2_onnx(
            model_name=model_name,
            checkpoint=Path(checkpoint),
            output_path=output_path,
            sam2_root=sam2_root,
            inputs=inputs,
            device=device,
            opset=opset,
        )
        return
    if model_name in SAM3_MODEL_NAMES:
        export_sam3_onnx(
            model_name=model_name,
            checkpoint=checkpoint,
            output_path=output_path,
            inputs=inputs,
            device=device,
            opset=opset,
            sam3_text_length=sam3_text_length,
        )
        return
    raise ValueError(f"Unsupported SAM model '{model_name}'. Supported: {', '.join(SAM_MODEL_NAMES)}")


def default_output_path(model_name: str) -> Path:
    if model_name in {"default", "sam"}:
        return Path("sam.onnx")
    if model_name in {"vit_b", "vit_l", "vit_h"}:
        return Path(f"sam_{model_name}.onnx")
    return Path(f"{model_name}.onnx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official SAM v1/SAM2/SAM3 checkpoints to an ONNX graph compatible with the segmentation sample"
    )
    parser.add_argument("-m", "--model", choices=SAM_MODEL_NAMES, default="sam_vit_b")
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        help="SAM v1/SAM2 checkpoint path, or SAM3 Hugging Face model id/local directory",
    )
    parser.add_argument("--sam-root", type=str, default="", help="Path to official segment-anything repository root")
    parser.add_argument("--sam2-root", type=str, default="", help="Path to official SAM2 repository root")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .onnx path")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch export device")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--sam3-text-length", type=int, default=32, help="Static SAM3 text prompt token length")
    return parser.parse_args()


def main() -> None:
    configure_stdio()
    args = parse_args()
    output_path = args.output if args.output is not None else default_output_path(args.model)
    export_sam_onnx(
        model_name=args.model,
        checkpoint=args.checkpoint,
        output_path=output_path,
        sam_root=args.sam_root,
        sam2_root=args.sam2_root,
        device=args.device,
        opset=args.opset,
        sam3_text_length=args.sam3_text_length,
    )
    print(f"Exported {args.model} ONNX to {output_path}")
    print(f"Inputs: {', '.join(SAM_INPUT_NAMES)}")
    print(f"Outputs: {', '.join(SAM_OUTPUT_NAMES)}")


if __name__ == "__main__":
    main()
