"""Export SAM v1, SAM2/SAM2.1, and SAM3 weights to InferRT .wts."""

from __future__ import annotations

import argparse
import importlib.util
import math
import struct
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch


DEFAULT_IMAGE_PATH = Path(__file__).resolve().parents[3] / "assets" / "pics" / "dog.jpg"

SAM_IMAGE_SIZE = 1024
SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
SAM_PIXEL_STD = (58.395, 57.12, 57.375)
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

SAM2_IMAGE_SIZE = 1024
SAM2_PIXEL_MEAN = (0.485, 0.456, 0.406)
SAM2_PIXEL_STD = (0.229, 0.224, 0.225)
SAM2_MODEL_CONFIGS = {
    "sam2_hiera_tiny": "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_small": "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_base_plus": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_large": "configs/sam2/sam2_hiera_l.yaml",
    "sam2_1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2_1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2_1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2_1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}
MODEL_CONFIGS = SAM2_MODEL_CONFIGS

SAM2_MODEL_SPECS = {
    "sam2_hiera_tiny": {
        "embed_dim": 96,
        "num_heads": 1,
        "stages": (1, 2, 7, 2),
        "global_att_blocks": (5, 7, 9),
        "window_spec": (8, 4, 14, 7),
        "pos_embed_size": 7,
    },
    "sam2_hiera_small": {
        "embed_dim": 96,
        "num_heads": 1,
        "stages": (1, 2, 11, 2),
        "global_att_blocks": (7, 10, 13),
        "window_spec": (8, 4, 14, 7),
        "pos_embed_size": 7,
    },
    "sam2_hiera_base_plus": {
        "embed_dim": 112,
        "num_heads": 2,
        "stages": (2, 3, 16, 3),
        "global_att_blocks": (12, 16, 20),
        "window_spec": (8, 4, 14, 7),
        "pos_embed_size": 14,
    },
    "sam2_hiera_large": {
        "embed_dim": 144,
        "num_heads": 2,
        "stages": (2, 6, 36, 4),
        "global_att_blocks": (23, 33, 43),
        "window_spec": (8, 4, 16, 8),
        "pos_embed_size": 7,
    },
    "sam2_1_hiera_tiny": {
        "embed_dim": 96,
        "num_heads": 1,
        "stages": (1, 2, 7, 2),
        "global_att_blocks": (5, 7, 9),
        "window_spec": (8, 4, 14, 7),
        "pos_embed_size": 7,
    },
    "sam2_1_hiera_small": {
        "embed_dim": 96,
        "num_heads": 1,
        "stages": (1, 2, 11, 2),
        "global_att_blocks": (7, 10, 13),
        "window_spec": (8, 4, 14, 7),
        "pos_embed_size": 7,
    },
    "sam2_1_hiera_base_plus": {
        "embed_dim": 112,
        "num_heads": 2,
        "stages": (2, 3, 16, 3),
        "global_att_blocks": (12, 16, 20),
        "window_spec": (8, 4, 14, 7),
        "pos_embed_size": 14,
    },
    "sam2_1_hiera_large": {
        "embed_dim": 144,
        "num_heads": 2,
        "stages": (2, 6, 36, 4),
        "global_att_blocks": (23, 33, 43),
        "window_spec": (8, 4, 16, 8),
        "pos_embed_size": 7,
    },
}

SAM3_MODEL_NAMES = ("sam3", "sam3_image")
EDGE_SAM_MODEL_NAMES = ("edge_sam",)
SAM_MODEL_NAMES = tuple(SAM_V1_ALIASES) + EDGE_SAM_MODEL_NAMES + tuple(SAM2_MODEL_CONFIGS) + SAM3_MODEL_NAMES


def elapsed_ms(start: float, end: float) -> float:
    return (end - start) * 1000.0


def import_segment_anything(sam_root: str | None) -> Any:
    if sam_root:
        root = Path(sam_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"SAM root does not exist: {root}")
        sys.path.insert(0, str(root))

    from segment_anything import sam_model_registry  # type: ignore

    return sam_model_registry


def import_sam2_builder(sam2_root: str | None) -> Any:
    if sam2_root:
        root = Path(sam2_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"SAM2 root does not exist: {root}")
        sys.path.insert(0, str(root))

    from sam2.build_sam import build_sam2  # type: ignore

    return build_sam2


def import_edge_sam_builder(edge_sam_root: str | None) -> Any:
    if edge_sam_root:
        root = Path(edge_sam_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"EdgeSAM root does not exist: {root}")
        sys.path.insert(0, str(root))
    install_edge_sam_optional_dependency_shims()

    from edge_sam.build_sam import build_edge_sam  # type: ignore

    return build_edge_sam


def install_optional_dependency_shims() -> None:
    if "hydra" not in sys.modules:
        hydra = types.ModuleType("hydra")
        hydra.initialize_config_module = lambda *args, **kwargs: None
        sys.modules["hydra"] = hydra

        hydra_core = types.ModuleType("hydra.core")
        global_hydra_mod = types.ModuleType("hydra.core.global_hydra")

        class _GlobalHydra:
            @staticmethod
            def instance() -> "_GlobalHydra":
                return _GlobalHydra()

            def is_initialized(self) -> bool:
                return True

        global_hydra_mod.GlobalHydra = _GlobalHydra
        sys.modules["hydra.core"] = hydra_core
        sys.modules["hydra.core.global_hydra"] = global_hydra_mod

    if "iopath" not in sys.modules:
        iopath = types.ModuleType("iopath")
        common = types.ModuleType("iopath.common")
        file_io = types.ModuleType("iopath.common.file_io")

        class _PathManager:
            @staticmethod
            def open(path: str, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
                return open(path, mode, *args, **kwargs)

        file_io.g_pathmgr = _PathManager()
        sys.modules["iopath"] = iopath
        sys.modules["iopath.common"] = common
        sys.modules["iopath.common.file_io"] = file_io


class _CfgNode(dict):
    """EdgeSAM 配置导入所需的最小 yacs CfgNode 兼容层。"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def clone(self) -> "_CfgNode":
        clone = _CfgNode()
        for key, value in self.items():
            clone[key] = value.clone() if isinstance(value, _CfgNode) else value
        return clone

    def defrost(self) -> None:
        return None

    def freeze(self) -> None:
        return None

    def merge_from_file(self, _path: str) -> None:
        return None

    def merge_from_list(self, _values: list[str]) -> None:
        return None


def install_edge_sam_optional_dependency_shims() -> None:
    """安装 EdgeSAM 推理路径不会实际用到的训练/RPN 依赖占位模块。"""

    if "loralib" not in sys.modules:
        loralib = types.ModuleType("loralib")
        loralib.Linear = torch.nn.Linear
        sys.modules["loralib"] = loralib

    if "yacs.config" not in sys.modules:
        yacs = types.ModuleType("yacs")
        config = types.ModuleType("yacs.config")
        config.CfgNode = _CfgNode
        sys.modules["yacs"] = yacs
        sys.modules["yacs.config"] = config

    if "mmengine" not in sys.modules:
        mmengine = types.ModuleType("mmengine")
        mmengine.ConfigDict = dict
        sys.modules["mmengine"] = mmengine

    class _UnusedRPNModule:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("EdgeSAM RPN dependencies are not available in this lightweight test path")

    if "mmdet.models.dense_heads" not in sys.modules:
        mmdet = types.ModuleType("mmdet")
        models = types.ModuleType("mmdet.models")
        dense_heads = types.ModuleType("mmdet.models.dense_heads")
        necks = types.ModuleType("mmdet.models.necks")
        dense_heads.RPNHead = _UnusedRPNModule
        dense_heads.CenterNetUpdateHead = _UnusedRPNModule
        necks.FPN = _UnusedRPNModule
        sys.modules["mmdet"] = mmdet
        sys.modules["mmdet.models"] = models
        sys.modules["mmdet.models.dense_heads"] = dense_heads
        sys.modules["mmdet.models.necks"] = necks

    if "projects.EfficientDet.efficientdet" not in sys.modules:
        projects = types.ModuleType("projects")
        efficient_det = types.ModuleType("projects.EfficientDet")
        efficientdet = types.ModuleType("projects.EfficientDet.efficientdet")
        efficientdet.BiFPN = _UnusedRPNModule
        efficientdet.EfficientDetSepBNHead = _UnusedRPNModule
        sys.modules["projects"] = projects
        sys.modules["projects.EfficientDet"] = efficient_det
        sys.modules["projects.EfficientDet.efficientdet"] = efficientdet


def write_wts(state_dict: dict[str, torch.Tensor], output_path: Path, *, verbose: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"{len(state_dict)}\n")
        for key, tensor in state_dict.items():
            if verbose:
                print(f"key: {key}\tvalue: {tuple(tensor.shape)}")
            values = tensor.detach().float().reshape(-1).cpu().numpy()
            file.write(f"{key} {len(values)}")
            for value in values:
                file.write(" ")
                file.write(struct.pack(">f", float(value)).hex())
            file.write("\n")


def compute_resize_shape(original_h: int, original_w: int, target_size: int = SAM_IMAGE_SIZE) -> tuple[int, int]:
    scale = target_size / max(original_h, original_w)
    return int(math.floor(original_h * scale + 0.5)), int(math.floor(original_w * scale + 0.5))


def preprocess_image(image_path: Path, device: torch.device) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_size = (image.shape[0], image.shape[1])
    resized_size = compute_resize_shape(*original_size)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (resized_size[1], resized_size[0]), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(SAM_PIXEL_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(SAM_PIXEL_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    normalized = (tensor - mean) / std
    padded = torch.zeros((1, 3, SAM_IMAGE_SIZE, SAM_IMAGE_SIZE), device=device, dtype=torch.float32)
    padded[:, :, : resized_size[0], : resized_size[1]] = normalized
    return padded, original_size, resized_size


def preprocess_sam2_image(image_path: Path, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_size = (image.shape[0], image.shape[1])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (SAM2_IMAGE_SIZE, SAM2_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    mean = torch.tensor(SAM2_PIXEL_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(SAM2_PIXEL_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean) / std, original_size


def resolve_sam3_image_size(model: Any) -> int:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    backbone_config = getattr(vision_config, "backbone_config", None)
    image_size = getattr(backbone_config, "image_size", None) or getattr(vision_config, "image_size", None)
    if isinstance(image_size, (tuple, list)):
        return int(image_size[0])
    return int(image_size or 1008)


def preprocess_sam3_image(image_path: Path, device: torch.device, image_size: int) -> torch.Tensor:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(resized).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0


def make_sam_v1_default_prompts(
    device: torch.device,
    resized_size: tuple[int, int],
) -> tuple[tuple[torch.Tensor, torch.Tensor], None, None]:
    resized_h, resized_w = resized_size
    point_coords = torch.tensor([[[resized_w / 2.0, resized_h / 2.0]]], device=device, dtype=torch.float32)
    point_labels = torch.tensor([[1]], device=device, dtype=torch.int64)
    return (point_coords, point_labels), None, None


def build_sam2_without_hydra(model_name: str, checkpoint: Path, sam2_root: str | None, device: torch.device) -> Any:
    if sam2_root:
        root = Path(sam2_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"SAM2 root does not exist: {root}")
        sys.path.insert(0, str(root))
    install_optional_dependency_shims()

    from sam2.modeling.backbones.hieradet import Hiera  # type: ignore
    from sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder  # type: ignore
    from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer  # type: ignore
    from sam2.modeling.memory_encoder import CXBlock, Fuser, MaskDownSampler, MemoryEncoder  # type: ignore
    from sam2.modeling.position_encoding import PositionEmbeddingSine  # type: ignore
    from sam2.modeling.sam.transformer import RoPEAttention  # type: ignore
    from sam2.modeling.sam2_base import SAM2Base  # type: ignore

    spec = SAM2_MODEL_SPECS[model_name]
    embed_dim = spec["embed_dim"]
    is_sam2_1 = model_name.startswith("sam2_1")
    trunk = Hiera(
        embed_dim=embed_dim,
        num_heads=spec["num_heads"],
        stages=spec["stages"],
        global_att_blocks=spec["global_att_blocks"],
        window_spec=spec["window_spec"],
        window_pos_embed_bkg_spatial_size=(spec["pos_embed_size"], spec["pos_embed_size"]),
    )
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            warmup_cache=False,
        ),
        d_model=256,
        backbone_channel_list=[embed_dim * 8, embed_dim * 4, embed_dim * 2, embed_dim],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )

    memory_attention = MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=MemoryAttentionLayer(
            activation="relu",
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            self_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=(64, 64),
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
            ),
            d_model=256,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            cross_attention=RoPEAttention(
                rope_theta=10000.0,
                feat_sizes=(64, 64),
                rope_k_repeat=True,
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                dropout=0.1,
                kv_in_dim=64,
            ),
        ),
        num_layers=4,
    )
    memory_encoder = MemoryEncoder(
        out_dim=64,
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=64,
            normalize=True,
            scale=None,
            temperature=10000,
            warmup_cache=False,
        ),
        mask_downsampler=MaskDownSampler(kernel_size=3, stride=2, padding=1),
        fuser=Fuser(
            layer=CXBlock(
                dim=256,
                kernel_size=7,
                padding=3,
                layer_scale_init_value=1e-6,
                use_dwconv=True,
            ),
            num_layers=2,
        ),
    )

    model = SAM2Base(
        image_encoder=ImageEncoder(trunk=trunk, neck=neck, scalp=1),
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        image_size=SAM2_IMAGE_SIZE,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        no_obj_embed_spatial=is_sam2_1,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=is_sam2_1,
        proj_tpos_enc_in_obj_ptrs=is_sam2_1,
        use_signed_tpos_enc_to_obj_ptrs=is_sam2_1,
        only_obj_ptrs_in_the_past_for_eval=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
        compile_image_encoder=False,
    )

    checkpoint_obj = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint_obj["model"] if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj else checkpoint_obj
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: missing keys while loading fallback SAM2 model: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Warning: unused checkpoint keys in fallback SAM2 model: {len(unexpected_keys)}")
    model.to(device)
    model.eval()
    return model


def load_sam_v1_model(model_name: str, checkpoint: Path, sam_root: str | None, device: torch.device) -> torch.nn.Module:
    model_type = SAM_V1_ALIASES[model_name]
    registry = import_segment_anything(sam_root)
    model = registry[model_type](checkpoint=str(checkpoint)).to(device)
    model.eval()
    return model


def load_sam2_model(model_name: str, checkpoint: Path, sam2_root: str | None, device: torch.device) -> torch.nn.Module:
    try:
        build_sam2 = import_sam2_builder(sam2_root)
        model = build_sam2(
            config_file=SAM2_MODEL_CONFIGS[model_name],
            ckpt_path=str(checkpoint),
            device=str(device),
            mode="eval",
            apply_postprocessing=False,
        )
    except ModuleNotFoundError as exc:
        print(f"Official SAM2 builder dependency missing ({exc.name}); using no-Hydra fallback builder.")
        model = build_sam2_without_hydra(model_name, checkpoint, sam2_root, device)
    model.eval()
    return model


def load_checkpoint_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    """读取 PyTorch checkpoint 中实际的 state_dict。"""

    checkpoint_obj = torch.load(checkpoint, map_location="cpu")
    if isinstance(checkpoint_obj, dict):
        if "model" in checkpoint_obj:
            return checkpoint_obj["model"]
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
        return checkpoint_obj
    raise TypeError(f"Unsupported checkpoint object type: {type(checkpoint_obj)!r}")


def load_edge_sam_model(checkpoint: Path, edge_sam_root: str | None, device: torch.device) -> torch.nn.Module:
    build_edge_sam = import_edge_sam_builder(edge_sam_root)
    model = build_edge_sam(str(checkpoint)).to(device)
    model.eval()
    return model


def load_sam3_model(checkpoint: str, device: torch.device, *, local_files_only: bool = False) -> torch.nn.Module:
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("SAM3 .wts export requires the 'transformers' Python package")

    from transformers import Sam3Model

    model = Sam3Model.from_pretrained(checkpoint, local_files_only=local_files_only).to(device)
    model.eval()
    return model


@torch.no_grad()
def run_sam_v1_reference_forward(
    model: torch.nn.Module,
    image: torch.Tensor,
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> None:
    encoder_start = time.perf_counter()
    image_embeddings = model.image_encoder(image)
    encoder_end = time.perf_counter()

    prompt_start = time.perf_counter()
    points, boxes, masks = make_sam_v1_default_prompts(image.device, resized_size)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(points=points, boxes=boxes, masks=masks)
    prompt_end = time.perf_counter()

    decoder_start = time.perf_counter()
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )
    decoder_end = time.perf_counter()

    post_start = time.perf_counter()
    masks_out = model.postprocess_masks(low_res_masks, input_size=resized_size, original_size=original_size)
    post_end = time.perf_counter()

    print(f"image_embeddings: {tuple(image_embeddings.shape)}")
    print(f"sparse_embeddings: {tuple(sparse_embeddings.shape)}")
    print(f"dense_embeddings: {tuple(dense_embeddings.shape)}")
    print(f"low_res_masks: {tuple(low_res_masks.shape)}")
    print(f"iou_predictions: {tuple(iou_predictions.shape)}")
    print(f"postprocessed_masks: {tuple(masks_out.shape)}")
    print(
        "Timing: "
        f"image_encoder={elapsed_ms(encoder_start, encoder_end):.3f} ms, "
        f"prompt_encoder={elapsed_ms(prompt_start, prompt_end):.3f} ms, "
        f"mask_decoder={elapsed_ms(decoder_start, decoder_end):.3f} ms, "
        f"postprocess={elapsed_ms(post_start, post_end):.3f} ms"
    )


@torch.no_grad()
def run_edge_sam_reference_forward(
    model: torch.nn.Module,
    image: torch.Tensor,
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> None:
    encoder_start = time.perf_counter()
    image_embeddings = model.image_encoder(image)
    encoder_end = time.perf_counter()

    prompt_start = time.perf_counter()
    points, boxes, masks = make_sam_v1_default_prompts(image.device, resized_size)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(points=points, boxes=boxes, masks=masks)
    prompt_end = time.perf_counter()

    decoder_start = time.perf_counter()
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        num_multimask_outputs=3,
    )
    decoder_end = time.perf_counter()

    post_start = time.perf_counter()
    masks_out = model.postprocess_masks(low_res_masks, input_size=resized_size, original_size=original_size)
    post_end = time.perf_counter()

    print(f"image_embeddings: {tuple(image_embeddings.shape)}")
    print(f"sparse_embeddings: {tuple(sparse_embeddings.shape)}")
    print(f"dense_embeddings: {tuple(dense_embeddings.shape)}")
    print(f"low_res_masks: {tuple(low_res_masks.shape)}")
    print(f"iou_predictions: {tuple(iou_predictions.shape)}")
    print(f"postprocessed_masks: {tuple(masks_out.shape)}")
    print(
        "Timing: "
        f"image_encoder={elapsed_ms(encoder_start, encoder_end):.3f} ms, "
        f"prompt_encoder={elapsed_ms(prompt_start, prompt_end):.3f} ms, "
        f"mask_decoder={elapsed_ms(decoder_start, decoder_end):.3f} ms, "
        f"postprocess={elapsed_ms(post_start, post_end):.3f} ms"
    )


@torch.no_grad()
def run_sam2_reference_forward(model: torch.nn.Module, image: torch.Tensor, original_size: tuple[int, int]) -> None:
    encoder_start = time.perf_counter()
    backbone_out = model.forward_image(image)
    image_embeddings = backbone_out["vision_features"]
    if getattr(model, "directly_add_no_mem_embed", False):
        no_mem_embed = model.no_mem_embed.reshape(1, 1, -1).permute(0, 2, 1).reshape(1, -1, 1, 1)
        image_embeddings = image_embeddings + no_mem_embed
    encoder_end = time.perf_counter()

    prompt_start = time.perf_counter()
    point_coords = torch.tensor([[[SAM2_IMAGE_SIZE / 2.0, SAM2_IMAGE_SIZE / 2.0]]], device=image.device)
    point_labels = torch.tensor([[1]], device=image.device, dtype=torch.int64)
    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=(point_coords, point_labels),
        boxes=None,
        masks=None,
    )
    prompt_end = time.perf_counter()

    high_res_features = [backbone_out["backbone_fpn"][0], backbone_out["backbone_fpn"][1]]
    decoder_start = time.perf_counter()
    low_res_masks, iou_predictions, _, _ = model.sam_mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=False,
        high_res_features=high_res_features,
    )
    decoder_end = time.perf_counter()

    post_start = time.perf_counter()
    high_res_masks = torch.nn.functional.interpolate(
        low_res_masks.float(),
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )
    post_end = time.perf_counter()

    print(f"image_embeddings: {tuple(backbone_out['vision_features'].shape)}")
    print(f"high_res_s0: {tuple(high_res_features[0].shape)}")
    print(f"high_res_s1: {tuple(high_res_features[1].shape)}")
    print(f"sparse_embeddings: {tuple(sparse_embeddings.shape)}")
    print(f"dense_embeddings: {tuple(dense_embeddings.shape)}")
    print(f"low_res_masks: {tuple(low_res_masks.shape)}")
    print(f"iou_predictions: {tuple(iou_predictions.shape)}")
    print(f"postprocessed_masks: {tuple(high_res_masks.shape)}")
    print(
        "Timing: "
        f"image_encoder={elapsed_ms(encoder_start, encoder_end):.3f} ms, "
        f"prompt_encoder={elapsed_ms(prompt_start, prompt_end):.3f} ms, "
        f"mask_decoder={elapsed_ms(decoder_start, decoder_end):.3f} ms, "
        f"postprocess={elapsed_ms(post_start, post_end):.3f} ms"
    )


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
    text_features = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    text_embeds = text_features.pooler_output
    if text_embeds.dim() == 2:
        text_embeds = text_embeds.unsqueeze(1)
    return text_embeds


@torch.no_grad()
def run_sam3_reference_forward(model: torch.nn.Module, image: torch.Tensor, text_length: int) -> None:
    text_start = time.perf_counter()
    text_embeds = make_sam3_text_embeds(model, image.device, text_length)
    text_end = time.perf_counter()

    forward_start = time.perf_counter()
    input_boxes = torch.tensor([[[0.5, 0.5, 1.0, 1.0]]], device=image.device, dtype=torch.float32)
    input_boxes_labels = torch.ones((1, 1), device=image.device, dtype=torch.int64)
    outputs = model(
        pixel_values=image,
        text_embeds=text_embeds,
        input_boxes=input_boxes,
        input_boxes_labels=input_boxes_labels,
        return_dict=True,
    )
    forward_end = time.perf_counter()

    print(f"text_embeds: {tuple(text_embeds.shape)}")
    print(f"pred_masks: {tuple(outputs.pred_masks.shape)}")
    print(f"pred_boxes: {tuple(outputs.pred_boxes.shape)}")
    if outputs.pred_logits is not None:
        print(f"pred_logits: {tuple(outputs.pred_logits.shape)}")
    print(
        "Timing: "
        f"text_encoder={elapsed_ms(text_start, text_end):.3f} ms, "
        f"sam3_forward={elapsed_ms(forward_start, forward_end):.3f} ms"
    )


def validate_model_checkpoint_pair(model_name: str, checkpoint: str) -> None:
    checkpoint_text = checkpoint.replace("\\", "/").lower()
    if model_name in SAM2_MODEL_CONFIGS:
        if ("sam2.1" in checkpoint_text or "sam2_1" in checkpoint_text) and not model_name.startswith("sam2_1"):
            suggested = model_name.replace("sam2_", "sam2_1_", 1)
            raise ValueError(f"Checkpoint looks like SAM2.1; use -m {suggested} instead of -m {model_name}")


def default_output_path(model_name: str) -> Path:
    if model_name in {"vit_b", "vit_l", "vit_h"}:
        return Path(f"sam_{model_name}.wts")
    if model_name == "default":
        return Path("sam_default.wts")
    return Path(f"{model_name}.wts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export official SAM/SAM2/SAM3 checkpoints to InferRT .wts")
    parser.add_argument("-m", "--model", choices=SAM_MODEL_NAMES, default="sam_vit_b")
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        help="SAM v1/SAM2 checkpoint path, or SAM3 Hugging Face model id/local directory",
    )
    parser.add_argument("--sam-root", type=str, default="", help="Path to official segment-anything repository root")
    parser.add_argument("--sam2-root", type=str, default="", help="Path to official SAM2 repository root")
    parser.add_argument("--edge-sam-root", type=str, default="", help="Path to official EdgeSAM repository root")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .wts path")
    parser.add_argument("-i", "--img-path", type=Path, default=DEFAULT_IMAGE_PATH, help="Image for forward smoke")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-forward", action="store_true", help="Only export weights, skip PyTorch forward smoke")
    parser.add_argument("--verbose", action="store_true", help="Print every exported weight key")
    parser.add_argument("--sam3-text-length", type=int, default=32, help="Static SAM3 text prompt token length")
    parser.add_argument("--local-files-only", action="store_true", help="Load SAM3 from the local HF cache only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_model_checkpoint_pair(args.model, args.checkpoint)
    output_path = args.output if args.output is not None else default_output_path(args.model)
    device = torch.device(args.device)
    state_dict: dict[str, torch.Tensor] | None = None

    load_start = time.perf_counter()
    if args.model == "edge_sam" and args.skip_forward:
        state_dict = load_checkpoint_state_dict(Path(args.checkpoint))
        model = None
        family = "EdgeSAM"
    elif args.model in SAM_V1_ALIASES:
        model = load_sam_v1_model(args.model, Path(args.checkpoint), args.sam_root or None, device)
        family = "SAM v1"
    elif args.model == "edge_sam":
        model = load_edge_sam_model(Path(args.checkpoint), args.edge_sam_root or None, device)
        family = "EdgeSAM"
    elif args.model in SAM2_MODEL_CONFIGS:
        model = load_sam2_model(args.model, Path(args.checkpoint), args.sam2_root or None, device)
        family = "SAM2"
    else:
        model = load_sam3_model(args.checkpoint, device, local_files_only=args.local_files_only)
        family = "SAM3"
    load_end = time.perf_counter()
    print(f"Loaded {family} {args.model} in {elapsed_ms(load_start, load_end):.3f} ms on {device}")

    if not args.skip_forward and model is not None:
        preprocess_start = time.perf_counter()
        if args.model in SAM_V1_ALIASES:
            image, original_size, resized_size = preprocess_image(args.img_path, device)
            preprocess_end = time.perf_counter()
            print(
                f"Preprocess: {elapsed_ms(preprocess_start, preprocess_end):.3f} ms, "
                f"original={original_size}, resized={resized_size}, padded={(SAM_IMAGE_SIZE, SAM_IMAGE_SIZE)}"
            )
            run_sam_v1_reference_forward(model, image, original_size, resized_size)
        elif args.model == "edge_sam":
            image, original_size, resized_size = preprocess_image(args.img_path, device)
            preprocess_end = time.perf_counter()
            print(
                f"Preprocess: {elapsed_ms(preprocess_start, preprocess_end):.3f} ms, "
                f"original={original_size}, resized={resized_size}, padded={(SAM_IMAGE_SIZE, SAM_IMAGE_SIZE)}"
            )
            run_edge_sam_reference_forward(model, image, original_size, resized_size)
        elif args.model in SAM2_MODEL_CONFIGS:
            image, original_size = preprocess_sam2_image(args.img_path, device)
            preprocess_end = time.perf_counter()
            print(f"Preprocess: {elapsed_ms(preprocess_start, preprocess_end):.3f} ms, original={original_size}")
            run_sam2_reference_forward(model, image, original_size)
        else:
            image_size = resolve_sam3_image_size(model)
            image = preprocess_sam3_image(args.img_path, device, image_size)
            preprocess_end = time.perf_counter()
            print(
                f"Preprocess: {elapsed_ms(preprocess_start, preprocess_end):.3f} ms, "
                f"resized={(image_size, image_size)}"
            )
            run_sam3_reference_forward(model, image, args.sam3_text_length)

    export_start = time.perf_counter()
    if state_dict is None:
        if model is None:
            raise RuntimeError("No model or checkpoint state_dict available for export")
        state_dict = model.state_dict()
    write_wts(state_dict, output_path, verbose=args.verbose)
    export_end = time.perf_counter()
    print(f"Exported {len(state_dict)} tensors to {output_path}")
    print(f"Weight export: {elapsed_ms(export_start, export_end):.3f} ms")


if __name__ == "__main__":
    main()
