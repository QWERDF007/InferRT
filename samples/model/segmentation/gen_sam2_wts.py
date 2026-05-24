"""导出官方 SAM2/SAM2.1 权重为 InferRT `.wts` 格式。"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import types
from pathlib import Path
from typing import Any

import cv2
import torch


DEFAULT_IMAGE_PATH = Path(__file__).resolve().parents[3] / "assets" / "pics" / "dog.jpg"
SAM2_IMAGE_SIZE = 1024
SAM2_PIXEL_MEAN = (0.485, 0.456, 0.406)
SAM2_PIXEL_STD = (0.229, 0.224, 0.225)

MODEL_CONFIGS = {
    "sam2_hiera_tiny": "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_small": "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_base_plus": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_large": "configs/sam2/sam2_hiera_l.yaml",
    "sam2_1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2_1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2_1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2_1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

MODEL_SPECS = {
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


def elapsed_ms(start: float, end: float) -> float:
    """将 ``perf_counter`` 的时间差转换为毫秒。

    Args:
        start: 起始时间戳。
        end: 结束时间戳。

    Returns:
        毫秒单位耗时。
    """

    return (end - start) * 1000.0


def import_sam2_builder(sam2_root: str | None) -> Any:
    """导入官方 ``sam2.build_sam.build_sam2``。

    Args:
        sam2_root: 官方 SAM2 仓库根目录；为空时使用当前 Python 环境。

    Returns:
        官方 ``build_sam2`` 函数。
    """

    if sam2_root:
        root = Path(sam2_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"SAM2 root does not exist: {root}")
        sys.path.insert(0, str(root))

    from sam2.build_sam import build_sam2  # type: ignore

    return build_sam2


def install_optional_dependency_shims() -> None:
    """为最小导出环境安装 Hydra/iopath 兼容桩。

    官方 SAM2 包的 ``__init__`` 会导入 Hydra；Hiera 文件会导入 iopath。
    导出 TensorRT 权重只需要模型类本身，因此在这些包缺失时用轻量桩绕过导入。
    """

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


def build_sam2_without_hydra(model_name: str, checkpoint: Path, sam2_root: str | None, device: torch.device) -> Any:
    """不依赖 Hydra 直接实例化官方 SAM2 图像分割模型。

    Args:
        model_name: InferRT/SAM2 模型 key。
        checkpoint: 官方 checkpoint 路径。
        sam2_root: 官方 SAM2 仓库根目录。
        device: 模型所在设备。

    Returns:
        已加载权重并切换到 eval 的官方模型。
    """

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

    spec = MODEL_SPECS[model_name]
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

    # 官方 SAM2/SAM2.1 的单图分割仍会在 checkpoint 中保存视频 memory 模块权重。
    # fallback builder 需要实例化这些模块，避免导出时因为未构建模块而丢失或错配权重。
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


def preprocess_image(image_path: Path, device: torch.device) -> tuple[torch.Tensor, tuple[int, int]]:
    """按官方 SAM2 图像流程预处理单张图片。

    SAM2 image predictor 使用固定 ``1024x1024`` resize、``0..1`` RGB 和 ImageNet 均值方差。

    Args:
        image_path: 输入图片路径。
        device: 输出张量所在设备。

    Returns:
        ``(image_tensor, original_size)``，其中张量形状为 ``1x3x1024x1024``。
    """

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


def write_wts(state_dict: dict[str, torch.Tensor], output_path: Path, *, verbose: bool = False) -> None:
    """将官方 SAM2 ``state_dict`` 写成 InferRT 文本权重格式。

    Args:
        state_dict: 官方 SAM2 权重。
        output_path: 输出 ``.wts`` 路径。
        verbose: 为 true 时打印每个权重 key 和 shape。
    """

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


@torch.no_grad()
def run_reference_forward(model: torch.nn.Module, image: torch.Tensor, original_size: tuple[int, int]) -> None:
    """跑一遍官方 SAM2 image encoder / prompt encoder / mask decoder。

    Args:
        model: 官方 ``SAM2Base`` 模型。
        image: 预处理后的 ``1x3x1024x1024`` 图像张量。
        original_size: 原图 ``(height, width)``，用于后处理计时。
    """

    encoder_start = time.perf_counter()
    backbone_out = model.forward_image(image)
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
        image_embeddings=backbone_out["vision_features"],
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


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数对象。
    """

    parser = argparse.ArgumentParser(description="Export official SAM2/SAM2.1 checkpoint to InferRT .wts")
    parser.add_argument("-m", "--model", choices=tuple(MODEL_CONFIGS), default="sam2_1_hiera_tiny")
    parser.add_argument("-c", "--checkpoint", required=True, type=Path, help="Official SAM2 checkpoint path")
    parser.add_argument("--sam2-root", type=str, default="", help="Path to official SAM2 repository root")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .wts path")
    parser.add_argument("-i", "--img-path", type=Path, default=DEFAULT_IMAGE_PATH, help="Image for forward smoke")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-forward", action="store_true", help="Only export weights, skip PyTorch forward smoke")
    parser.add_argument("--verbose", action="store_true", help="Print every exported weight key")
    return parser.parse_args()


def main() -> None:
    """加载官方 SAM2、执行可选前向验证并导出权重。"""

    args = parse_args()
    output_path = args.output if args.output is not None else Path(f"{args.model}.wts")
    device = torch.device(args.device)

    load_start = time.perf_counter()
    try:
        build_sam2 = import_sam2_builder(args.sam2_root or None)
        model = build_sam2(
            config_file=MODEL_CONFIGS[args.model],
            ckpt_path=str(args.checkpoint),
            device=str(device),
            mode="eval",
            apply_postprocessing=False,
        )
    except ModuleNotFoundError as exc:
        print(f"Official SAM2 builder dependency missing ({exc.name}); using no-Hydra fallback builder.")
        model = build_sam2_without_hydra(args.model, args.checkpoint, args.sam2_root or None, device)
    load_end = time.perf_counter()
    print(f"Loaded official {args.model} in {elapsed_ms(load_start, load_end):.3f} ms on {device}")

    if not args.skip_forward:
        preprocess_start = time.perf_counter()
        image, original_size = preprocess_image(args.img_path, device)
        preprocess_end = time.perf_counter()
        print(f"Preprocess: {elapsed_ms(preprocess_start, preprocess_end):.3f} ms, original={original_size}")
        run_reference_forward(model, image, original_size)

    export_start = time.perf_counter()
    write_wts(model.state_dict(), output_path, verbose=args.verbose)
    export_end = time.perf_counter()
    print(f"Exported {len(model.state_dict())} tensors to {output_path}")
    print(f"Weight export: {elapsed_ms(export_start, export_end):.3f} ms")


if __name__ == "__main__":
    main()
