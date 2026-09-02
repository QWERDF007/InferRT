"""DINOv2、DINOv3 和 LingBot-Vision 的 PyTorch 模型加载适配。"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import torch

from classification_model_zoo import (
    TRANSFORMERS_DINOV3_MODEL_IDS,
    TORCHHUB_DINO_MODEL_ALIASES,
    TORCHHUB_DINO_MODEL_REPOS,
    create_model as create_classification_model,
    export_model_state_dict,
    list_supported_models as list_classification_models,
    normalize_image_size,
    parse_image_size,
    resolve_input_size,
)


DINO_BACKENDS = ("torchhub", "transformers", "timm", "lingbot")
DEFAULT_LINGBOT_ROOT = Path(
    os.environ.get("INFERRT_LINGBOT_VISION_REPO", "F:/Github/lingbot-vision")
).resolve()
DEFAULT_LINGBOT_MODEL_DIR = Path(
    os.environ.get("INFERRT_LINGBOT_VISION_ROOT", "F:/models/lingbot-vision")
).resolve()

LINGBOT_MODEL_CONFIGS = {
    "lingbot_vision_vits16": ("lbot_vision_vits.yaml", ("lingbot-vision-vit-small.pt", "lbotv_vit_small.pt")),
    "lingbot_vision_vitb16": ("lbot_vision_vitb.yaml", ("lingbot-vision-vit-base.pt", "lbotv_vit_base.pt")),
    "lingbot_vision_vitl16": ("lbot_vision_vitl.yaml", ("lingbot-vision-vit-large.pt", "lbotv_vit_large.pt")),
    "lingbot_vision_vitg16": ("lbot_vision_vitg.yaml", ("lingbot-vision-vit-giant.pt", "lbotv_vit_giant.pt")),
}


def list_supported_models(backend: str) -> list[str]:
    """列出指定 DINO 模型来源支持的模型名。"""

    if backend == "torchhub":
        return [*TORCHHUB_DINO_MODEL_REPOS, *TORCHHUB_DINO_MODEL_ALIASES]
    if backend == "transformers":
        return list(TRANSFORMERS_DINOV3_MODEL_IDS)
    if backend == "timm":
        return [name for name in list_classification_models("timm") if "dino" in name.lower()]
    if backend == "lingbot":
        return list(LINGBOT_MODEL_CONFIGS)
    if backend == "all":
        return sorted(
            {
                *list_supported_models("torchhub"),
                *list_supported_models("transformers"),
                *list_supported_models("timm"),
                *list_supported_models("lingbot"),
            }
        )
    raise ValueError(f"Unsupported DINO backend: {backend}")


def resolve_lingbot_checkpoint(
    model_name: str,
    checkpoint: str | Path | None,
    model_dir: str | Path = DEFAULT_LINGBOT_MODEL_DIR,
) -> Path:
    """解析 LingBot-Vision checkpoint，支持显式路径和默认模型目录。"""

    if model_name not in LINGBOT_MODEL_CONFIGS:
        available = ", ".join(LINGBOT_MODEL_CONFIGS)
        raise ValueError(f"Unsupported LingBot-Vision model: {model_name}. Available: {available}")
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"LingBot-Vision checkpoint not found: {path}")
        return path

    root = Path(model_dir).expanduser().resolve()
    names = {name.lower() for name in LINGBOT_MODEL_CONFIGS[model_name][1]}
    candidates = [path for path in root.rglob("*") if path.is_file() and path.name.lower() in names] if root.exists() else []
    if not candidates:
        expected = ", ".join(LINGBOT_MODEL_CONFIGS[model_name][1])
        raise FileNotFoundError(f"Could not find {expected} under {root}")
    return sorted(candidates)[0]


def _add_lingbot_root(root: str | Path | None) -> Path:
    path = Path(root or DEFAULT_LINGBOT_ROOT).expanduser().resolve()
    package_dir = path / "lingbot_vision"
    if not package_dir.is_dir():
        raise FileNotFoundError(f"LingBot-Vision source package not found: {package_dir}")
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    return path


def load_lingbot_model(
    model_name: str,
    *,
    checkpoint: str | Path | None = None,
    model_dir: str | Path = DEFAULT_LINGBOT_MODEL_DIR,
    lingbot_root: str | Path | None = DEFAULT_LINGBOT_ROOT,
    device: str = "cpu",
) -> torch.nn.Module:
    """从 LingBot-Vision 官方配置和 checkpoint 加载冻结 backbone。"""

    root = _add_lingbot_root(lingbot_root)
    checkpoint_path = resolve_lingbot_checkpoint(model_name, checkpoint, model_dir)
    config_name = LINGBOT_MODEL_CONFIGS[model_name][0]

    from lingbot_vision import load_backbone, load_backbone_state, load_config

    config_path = root / "lingbot_vision" / "configs" / config_name
    config = load_config(config_path)
    model, _ = load_backbone(
        config,
        load_backbone_state(checkpoint_path),
        device=device,
        dtype=torch.float32,
        verbose=False,
    )
    input_size = int(config.crops.global_crops_size)
    model.default_cfg = {"input_size": (3, input_size, input_size)}
    model.pretrained_cfg = model.default_cfg
    return model.eval()


def create_model(
    model_name: str,
    backend: str,
    *,
    checkpoint: str | Path | None = None,
    model_dir: str | Path = DEFAULT_LINGBOT_MODEL_DIR,
    lingbot_root: str | Path | None = DEFAULT_LINGBOT_ROOT,
    device: str = "cpu",
    hub_repo: str | None = None,
    hub_source: str = "github",
    pretrained: bool = True,
    hub_weights: str | None = None,
    hf_model_id: str | None = None,
    local_files_only: bool = False,
) -> torch.nn.Module:
    """创建 DINOv2/DINOv3/timm DINO 或 LingBot-Vision 模型。"""

    if backend == "lingbot":
        return load_lingbot_model(
            model_name,
            checkpoint=checkpoint,
            model_dir=model_dir,
            lingbot_root=lingbot_root,
            device=device,
        )
    if model_name not in list_supported_models(backend):
        available = ", ".join(list_supported_models(backend))
        raise ValueError(f"Unsupported DINO model: {model_name}. Available: {available}")
    return create_classification_model(
        model_name,
        backend,
        hub_repo=hub_repo,
        hub_source=hub_source,
        pretrained=pretrained,
        hub_weights=hub_weights,
        hf_model_id=hf_model_id,
        local_files_only=local_files_only,
    )


__all__ = [
    "DINO_BACKENDS",
    "DEFAULT_LINGBOT_MODEL_DIR",
    "DEFAULT_LINGBOT_ROOT",
    "LINGBOT_MODEL_CONFIGS",
    "create_model",
    "export_model_state_dict",
    "list_supported_models",
    "normalize_image_size",
    "parse_image_size",
    "resolve_input_size",
    "resolve_lingbot_checkpoint",
]
