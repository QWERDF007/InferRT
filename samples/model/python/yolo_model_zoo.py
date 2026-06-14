from __future__ import annotations

from collections.abc import Mapping
import importlib.util
import os
from pathlib import Path
import struct
import sys
import types
from typing import Any

import torch


YOLO_DETECTION_MODEL_NAMES = [
    "yolov5",
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
    "yolov8",
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
]

YOLOV8_SEGMENTATION_MODEL_NAMES = [
    "yolov8_seg",
    "yolov8n_seg",
    "yolov8s_seg",
    "yolov8m_seg",
    "yolov8l_seg",
    "yolov8x_seg",
    "yolov8-seg",
    "yolov8n-seg",
    "yolov8s-seg",
    "yolov8m-seg",
    "yolov8l-seg",
    "yolov8x-seg",
]

YOLO_MODEL_NAMES = [
    *YOLO_DETECTION_MODEL_NAMES,
    *YOLOV8_SEGMENTATION_MODEL_NAMES,
]

YOLO_MODEL_WEIGHTS = {
    "yolov5": "yolov5su.pt",
    "yolov8": "yolov8n.pt",
    "yolov8_seg": "yolov8n-seg.pt",
    "yolov8-seg": "yolov8n-seg.pt",
    **{
        name: f"{name}u.pt" if name.startswith("yolov5") else f"{name}.pt"
        for name in YOLO_DETECTION_MODEL_NAMES
        if name not in {"yolov5", "yolov8"}
    },
    **{
        name: f"{name.replace('_seg', '-seg')}.pt" if name.endswith("_seg") else f"{name}.pt"
        for name in YOLOV8_SEGMENTATION_MODEL_NAMES
        if name not in {"yolov8_seg", "yolov8-seg"}
    },
}


def ensure_ultralytics_config_dir() -> None:
    """为 Ultralytics 准备项目内可写配置目录，避免用户目录权限影响导出。"""

    if os.environ.get("YOLO_CONFIG_DIR"):
        return
    config_dir = Path(__file__).resolve().parents[3] / "build" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)


def list_supported_models() -> list[str]:
    """列出当前 InferRT 原生 YOLO 检测构建器支持的模型 key。

    Returns:
        模型 key 列表，例如 ``yolov5n`` 和 ``yolov8s``。
    """

    return YOLO_MODEL_NAMES.copy()


def add_ultralytics_repo(repo: str | Path | None) -> None:
    """将本地 ultralytics 仓库加入 ``sys.path``。

    Args:
        repo: 本地仓库根目录；为空时不修改导入路径。

    该函数只做路径注入，不导入 ``ultralytics``，便于测试中使用 mock 模块。
    """

    if repo is None:
        return
    repo_path = str(Path(repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def add_yolov5_repo(repo: str | Path | None) -> None:
    """将本地 YOLOv5 仓库加入 ``sys.path``，用于加载旧版 YOLOv5 checkpoint。"""

    if repo is None:
        return
    repo_path = str(Path(repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def infer_yolov5_repo(ultralytics_repo: str | Path | None) -> Path | None:
    """从显式环境变量或 Ultralytics 仓库旁推断本地 YOLOv5 仓库。"""

    env_repo = os.environ.get("INFERRT_YOLOV5_REPO")
    if env_repo:
        return Path(env_repo)
    if ultralytics_repo is not None:
        sibling = Path(ultralytics_repo).resolve().parent / "yolov5"
        if sibling.exists():
            return sibling
    default = Path("F:/Github/CV/yolov5")
    return default if default.exists() else None


def is_legacy_yolov5_source(model_name: str, source: str) -> bool:
    """判断显式权重是否应按旧版 YOLOv5 repo 加载。"""

    path = Path(source)
    return (
        model_name.startswith("yolov5")
        and path.suffix.lower() == ".pt"
        and path.stem.startswith("yolov5")
        and not path.stem.endswith("u")
    )


def load_legacy_yolov5_model(source: str, *, repo: str | Path | None, ultralytics_repo: str | Path | None) -> Any:
    """使用本地 YOLOv5 仓库加载旧版 anchor-head checkpoint。"""

    yolov5_repo = Path(repo) if repo is not None else infer_yolov5_repo(ultralytics_repo)
    if yolov5_repo is None or not yolov5_repo.exists():
        raise FileNotFoundError("Legacy YOLOv5 checkpoint requires --yolov5-repo or INFERRT_YOLOV5_REPO")

    add_ultralytics_repo(ultralytics_repo)
    add_yolov5_repo(yolov5_repo)
    if "seaborn" not in sys.modules and importlib.util.find_spec("seaborn") is None:
        sys.modules.setdefault("seaborn", types.SimpleNamespace())

    from models.experimental import attempt_load  # type: ignore

    return attempt_load(source, device="cpu", inplace=True, fuse=False)


def resolve_model_weights(model_name: str, weights: str | Path | None) -> str:
    """解析传给 ``ultralytics.YOLO`` 的权重或模型名。

    Args:
        model_name: InferRT YOLO 模型 key。
        weights: 用户显式传入的 ``.pt`` 路径或模型名。

    Returns:
        可直接传给 ``YOLO(...)`` 的参数。

    Raises:
        ValueError: 模型 key 不在支持列表中时抛出。
    """

    if weights:
        return str(weights)
    try:
        return YOLO_MODEL_WEIGHTS[model_name]
    except KeyError as exc:
        available = ", ".join(YOLO_MODEL_NAMES)
        raise ValueError(f"Unsupported YOLO model: {model_name}. Available: {available}") from exc


def load_ultralytics_model(
    model_name: str,
    *,
    weights: str | Path | None = None,
    repo: str | Path | None = None,
    yolov5_repo: str | Path | None = None,
) -> Any:
    """使用 Ultralytics 加载 YOLOv5/YOLOv8 PyTorch 模型。

    Args:
        model_name: InferRT YOLO 模型 key。
        weights: 可选的 ``.pt`` 权重路径；为空时使用 ``<model>.pt``。
        repo: 可选的本地 ``D:/Github/ultralytics`` 仓库路径。

    Returns:
        已切换到 ``eval`` 模式的 FP32 PyTorch 模型。
    """

    source = resolve_model_weights(model_name, weights)
    ensure_ultralytics_config_dir()
    if is_legacy_yolov5_source(model_name, source):
        model = load_legacy_yolov5_model(source, repo=yolov5_repo, ultralytics_repo=repo)
        return model.float().eval()

    add_ultralytics_repo(repo)
    try:
        from ultralytics import YOLO

        wrapper = YOLO(source)
        model = wrapper.model
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        from ultralytics.nn.tasks import DetectionModel, load_checkpoint

        if Path(source).suffix.lower() in {".yaml", ".yml"}:
            model = DetectionModel(str(source))
        else:
            model, _ = load_checkpoint(source)
    return model.float().eval()


def export_state_dict(model: Any) -> Mapping[str, torch.Tensor]:
    """获取适合 InferRT ``.wts`` 导出的 YOLO 权重表。

    Args:
        model: Ultralytics ``DetectionModel`` 或兼容对象。

    Returns:
        ``state_dict`` 权重表。

    YOLOv5u 和 YOLOv8 的原生 C++ 构建器均读取 Ultralytics ``model.*`` 命名；
    若调用方显式传入旧版 YOLOv5 权重，C++ 构建器也会回退到 tensorrtx 的
    ``model.24.m.*`` 检测头。
    """

    state_dict = model.state_dict()
    if not isinstance(state_dict, Mapping):
        raise TypeError("YOLO model state_dict() must return a mapping")
    return state_dict


def write_wts(model: Any, output_path: str | Path, *, verbose: bool = True) -> None:
    """将 Ultralytics YOLO 权重导出为 InferRT 文本 ``.wts``。

    Args:
        model: Ultralytics PyTorch 模型。
        output_path: 输出 ``.wts`` 文件路径。
        verbose: 为 true 时打印每个权重 key 和 shape。
    """

    state_dict = export_state_dict(model)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{len(state_dict)}\n")
        for key, tensor in state_dict.items():
            value = tensor.detach().reshape(-1).cpu().numpy()
            if verbose:
                print(f"key: {key}\tvalue: {tuple(tensor.shape)}")
            f.write(f"{key} {len(value)}")
            for item in value:
                f.write(" ")
                f.write(struct.pack(">f", float(item)).hex())
            f.write("\n")
