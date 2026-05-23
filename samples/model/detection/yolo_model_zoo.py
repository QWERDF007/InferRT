from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import struct
import sys
from typing import Any

import torch


YOLO_MODEL_NAMES = [
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

YOLO_MODEL_WEIGHTS = {
    "yolov5": "yolov5su.pt",
    "yolov8": "yolov8n.pt",
    **{
        name: f"{name}u.pt" if name.startswith("yolov5") else f"{name}.pt"
        for name in YOLO_MODEL_NAMES
        if name not in {"yolov5", "yolov8"}
    },
}


def ensure_ultralytics_config_dir() -> None:
    """@brief 为 Ultralytics 准备项目内可写配置目录，避免用户目录权限影响导出。"""

    if os.environ.get("YOLO_CONFIG_DIR"):
        return
    config_dir = Path(__file__).resolve().parents[3] / "build" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)


def list_supported_models() -> list[str]:
    """@brief 列出当前 InferRT 原生 YOLO 检测构建器支持的模型 key。

    @return 模型 key 列表，例如 ``yolov5n`` 和 ``yolov8s``。
    """

    return YOLO_MODEL_NAMES.copy()


def add_ultralytics_repo(repo: str | Path | None) -> None:
    """@brief 将本地 ultralytics 仓库加入 ``sys.path``。

    @param repo 本地仓库根目录；为空时不修改导入路径。

    该函数只做路径注入，不导入 ``ultralytics``，便于测试中使用 mock 模块。
    """

    if repo is None:
        return
    repo_path = str(Path(repo).resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def resolve_model_weights(model_name: str, weights: str | Path | None) -> str:
    """@brief 解析传给 ``ultralytics.YOLO`` 的权重或模型名。

    @param model_name InferRT YOLO 模型 key。
    @param weights 用户显式传入的 ``.pt`` 路径或模型名。
    @return 可直接传给 ``YOLO(...)`` 的参数。
    @exception ValueError 模型 key 不在支持列表中时抛出。
    """

    if weights:
        return str(weights)
    try:
        return YOLO_MODEL_WEIGHTS[model_name]
    except KeyError as exc:
        available = ", ".join(YOLO_MODEL_NAMES)
        raise ValueError(f"Unsupported YOLO model: {model_name}. Available: {available}") from exc


def load_ultralytics_model(model_name: str, *, weights: str | Path | None = None, repo: str | Path | None = None) -> Any:
    """@brief 使用 Ultralytics 加载 YOLOv5/YOLOv8 PyTorch 模型。

    @param model_name InferRT YOLO 模型 key。
    @param weights 可选的 ``.pt`` 权重路径；为空时使用 ``<model>.pt``。
    @param repo 可选的本地 ``D:/Github/ultralytics`` 仓库路径。
    @return 已切换到 ``eval`` 模式的 FP32 PyTorch 模型。
    """

    source = resolve_model_weights(model_name, weights)
    ensure_ultralytics_config_dir()
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
    """@brief 获取适合 InferRT ``.wts`` 导出的 YOLO 权重表。

    @param model Ultralytics ``DetectionModel`` 或兼容对象。
    @return ``state_dict`` 权重表。

    YOLOv5u 和 YOLOv8 的原生 C++ 构建器均读取 Ultralytics ``model.*`` 命名；
    若调用方显式传入旧版 YOLOv5 权重，C++ 构建器也会回退到 tensorrtx 的
    ``model.24.m.*`` 检测头。
    """

    state_dict = model.state_dict()
    if not isinstance(state_dict, Mapping):
        raise TypeError("YOLO model state_dict() must return a mapping")
    return state_dict


def write_wts(model: Any, output_path: str | Path, *, verbose: bool = True) -> None:
    """@brief 将 Ultralytics YOLO 权重导出为 InferRT 文本 ``.wts``。

    @param model Ultralytics PyTorch 模型。
    @param output_path 输出 ``.wts`` 文件路径。
    @param verbose 为 true 时打印每个权重 key 和 shape。
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
