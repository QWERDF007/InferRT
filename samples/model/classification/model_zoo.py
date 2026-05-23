from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any, Union

import cv2
import numpy as np
import torch
from torchvision.models import (
    AlexNet_Weights,
    GoogLeNet_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    VGG11_Weights,
    VGG13_Weights,
    VGG16_Weights,
    VGG19_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    alexnet,
    googlenet,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    vgg11,
    vgg13,
    vgg16,
    vgg19,
    wide_resnet50_2,
    wide_resnet101_2,
)


TORCHVISION_MODEL_ZOO = {
    "alexnet": (alexnet, AlexNet_Weights.IMAGENET1K_V1),
    "googlenet": (googlenet, GoogLeNet_Weights.IMAGENET1K_V1),
    "mobilenet_v2": (mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V2),
    "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "vgg11": (vgg11, VGG11_Weights.IMAGENET1K_V1),
    "vgg13": (vgg13, VGG13_Weights.IMAGENET1K_V1),
    "vgg16": (vgg16, VGG16_Weights.IMAGENET1K_V1),
    "vgg19": (vgg19, VGG19_Weights.IMAGENET1K_V1),
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2),
    "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V2),
    "wide_resnet50_2": (wide_resnet50_2, Wide_ResNet50_2_Weights.IMAGENET1K_V2),
    "wide_resnet101_2": (wide_resnet101_2, Wide_ResNet101_2_Weights.IMAGENET1K_V2),
}

DEFAULT_IMAGE_SIZE = (224, 224)
ImageSizeLike = Union[int, Sequence[int]]


def read_imagenet_labels(labels_path: str) -> dict[int, str]:
    clsid2label = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            k, v = line.split(": ")
            clsid2label.setdefault(int(k), v[1:-3])
    return clsid2label


def _get_config_value(config: Any, key: str) -> Any:
    """@brief 从 dict 或对象配置中读取指定字段。

    @param config 模型配置对象，通常来自 timm 的 ``default_cfg`` 或 ``pretrained_cfg``。
    @param key 待读取的字段名。
    @return 字段存在时返回对应值，否则返回 ``None``。
    """

    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def normalize_image_size(image_size: ImageSizeLike) -> tuple[int, int]:
    """@brief 将不同格式的输入尺寸统一转换为 ``(height, width)``。

    @param image_size 输入尺寸，可为单个整数、``(H, W)``、``(C, H, W)`` 或 ``(N, C, H, W)``。
    @return 归一化后的图像高宽。
    @exception ValueError 输入尺寸格式非法或非正数时抛出。
    """

    if isinstance(image_size, int):
        height = width = image_size
    elif isinstance(image_size, Sequence) and not isinstance(image_size, (str, bytes)):
        dims = tuple(int(dim) for dim in image_size)
        if len(dims) == 2:
            height, width = dims
        elif len(dims) == 3:
            _, height, width = dims
        elif len(dims) == 4:
            _, _, height, width = dims
        else:
            raise ValueError(f"Unsupported image size rank: {len(dims)}")
    else:
        raise ValueError(f"Unsupported image size value: {image_size!r}")

    if height <= 0 or width <= 0:
        raise ValueError(f"Image size must be positive, got H={height}, W={width}")
    return height, width


def parse_image_size(value: str) -> tuple[int, int]:
    """@brief 解析命令行传入的图像输入尺寸。

    @param value 尺寸字符串，支持 ``384``、``384x384``、``384,384``、``3x384x384``
        或 ``1x3x384x384``。
    @return 归一化后的 ``(height, width)``。
    @exception ValueError 输入字符串为空或格式非法时抛出。
    """

    text = value.strip().lower()
    if not text:
        raise ValueError("Image size must not be empty")

    tokens = [token for token in re.split(r"[x,\s*]+", text) if token]
    try:
        dims = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(f"Invalid image size: {value!r}") from exc

    if len(dims) == 1:
        return normalize_image_size(dims[0])
    return normalize_image_size(dims)


def resolve_input_size(model: torch.nn.Module, default: ImageSizeLike = DEFAULT_IMAGE_SIZE) -> tuple[int, int]:
    """@brief 根据模型元信息解析预处理输入尺寸。

    @param model 已创建的 PyTorch/timm 模型实例。
    @param default 模型未暴露尺寸信息时使用的默认尺寸。
    @return 适用于 ``preprocess`` 的 ``(height, width)``。

    timm 的 ViT 会在 ``patch_embed.img_size`` 中记录真实输入尺寸；若该字段不存在，
    则回退读取 ``pretrained_cfg`` / ``default_cfg`` 的 ``input_size``，最后使用默认 224。
    """

    patch_embed = getattr(model, "patch_embed", None)
    patch_image_size = getattr(patch_embed, "img_size", None)
    if patch_image_size is not None:
        return normalize_image_size(patch_image_size)

    for config_name in ("pretrained_cfg", "default_cfg"):
        input_size = _get_config_value(getattr(model, config_name, None), "input_size")
        if input_size is not None:
            return normalize_image_size(input_size)

    return normalize_image_size(default)


def preprocess(img: np.ndarray, image_size: ImageSizeLike = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """@brief 按 ImageNet 约定预处理输入图像。

    @param img OpenCV 读取的 BGR 图像。
    @param image_size 目标输入尺寸，按 ``(height, width)`` 解析。
    @return 归一化后的 ``NCHW`` PyTorch 张量。
    """

    height, width = normalize_image_size(image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) * (1.0 / 255.0)
    img = cv2.subtract(img, (0.485, 0.456, 0.406, 0.0))
    img = cv2.divide(img, (0.229, 0.224, 0.225, 1.0))
    img = img.transpose(2, 0, 1)[None, ...]
    return torch.from_numpy(img)


def list_supported_models(backend: str) -> list[str]:
    if backend == "torchvision":
        return list(TORCHVISION_MODEL_ZOO.keys())
    if backend == "timm":
        from timm import list_models

        names: list[str] = []
        for pattern in ("resnet*", "wide_resnet*", "vit*"):
            names.extend(list_models(pattern))
        return list(dict.fromkeys(names))
    raise ValueError(f"Unsupported backend: {backend}")


def create_model(model_name: str, backend: str) -> torch.nn.Module:
    if backend == "torchvision":
        if model_name not in TORCHVISION_MODEL_ZOO:
            available = ", ".join(TORCHVISION_MODEL_ZOO.keys())
            raise ValueError(f"Unsupported torchvision model: {model_name}. Available: {available}")

        model_fn, weights = TORCHVISION_MODEL_ZOO[model_name]
        if model_name == "googlenet":
            return model_fn(weights=weights, transform_input=False)
        return model_fn(weights=weights)

    if backend == "timm":
        from timm import create_model

        return create_model(model_name, pretrained=True)

    raise ValueError(f"Unsupported backend: {backend}")
