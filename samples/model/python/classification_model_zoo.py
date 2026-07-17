from __future__ import annotations

from collections.abc import Mapping, Sequence
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

TORCHHUB_DINOV2_REPO = "facebookresearch/dinov2"
TORCHHUB_DINOV2_MODEL_NAMES = [
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
    "dinov2_vitl14_reg",
    "dinov2_vitg14_reg",
]
TORCHHUB_DINO_MODEL_REPOS = {
    **{name: TORCHHUB_DINOV2_REPO for name in TORCHHUB_DINOV2_MODEL_NAMES},
}
TRANSFORMERS_DINOV3_MODEL_IDS = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_vitl16plus": "facebook/dinov3-vitl16plus-pretrain-lvd1689m",
    "dinov3_vith16plus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "dinov3_vit7b16": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}


def read_imagenet_labels(labels_path: str) -> dict[int, str]:
    clsid2label = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            k, v = line.split(": ")
            clsid2label.setdefault(int(k), v[1:-3])
    return clsid2label


def _get_config_value(config: Any, key: str) -> Any:
    """从 dict 或对象配置中读取指定字段。

    Args:
        config: 模型配置对象，通常来自 timm 的 ``default_cfg`` 或 ``pretrained_cfg``。
        key: 待读取的字段名。

    Returns:
        字段存在时返回对应值，否则返回 ``None``。
    """

    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    return getattr(config, key, None)


def _get_image_processor_size(image_processor: Any) -> tuple[int, int] | None:
    """从 Hugging Face image processor 中解析输入尺寸。

    Args:
        image_processor: ``transformers.pipeline`` 持有的图像预处理器。

    Returns:
        若存在固定尺寸则返回 ``(height, width)``，否则返回 ``None``。
    """

    size = getattr(image_processor, "size", None)
    if isinstance(size, Mapping):
        height = size.get("height") or size.get("shortest_edge")
        width = size.get("width") or size.get("shortest_edge")
        if height and width:
            return normalize_image_size((int(height), int(width)))
    height = getattr(size, "height", None)
    width = getattr(size, "width", None)
    shortest_edge = getattr(size, "shortest_edge", None)
    if (height and width) or shortest_edge:
        return normalize_image_size((int(height or shortest_edge), int(width or shortest_edge)))
    if size is not None:
        return normalize_image_size(size)
    return None


def _require_key(state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    """读取 state_dict 中的必需权重，缺失时给出明确错误。

    Args:
        state_dict: PyTorch 权重表。
        key: 待读取的权重名。

    Returns:
        对应的张量。

    Raises:
        KeyError: 权重不存在时抛出。
    """

    try:
        return state_dict[key]
    except KeyError as exc:
        raise KeyError(f"Missing DINOv3 Transformers weight: {key}") from exc


def _optional_key(state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor | None:
    """读取可选权重，缺失时返回 ``None``。"""

    return state_dict.get(key)


def _transformers_dinov3_layer_prefix(state_dict: Mapping[str, torch.Tensor], index: int) -> str:
    """解析 Transformers DINOv3 各层的 state_dict 前缀。

    不同版本的 Hugging Face 权重命名不一致：
    - 新版：``layer.{i}.attention.*``（无 ``model.`` 包裹）
    - 旧版：``model.layer.{i}.attention.*``（顶层带 ``model.``）

    用 ``norm1.weight`` 探测前缀，供 ``convert_transformers_dinov3_state_dict`` 统一读取。

    Args:
        state_dict: Hugging Face ``DINOv3ViTModel`` 的原始 ``state_dict``。
        index: Transformer 层索引（从 0 开始）。

    Returns:
        该层权重在 ``state_dict`` 中的前缀，形如 ``layer.{index}`` 或 ``model.layer.{index}``。
    """

    direct_prefix = f"layer.{index}"  # 新版 Transformers 命名
    legacy_prefix = f"model.layer.{index}"  # 旧版带 model. 包裹
    if f"{direct_prefix}.norm1.weight" in state_dict:
        return direct_prefix
    return legacy_prefix


def convert_transformers_dinov3_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    """将 Transformers DINOv3 权重转换为 InferRT DINOv3 构建器使用的 key。

    Args:
        state_dict: Hugging Face ``DINOv3ViTModel`` 的原始 ``state_dict``。
        config: Hugging Face DINOv3 配置对象。

    Returns:
        转换后的权重表，key 与 ``src/model/priv/DINO.cpp`` 的读取逻辑一致。

    Transformers 使用拆分的 ``q_proj/k_proj/v_proj`` 和 ``embeddings.*`` 命名；
    当前 TensorRT DINO 实现复用官方/timm 风格的 ``blocks.*.attn.qkv``、
    ``patch_embed.proj``、``cls_token`` 等命名。本函数只在导出阶段做一次机械映射，
    避免 C++ 侧同时维护多套权重命名分支。
    """

    converted: dict[str, torch.Tensor] = {}
    hidden_size = int(getattr(config, "hidden_size"))
    num_heads = int(getattr(config, "num_attention_heads"))
    depth = int(getattr(config, "num_hidden_layers"))
    head_dim = hidden_size // num_heads

    converted["cls_token"] = _require_key(state_dict, "embeddings.cls_token")
    converted["storage_tokens"] = _require_key(state_dict, "embeddings.register_tokens")
    converted["patch_embed.proj.weight"] = _require_key(state_dict, "embeddings.patch_embeddings.weight")
    converted["patch_embed.proj.bias"] = _require_key(state_dict, "embeddings.patch_embeddings.bias")
    converted["norm.weight"] = _require_key(state_dict, "norm.weight")
    converted["norm.bias"] = _require_key(state_dict, "norm.bias")

    rope_theta = float(getattr(config, "rope_theta", 100.0))
    converted["rope_embed.periods"] = torch.pow(
        torch.tensor(rope_theta, dtype=torch.float32),
        torch.arange(0, head_dim // 4, dtype=torch.float32) * (4.0 / float(head_dim)),
    )

    for index in range(depth):
        src = _transformers_dinov3_layer_prefix(state_dict, index)
        dst = f"blocks.{index}"

        for norm_name in ("norm1", "norm2"):
            converted[f"{dst}.{norm_name}.weight"] = _require_key(state_dict, f"{src}.{norm_name}.weight")
            converted[f"{dst}.{norm_name}.bias"] = _require_key(state_dict, f"{src}.{norm_name}.bias")

        q_weight = _require_key(state_dict, f"{src}.attention.q_proj.weight")
        k_weight = _require_key(state_dict, f"{src}.attention.k_proj.weight")
        v_weight = _require_key(state_dict, f"{src}.attention.v_proj.weight")
        converted[f"{dst}.attn.qkv.weight"] = torch.cat((q_weight, k_weight, v_weight), dim=0)

        q_bias = _optional_key(state_dict, f"{src}.attention.q_proj.bias")
        k_bias = _optional_key(state_dict, f"{src}.attention.k_proj.bias")
        v_bias = _optional_key(state_dict, f"{src}.attention.v_proj.bias")
        if q_bias is not None or k_bias is not None or v_bias is not None:
            reference = q_bias if q_bias is not None else v_bias if v_bias is not None else k_bias
            assert reference is not None
            zero_bias = torch.zeros_like(reference)
            converted[f"{dst}.attn.qkv.bias"] = torch.cat(
                (
                    q_bias if q_bias is not None else zero_bias,
                    k_bias if k_bias is not None else zero_bias,
                    v_bias if v_bias is not None else zero_bias,
                ),
                dim=0,
            )

        converted[f"{dst}.attn.proj.weight"] = _require_key(state_dict, f"{src}.attention.o_proj.weight")
        converted[f"{dst}.attn.proj.bias"] = _require_key(state_dict, f"{src}.attention.o_proj.bias")
        converted[f"{dst}.ls1.gamma"] = _require_key(state_dict, f"{src}.layer_scale1.lambda1")
        converted[f"{dst}.ls2.gamma"] = _require_key(state_dict, f"{src}.layer_scale2.lambda1")

        if f"{src}.mlp.gate_proj.weight" in state_dict:
            converted[f"{dst}.mlp.w1.weight"] = _require_key(state_dict, f"{src}.mlp.gate_proj.weight")
            converted[f"{dst}.mlp.w1.bias"] = _require_key(state_dict, f"{src}.mlp.gate_proj.bias")
            converted[f"{dst}.mlp.w2.weight"] = _require_key(state_dict, f"{src}.mlp.up_proj.weight")
            converted[f"{dst}.mlp.w2.bias"] = _require_key(state_dict, f"{src}.mlp.up_proj.bias")
            converted[f"{dst}.mlp.w3.weight"] = _require_key(state_dict, f"{src}.mlp.down_proj.weight")
            converted[f"{dst}.mlp.w3.bias"] = _require_key(state_dict, f"{src}.mlp.down_proj.bias")
        else:
            converted[f"{dst}.mlp.fc1.weight"] = _require_key(state_dict, f"{src}.mlp.up_proj.weight")
            converted[f"{dst}.mlp.fc1.bias"] = _require_key(state_dict, f"{src}.mlp.up_proj.bias")
            converted[f"{dst}.mlp.fc2.weight"] = _require_key(state_dict, f"{src}.mlp.down_proj.weight")
            converted[f"{dst}.mlp.fc2.bias"] = _require_key(state_dict, f"{src}.mlp.down_proj.bias")

    return converted


class TransformersDINOv3Model(torch.nn.Module):
    """基于 Hugging Face pipeline 的 DINOv3 导出适配器。

    pipeline 负责按官方模型 id 加载 pretrained 权重；本适配器提供普通
    ``torch.nn.Module`` 风格的 ``forward``、``forward_features`` 和
    ``export_state_dict``，使 DINO 导出脚本与现有测试无需感知 pipeline 包装层。
    """

    def __init__(self, feature_pipeline: Any, model_name: str, model_id: str):
        super().__init__()
        self.feature_pipeline = feature_pipeline
        self.hf_model = feature_pipeline.model
        self.image_processor = getattr(feature_pipeline, "image_processor", None)
        self.config = self.hf_model.config
        self.model_name = model_name
        self.model_id = model_id

        processor_size = _get_image_processor_size(self.image_processor)
        config_size = normalize_image_size(getattr(self.config, "image_size", DEFAULT_IMAGE_SIZE))
        height, width = processor_size if processor_size is not None else config_size
        self.default_cfg = {"input_size": (3, height, width)}
        self.pretrained_cfg = self.default_cfg

    def _to_model_device(self, x: torch.Tensor) -> torch.Tensor:
        """将输入张量移动到 HF 模型当前设备。"""

        parameter = next(self.hf_model.parameters(), None)
        if parameter is None:
            return x
        return x.to(device=parameter.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 DINOv3 CLS pooled 特征，供 DINO 导出脚本使用。"""

        x = self._to_model_device(x)
        return self.hf_model(pixel_values=x).pooler_output

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """返回与官方 DINOv3 ``forward_features`` 兼容的常用特征字典。"""

        x = self._to_model_device(x)
        outputs = self.hf_model(pixel_values=x)
        tokens = outputs.last_hidden_state
        extra_tokens = int(getattr(self.config, "num_register_tokens", 0))
        return {
            "x_norm_clstoken": tokens[:, 0],
            "x_storage_tokens": tokens[:, 1 : 1 + extra_tokens],
            "x_norm_patchtokens": tokens[:, 1 + extra_tokens :],
        }

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        """导出 InferRT DINOv3 构建器可读取的权重表。"""

        return convert_transformers_dinov3_state_dict(self.hf_model.state_dict(), self.config)


def normalize_image_size(image_size: ImageSizeLike) -> tuple[int, int]:
    """将不同格式的输入尺寸统一转换为 ``(height, width)``。

    Args:
        image_size: 输入尺寸，可为单个整数、``(H, W)``、``(C, H, W)`` 或 ``(N, C, H, W)``。

    Returns:
        归一化后的图像高宽。

    Raises:
        ValueError: 输入尺寸格式非法或非正数时抛出。
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
    """解析命令行传入的图像输入尺寸。

    Args:
        value: 尺寸字符串，支持 ``384``、``384x384``、``384,384``、``3x384x384``
        或 ``1x3x384x384``。

    Returns:
        归一化后的 ``(height, width)``。

    Raises:
        ValueError: 输入字符串为空或格式非法时抛出。
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
    """根据模型元信息解析预处理输入尺寸。

    Args:
        model: 已创建的 PyTorch/timm 模型实例。
        default: 模型未暴露尺寸信息时使用的默认尺寸。

    Returns:
        适用于 ``preprocess`` 的 ``(height, width)``。

    timm 的 ViT 会在 ``patch_embed.img_size`` 中记录真实输入尺寸；若该字段不存在，
    则回退读取 ``pretrained_cfg`` / ``default_cfg`` 的 ``input_size``，最后使用默认 224。
    """

    patch_embed = getattr(model, "patch_embed", None)
    patch_image_size = getattr(patch_embed, "img_size", None)
    if patch_image_size is not None:
        return normalize_image_size(patch_image_size)

    config_image_size = _get_config_value(getattr(model, "config", None), "image_size")
    if config_image_size is not None:
        return normalize_image_size(config_image_size)

    for config_name in ("pretrained_cfg", "default_cfg"):
        input_size = _get_config_value(getattr(model, config_name, None), "input_size")
        if input_size is not None:
            return normalize_image_size(input_size)

    return normalize_image_size(default)


def preprocess(img: np.ndarray, image_size: ImageSizeLike = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """按 ImageNet 约定预处理输入图像。

    Args:
        img: OpenCV 读取的 BGR 图像。
        image_size: 目标输入尺寸，按 ``(height, width)`` 解析。

    Returns:
        归一化后的 ``NCHW`` PyTorch 张量。
    """

    height, width = normalize_image_size(image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) * (1.0 / 255.0)
    img = cv2.subtract(img, (0.485, 0.456, 0.406, 0.0))
    img = cv2.divide(img, (0.229, 0.224, 0.225, 1.0))
    img = img.transpose(2, 0, 1)[None, ...]
    return torch.from_numpy(img)


def export_model_state_dict(model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
    """获取用于 ``.wts`` 导出的权重表。

    Args:
        model: PyTorch 模型或导出适配器。

    Returns:
        适合写入 InferRT ``.wts`` 的 state_dict。

    普通 torchvision/timm/torchhub 模型直接使用 ``state_dict``；Transformers DINOv3
    通过 ``export_state_dict`` 做 key 转换后再导出。
    """

    export_fn = getattr(model, "export_state_dict", None)
    if callable(export_fn):
        return export_fn()
    return model.state_dict()


def list_supported_models(backend: str) -> list[str]:
    """列出指定后端可导出的模型名称。

    Args:
        backend: 模型来源，支持 ``torchvision``、``timm``、``torchhub`` 和 ``transformers``。

    Returns:
        模型名称列表。

    Raises:
        ValueError: 后端名称不受支持时抛出。
    """

    if backend == "torchvision":
        return list(TORCHVISION_MODEL_ZOO.keys())
    if backend == "timm":
        from timm import list_models

        names: list[str] = []
        for pattern in ("resnet*", "wide_resnet*", "vit*", "*dinov2*", "*dinov3*"):
            names.extend(list_models(pattern))
        return list(dict.fromkeys(names))
    if backend == "torchhub":
        return list(TORCHHUB_DINO_MODEL_REPOS.keys())
    if backend == "transformers":
        return list(TRANSFORMERS_DINOV3_MODEL_IDS.keys())
    raise ValueError(f"Unsupported backend: {backend}")


def _resolve_torchhub_repo(model_name: str, hub_repo: str | None) -> str:
    """根据模型名解析默认 PyTorch Hub 仓库。

    Args:
        model_name: DINO 官方 hub 模型名。
        hub_repo: 调用方显式传入的仓库；为空时按模型系列自动选择。

    Returns:
        可传给 ``torch.hub.load`` 的仓库名或本地目录。

    Raises:
        ValueError: 模型名不属于已注册的 torchhub DINO 系列时抛出。
    """

    if hub_repo:
        return hub_repo
    try:
        return TORCHHUB_DINO_MODEL_REPOS[model_name]
    except KeyError as exc:
        available = ", ".join(TORCHHUB_DINO_MODEL_REPOS.keys())
        raise ValueError(f"Unsupported torchhub model: {model_name}. Available: {available}") from exc


def _resolve_transformers_model_id(model_name: str, model_id: str | None) -> str:
    """根据 DINOv3 key 解析 Hugging Face 模型 id。

    Args:
        model_name: InferRT / DINOv3 官方模型 key。
        model_id: 调用方显式传入的 Hugging Face 模型 id；为空时使用内置映射。

    Returns:
        可传给 ``transformers.pipeline(model=...)`` 的模型 id。

    Raises:
        ValueError: 模型名不属于已支持的 DINOv3 Transformers key 时抛出。
    """

    if model_id:
        return model_id
    try:
        return TRANSFORMERS_DINOV3_MODEL_IDS[model_name]
    except KeyError as exc:
        available = ", ".join(TRANSFORMERS_DINOV3_MODEL_IDS.keys())
        raise ValueError(f"Unsupported transformers DINOv3 model: {model_name}. Available: {available}") from exc


def create_model(
    model_name: str,
    backend: str,
    *,
    hub_repo: str | None = None,
    hub_source: str = "github",
    pretrained: bool = True,
    hub_weights: str | None = None,
    hf_model_id: str | None = None,
    local_files_only: bool = False,
) -> torch.nn.Module:
    """根据后端创建 PyTorch 参考模型。

    Args:
        model_name: 模型名称。
        backend: 模型来源，支持 ``torchvision``、``timm``、``torchhub`` 和 ``transformers``。
        hub_repo: torch.hub 仓库或本地目录；为空时按 DINOv2 模型名自动选择官方仓库。
        hub_source: torch.hub source，通常为 ``github`` 或 ``local``。
        pretrained: 是否加载预训练权重。
        hub_weights: DINO hub 的 ``weights`` 参数，可传本地 ``.pth`` 或 URL。
        hf_model_id: DINOv3 Hugging Face 模型 id；为空时按 ``model_name`` 自动选择。
        local_files_only: 是否只从本地 Hugging Face 缓存加载。

    Returns:
        已创建的 PyTorch 模型实例。

    Raises:
        ValueError: 后端或模型名称不受支持时抛出。
    """

    if backend == "torchvision":
        if model_name not in TORCHVISION_MODEL_ZOO:
            available = ", ".join(TORCHVISION_MODEL_ZOO.keys())
            raise ValueError(f"Unsupported torchvision model: {model_name}. Available: {available}")

        model_fn, weights = TORCHVISION_MODEL_ZOO[model_name]
        selected_weights = weights if pretrained else None
        if model_name == "googlenet":
            return model_fn(weights=selected_weights, transform_input=False)
        return model_fn(weights=selected_weights)

    if backend == "timm":
        from timm import create_model

        if "dinov3" in model_name.lower():
            return create_model(model_name, pretrained=pretrained, global_pool="token")
        return create_model(model_name, pretrained=pretrained)

    if backend == "torchhub":
        if model_name not in TORCHHUB_DINO_MODEL_REPOS:
            available = ", ".join(TORCHHUB_DINO_MODEL_REPOS.keys())
            raise ValueError(f"Unsupported torchhub model: {model_name}. Available: {available}")

        repo_or_dir = _resolve_torchhub_repo(model_name, hub_repo)
        hub_kwargs: dict[str, object] = {
            "source": hub_source,
            "pretrained": pretrained,
        }
        if hub_weights:
            hub_kwargs["weights"] = hub_weights
        if hub_source == "github":
            # 避免交互式 trust 提示，并跳过 GitHub API 校验以减少离线环境依赖。
            hub_kwargs["trust_repo"] = True
            hub_kwargs["skip_validation"] = True
        return torch.hub.load(repo_or_dir, model_name, **hub_kwargs)

    if backend == "transformers":
        if not pretrained:
            raise ValueError("Transformers DINOv3 export requires pretrained=True")
        model_id = _resolve_transformers_model_id(model_name, hf_model_id)
        from transformers import pipeline

        feature_pipeline = pipeline(
            model=model_id,
            task="image-feature-extraction",
            local_files_only=local_files_only,
        )
        return TransformersDINOv3Model(feature_pipeline, model_name=model_name, model_id=model_id)

    raise ValueError(f"Unsupported backend: {backend}")
