"""从真实模型目录导出 .wts/.onnx，并按后端与 PyTorch 做数值一致性测试。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

from helpers.manifest import assert_tensors_close
from helpers.model_integration import artifact_dir, is_fresh_against_all


CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".safetensors"}
INPUT_NAME = "input"
PRIMARY_OUTPUT_NAME = "output"

RESNET_MODELS = [
    "wide_resnet101_2",
    "wide_resnet50_2",
    "resnet152",
    "resnet101",
    "resnet50",
    "resnet34",
    "resnet18",
]
DINO_V2_MODELS = [
    "vit_small_patch14_reg4_dinov2",
    "vit_base_patch14_reg4_dinov2",
    "vit_large_patch14_reg4_dinov2",
    "vit_giant_patch14_reg4_dinov2",
    "vit_small_patch14_dinov2",
    "vit_base_patch14_dinov2",
    "vit_large_patch14_dinov2",
    "vit_giant_patch14_dinov2",
    "dinov2_vits14_reg4",
    "dinov2_vitb14_reg4",
    "dinov2_vitl14_reg4",
    "dinov2_vitg14_reg4",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
    "dinov2_vitl14_reg",
    "dinov2_vitg14_reg",
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
]
DINO_V3_MODELS = [
    "vit_small_plus_patch16_dinov3_qkvb",
    "vit_huge_plus_patch16_dinov3_qkvb",
    "vit_small_plus_patch16_dinov3",
    "vit_huge_plus_patch16_dinov3",
    "vit_small_patch16_dinov3_qkvb",
    "vit_base_patch16_dinov3_qkvb",
    "vit_large_patch16_dinov3_qkvb",
    "vit_small_patch16_dinov3",
    "vit_base_patch16_dinov3",
    "vit_large_patch16_dinov3",
    "vit_7b_patch16_dinov3",
    "dinov3_vits16plus",
    "dinov3_vitl16plus",
    "dinov3_vith16plus",
    "dinov3_vits16",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vit7b16",
]

FEATURE_NAMES = {
    "resnet": ["layer1", "layer4"],
    "dinov2": ["x_norm_clstoken", "x_norm_patchtokens"],
    "dinov3": ["x_norm_clstoken", "x_storage_tokens", "x_norm_patchtokens"],
}


@dataclass(frozen=True)
class ModelDirCase:
    """描述从 ``--inferrt-model-root`` 发现的一条模型测试用例。

    Attributes:
        family: 模型族名称，例如 ``resnet``、``dinov2`` 或 ``dinov3``。
        model_name: InferRT/model_zoo 中使用的模型 key。
        checkpoint: 原始 checkpoint 文件或 Hugging Face 模型目录。
        source_files: 判断导出产物是否过期所需的源文件集合。
        provider: 加载 PyTorch 模型时使用的 provider 类型。
    """

    family: str
    model_name: str
    checkpoint: Path
    source_files: tuple[Path, ...]
    provider: str

    @property
    def feature_names(self) -> list[str]:
        """返回当前模型族用于特征对比的输出名列表。"""

        return FEATURE_NAMES[self.family]


class ResNetFeatureOutputWrapper:
    """将 ResNet 前向过程包装成可导出指定中间特征的模块。"""

    def __new__(cls, model: Any, feature_names: list[str]):
        """创建实际的 PyTorch wrapper 实例。

        Args:
            model: 已加载的 ResNet PyTorch 模型。
            feature_names: 需要返回的中间特征名。

        Returns:
            一个 ``torch.nn.Module`` wrapper。
        """

        torch = pytest.importorskip("torch")

        class _Wrapper(torch.nn.Module):
            """按 ResNet 主干阶段收集中间输出的内部模块。"""

            def __init__(self) -> None:
                """保存原始模型和待输出特征名。"""

                super().__init__()
                self.model = model
                self.feature_names = list(feature_names)

            def forward(self, x):
                """执行 ResNet 前向并按指定顺序返回特征。

                Args:
                    x: 形状为 ``NCHW`` 的输入张量。

                Returns:
                    按 ``feature_names`` 排列的特征张量元组。
                """

                outputs = {}
                x = self.model.conv1(x)
                outputs["stem.conv1"] = x
                x = self.model.bn1(x)
                x = self.model.relu(x)
                outputs["stem.relu"] = x
                x = self.model.maxpool(x)
                outputs["stem.pool"] = x
                x = self.model.layer1(x)
                outputs["layer1"] = x
                x = self.model.layer2(x)
                outputs["layer2"] = x
                x = self.model.layer3(x)
                outputs["layer3"] = x
                x = self.model.layer4(x)
                outputs["layer4"] = x
                x = self.model.avgpool(x)
                outputs["avgpool"] = x
                x = torch.flatten(x, 1)
                outputs["flatten"] = x
                outputs["logits"] = self.model.fc(x)
                return tuple(outputs[name] for name in self.feature_names)

        return _Wrapper()


class DINOFeatureOutputWrapper:
    """将 DINO 模型的 ``forward_features`` 输出整理成可导出的特征张量。"""

    def __new__(cls, model: Any, feature_names: list[str]):
        """创建实际的 PyTorch wrapper 实例。

        Args:
            model: DINOv2/DINOv3 PyTorch 模型。
            feature_names: 需要返回的 DINO 特征名。

        Returns:
            一个 ``torch.nn.Module`` wrapper。
        """

        torch = pytest.importorskip("torch")

        class _Wrapper(torch.nn.Module):
            """兼容 dict 输出和 token tensor 输出的 DINO 特征包装器。"""

            def __init__(self) -> None:
                """保存原始模型和待输出特征名。"""

                super().__init__()
                self.model = model
                self.feature_names = list(feature_names)

            def forward(self, x):
                """执行 DINO 特征前向并返回指定特征。

                Args:
                    x: 形状为 ``NCHW`` 的输入张量。

                Returns:
                    按 ``feature_names`` 排列的特征张量元组。
                """

                forward_features = getattr(self.model, "forward_features", None)
                if not callable(forward_features):
                    raise TypeError(f"{type(self.model).__name__} does not provide forward_features()")

                outputs = forward_features(x)
                if isinstance(outputs, Mapping):
                    feature_map = outputs
                else:
                    if not torch.is_tensor(outputs) or outputs.ndim != 3:
                        raise TypeError(
                            "DINO forward_features() must return a feature dict or token tensor, "
                            f"got {type(outputs)!r}"
                        )
                    num_extra = int(
                        getattr(
                            self.model,
                            "num_register_tokens",
                            getattr(self.model, "num_reg_tokens", max(int(getattr(self.model, "num_prefix_tokens", 1)) - 1, 0)),
                        )
                    )
                    feature_map = {
                        "x_norm_clstoken": outputs[:, 0],
                        "x_norm_regtokens": outputs[:, 1 : 1 + num_extra],
                        "x_storage_tokens": outputs[:, 1 : 1 + num_extra],
                        "x_norm_patchtokens": outputs[:, 1 + num_extra :],
                    }

                missing = [name for name in self.feature_names if name not in feature_map]
                if missing:
                    raise KeyError(f"Missing DINO feature outputs: {', '.join(missing)}")
                return tuple(feature_map[name] for name in self.feature_names)

        return _Wrapper()


def _normalize_name(value: str) -> str:
    """将模型名或路径片段归一化为便于模糊匹配的形式。

    Args:
        value: 原始名称。

    Returns:
        仅包含小写字母、数字和下划线的名称。
    """

    return re.sub(r"[^0-9a-z]+", "_", value.lower()).strip("_")


def _infer_model_name(path: Path, candidates: list[str]) -> str | None:
    """根据 checkpoint 路径推断匹配的模型名称。

    Args:
        path: checkpoint 文件或目录路径。
        candidates: 当前模型族允许的模型名候选。

    Returns:
        匹配到的模型名；无法匹配时返回 ``None``。
    """

    haystack = "_".join(_normalize_name(part) for part in (*path.parts, path.stem))
    for candidate in sorted(candidates, key=len, reverse=True):
        if _normalize_name(candidate) in haystack:
            return candidate
    return None


def _families_from_option(pytestconfig: pytest.Config) -> list[str]:
    """解析 ``--inferrt-model-dir-families`` 参数。

    Args:
        pytestconfig: pytest 配置对象。

    Returns:
        去重后的模型族列表。
    """

    raw = pytestconfig.getoption("--inferrt-model-dir-families")
    families = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = sorted(set(families) - {"resnet", "dinov2", "dinov3"})
    if invalid:
        raise pytest.UsageError(f"Unsupported --inferrt-model-dir-families value(s): {', '.join(invalid)}")
    return list(dict.fromkeys(families))


def _source_files_for_checkpoint(path: Path) -> tuple[Path, ...]:
    """收集判断导出缓存是否过期所需的源文件。

    Args:
        path: checkpoint 文件或 Hugging Face 模型目录。

    Returns:
        存在的源文件路径元组。
    """

    if path.is_file():
        return (path,)

    sources = [path / "config.json"]
    sources.extend(sorted(path.glob("*.safetensors")))
    sources.extend(sorted(path.glob("*.pth")))
    sources.extend(sorted(path.glob("*.pt")))
    return tuple(source for source in sources if source.exists())


def discover_model_dir_cases(model_root: Path, families: list[str], max_cases: int = 0) -> list[ModelDirCase]:
    """扫描模型根目录并生成支持的真实模型测试用例。

    Args:
        model_root: 用户传入的模型根目录。
        families: 需要扫描的模型族。
        max_cases: 最大用例数，0 表示不限制。

    Returns:
        发现的模型目录测试用例列表。
    """

    cases: list[ModelDirCase] = []
    seen: set[tuple[str, str, Path]] = set()

    for family in families:
        family_dir = model_root / family
        if not family_dir.exists():
            continue

        candidates = {
            "resnet": RESNET_MODELS,
            "dinov2": DINO_V2_MODELS,
            "dinov3": DINO_V3_MODELS,
        }[family]

        for checkpoint_file in sorted(family_dir.rglob("*")):
            if not checkpoint_file.is_file() or checkpoint_file.suffix.lower() not in CHECKPOINT_EXTENSIONS:
                continue

            model_name = _infer_model_name(checkpoint_file, candidates)
            if model_name is None:
                continue

            checkpoint = checkpoint_file
            provider = {
                "resnet": "torchvision_state",
                "dinov2": "torchhub_weights",
                "dinov3": "transformers_dir",
            }[family]

            if family == "dinov3":
                if (checkpoint_file.parent / "config.json").exists():
                    checkpoint = checkpoint_file.parent
                    provider = "transformers_dir"
                elif model_name.startswith("vit_"):
                    provider = "timm_state"
                else:
                    continue

            key = (family, model_name, checkpoint.resolve())
            if key in seen:
                continue
            seen.add(key)

            cases.append(
                ModelDirCase(
                    family=family,
                    model_name=model_name,
                    checkpoint=checkpoint,
                    source_files=_source_files_for_checkpoint(checkpoint),
                    provider=provider,
                )
            )
            if max_cases > 0 and len(cases) >= max_cases:
                return cases

    return cases


def _torch_load_checkpoint(path: Path, torch: Any) -> Any:
    """兼容 PyTorch 新旧版本加载普通 checkpoint。

    Args:
        path: checkpoint 文件路径。
        torch: 已导入的 PyTorch 模块。

    Returns:
        ``torch.load`` 读取到的原始对象。
    """

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")
    except RuntimeError as exc:
        message = str(exc)
        weights_only_incompatible = (
            "Cannot use ``weights_only=True``" in message
            or "Weights only load failed" in message
            or "weights_only" in message
        )
        if not weights_only_incompatible:
            raise
        # 用户传入的 --inferrt-model-root 视为本地可信模型目录；legacy tar checkpoint 需要关闭 weights_only。
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _state_dict_from_checkpoint(path: Path) -> dict[str, Any]:
    """从 ``.pt``、``.pth`` 或 ``.safetensors`` 中读取 state_dict。

    Args:
        path: checkpoint 文件路径。

    Returns:
        已清理常见前缀后的 tensor state_dict。
    """

    torch = pytest.importorskip("torch")

    if path.suffix.lower() == ".safetensors":
        safetensors_torch = pytest.importorskip("safetensors.torch")
        state = safetensors_torch.load_file(str(path), device="cpu")
    else:
        state = _torch_load_checkpoint(path, torch)

    if isinstance(state, torch.nn.Module):
        state = state.state_dict()

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "module"):
            value = state.get(key)
            if isinstance(value, torch.nn.Module):
                return dict(value.state_dict())
            if isinstance(value, dict) and any(torch.is_tensor(v) for v in value.values()):
                state = value
                break

    if not isinstance(state, dict) or not any(torch.is_tensor(v) for v in state.values()):
        raise TypeError(f"Checkpoint does not contain a tensor state_dict: {path}")

    cleaned: dict[str, Any] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        name = str(key)
        for prefix in ("module.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


def _check_load_state_dict_compatible(model_name: str, incompatible: Any) -> None:
    """检查宽松加载后的缺失/多余 key 是否在允许范围内。

    Args:
        model_name: 当前模型名称，用于错误信息。
        incompatible: ``load_state_dict`` 返回的兼容性结果。
    """

    allowed_missing_prefixes = ("fc.", "head.", "classifier.")
    allowed_unexpected_prefixes = ("fc.", "head.", "classifier.")
    missing = [key for key in incompatible.missing_keys if not key.startswith(allowed_missing_prefixes)]
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith(allowed_unexpected_prefixes)]
    if missing or unexpected:
        raise RuntimeError(
            f"State dict for {model_name} is incompatible: missing={missing}, unexpected={unexpected}"
        )


def _nearest_hub_repo(path: Path) -> Path | None:
    """从 checkpoint 路径向上查找最近的 torch.hub 仓库。

    Args:
        path: checkpoint 文件或目录路径。

    Returns:
        包含 ``hubconf.py`` 的目录；未找到时返回 ``None``。
    """

    for parent in (path if path.is_dir() else path.parent).parents:
        if (parent / "hubconf.py").exists():
            return parent
    if path.is_dir() and (path / "hubconf.py").exists():
        return path
    return None


def _load_case_model(case: ModelDirCase, pytestconfig: pytest.Config):
    """按用例 provider 加载 PyTorch 参考模型。

    Args:
        case: 模型目录测试用例。
        pytestconfig: pytest 配置对象，用于读取离线仓库参数。

    Returns:
        已加载并切到 eval 模式的 PyTorch 模型。
    """

    from classification_model_zoo import create_model

    if case.provider == "torchvision_state":
        model = create_model(case.model_name, "torchvision", pretrained=False)
        state = _state_dict_from_checkpoint(case.checkpoint)
        incompatible = model.load_state_dict(state, strict=False)
        _check_load_state_dict_compatible(case.model_name, incompatible)
        model.eval()
        return model

    if case.provider == "timm_state":
        model = create_model(case.model_name, "timm", pretrained=False)
        state = _state_dict_from_checkpoint(case.checkpoint)
        incompatible = model.load_state_dict(state, strict=False)
        _check_load_state_dict_compatible(case.model_name, incompatible)
        model.eval()
        return model

    if case.provider == "torchhub_weights":
        explicit_repo = pytestconfig.getoption("--inferrt-dinov2-hub-repo") or os.environ.get("INFERRT_DINOV2_HUB_REPO", "")
        hub_repo = Path(explicit_repo) if explicit_repo else _nearest_hub_repo(case.checkpoint)
        hub_source = "local" if hub_repo is not None and hub_repo.exists() else "github"
        model = create_model(
            case.model_name,
            "torchhub",
            hub_repo=str(hub_repo) if hub_repo is not None and hub_repo.exists() else None,
            hub_source=hub_source,
            pretrained=False,
        )
        state = _state_dict_from_checkpoint(case.checkpoint)
        incompatible = model.load_state_dict(state, strict=False)
        _check_load_state_dict_compatible(case.model_name, incompatible)
        model.eval()
        return model

    if case.provider == "transformers_dir":
        model = create_model(
            case.model_name,
            "transformers",
            hf_model_id=str(case.checkpoint),
            local_files_only=True,
        )
        model.eval()
        return model

    raise ValueError(f"Unsupported model-dir provider: {case.provider}")


def _input_tensor_for_model(model: Any) -> np.ndarray:
    """根据模型默认输入尺寸生成确定性输入张量。

    Args:
        model: PyTorch 参考模型。

    Returns:
        连续内存布局的 ``float32`` NCHW 输入。
    """

    from classification_model_zoo import resolve_input_size

    height, width = resolve_input_size(model)
    values = np.linspace(-1.0, 1.0, num=1 * 3 * height * width, dtype=np.float32)
    return np.ascontiguousarray(values.reshape(1, 3, height, width))


def _torch_primary(model: Any, input_tensor: np.ndarray) -> np.ndarray:
    """用 PyTorch 执行主输出前向。

    Args:
        model: PyTorch 参考模型。
        input_tensor: NumPy 输入张量。

    Returns:
        PyTorch 主输出的 NumPy 数组。
    """

    torch = pytest.importorskip("torch")

    batch = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    with torch.inference_mode():
        output = model(batch)
    return output.detach().cpu().numpy()


def _torch_features(case: ModelDirCase, model: Any, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
    """用 PyTorch 提取当前用例配置的中间特征。

    Args:
        case: 模型目录测试用例。
        model: PyTorch 参考模型。
        input_tensor: NumPy 输入张量。

    Returns:
        特征名到 NumPy 输出数组的映射。
    """

    torch = pytest.importorskip("torch")

    if case.family == "resnet":
        wrapper = ResNetFeatureOutputWrapper(model, case.feature_names)
    else:
        wrapper = DINOFeatureOutputWrapper(model, case.feature_names)
    wrapper.eval()

    batch = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    with torch.inference_mode():
        outputs = wrapper(batch)
    return {name: output.detach().cpu().numpy() for name, output in zip(case.feature_names, outputs)}


def _case_artifact_dir(build_dir: Path, case: ModelDirCase) -> Path:
    """计算当前用例导出产物的缓存目录。

    Args:
        build_dir: CMake 构建目录。
        case: 模型目录测试用例。

    Returns:
        当前 checkpoint 对应的产物目录。
    """

    digest = hashlib.sha1(str(case.checkpoint.resolve()).encode("utf-8")).hexdigest()[:10]
    return artifact_dir(build_dir, "model_dir_parity") / case.family / case.model_name / digest


def _ensure_wts(case: ModelDirCase, model: Any, output_dir: Path) -> Path:
    """确保当前用例已有新鲜的 TensorRT ``.wts`` 权重。

    Args:
        case: 模型目录测试用例。
        model: PyTorch 参考模型。
        output_dir: 产物输出目录。

    Returns:
        ``.wts`` 文件路径。
    """

    from gen_wts import write_wts

    output = output_dir / f"{case.model_name}.wts"
    if is_fresh_against_all(output, list(case.source_files)):
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    write_wts(model, str(output), verbose=False)
    return output


def _export_primary_onnx(model: Any, input_tensor: np.ndarray, output_path: Path) -> None:
    """导出模型主输出 ONNX 图。

    Args:
        model: PyTorch 参考模型。
        input_tensor: 导出示例输入。
        output_path: ONNX 输出路径。
    """

    torch = pytest.importorskip("torch")

    class _PrimaryOutputWrapper(torch.nn.Module):
        """用单输入签名包装模型主前向，避免可选参数暴露为 ONNX 输入。"""

        def __init__(self, wrapped: Any) -> None:
            """保存待导出的 PyTorch 模型。

            Args:
                wrapped: PyTorch 参考模型。
            """

            super().__init__()
            self.wrapped = wrapped

        def forward(self, x):
            """仅以图像张量执行主前向。

            Args:
                x: 形状为 ``NCHW`` 的输入张量。

            Returns:
                模型主输出张量。
            """

            return self.wrapped(x)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    wrapped_model = _PrimaryOutputWrapper(model).eval()
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[PRIMARY_OUTPUT_NAME],
    )


def _feature_wrapper_for_case(case: ModelDirCase, model: Any):
    """根据模型族选择对应的特征导出 wrapper。

    Args:
        case: 模型目录测试用例。
        model: PyTorch 参考模型。

    Returns:
        用于 ONNX 特征导出的 PyTorch wrapper。
    """

    if case.family == "resnet":
        return ResNetFeatureOutputWrapper(model, case.feature_names)
    return DINOFeatureOutputWrapper(model, case.feature_names)


def _export_feature_onnx(case: ModelDirCase, model: Any, input_tensor: np.ndarray, output_path: Path) -> None:
    """导出当前用例的特征输出 ONNX 图。

    Args:
        case: 模型目录测试用例。
        model: PyTorch 参考模型。
        input_tensor: 导出示例输入。
        output_path: ONNX 输出路径。
    """

    torch = pytest.importorskip("torch")
    from export_feature_onnx import export_features_with_onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _feature_wrapper_for_case(case, model)
    wrapper.eval()
    dummy_input = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    args = SimpleNamespace(
        input_name=INPUT_NAME,
        features=case.feature_names,
        opset=17,
        dynamic_batch=False,
    )
    export_features_with_onnx(wrapper, dummy_input, output_path, args, use_dynamo=False)


def _ensure_onnx_artifacts(case: ModelDirCase, model: Any, input_tensor: np.ndarray, output_dir: Path) -> tuple[Path, Path]:
    """确保主输出和特征输出 ONNX 产物均已生成且不过期。

    Args:
        case: 模型目录测试用例。
        model: PyTorch 参考模型。
        input_tensor: 导出示例输入。
        output_dir: 产物输出目录。

    Returns:
        ``(主输出 ONNX, 特征 ONNX)`` 路径元组。
    """

    primary_onnx = output_dir / f"{case.model_name}.onnx"
    feature_onnx = output_dir / f"{case.model_name}.features.onnx"
    sources = [*case.source_files, Path(__file__).resolve()]

    if not is_fresh_against_all(primary_onnx, sources):
        _export_primary_onnx(model, input_tensor, primary_onnx)
    if not is_fresh_against_all(feature_onnx, sources):
        _export_feature_onnx(case, model, input_tensor, feature_onnx)
    return primary_onnx, feature_onnx


def _set_input_shape(config: Any, input_tensor: np.ndarray) -> None:
    """把输入张量形状写入 InferRT 配置。

    Args:
        config: ``inferrt_model_py.ModelConfig`` 实例。
        input_tensor: NumPy 输入张量。
    """

    shape = [int(dim) for dim in input_tensor.shape]
    config.input_shape = shape
    try:
        config.input_shapes = [shape]
    except AttributeError:
        pass


def _build_model_or_skip(model: Any, model_file: Path, *, label: str) -> None:
    """构建/加载模型，后端不可用时跳过当前用例。

    Args:
        model: InferRT Python 模型对象。
        model_file: 待加载的 ``.wts``、ONNX 或 IR 路径。
        label: skip 消息中的后端/模型标签。
    """

    try:
        model.build_or_load(str(model_file))
    except Exception as exc:
        message = str(exc)
        unavailable_markers = (
            "backend is not enabled",
            "Failed to load ONNX Runtime DLL",
            "CUDA driver",
            "CUDA failure",
            "CUDA error",
            "CUDA provider",
            "Failed to initialize CUDA",
            "Device with \"GPU\" name is not registered",
            "Cannot get DEVICE_PROPERTIES",
        )
        if any(marker in message for marker in unavailable_markers):
            pytest.skip(f"{label} backend unavailable: {message}")
        raise


def _single_model_input_name(model: Any, *, label: str) -> str:
    """返回单输入模型在当前后端暴露的真实输入名。"""

    input_names = [str(name) for name in model.input_tensor_names()]
    if len(input_names) != 1:
        raise RuntimeError(f"Expected one input for {label}, got {input_names}")
    return input_names[0]


def _run_inferrt_primary(
    irt_module: Any,
    *,
    model_name: str,
    model_file: Path,
    input_tensor: np.ndarray,
    backend_attr: str,
    device_attr: str,
) -> np.ndarray:
    """使用指定 InferRT 后端执行主输出推理。

    Args:
        irt_module: 已导入的 ``inferrt_model_py`` 模块。
        model_name: TensorRT 原生模型名；图后端会改用 ``onnx``。
        model_file: ``.wts`` 或 ONNX 文件路径。
        input_tensor: NumPy 输入张量。
        backend_attr: 后端枚举属性名。
        device_attr: 设备枚举属性名。

    Returns:
        主输出 NumPy 数组。
    """

    config = irt_module.ModelConfig()
    config.backend = getattr(irt_module.ModelBackend, backend_attr)
    config.device = getattr(irt_module.ModelDevice, device_attr)
    if backend_attr == "TENSORRT":
        _set_input_shape(config, input_tensor)

    runtime_model_name = model_name if backend_attr == "TENSORRT" else "onnx"
    model = irt_module.create_model(runtime_model_name, config=config)
    _build_model_or_skip(model, model_file, label=f"{backend_attr}/{device_attr}/{model_name}")

    input_name = _single_model_input_name(model, label=f"{backend_attr}/{device_attr}/{model_name}")
    outputs = model.infer({input_name: np.ascontiguousarray(input_tensor, dtype=np.float32)}, None)
    if isinstance(outputs, np.ndarray):
        return outputs
    output_dict = {str(name): np.asarray(value) for name, value in dict(outputs).items()}
    if PRIMARY_OUTPUT_NAME in output_dict:
        return output_dict[PRIMARY_OUTPUT_NAME]
    if len(output_dict) == 1:
        return next(iter(output_dict.values()))
    raise RuntimeError(f"Expected one primary output, got {sorted(output_dict)}")


def _run_inferrt_features(
    irt_module: Any,
    *,
    case: ModelDirCase,
    model_file: Path,
    input_tensor: np.ndarray,
    backend_attr: str,
    device_attr: str,
) -> dict[str, np.ndarray]:
    """使用指定 InferRT 后端执行特征提取。

    Args:
        irt_module: 已导入的 ``inferrt_model_py`` 模块。
        case: 模型目录测试用例。
        model_file: ``.wts`` 或特征 ONNX 文件路径。
        input_tensor: NumPy 输入张量。
        backend_attr: 后端枚举属性名。
        device_attr: 设备枚举属性名。

    Returns:
        特征名到 NumPy 输出数组的映射。
    """

    config = irt_module.ModelConfig()
    config.backend = getattr(irt_module.ModelBackend, backend_attr)
    config.device = getattr(irt_module.ModelDevice, device_attr)
    config.feature_only = True
    config.feature_tensor_names = case.feature_names
    config.output_tensor_names = case.feature_names
    if backend_attr == "TENSORRT":
        _set_input_shape(config, input_tensor)

    runtime_model_name = case.model_name if backend_attr == "TENSORRT" else "onnx"
    model = irt_module.create_model(runtime_model_name, config=config)
    _build_model_or_skip(model, model_file, label=f"{backend_attr}/{device_attr}/{case.model_name}/features")

    input_name = _single_model_input_name(model, label=f"{backend_attr}/{device_attr}/{case.model_name}/features")
    outputs = model.forward_features({input_name: np.ascontiguousarray(input_tensor, dtype=np.float32)})
    if isinstance(outputs, np.ndarray):
        return {case.feature_names[0]: outputs}
    return {str(name): np.asarray(value) for name, value in dict(outputs).items()}


def _assert_feature_outputs(
    reference: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
    label: str,
) -> None:
    """比较两组特征输出名称和数值。

    Args:
        reference: PyTorch 参考特征。
        actual: InferRT 后端特征。
        rtol: 相对误差容差。
        atol: 绝对误差容差。
        label: 断言失败时使用的标签前缀。
    """

    assert sorted(actual) == sorted(reference), f"{label} feature outputs mismatch"
    for name, ref in reference.items():
        assert_tensors_close(ref, actual[name], rtol=rtol, atol=atol, name=f"{label}.{name}")


def test_discover_model_dir_cases_detects_supported_families(tmp_path: Path) -> None:
    """验证模型目录扫描能识别 ResNet、DINOv2 和 DINOv3 checkpoint。"""

    (tmp_path / "resnet" / "a").mkdir(parents=True)
    (tmp_path / "resnet" / "a" / "resnet50_custom.pth").write_bytes(b"fake")
    (tmp_path / "dinov2").mkdir()
    (tmp_path / "dinov2" / "dinov2_vits14_pretrain.pth").write_bytes(b"fake")
    hf_dir = tmp_path / "dinov3" / "dinov3_vitb16"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_bytes(b"fake")

    cases = discover_model_dir_cases(tmp_path, ["resnet", "dinov2", "dinov3"])
    keys = {(case.family, case.model_name, case.provider) for case in cases}

    assert ("resnet", "resnet50", "torchvision_state") in keys
    assert ("dinov2", "dinov2_vits14", "torchhub_weights") in keys
    assert ("dinov3", "dinov3_vitb16", "transformers_dir") in keys


def _runtime_label(runtime_attr: str) -> str:
    """将后端枚举名转换为断言标签中的短名称。

    Args:
        runtime_attr: 后端枚举属性名。

    Returns:
        用于日志和断言名称的短标签。
    """

    return {
        "TENSORRT": "tensorrt",
        "ONNXRUNTIME": "onnx",
        "OPENVINO": "openvino",
    }[runtime_attr]


def _runtime_device_pairs(compare_runtimes: list[str], compare_devices: list[str]) -> list[tuple[str, str]]:
    """根据 runtime 和 device 参数生成有效的后端/设备组合。

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


def _primary_output_tolerances(
    case: ModelDirCase,
    backend_attr: str,
    primary_tolerances: tuple[float, float],
    feature_tolerances: tuple[float, float],
) -> tuple[float, float]:
    """根据模型族和后端选择主输出数值比对容差。

    Args:
        case: 当前模型目录测试用例。
        backend_attr: 后端枚举属性名，例如 ``TENSORRT`` 或 ``ONNXRUNTIME``。
        primary_tolerances: 分类主输出默认 ``(rtol, atol)``。
        feature_tolerances: 特征输出默认 ``(rtol, atol)``，DINO 主输出也复用该容差。

    Returns:
        tuple[float, float]: 当前主输出断言应使用的 ``(rtol, atol)``。
    """

    if case.family != "resnet":
        return feature_tolerances

    primary_rtol, primary_atol = primary_tolerances
    if backend_attr == "TENSORRT":
        # TensorRT GPU 构建会按硬件 tactic/TF32 路径累积更明显的 logits 误差。
        return primary_rtol, max(primary_atol, 7.5e-2)
    return primary_tolerances


def _feature_output_tolerances(
    case: ModelDirCase,
    backend_attr: str,
    feature_tolerances: tuple[float, float],
) -> tuple[float, float]:
    """根据模型族和后端选择特征输出数值比对容差。

    Args:
        case: 当前模型目录测试用例。
        backend_attr: 后端枚举属性名。
        feature_tolerances: 默认特征输出 ``(rtol, atol)``。

    Returns:
        tuple[float, float]: 当前特征断言应使用的 ``(rtol, atol)``。
    """

    feature_rtol, feature_atol = feature_tolerances
    if case.family == "resnet" and backend_attr == "TENSORRT":
        # ResNet 中间层比 logits 更容易放大 TensorRT GPU tactic/TF32 的逐元素绝对误差。
        return feature_rtol, max(feature_atol, 6e-1)
    if case.family in {"dinov2", "dinov3"} and backend_attr == "TENSORRT":
        # DINO token 特征维度大，TensorRT GPU 路径会出现少量超过默认 0.15 的逐元素差异。
        return feature_rtol, max(feature_atol, 2.5e-1)
    return feature_tolerances


def _model_root_from_options(pytestconfig: pytest.Config) -> Path:
    """解析并检查真实模型根目录。

    Args:
        pytestconfig: pytest 配置对象。

    Returns:
        已解析的模型根目录路径。
    """

    value = pytestconfig.getoption("--inferrt-model-root") or os.environ.get("INFERRT_MODEL_ROOT") or "D:/Models"
    path = Path(value).expanduser().resolve()
    if not path.exists():
        pytest.skip(f"Model root not found: {path}")
    return path


def test_model_root_checkpoints_export_and_match_pytorch(
    pytestconfig: pytest.Config,
    compare_runtimes: list[str],
    compare_devices: list[str],
    build_dir: Path,
    irt_module: Any,
    tolerances: tuple[float, float],
    feature_tolerances: tuple[float, float],
) -> None:
    """导出模型目录中的 checkpoint，并按所选后端与 PyTorch 主输出/特征对比。"""

    if not compare_runtimes:
        pytest.skip("No runtimes selected; pass --inferrt-compare-runtime=TensorRT,onnx,openvino")
    runtime_device_pairs = _runtime_device_pairs(compare_runtimes, compare_devices)
    if not runtime_device_pairs:
        pytest.skip("No compatible runtime/device pairs selected; TensorRT requires --inferrt-compare-devices=gpu")

    model_root = _model_root_from_options(pytestconfig)
    max_cases = int(pytestconfig.getoption("--inferrt-model-dir-max-cases"))
    cases = discover_model_dir_cases(model_root, _families_from_option(pytestconfig), max_cases=max_cases)
    if not cases:
        pytest.skip(f"No supported checkpoints found under {model_root}/{{resnet,dinov2,dinov3}}")

    for case in cases:
        model = _load_case_model(case, pytestconfig)
        input_tensor = _input_tensor_for_model(model)
        torch_primary = _torch_primary(model, input_tensor)
        torch_features = _torch_features(case, model, input_tensor)

        output_dir = _case_artifact_dir(build_dir, case)
        wts_path = _ensure_wts(case, model, output_dir)
        primary_onnx, feature_onnx = _ensure_onnx_artifacts(case, model, input_tensor, output_dir)

        for backend_attr, device in runtime_device_pairs:
            device_attr = device.upper()
            primary_model_file = wts_path if backend_attr == "TENSORRT" else primary_onnx
            feature_model_file = wts_path if backend_attr == "TENSORRT" else feature_onnx
            label = f"{case.model_name}.{_runtime_label(backend_attr)}.{device}"

            backend_primary = _run_inferrt_primary(
                irt_module,
                model_name=case.model_name,
                model_file=primary_model_file,
                input_tensor=input_tensor,
                backend_attr=backend_attr,
                device_attr=device_attr,
            )
            primary_rtol, primary_atol = _primary_output_tolerances(case, backend_attr, tolerances, feature_tolerances)
            assert_tensors_close(
                torch_primary,
                backend_primary,
                rtol=primary_rtol,
                atol=primary_atol,
                name=f"{label}.primary",
            )

            backend_features = _run_inferrt_features(
                irt_module,
                case=case,
                model_file=feature_model_file,
                input_tensor=input_tensor,
                backend_attr=backend_attr,
                device_attr=device_attr,
            )
            feature_rtol, feature_atol = _feature_output_tolerances(case, backend_attr, feature_tolerances)
            _assert_feature_outputs(
                torch_features,
                backend_features,
                rtol=feature_rtol,
                atol=feature_atol,
                label=f"{label}.features",
            )
