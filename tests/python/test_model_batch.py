"""TensorRT 手写模型动态 batch 推理与特征提取测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers.manifest import assert_tensors_close
from helpers.model_integration import resolve_model_file

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _resnet18_weights(model_root: Path) -> Path:
    """返回配置模型根目录下的 ResNet18 权重。"""

    return resolve_model_file(
        model_root,
        ("resnet/resnet18.wts", "resnet/resnet18-f37072fd/resnet18.wts"),
        "ResNet18 weights",
    )


def _dinov2_vits14_weights(model_root: Path) -> Path:
    """返回配置模型根目录下的 DINOv2 ViT-S/14 权重。"""

    return resolve_model_file(
        model_root,
        ("dinov2/dinov2_vits14.wts", "dinov2/dinov2_vits14_pretrain/dinov2_vits14.wts"),
        "DINOv2 ViT-S/14 weights",
    )


def _make_inputs(batch: int) -> np.ndarray:
    """生成确定性的 ImageNet 尺寸 NCHW 输入。"""

    values = np.linspace(-1.0, 1.0, num=batch * 3 * 224 * 224, dtype=np.float32)
    return np.ascontiguousarray(values.reshape(batch, 3, 224, 224))


def _make_dino_inputs(batch: int) -> np.ndarray:
    """生成确定性的 DINOv2 518x518 NCHW 输入。"""

    values = np.linspace(-1.0, 1.0, num=batch * 3 * 518 * 518, dtype=np.float32)
    return np.ascontiguousarray(values.reshape(batch, 3, 518, 518))


def _build_or_skip(model: Any, weights: Path) -> None:
    """构建 TensorRT engine；GPU/TensorRT 不可用时跳过。"""

    try:
        model.build_or_load(str(weights))
    except Exception as exc:  # noqa: BLE001 - pytest 需要兼容 pybind 异常类型
        message = str(exc)
        unavailable_markers = (
            "CUDA driver",
            "CUDA failure",
            "CUDA error",
            "Failed to initialize CUDA",
            "TensorRT backend is not enabled",
        )
        if any(marker in message for marker in unavailable_markers):
            pytest.skip(f"TensorRT unavailable: {message}")
        raise


def _torch_resnet18_logits(input_tensor: np.ndarray) -> np.ndarray:
    """使用 torchvision ResNet18 计算 PyTorch logits。"""

    torch = pytest.importorskip("torch")
    from classification_model_zoo import create_model

    model = create_model("resnet18", "torchvision")
    model.eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32)))
    return output.detach().cpu().numpy()


def _torch_resnet18_features(input_tensor: np.ndarray) -> dict[str, np.ndarray]:
    """使用 PyTorch 主干提取与 InferRT 同名的 ResNet18 中间特征。"""

    torch = pytest.importorskip("torch")
    from classification_model_zoo import create_model

    model = create_model("resnet18", "torchvision")
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    with torch.inference_mode():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        layer1 = model.layer1(x)
        x = model.layer2(layer1)
        x = model.layer3(x)
        layer4 = model.layer4(x)
    return {
        "layer1": layer1.detach().cpu().numpy(),
        "layer4": layer4.detach().cpu().numpy(),
    }


def _load_torch_dinov2_vits14() -> Any:
    """加载 PyTorch DINOv2 ViT-S/14；离线或依赖缺失时跳过。"""

    pytest.importorskip("torch")
    from classification_model_zoo import create_model

    try:
        model = create_model("dinov2_vits14", "torchhub")
    except Exception as exc:  # noqa: BLE001 - torch.hub 离线缓存缺失时跳过
        pytest.skip(f"PyTorch DINOv2 model unavailable: {exc}")
    model.eval()
    return model


def _torch_dinov2_primary(input_tensor: np.ndarray) -> np.ndarray:
    """使用 PyTorch DINOv2 计算主输出 CLS 特征。"""

    torch = pytest.importorskip("torch")
    model = _load_torch_dinov2_vits14()
    with torch.inference_mode():
        output = model(torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32)))
    return output.detach().cpu().numpy()


def _torch_dinov2_features(input_tensor: np.ndarray) -> dict[str, np.ndarray]:
    """使用 PyTorch DINOv2 提取与 InferRT 同名的中间特征。"""

    torch = pytest.importorskip("torch")
    model = _load_torch_dinov2_vits14()
    with torch.inference_mode():
        outputs = model.forward_features(torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32)))
    if not isinstance(outputs, dict):
        pytest.skip("PyTorch DINOv2 forward_features() did not return a feature dict")
    return {
        "x_norm_clstoken": outputs["x_norm_clstoken"].detach().cpu().numpy(),
        "x_norm_patchtokens": outputs["x_norm_patchtokens"].detach().cpu().numpy(),
    }


def _dynamic_resnet_config(irt_module: Any, *, batch: int, feature_only: bool = False) -> Any:
    """创建动态 batch ResNet 配置。"""

    config = irt_module.ModelConfig()
    config.input_shape = [1, 3, 224, 224]
    config.dynamic_batch_range = [1, batch, batch]
    if feature_only:
        config.feature_only = True
        config.feature_tensor_names = ["layer1", "layer4"]
        config.output_tensor_names = ["layer1", "layer4"]
    return config


def _dynamic_dino_config(irt_module: Any, *, batch: int, feature_only: bool = False) -> Any:
    """创建动态 batch DINOv2 配置。"""

    config = irt_module.ModelConfig()
    config.input_shape = [1, 3, 518, 518]
    config.dynamic_batch_range = [1, batch, batch]
    if feature_only:
        config.feature_only = True
        config.feature_tensor_names = ["x_norm_clstoken", "x_norm_patchtokens"]
        config.output_tensor_names = ["x_norm_clstoken", "x_norm_patchtokens"]
    return config


def _dino_feature_atol(name: str, feature_atol: float) -> float:
    """DINO patch token 大张量在 TensorRT GPU 路径下允许少量 TF32/tactic 累积误差。"""

    return max(feature_atol, 3e-1) if name == "x_norm_patchtokens" else feature_atol


def test_resnet18_dynamic_batch_infer_matches_pytorch(
    irt_module: Any,
    model_root: Path,
    tolerances: tuple[float, float],
) -> None:
    """同一个动态 batch engine 应能先跑 batch=1，再跑 batch=2，并与 PyTorch 对齐。"""

    weights = _resnet18_weights(model_root)
    config = _dynamic_resnet_config(irt_module, batch=2)
    model = irt_module.create_model("resnet18", config)
    _build_or_skip(model, weights)

    input_name = model.input_tensor_names()[0]

    single_input = _make_inputs(1)
    single_output = model.infer({input_name: single_input}, None)
    assert tuple(single_output.shape) == (1, 1000)
    assert_tensors_close(
        _torch_resnet18_logits(single_input),
        single_output,
        rtol=tolerances[0],
        atol=tolerances[1],
        name="resnet18.dynamic_batch1",
    )

    batch_input = _make_inputs(2)
    batch_output = model.infer({input_name: batch_input}, None)
    assert tuple(batch_output.shape) == (2, 1000)
    assert_tensors_close(
        _torch_resnet18_logits(batch_input),
        batch_output,
        rtol=tolerances[0],
        atol=tolerances[1],
        name="resnet18.dynamic_batch2",
    )


def test_dinov2_dynamic_batch_infer_matches_pytorch(
    irt_module: Any,
    model_root: Path,
    feature_tolerances: tuple[float, float],
) -> None:
    """DINO 动态 batch engine 应支持 batch=1/2 主输出，并与 PyTorch CLS 特征对齐。"""

    weights = _dinov2_vits14_weights(model_root)
    config = _dynamic_dino_config(irt_module, batch=2)
    model = irt_module.create_model("dinov2_vits14", config)
    _build_or_skip(model, weights)

    input_name = model.input_tensor_names()[0]
    for batch in (1, 2):
        input_tensor = _make_dino_inputs(batch)
        output = model.infer({input_name: input_tensor}, None)
        assert tuple(output.shape) == (batch, 384)
        assert_tensors_close(
            _torch_dinov2_primary(input_tensor),
            output,
            rtol=feature_tolerances[0],
            atol=feature_tolerances[1],
            name=f"dinov2.dynamic_batch{batch}",
        )


def test_resnet18_dynamic_batch_forward_features_matches_pytorch(
    irt_module: Any,
    model_root: Path,
    feature_tolerances: tuple[float, float],
) -> None:
    """feature-only engine 的输出 batch 维应与输入一致，并与 PyTorch 中间层对齐。"""

    weights = _resnet18_weights(model_root)
    config = _dynamic_resnet_config(irt_module, batch=2, feature_only=True)
    model = irt_module.create_model("resnet18", config)
    _build_or_skip(model, weights)

    input_tensor = _make_inputs(2)
    input_name = model.input_tensor_names()[0]
    outputs = model.forward_features({input_name: input_tensor})
    actual = {str(name): np.asarray(value) for name, value in dict(outputs).items()}
    reference = _torch_resnet18_features(input_tensor)

    assert sorted(actual) == ["layer1", "layer4"]
    for name, expected in reference.items():
        assert actual[name].shape[0] == input_tensor.shape[0]
        assert_tensors_close(
            expected,
            actual[name],
            rtol=feature_tolerances[0],
            atol=feature_tolerances[1],
            name=f"resnet18.features.{name}",
        )


def test_dinov2_dynamic_batch_forward_features_matches_pytorch(
    irt_module: Any,
    model_root: Path,
    feature_tolerances: tuple[float, float],
) -> None:
    """DINO feature-only 动态 batch 输出应保持输入 batch，并与 PyTorch 中间特征对齐。"""

    weights = _dinov2_vits14_weights(model_root)
    config = _dynamic_dino_config(irt_module, batch=2, feature_only=True)
    model = irt_module.create_model("dinov2_vits14", config)
    _build_or_skip(model, weights)

    input_tensor = _make_dino_inputs(2)
    input_name = model.input_tensor_names()[0]
    outputs = model.forward_features({input_name: input_tensor})
    actual = {str(name): np.asarray(value) for name, value in dict(outputs).items()}
    reference = _torch_dinov2_features(input_tensor)

    assert sorted(actual) == ["x_norm_clstoken", "x_norm_patchtokens"]
    for name, expected in reference.items():
        assert actual[name].shape[0] == input_tensor.shape[0]
        assert_tensors_close(
            expected,
            actual[name],
            rtol=feature_tolerances[0],
            atol=_dino_feature_atol(name, feature_tolerances[1]),
            name=f"dinov2.features.{name}",
        )
