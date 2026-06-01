"""TensorRT 手写模型动态 batch 推理与特征提取测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers.manifest import assert_tensors_close

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _resnet18_weights(repo_root: Path) -> Path:
    """返回示例 ResNet18 权重；缺失时跳过依赖真实权重的测试。"""

    weights = repo_root / "samples" / "model" / "classification" / "resnet18.wts"
    if not weights.exists():
        pytest.skip(f"ResNet18 weights not found: {weights}")
    return weights


def _make_inputs(batch: int) -> np.ndarray:
    """生成确定性的 ImageNet 尺寸 NCHW 输入。"""

    values = np.linspace(-1.0, 1.0, num=batch * 3 * 224 * 224, dtype=np.float32)
    return np.ascontiguousarray(values.reshape(batch, 3, 224, 224))


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


def test_resnet18_dynamic_batch_infer_matches_pytorch(
    irt_module: Any,
    repo_root: Path,
    tolerances: tuple[float, float],
) -> None:
    """同一个动态 batch engine 应能先跑 batch=1，再跑 batch=2，并与 PyTorch 对齐。"""

    weights = _resnet18_weights(repo_root)
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


def test_resnet18_dynamic_batch_forward_features_matches_pytorch(
    irt_module: Any,
    repo_root: Path,
    feature_tolerances: tuple[float, float],
) -> None:
    """feature-only engine 的输出 batch 维应与输入一致，并与 PyTorch 中间层对齐。"""

    weights = _resnet18_weights(repo_root)
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
