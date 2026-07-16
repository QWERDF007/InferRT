"""LingBot-Vision PyTorch 与 InferRT TensorRT 路径的一致性测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers.manifest import assert_tensors_close
from helpers.model_integration import artifact_dir, is_fresh_against_all
from helpers.runtime import run_python_classification, run_python_features

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_VARIANTS = (
    ("small", "lingbot_vision_vits16", "lingbot-vision-vit-small.pt", "lbot_vision_vits.yaml"),
    ("base", "lingbot_vision_vitb16", "lingbot-vision-vit-base.pt", "lbot_vision_vitb.yaml"),
    ("large", "lingbot_vision_vitl16", "lingbot-vision-vit-large.pt", "lbot_vision_vitl.yaml"),
    ("giant", "lingbot_vision_vitg16", "lingbot-vision-vit-giant.pt", "lbot_vision_vitg.yaml"),
)
_FEATURE_NAMES = ["x_norm_clstoken", "x_storage_tokens", "x_norm_patchtokens"]
_INPUT_SIZE = 512


def _load_lingbot_model(repo_root: Path, checkpoint: Path, config_name: str) -> Any:
    """从上游源码和 checkpoint 加载冻结的 PyTorch 参考模型。"""

    torch = pytest.importorskip("torch")
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)

    from lingbot_vision import load_backbone, load_backbone_state, load_config

    config_path = repo_root / "lingbot_vision" / "configs" / config_name
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_backbone(
        config,
        load_backbone_state(checkpoint),
        device=device,
        dtype=torch.float32,
        verbose=False,
    )
    return model


def _reference_outputs(model: Any) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """执行 PyTorch 前向并返回主输出、特征输出和确定性输入。"""

    torch = pytest.importorskip("torch")
    values = np.linspace(
        -1.0,
        1.0,
        num=1 * 3 * _INPUT_SIZE * _INPUT_SIZE,
        dtype=np.float32,
    )
    input_tensor = np.ascontiguousarray(values.reshape(1, 3, _INPUT_SIZE, _INPUT_SIZE))
    device = next(model.parameters()).device
    batch = torch.from_numpy(input_tensor).to(device)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    with torch.inference_mode():
        outputs = model.forward_features(batch)

    features = {
        name: outputs[name].detach().cpu().numpy()
        for name in _FEATURE_NAMES
    }
    return features["x_norm_clstoken"], features, input_tensor


def _ensure_wts(model: Any, checkpoint: Path, model_name: str, build_dir: Path) -> Path:
    """将 checkpoint 对应的 PyTorch state_dict 缓存为 InferRT ``.wts``。"""

    from classification_gen_wts import write_wts

    output = artifact_dir(build_dir, "lingbot_vision") / f"{model_name}.wts"
    sources = [checkpoint, Path(__file__).resolve()]
    if is_fresh_against_all(output, sources):
        return output

    write_wts(model, str(output), verbose=False)
    return output


def _run_inferrt(
    model_name: str, weights_path: Path, input_tensor: np.ndarray, irt_module: Any
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """执行 InferRT 主输出与 feature-only 输出。"""

    try:
        primary_result = run_python_classification(
            irt_module,
            model_name=model_name,
            weights_path=weights_path,
            input_tensor=input_tensor,
        )
        features = run_python_features(
            irt_module,
            model_name=model_name,
            weights_path=weights_path,
            input_tensor=input_tensor,
            feature_names=_FEATURE_NAMES,
        )
    except Exception as exc:
        message = str(exc)
        unavailable_markers = (
            "backend is not enabled",
            "CUDA driver",
            "CUDA failure",
            "CUDA error",
            "Failed to initialize CUDA",
            "Device with \"GPU\" name is not registered",
            "Cannot get DEVICE_PROPERTIES",
        )
        if any(marker in message for marker in unavailable_markers):
            pytest.skip(f"TensorRT backend unavailable: {message}")
        raise

    if isinstance(primary_result, dict):
        if "output" in primary_result:
            primary = primary_result["output"]
        elif len(primary_result) == 1:
            primary = next(iter(primary_result.values()))
        else:
            raise RuntimeError(f"Expected one primary output, got {sorted(primary_result)}")
    else:
        primary = primary_result
    return np.asarray(primary), {str(name): np.asarray(value) for name, value in features.items()}


@pytest.mark.parametrize("variant,model_name,checkpoint_name,config_name", _VARIANTS)
def test_lingbot_vision_tensorrt_matches_pytorch(
    variant: str,
    model_name: str,
    checkpoint_name: str,
    config_name: str,
    compare_runtimes: list[str],
    compare_devices: list[str],
    irt_module: Any,
    build_dir: Path,
    lingbot_vision_root: Path,
    lingbot_vision_repo: Path,
    feature_tolerances: tuple[float, float],
) -> None:
    """比较 LingBot-Vision 的 CLS 和 patch/storage 特征。"""

    if "TENSORRT" not in compare_runtimes:
        pytest.skip("LingBot-Vision parity requires --inferrt-compare-runtime=tensorrt")
    if "gpu" not in compare_devices:
        pytest.skip("LingBot-Vision TensorRT parity requires --inferrt-compare-devices=gpu")

    checkpoint = lingbot_vision_root / checkpoint_name
    if not checkpoint.is_file():
        pytest.skip(f"LingBot-Vision {variant} checkpoint not found: {checkpoint}")

    model = _load_lingbot_model(lingbot_vision_repo, checkpoint, config_name)
    torch_primary, torch_features, input_tensor = _reference_outputs(model)
    weights_path = _ensure_wts(model, checkpoint, model_name, build_dir)
    inferrt_primary, inferrt_features = _run_inferrt(model_name, weights_path, input_tensor, irt_module)

    rtol, atol = feature_tolerances
    assert_tensors_close(torch_primary, inferrt_primary, rtol=rtol, atol=atol, name=f"{model_name}.primary")
    assert sorted(inferrt_features) == sorted(torch_features)
    for name, reference in torch_features.items():
        assert_tensors_close(
            reference,
            inferrt_features[name],
            rtol=rtol,
            atol=atol,
            name=f"{model_name}.{name}",
        )
