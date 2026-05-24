"""特征提取：PyTorch 与 InferRT pybind11 ``forward_features()`` 输出一致性测试。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from helpers.manifest import assert_tensors_close
from helpers.runtime import run_python_features, run_torch_features

# parametrize 参数说明见 ``test_forward_features_matches_pytorch`` 的 Args
FEATURE_CASES = [
    pytest.param(
        "alexnet",
        "samples/model/classification/alexnet.wts",
        ("pool1", "fc2"),
        id="alexnet",
    ),
    pytest.param(
        "googlenet",
        "samples/model/classification/googlenet.wts",
        ("inception3a", "inception5b"),
        id="googlenet",
    ),
    pytest.param(
        "resnet18",
        "samples/model/classification/resnet18.wts",
        ("layer1", "layer4"),
        id="resnet18",
    ),
    pytest.param(
        "resnet50",
        "samples/model/classification/resnet50.wts",
        ("layer1", "layer3"),
        id="resnet50",
    ),
    pytest.param(
        "vgg11",
        "samples/model/classification/vgg11.wts",
        ("block1", "avgpool"),
        id="vgg11",
    ),
    pytest.param(
        "mobilenet_v2",
        "samples/model/classification/mobilenet_v2.wts",
        ("stem", "features.1"),
        id="mobilenet_v2",
    ),
]


DINO_OFFICIAL_FEATURE_CASES = [
    pytest.param(
        "dinov2_vits14",
        ["x_norm_clstoken", "x_norm_patchtokens"],
        id="dinov2_vits14",
    ),
    pytest.param(
        "dinov3_vitb16",
        ["x_norm_clstoken", "x_storage_tokens", "x_norm_patchtokens"],
        id="dinov3_vitb16",
    ),
]


def _load_official_dino_pretrained(model_name: str) -> object:
    """通过官方后端加载 DINO 系列预训练模型。

    Args:
        model_name: 官方模型名，例如 ``dinov2_vits14`` 或 ``dinov3_vitb16``。

    Returns:
        已切换到 ``eval`` 模式的 PyTorch 模型。

    DINOv2 使用 PyTorch Hub；DINOv3 使用 Hugging Face
    ``pipeline(task="image-feature-extraction")``。若当前环境无法访问官方模型或
    缓存不存在，则跳过该集成测试，避免因为网络/缓存条件导致普通单元测试失败。
    """

    from model_zoo import create_model

    backend = "transformers" if model_name.startswith("dinov3_") else "torchhub"
    try:
        model = create_model(model_name, backend)
    except Exception as exc:
        pytest.skip(f"Official {model_name} pretrained model unavailable: {exc}")

    model.eval()
    return model


@pytest.mark.parametrize("model_name,weights_rel,feature_names", FEATURE_CASES)
def test_forward_features_matches_pytorch(
    model_name: str,
    weights_rel: str,
    feature_names: tuple[str, ...],
    irt_module: object,
    input_tensor,
    weights_path,
    feature_tolerances: tuple[float, float],
) -> None:
    """PyTorch 参考特征应与 InferRT Python ``forward_features()`` 在容差内一致。

    Args:
        model_name: ``model_zoo`` / ``create_model`` 使用的模型名，如 ``resnet18``。
        weights_rel: 相对仓库根的 ``.wts`` 路径，供 InferRT ``build_or_load``；缺失时跳过。
        feature_names: 建网层 key；同时作为 ``feature_tensor_names`` 与
            ``output_tensor_names``。
        irt_module: ``inferrt_model_py`` 模块（``conftest.irt_module``）。
        input_tensor: ImageNet 预处理后的 NumPy 输入，形状 ``(1, 3, 224, 224)``；
            同时作为 PyTorch 与 InferRT 的输入，保证同图同预处理。
        weights_path: 将 ``weights_rel`` 解析为绝对 ``.wts`` 路径的 callable。
        feature_tolerances: ``(rtol, atol)``，跨框架中间层差异较大，默认较分类宽松。

    参考侧为 ``torchvision`` 预训练模型（与 ``gen_wts.py`` 一致），通过 forward hook
    捕获中间层输出；待测侧为加载对应 ``.wts`` 的 TensorRT engine，经 pybind11
    ``forward_features()`` 推理。
    """

    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    rtol, atol = feature_tolerances
    weights = weights_path(weights_rel)
    names = list(feature_names)

    torch_features = run_torch_features(model_name, input_tensor, names)
    irt_features = run_python_features(
        irt_module,
        model_name=model_name,
        weights_path=weights,
        input_tensor=input_tensor,
        feature_names=names,
    )

    for tensor_name in names:
        assert tensor_name in torch_features, f"PyTorch result missing tensor '{tensor_name}'"
        assert tensor_name in irt_features, f"InferRT result missing tensor '{tensor_name}'"
        assert_tensors_close(
            torch_features[tensor_name],
            irt_features[tensor_name],
            rtol=rtol,
            atol=atol,
            name=tensor_name,
        )


@pytest.mark.parametrize("model_name,feature_names", DINO_OFFICIAL_FEATURE_CASES)
def test_dino_forward_features_matches_official_pretrained(
    model_name: str,
    feature_names: list[str],
    irt_module: object,
    default_image: Path,
    build_dir: Path,
    feature_tolerances: tuple[float, float],
) -> None:
    """官方 DINO ``forward_features`` 应与 InferRT pybind11 特征输出一致。

    Args:
        model_name: 官方 DINO 模型名；DINOv2 走 PyTorch Hub，DINOv3 走 Hugging Face pipeline。
        feature_names: 需要同时从 PyTorch 和 InferRT 读取的特征张量名。
        irt_module: ``inferrt_model_py`` 模块。
        default_image: 默认测试图片。
        build_dir: CMake 构建目录，用于写出本测试专属 ``.wts`` 和 engine。
        feature_tolerances: ``(rtol, atol)``，复用中间特征容差。

    本测试通过官方权重入口加载 pretrained 模型，不再依赖本地源码目录等固定路径。
    随后把同一份 ``state_dict`` 导出为 ``.wts``，通过 pybind11
    ``forward_features()`` 构建 InferRT feature-only engine。
    """

    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")

    from gen_wts import write_wts
    from model_zoo import preprocess, resolve_input_size

    torch_model = _load_official_dino_pretrained(model_name)

    image = cv2.imread(str(default_image), cv2.IMREAD_COLOR)
    if image is None:
        pytest.skip(f"Failed to read default image: {default_image}")
    input_tensor = preprocess(image, image_size=resolve_input_size(torch_model)).numpy()

    batch = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    with torch.inference_mode():
        torch_outputs = torch_model.forward_features(batch)
    torch_features = {name: torch_outputs[name].detach().cpu().numpy() for name in feature_names}

    artifact_dir = build_dir / "python_test_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_file = artifact_dir / f"{model_name}_pretrained_{os.getpid()}.wts"
    write_wts(torch_model, str(weights_file), verbose=False)

    irt_features = run_python_features(
        irt_module,
        model_name=model_name,
        weights_path=weights_file,
        input_tensor=input_tensor,
        feature_names=feature_names,
    )

    rtol, atol = feature_tolerances
    for tensor_name in feature_names:
        assert tensor_name in irt_features, f"InferRT result missing tensor '{tensor_name}'"
        assert_tensors_close(
            torch_features[tensor_name],
            irt_features[tensor_name],
            rtol=rtol,
            atol=atol,
            name=tensor_name,
        )
