"""特征提取：PyTorch 与 InferRT pybind11 ``forward_features()`` 输出一致性测试。"""

from __future__ import annotations

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


@pytest.mark.parametrize("model_name,weights_rel,feature_names", FEATURE_CASES)
def test_forward_features_matches_pytorch(
    model_name: str,
    weights_rel: str,
    feature_names: tuple[str, ...],
    irt_module: object,
    input_tensor,
    weights_path,
    tolerances: tuple[float, float],
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
        tolerances: ``(rtol, atol)``，TensorRT 与 PyTorch 对比宜略宽，默认见 ``conftest``。

    参考侧为 ``torchvision`` 预训练模型（与 ``gen_wts.py`` 一致），通过 forward hook
    捕获中间层输出；待测侧为加载对应 ``.wts`` 的 TensorRT engine，经 pybind11
    ``forward_features()`` 推理。
    """

    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    rtol, atol = tolerances
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
