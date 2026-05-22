"""特征提取：C++ sample 与 pybind11 ``forward_features()`` 输出一致性测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

from helpers.manifest import assert_tensors_close, load_tensor_from_dump, parse_manifest
from helpers.runtime import run_cpp_feature_dump, run_python_features

# parametrize 参数说明见 ``test_forward_features_matches_cpp_sample`` 的 Args
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
def test_forward_features_matches_cpp_sample(
    model_name: str,
    weights_rel: str,
    feature_names: tuple[str, ...],
    build_dir: Path,
    repo_root: Path,
    irt_module: object,
    input_tensor,
    weights_path,
    tolerances: tuple[float, float],
    tmp_path: Path,
) -> None:
    """C++ 特征 sample dump 应与 Python ``forward_features()`` 在容差内一致。

    Args:
        model_name: 内置模型名，如 ``resnet18``。
        weights_rel: 相对 ``repo_root`` 的 ``.wts`` 路径。
        feature_names: 要导出的中间层张量名元组，传给 C++ ``--features`` 与
            ``ModelConfig.feature_tensor_names``（逗号连接）。
        build_dir: 构建目录，定位 ``inferrt_sample_feature_extract``。
        repo_root: 仓库根目录。
        irt_module: ``inferrt_model_py`` 模块。
        input_tensor: 预处理后的 NumPy 输入张量。
        weights_path: 权重路径解析 callable。
        tolerances: ``(rtol, atol)`` 数值容差。
        tmp_path: 临时目录，C++ 特征 dump 写入 ``cpp_features/``。

    ``feature_only=True`` 时仅导出指定中间层，避免完整分类头干扰比对。
    """

    rtol, atol = tolerances
    weights = weights_path(weights_rel)
    image_path = repo_root / "assets" / "pics" / "dog.jpg"
    cpp_dump_dir = tmp_path / "cpp_features"

    run_cpp_feature_dump(
        build_dir=build_dir,
        model_name=model_name,
        weights_path=weights,
        image_path=image_path,
        feature_names=list(feature_names),
        dump_dir=cpp_dump_dir,
        cwd=repo_root,
    )

    py_tensors = run_python_features(
        irt_module,
        model_name=model_name,
        weights_path=weights,
        input_tensor=input_tensor,
        feature_names=list(feature_names),
    )

    _, specs = parse_manifest(cpp_dump_dir / "manifest.txt")
    for tensor_name, spec in specs.items():
        cpp_tensor = load_tensor_from_dump(cpp_dump_dir, spec)
        assert tensor_name in py_tensors, f"Python result missing tensor '{tensor_name}'"
        assert_tensors_close(cpp_tensor, py_tensors[tensor_name], rtol=rtol, atol=atol, name=tensor_name)
