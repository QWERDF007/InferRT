"""分类模型：C++ sample 与 pybind11 ``infer()`` 输出一致性测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

from helpers.manifest import assert_tensors_close, load_tensor_from_dump, parse_manifest
from helpers.runtime import run_cpp_classification_dump, run_python_classification

# parametrize 参数说明见 ``test_classification_matches_cpp_sample`` 的 Args
CLASSIFICATION_CASES = [
    pytest.param("alexnet", "samples/model/classification/alexnet.wts", id="alexnet"),
    pytest.param("resnet18", "samples/model/classification/resnet18.wts", id="resnet18"),
    pytest.param("resnet50", "samples/model/classification/resnet50.wts", id="resnet50"),
    pytest.param("vgg11", "samples/model/classification/vgg11.wts", id="vgg11"),
    pytest.param("mobilenet_v2", "samples/model/classification/mobilenet_v2.wts", id="mobilenet_v2"),
    # pytest.param("mobilenet_v3_large", "samples/model/classification/mobilenet_v3_large.wts", id="mobilenet_v3_large"),
    pytest.param("mobilenet_v3_small", "samples/model/classification/mobilenet_v3_small.wts", id="mobilenet_v3_small"),
]


@pytest.mark.parametrize("model_name,weights_rel", CLASSIFICATION_CASES)
def test_classification_matches_cpp_sample(
    model_name: str,
    weights_rel: str,
    build_dir: Path,
    repo_root: Path,
    irt_module: object,
    input_tensor,
    weights_path,
    tolerances: tuple[float, float],
    tmp_path: Path,
) -> None:
    """C++ 分类 sample dump 的张量应与 Python ``infer()`` 结果在容差内一致。

    Args:
        model_name: 内置模型名，传给 C++ ``--model`` 与 ``create_model()``，
            如 ``resnet18``、``alexnet``。
        weights_rel: 相对 ``repo_root`` 的 ``.wts`` 路径；缺失时跳过。
        build_dir: CMake 构建目录，用于定位 ``inferrt_sample_classification``。
        repo_root: 仓库根目录，作为 C++ sample 的工作目录与资源解析基准。
        irt_module: ``inferrt_model_py`` 模块（``conftest.irt_module``）。
        input_tensor: ImageNet 预处理后的 NumPy 输入，形状 ``(1,3,224,224)``。
        weights_path: callable，将 ``weights_rel`` 解析为绝对权重路径。
        tolerances: ``(rtol, atol)``，来自 ``--inferrt-rtol`` / ``--inferrt-atol``。
        tmp_path: pytest 提供的临时目录，存放本次 C++ dump（``cpp_dump/``）。

    流程：运行 C++ sample 写出 ``manifest.txt`` 与 ``.bin``；
    再用 Python ``infer()`` 推理；对 manifest 中每个张量做 ``np.allclose`` 比对。
    """

    rtol, atol = tolerances
    weights = weights_path(weights_rel)
    image_path = repo_root / "assets" / "pics" / "dog.jpg"
    cpp_dump_dir = tmp_path / "cpp_dump"

    run_cpp_classification_dump(
        build_dir=build_dir,
        model_name=model_name,
        weights_path=weights,
        image_path=image_path,
        dump_dir=cpp_dump_dir,
        cwd=repo_root,
    )

    py_tensors = run_python_classification(
        irt_module,
        model_name=model_name,
        weights_path=weights,
        input_tensor=input_tensor,
    )

    _, specs = parse_manifest(cpp_dump_dir / "manifest.txt")
    for tensor_name, spec in specs.items():
        cpp_tensor = load_tensor_from_dump(cpp_dump_dir, spec)
        assert tensor_name in py_tensors, f"Python result missing tensor '{tensor_name}'"
        assert_tensors_close(cpp_tensor, py_tensors[tensor_name], rtol=rtol, atol=atol, name=tensor_name)
