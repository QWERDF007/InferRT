"""Parity tests between C++ feature sample and pybind11 forward_features API."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

from helpers.manifest import assert_tensors_close, load_tensor_from_dump, parse_manifest
from helpers.runtime import run_cpp_feature_dump, run_python_features


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
