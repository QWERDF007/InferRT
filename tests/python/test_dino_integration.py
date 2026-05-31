"""使用真实 DINO checkpoint 验证导出产物与 feature_extract sample。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers.manifest import load_tensor_from_dump, parse_manifest
from helpers.model_integration import artifact_dir, require_sample
from helpers.runtime import run_process_capture
from test_dino_parity import DINO_FAMILIES, _discover_dino_cases
from test_model_dir_parity import ModelDirCase, _case_artifact_dir, _ensure_wts, _load_case_model

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _representative_dino_cases(model_root: Path, pytestconfig: pytest.Config) -> list[ModelDirCase]:
    """每个 DINO 家族选择一个代表性 checkpoint 运行 sample 集成测试。

    Args:
        model_root: 真实模型根目录。
        pytestconfig: pytest 配置对象。

    Returns:
        按 ``dinov2``、``dinov3`` 顺序筛出的代表用例列表。
    """

    cases = _discover_dino_cases(model_root, pytestconfig)
    selected: dict[str, ModelDirCase] = {}
    for case in cases:
        selected.setdefault(case.family, case)
    return [selected[family] for family in DINO_FAMILIES if family in selected]


def _assert_feature_dump(dump_dir: Path, case: ModelDirCase) -> None:
    """校验 feature_extract sample 写出的 manifest 与特征张量。

    Args:
        dump_dir: sample 输出目录。
        case: 当前 DINO 测试用例。
    """

    metadata, tensors = parse_manifest(dump_dir / "manifest.txt")
    assert metadata["model_name"] == case.model_name
    assert metadata["feature_only"] == "true"
    assert metadata["feature_tensor_names"] == ",".join(case.feature_names)
    assert "input" in tensors

    input_tensor = load_tensor_from_dump(dump_dir, tensors["input"])
    assert input_tensor.shape[0] == 1
    assert input_tensor.shape[1] == 3
    assert input_tensor.dtype == np.float32

    for feature_name in case.feature_names:
        assert feature_name in tensors
        values = load_tensor_from_dump(dump_dir, tensors[feature_name])
        assert values.dtype == np.float32
        assert values.size > 0


def test_dino_feature_extract_sample_runs_with_models_root(
    pytestconfig: pytest.Config,
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    default_image: Path,
) -> None:
    """使用模型根目录中的 DINO checkpoint 导出 ``.wts`` 并运行 C++ 特征提取 sample。

    Args:
        pytestconfig: pytest 配置对象。
        repo_root: 仓库根目录。
        build_dir: CMake 构建目录。
        model_root: 真实模型根目录。
        default_image: 默认测试图片。
    """

    executable = require_sample(build_dir, "feature_extract")

    for case in _representative_dino_cases(model_root, pytestconfig):
        model = _load_case_model(case, pytestconfig)
        output_dir = _case_artifact_dir(model_root, case)
        weights_path = _ensure_wts(case, model, output_dir)
        dump_dir = artifact_dir(build_dir, "dino_feature_extract") / case.family / case.model_name

        completed = run_process_capture(
            [
                str(executable),
                "--model",
                case.model_name,
                "--weights-file",
                str(weights_path),
                "--features",
                ",".join(case.feature_names),
                "--image-path",
                str(default_image),
                "--output-dir",
                str(dump_dir),
            ],
            cwd=repo_root,
        )
        output = completed.stdout + completed.stderr

        assert "Building or loading feature-only model" in output
        assert "Running feature forward" in output
        assert "Saved feature dump to:" in output
        for feature_name in case.feature_names:
            assert f"{feature_name} dims=[" in output
        _assert_feature_dump(dump_dir, case)
