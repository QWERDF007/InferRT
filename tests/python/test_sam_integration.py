"""使用真实 SAM checkpoint 验证导出脚本与 segmentation sample。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

from helpers.model_integration import (
    artifact_dir,
    assert_output_image,
    is_fresh,
    require_file,
    require_sample,
    run_export_or_skip,
)
from helpers.runtime import run_process_capture


def _ensure_sam_v1_wts(*, repo_root: Path, build_dir: Path, checkpoint: Path, sam_root: Path) -> Path:
    """使用官方 Segment Anything v1 checkpoint 导出 ``sam_vit_b`` 的 ``.wts``。

    Args:
        repo_root: 仓库根目录。
        build_dir: CMake 构建目录。
        checkpoint: SAM v1 官方 ``.pth`` checkpoint。
        sam_root: 本地 Segment Anything v1 仓库路径。

    Returns:
        导出的 ``.wts`` 路径。
    """

    checkpoint = require_file(checkpoint, "SAM ViT-B checkpoint")
    output = artifact_dir(build_dir, "sam") / "sam_vit_b.wts"
    if is_fresh(output, checkpoint):
        return output

    command = [
        sys.executable,
        "samples/model/segmentation/gen_sam_wts.py",
        "--model",
        "vit_b",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--skip-forward",
    ]
    if sam_root.exists():
        command.extend(["--sam-root", str(sam_root)])
    elif importlib.util.find_spec("segment_anything") is None:
        pytest.skip(f"segment-anything package/repo not found: {sam_root}")

    run_export_or_skip(command, cwd=repo_root)
    return require_file(output, "SAM ViT-B exported .wts")


def _ensure_sam2_wts(*, repo_root: Path, build_dir: Path, checkpoint: Path, sam2_root: Path) -> Path:
    """使用官方 SAM2.1 Hiera-Tiny checkpoint 导出 InferRT ``.wts``。

    Args:
        repo_root: 仓库根目录。
        build_dir: CMake 构建目录。
        checkpoint: SAM2.1 官方 ``.pt`` checkpoint。
        sam2_root: 本地 SAM2 仓库路径。

    Returns:
        导出的 ``.wts`` 路径。
    """

    checkpoint = require_file(checkpoint, "SAM2.1 Hiera-Tiny checkpoint")
    output = artifact_dir(build_dir, "sam") / "sam2_1_hiera_tiny.wts"
    if is_fresh(output, checkpoint):
        return output

    command = [
        sys.executable,
        "samples/model/segmentation/gen_sam2_wts.py",
        "--model",
        "sam2_1_hiera_tiny",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--skip-forward",
    ]
    if sam2_root.exists():
        command.extend(["--sam2-root", str(sam2_root)])
    elif importlib.util.find_spec("sam2") is None:
        pytest.skip(f"SAM2 package/repo not found: {sam2_root}")

    run_export_or_skip(command, cwd=repo_root)
    return require_file(output, "SAM2.1 Hiera-Tiny exported .wts")


def test_sam_v1_checkpoint_exports_with_models_root(
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    sam_root: Path,
) -> None:
    """使用 ``D:/Models/sam`` 中的 SAM ViT-B checkpoint 验证 v1 导出脚本。

    本用例只验证 checkpoint 到 ``.wts`` 的真实权重导出，避免默认 pytest 运行时构建较重的
    SAM ViT-B TensorRT engine；完整分割 sample 由 SAM2.1 tiny 用例覆盖。
    """

    weights = _ensure_sam_v1_wts(
        repo_root=repo_root,
        build_dir=build_dir,
        checkpoint=model_root / "sam" / "sam_vit_b_01ec64.pth",
        sam_root=sam_root,
    )

    with weights.open("r", encoding="utf-8") as file:
        first_line = file.readline().strip()
    assert int(first_line) > 100


def test_sam2_sample_runs_with_models_root(
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    default_image: Path,
    sam2_root: Path,
) -> None:
    """使用 ``D:/Models/sam`` 中的 SAM2.1 tiny checkpoint 运行 segmentation sample。"""

    executable = require_sample(build_dir, "segmentation")
    weights = _ensure_sam2_wts(
        repo_root=repo_root,
        build_dir=build_dir,
        checkpoint=model_root / "sam" / "sam2.1_hiera_tiny.pt",
        sam2_root=sam2_root,
    )
    output_image = artifact_dir(build_dir, "sam") / "sam2_1_hiera_tiny_mask.jpg"

    completed = run_process_capture(
        [
            str(executable),
            "--model",
            "sam2_1_hiera_tiny",
            "--weights-file",
            str(weights),
            "--image-path",
            str(default_image),
            "--output-image",
            str(output_image),
            "--point-x",
            "0.5",
            "--point-y",
            "0.5",
        ],
        cwd=repo_root,
    )
    output = completed.stdout + completed.stderr

    assert "Output masks dims=[1,3,256,256]" in output
    assert "Output iou_predictions dims=[1,3,1,1]" in output
    assert "Saved mask overlay to:" in output
    assert_output_image(output_image)
