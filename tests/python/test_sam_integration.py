"""使用真实 SAM checkpoint 验证导出脚本与 segmentation sample。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

from helpers.model_integration import (
    artifact_dir,
    assert_output_image,
    ensure_edge_sam_wts,
    ensure_sam2_wts,
    ensure_sam_v1_wts,
    require_sample,
)
from helpers.runtime import run_process_capture


def test_sam_v1_checkpoint_exports_with_models_root(
    repo_root: Path,
    model_root: Path,
    sam_root: Path,
) -> None:
    """使用 ``D:/Models/sam`` 中的 SAM ViT-B checkpoint 验证 v1 导出脚本。

    本用例只验证 checkpoint 到 ``.wts`` 的真实权重导出，避免默认 pytest 运行时构建较重的
    SAM ViT-B TensorRT engine；完整分割 sample 由 SAM2.1 tiny 用例覆盖。
    """

    weights = ensure_sam_v1_wts(
        repo_root=repo_root,
        model_root=model_root,
        checkpoint=model_root / "sam" / "sam_vit_b_01ec64.pth",
        sam_root=sam_root,
    )

    with weights.open("r", encoding="utf-8") as file:
        first_line = file.readline().strip()
    assert int(first_line) > 100


def test_edge_sam_checkpoint_exports_with_models_root(
    repo_root: Path,
    model_root: Path,
    edge_sam_root: Path,
    edge_sam_checkpoint: Path,
) -> None:
    """使用本地 EdgeSAM checkpoint 验证 ``.wts`` 导出。"""

    weights = ensure_edge_sam_wts(
        repo_root=repo_root,
        model_root=model_root,
        checkpoint=edge_sam_checkpoint,
        edge_sam_root=edge_sam_root,
    )

    with weights.open("r", encoding="utf-8") as file:
        first_line = file.readline().strip()
    assert int(first_line) > 500


def test_sam2_sample_runs_with_models_root(
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    default_image: Path,
    sam2_root: Path,
) -> None:
    """使用 ``D:/Models/sam`` 中的 SAM2.1 tiny checkpoint 运行 segmentation sample。"""

    executable = require_sample(build_dir, "segmentation")
    weights = ensure_sam2_wts(
        repo_root=repo_root,
        model_root=model_root,
        checkpoint=model_root / "sam" / "sam2.1_hiera_tiny.pt",
        sam2_root=sam2_root,
    )
    output_image = artifact_dir(build_dir, "sam") / f"sam2_1_hiera_tiny_mask_{os.getpid()}.jpg"

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
