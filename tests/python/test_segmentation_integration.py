"""使用真实 RF-DETR-Seg checkpoint 验证单图 segmentation sample。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

from helpers.model_integration import artifact_dir, assert_output_image, ensure_rfdetr_wts, ensure_yolo_wts, require_sample
from helpers.runtime import run_process_capture


def test_rfdetr_seg_sample_runs_with_models_root(
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    default_image: Path,
    rfdetr_root: Path,
) -> None:
    """运行 RF-DETR-Seg 单图分割 sample，不向模型传入 prompt 等额外输入。

    Args:
        repo_root: 仓库根目录。
        build_dir: CMake 构建目录。
        model_root: 真实模型根目录，包含 ``rfdetr`` 子目录。
        default_image: 默认 dog 测试图。
        rfdetr_root: 本地 RF-DETR 上游源码仓库。
    """

    executable = require_sample(build_dir, "segmentation")
    weights = ensure_rfdetr_wts(
        repo_root=repo_root,
        build_dir=build_dir,
        model_name="rfdetr_seg_nano",
        checkpoint=model_root / "rfdetr" / "rf-detr-seg-nano.pt",
        rfdetr_root=rfdetr_root,
    )
    output_image = artifact_dir(build_dir, "rfdetr") / f"rfdetr_seg_nano_dog_{os.getpid()}.jpg"

    completed = run_process_capture(
        [
            str(executable),
            "--model",
            "rfdetr_seg_nano",
            "--weights-file",
            str(weights),
            "--image-path",
            str(default_image),
            "--output-image",
            str(output_image),
            "--conf-threshold",
            "0.35",
            "--max-instances",
            "20",
        ],
        cwd=repo_root,
    )
    output = completed.stdout + completed.stderr

    assert "Model loaded successfully." in output
    assert "Output dets dims=" in output
    assert "Output labels dims=" in output
    assert "Output masks dims=" in output
    assert "Instances:" in output
    assert_output_image(output_image)


def test_yolov8_seg_sample_runs_with_models_root(
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    default_image: Path,
    ultralytics_repo: Path,
    yolov5_repo: Path,
) -> None:
    """运行 YOLOv8-Seg 单图分割 sample，验证 TensorRT 原生图输出 proto mask。
    Args:
        repo_root: 仓库根目录。
        build_dir: CMake 构建目录。
        model_root: 真实模型根目录，包含 ``yolov8`` 子目录。
        default_image: 默认 dog 测试图。
        ultralytics_repo: 本地 Ultralytics 仓库路径。
        yolov5_repo: 本地 YOLOv5 仓库路径；保持 helper 签名一致。
    """

    executable = require_sample(build_dir, "segmentation")
    weights = ensure_yolo_wts(
        repo_root=repo_root,
        build_dir=build_dir,
        model_root=model_root,
        model_name="yolov8n_seg",
        checkpoint=model_root / "yolov8" / "yolov8n-seg.pt",
        ultralytics_repo=ultralytics_repo,
        yolov5_repo=yolov5_repo,
    )
    output_image = artifact_dir(build_dir, "yolo") / f"yolov8n_seg_dog_{os.getpid()}.jpg"

    completed = run_process_capture(
        [
            str(executable),
            "--model",
            "yolov8n_seg",
            "--weights-file",
            str(weights),
            "--image-path",
            str(default_image),
            "--output-image",
            str(output_image),
            "--conf-threshold",
            "0.10",
            "--max-instances",
            "20",
        ],
        cwd=repo_root,
    )
    output = completed.stdout + completed.stderr

    assert "Model loaded successfully." in output
    assert "Output output0 dims=" in output
    assert "Output output1 dims=" in output
    assert "Output output2 dims=" in output
    assert "Output proto dims=" in output
    assert "Instances:" in output
    assert_output_image(output_image)
