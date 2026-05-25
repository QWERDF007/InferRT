"""使用真实 YOLO checkpoint 验证导出脚本与 detection sample。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

from helpers.model_integration import artifact_dir, assert_output_image, ensure_yolo_wts, require_sample
from helpers.runtime import run_process_capture


@pytest.mark.parametrize(
    "model_name,relative_checkpoint",
    [
        pytest.param("yolov8n", Path("yolov8") / "yolov8n.pt", id="yolov8n"),
        pytest.param("yolov5n", Path("yolov5") / "yolov5n.pt", id="yolov5n_legacy"),
        pytest.param("yolov5s", Path("yolov5") / "yolov5s.pt", id="yolov5s_legacy"),
    ],
)
def test_yolo_sample_runs_with_models_root(
    model_name: str,
    relative_checkpoint: Path,
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    default_image: Path,
    ultralytics_repo: Path,
    yolov5_repo: Path,
) -> None:
    """使用 ``D:/Models`` 中的 YOLO 权重导出 ``.wts`` 并运行 detection sample。

    Args:
        model_name: InferRT YOLO 模型 key。
        relative_checkpoint: 相对 ``--inferrt-model-root`` 的 checkpoint 路径。
        repo_root: 仓库根目录。
        build_dir: CMake 构建目录。
        model_root: 真实模型根目录，默认 ``D:/Models``。
        default_image: 默认 dog 测试图。
        ultralytics_repo: 本地 ultralytics 仓库路径。
    """

    executable = require_sample(build_dir, "detection")
    weights = ensure_yolo_wts(
        repo_root=repo_root,
        build_dir=build_dir,
        model_name=model_name,
        checkpoint=model_root / relative_checkpoint,
        ultralytics_repo=ultralytics_repo,
        yolov5_repo=yolov5_repo,
        family=f"yolo_sample_{relative_checkpoint.stem}",
    )
    output_image = artifact_dir(build_dir, "yolo") / f"{model_name}_dog_{os.getpid()}.jpg"

    completed = run_process_capture(
        [
            str(executable),
            "--model",
            model_name,
            "--weights-file",
            str(weights),
            "--image-path",
            str(default_image),
            "--output-image",
            str(output_image),
            "--conf-threshold",
            "0.10",
            "--max-detections",
            "20",
        ],
        cwd=repo_root,
    )
    output = completed.stdout + completed.stderr

    assert "Model loaded successfully." in output
    assert "Output output0 shape:" in output
    assert "Detections:" in output
    assert_output_image(output_image)
