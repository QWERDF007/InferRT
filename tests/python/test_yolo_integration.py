"""使用真实 YOLO checkpoint 验证导出脚本与 detection sample。"""

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


def _ensure_yolo_wts(
    *,
    repo_root: Path,
    build_dir: Path,
    model_name: str,
    checkpoint: Path,
    ultralytics_repo: Path,
) -> Path:
    """使用 Ultralytics checkpoint 导出 InferRT YOLO ``.wts``。

    Args:
        repo_root: 仓库根目录。
        build_dir: CMake 构建目录。
        model_name: InferRT YOLO 模型 key。
        checkpoint: ``--inferrt-model-root`` 下的 ``.pt`` 权重。
        ultralytics_repo: 本地 ultralytics 仓库；未安装包时用于导入。

    Returns:
        导出的 ``.wts`` 路径。
    """

    checkpoint = require_file(checkpoint, f"{model_name} checkpoint")
    output = artifact_dir(build_dir, "yolo") / f"{model_name}.wts"
    if is_fresh(output, checkpoint):
        return output

    command = [
        sys.executable,
        "samples/model/detection/gen_wts.py",
        "--model",
        model_name,
        "--weights",
        str(checkpoint),
        "--output",
        str(output),
        "--quiet",
    ]
    if ultralytics_repo.exists():
        command.extend(["--ultralytics-repo", str(ultralytics_repo)])
    elif importlib.util.find_spec("ultralytics") is None:
        pytest.skip(f"Ultralytics package/repo not found: {ultralytics_repo}")

    run_export_or_skip(command, cwd=repo_root)
    return require_file(output, f"{model_name} exported .wts")


@pytest.mark.parametrize(
    "model_name,relative_checkpoint",
    [
        pytest.param("yolov8n", Path("yolov8") / "yolov8n.pt", id="yolov8n"),
        pytest.param("yolov5n", Path("yolov5") / "yolov5nu.pt", id="yolov5n"),
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
    weights = _ensure_yolo_wts(
        repo_root=repo_root,
        build_dir=build_dir,
        model_name=model_name,
        checkpoint=model_root / relative_checkpoint,
        ultralytics_repo=ultralytics_repo,
    )
    output_image = artifact_dir(build_dir, "yolo") / f"{model_name}_dog.jpg"

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
