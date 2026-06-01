"""真实模型集成测试的公共辅助函数。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from helpers.runtime import run_process_capture, sample_executable


def artifact_dir(build_dir: Path, family: str) -> Path:
    """返回真实模型集成测试的临时产物目录。

    Args:
        build_dir: CMake 构建目录。
        family: 模型族名称，例如 ``yolo`` 或 ``sam``。

    Returns:
        存放 engine cache、输出图片与 sample dump 等临时产物的目录。
    """

    path = build_dir / "python_test_artifacts" / family
    path.mkdir(parents=True, exist_ok=True)
    return path


def conversion_artifact_dir(model_root: Path, checkpoint: Path) -> Path:
    """返回真实模型转换产物在模型根目录下的目录。

    文件 checkpoint 会映射到去掉后缀后的同名子目录，例如
    ``<root>/yolov8/yolov8n.pt`` 映射为 ``<root>/yolov8/yolov8n``；
    目录 checkpoint 会直接映射到该目录本身。
    """

    root = model_root.expanduser().resolve()
    source = checkpoint.expanduser().resolve()
    try:
        relative_source = source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Checkpoint must be under model root: checkpoint={source}, model_root={root}") from exc

    relative_output = relative_source if source.is_dir() else relative_source.with_suffix("")
    path = root / relative_output
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path, label: str) -> Path:
    """确保真实模型文件存在，不存在时跳过当前集成测试。

    Args:
        path: 待检查的文件路径。
        label: 用于 skip 消息的模型描述。

    Returns:
        已存在的文件路径。
    """

    if not path.exists():
        pytest.skip(f"{label} not found: {path}")
    return path


def require_sample(build_dir: Path, name: str) -> Path:
    """查找 C++ sample 可执行文件，不存在时跳过当前集成测试。

    Args:
        build_dir: CMake 构建目录。
        name: sample 名称，例如 ``detection`` 或 ``segmentation``。

    Returns:
        sample 可执行文件路径。
    """

    try:
        return sample_executable(build_dir, name)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def run_export_or_skip(command: list[str], *, cwd: Path) -> None:
    """运行权重导出命令；缺少第三方导出依赖时跳过而不是失败。

    Args:
        command: Python 导出脚本命令行。
        cwd: 子进程工作目录。
    """

    try:
        run_process_capture(command, cwd=cwd)
    except RuntimeError as exc:
        message = str(exc)
        missing_dependency = (
            "ModuleNotFoundError" in message
            or "No module named" in message
            or "DLL load failed" in message
            or "ImportError" in message
        )
        if missing_dependency:
            pytest.skip(message)
        raise


def is_fresh(output: Path, source: Path) -> bool:
    """判断缓存的 ``.wts`` 是否可复用。

    Args:
        output: 已导出的 ``.wts``。
        source: 原始 checkpoint。

    Returns:
        ``output`` 存在、非空且不早于 ``source`` 时返回 ``True``。
    """

    return output.exists() and output.stat().st_size > 0 and output.stat().st_mtime >= source.stat().st_mtime


def is_fresh_against_all(output: Path, sources: list[Path]) -> bool:
    """判断缓存文件是否不早于所有存在的源文件。"""

    if not output.exists() or output.stat().st_size <= 0:
        return False
    output_mtime = output.stat().st_mtime
    return all(output_mtime >= source.stat().st_mtime for source in sources if source.exists())


def assert_output_image(path: Path) -> None:
    """检查 sample 输出图片已生成且非空。

    Args:
        path: 输出图片路径。
    """

    assert path.exists(), f"output image missing: {path}"
    assert path.stat().st_size > 0, f"output image is empty: {path}"


def ensure_yolo_wts(
    *,
    repo_root: Path,
    build_dir: Path,
    model_root: Path,
    model_name: str,
    checkpoint: Path,
    ultralytics_repo: Path,
    yolov5_repo: Path | None = None,
) -> Path:
    """使用 Ultralytics checkpoint 导出 InferRT YOLO ``.wts``。

    Args:
        repo_root: 仓库根目录。
        build_dir: CMake 构建目录。
        model_root: 真实模型根目录。
        model_name: InferRT YOLO 模型 key。
        checkpoint: ``--inferrt-model-root`` 下的 ``.pt`` 权重。
        ultralytics_repo: 本地 ultralytics 仓库；未安装包时用于导入。
        yolov5_repo: 本地 YOLOv5 仓库；旧版 YOLOv5 checkpoint 导出时使用。

    Returns:
        导出的 ``.wts`` 路径。
    """

    checkpoint = require_file(checkpoint, f"{model_name} checkpoint")
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(build_dir / "ultralytics_config"))
    output = conversion_artifact_dir(model_root, checkpoint) / f"{model_name}.wts"
    exporter_sources = [
        checkpoint,
        repo_root / "samples" / "model" / "python" / "detection_gen_wts.py",
        repo_root / "samples" / "model" / "python" / "yolo_model_zoo.py",
    ]
    if is_fresh_against_all(output, exporter_sources):
        return output

    command = [
        sys.executable,
        "samples/model/python/detection_gen_wts.py",
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
    if yolov5_repo is not None and yolov5_repo.exists():
        command.extend(["--yolov5-repo", str(yolov5_repo)])

    run_export_or_skip(command, cwd=repo_root)
    return require_file(output, f"{model_name} exported .wts")


def ensure_sam_v1_wts(*, repo_root: Path, model_root: Path, checkpoint: Path, sam_root: Path) -> Path:
    """使用官方 Segment Anything v1 checkpoint 导出 ``sam_vit_b`` 的 ``.wts``。

    Args:
        repo_root: 仓库根目录。
        model_root: 真实模型根目录。
        checkpoint: SAM v1 官方 ``.pth`` checkpoint。
        sam_root: 本地 Segment Anything v1 仓库路径。

    Returns:
        导出的 ``.wts`` 路径。
    """

    checkpoint = require_file(checkpoint, "SAM ViT-B checkpoint")
    output = conversion_artifact_dir(model_root, checkpoint) / "sam_vit_b.wts"
    if is_fresh(output, checkpoint):
        return output

    command = [
        sys.executable,
        "samples/model/python/gen_sam_wts.py",
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


def ensure_sam2_wts(
    *,
    repo_root: Path,
    model_root: Path,
    checkpoint: Path,
    sam2_root: Path,
) -> Path:
    """使用官方 SAM2.1 Hiera-Tiny checkpoint 导出 InferRT ``.wts``。

    Args:
        repo_root: 仓库根目录。
        model_root: 真实模型根目录。
        checkpoint: SAM2.1 官方 ``.pt`` checkpoint。
        sam2_root: 本地 SAM2 仓库路径。

    Returns:
        导出的 ``.wts`` 路径。
    """

    checkpoint = require_file(checkpoint, "SAM2.1 Hiera-Tiny checkpoint")
    output = conversion_artifact_dir(model_root, checkpoint) / "sam2_1_hiera_tiny.wts"
    if is_fresh(output, checkpoint):
        return output

    command = [
        sys.executable,
        "samples/model/python/gen_sam_wts.py",
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


def ensure_edge_sam_wts(*, repo_root: Path, model_root: Path, checkpoint: Path, edge_sam_root: Path) -> Path:
    """使用官方 EdgeSAM checkpoint 导出 InferRT ``.wts``。

    Args:
        repo_root: 仓库根目录。
        model_root: 真实模型根目录。
        checkpoint: EdgeSAM 官方 ``.pth`` checkpoint。
        edge_sam_root: 本地 EdgeSAM 仓库路径。

    Returns:
        导出的 ``.wts`` 路径。
    """

    checkpoint = require_file(checkpoint, "EdgeSAM checkpoint")
    try:
        output_dir = conversion_artifact_dir(model_root, checkpoint)
    except ValueError:
        output_dir = checkpoint.with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "edge_sam.wts"
    exporter_sources = [checkpoint, repo_root / "samples" / "model" / "python" / "gen_sam_wts.py"]
    if is_fresh_against_all(output, exporter_sources):
        return output

    command = [
        sys.executable,
        "samples/model/python/gen_sam_wts.py",
        "--model",
        "edge_sam",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--skip-forward",
    ]
    if edge_sam_root.exists():
        command.extend(["--edge-sam-root", str(edge_sam_root)])

    run_export_or_skip(command, cwd=repo_root)
    return require_file(output, "EdgeSAM exported .wts")
