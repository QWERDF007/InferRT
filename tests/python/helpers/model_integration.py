"""真实模型集成测试的公共辅助函数。"""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers.runtime import run_process_capture, sample_executable


def artifact_dir(build_dir: Path, family: str) -> Path:
    """返回真实模型集成测试的产物目录。

    Args:
        build_dir: CMake 构建目录。
        family: 模型族名称，例如 ``yolo`` 或 ``sam``。

    Returns:
        存放导出 ``.wts``、engine cache 与输出图片的目录。
    """

    path = build_dir / "python_test_artifacts" / family
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


def assert_output_image(path: Path) -> None:
    """检查 sample 输出图片已生成且非空。

    Args:
        path: 输出图片路径。
    """

    assert path.exists(), f"output image missing: {path}"
    assert path.stat().st_size > 0, f"output image is empty: {path}"
