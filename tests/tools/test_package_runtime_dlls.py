from __future__ import annotations

from pathlib import Path

import pytest

from tools.dependency_utils import resolve_dependency_root
from tools.package_runtime_dlls import copy_runtime_files, dependency_files, project_runtime_files


def write_cache(build_dir: Path, **values: str) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}:BOOL={value}" for key, value in values.items()]
    (build_dir / "CMakeCache.txt").write_text("\n".join(lines), encoding="utf-8")


def test_disabled_dependency_is_not_required(tmp_path):
    build_dir = tmp_path / "build"
    root = tmp_path / "optional"
    root.mkdir()
    (root / "optional.dll").write_bytes(b"optional")
    write_cache(build_dir, INFERRT_BUILD_ONNX="OFF")
    manifest = tmp_path / "dependencies.yaml"
    manifest.write_text(
        "dependencies:\n"
        f"  - name: optional\n    requires: [INFERRT_BUILD_ONNX]\n    root: {root.as_posix()}\n"
        "    windows: [optional.dll]\n",
        encoding="utf-8",
    )

    assert dependency_files(build_dir, manifest, "Release", strict=True) == []


def test_strict_mode_rejects_missing_wildcard(tmp_path):
    build_dir = tmp_path / "build"
    root = tmp_path / "runtime"
    root.mkdir()
    write_cache(build_dir, INFERRT_ENABLE_CUDA="ON")
    manifest = tmp_path / "dependencies.yaml"
    manifest.write_text(
        "dependencies:\n"
        f"  - name: runtime\n    root: {root.as_posix()}\n    windows: [missing-*.dll]\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="matched no files"):
        dependency_files(build_dir, manifest, "Release", strict=True)


def test_copy_runtime_files_rejects_same_name_from_different_sources(tmp_path):
    first = tmp_path / "first" / "runtime.dll"
    second = tmp_path / "second" / "runtime.dll"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    with pytest.raises(RuntimeError, match="name collision"):
        copy_runtime_files([first, second], tmp_path / "bin")


def test_strict_mode_rejects_missing_project_output(tmp_path):
    with pytest.raises(RuntimeError, match="project runtime output directory is missing"):
        project_runtime_files(tmp_path / "build", "Release", strict=True)


def test_project_runtime_files_filters_python_abi_from_cmake_cache(tmp_path):
    build_dir = tmp_path / "build"
    output_dir = build_dir / "bin"
    output_dir.mkdir(parents=True)
    (output_dir / "inferrt_model_py.cp312-win_amd64.pyd").write_bytes(b"py312")
    (output_dir / "inferrt_model_py.cp39-win_amd64.pyd").write_bytes(b"py39")
    (build_dir / "CMakeCache.txt").write_text(
        "PYTHON_MODULE_EXTENSION:INTERNAL=.cp312-win_amd64.pyd\n",
        encoding="utf-8",
    )

    files = project_runtime_files(build_dir, "Release", strict=True)

    assert [path.name for path in files] == ["inferrt_model_py.cp312-win_amd64.pyd"]


def test_platform_specific_dependency_root_is_used(tmp_path):
    runtime_root = tmp_path / "opencv-bin"
    runtime_root.mkdir()
    dependency = {
        "root": str(tmp_path / "wrong-root"),
        "windows_root": runtime_root.as_posix(),
    }

    assert resolve_dependency_root(
        dependency,
        tmp_path / "build",
        platform="windows",
    ) == runtime_root.resolve()
