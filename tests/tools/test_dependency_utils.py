from __future__ import annotations

import os

import pytest

from tools.dependency_utils import link_file, resolve_dependency_root


def test_copy_mode_replaces_existing_symlink(tmp_path):
    source = tmp_path / "source.dll"
    destination = tmp_path / "runtime.dll"
    source.write_bytes(b"runtime")

    try:
        os.symlink(source, destination)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    link_file(source, destination, mode="copy")

    assert not destination.is_symlink()
    assert destination.read_bytes() == b"runtime"


def test_manifest_default_precedes_stale_cmake_cache(tmp_path, monkeypatch):
    default_root = tmp_path / "default-opencv"
    stale_root = tmp_path / "stale-opencv"
    default_root.mkdir()
    stale_root.mkdir()

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "CMakeCache.txt").write_text(
        f"OpenCV_HOME:PATH={stale_root.as_posix()}\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("OpenCV_HOME", raising=False)
    dependency = {
        "name": "opencv",
        "root": "OpenCV_HOME",
        "cmake": "cmake/ConfigOpenCV.cmake",
        "default": default_root.as_posix(),
    }

    assert resolve_dependency_root(dependency, build_dir, repo_root=tmp_path, platform="windows") == (
        default_root.resolve()
    )
