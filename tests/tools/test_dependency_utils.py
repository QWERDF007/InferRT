from __future__ import annotations

import os

import pytest

from tools.dependency_utils import link_file


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
