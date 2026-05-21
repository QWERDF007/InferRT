"""Pytest fixtures for InferRT Python binding integration tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLES_PYTHON = ROOT / "samples" / "model" / "python"

if str(SAMPLES_PYTHON) not in sys.path:
    sys.path.insert(0, str(SAMPLES_PYTHON))

from util import ensure_module_path, preprocess_image  # noqa: E402

from helpers.runtime import default_build_dir, project_root, require_weights  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--inferrt-build-dir",
        action="store",
        default="",
        help="CMake build directory containing inferrt_model_py and sample executables",
    )
    parser.addoption("--inferrt-rtol", action="store", type=float, default=1e-2, help="Relative tolerance")
    parser.addoption("--inferrt-atol", action="store", type=float, default=1.2e-1, help="Absolute tolerance")


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return project_root()


@pytest.fixture(scope="session")
def build_dir(repo_root: Path, pytestconfig: pytest.Config) -> Path:
    override = pytestconfig.getoption("--inferrt-build-dir")
    path = Path(override).resolve() if override else default_build_dir(repo_root)
    if not path.exists():
        pytest.skip(f"Build directory not found: {path}")
    return path


@pytest.fixture(scope="session")
def irt_module(build_dir: Path):
    ensure_module_path(build_dir)
    import inferrt_model_py as irt

    return irt


@pytest.fixture(scope="session")
def default_image(repo_root: Path) -> Path:
    image_path = repo_root / "assets" / "pics" / "dog.jpg"
    if not image_path.exists():
        pytest.skip(f"Default test image missing: {image_path}")
    return image_path


@pytest.fixture(scope="session")
def input_tensor(default_image: Path):
    return preprocess_image(default_image)


@pytest.fixture(scope="session")
def tolerances(pytestconfig: pytest.Config) -> tuple[float, float]:
    return pytestconfig.getoption("--inferrt-rtol"), pytestconfig.getoption("--inferrt-atol")


@pytest.fixture
def weights_path(repo_root: Path):
    def _resolve(relative_path: str) -> Path:
        try:
            return require_weights(repo_root, relative_path)
        except FileNotFoundError as exc:
            pytest.skip(str(exc))

    return _resolve
