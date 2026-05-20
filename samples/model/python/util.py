"""Shared helpers for InferRT Python samples."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_DLL_DIRECTORY_HANDLES: list[object] = []


def resolve_project_root() -> Path:
    """Return the repository root resolved from this file location."""

    return Path(__file__).resolve().parents[3]


def read_cmake_cache(cache_path: Path) -> dict[str, str]:
    """Read `CMakeCache.txt` and return a key-value mapping."""

    if not cache_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("//") or line.startswith("#") or "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0].strip()
        values[key] = value.strip()
    return values


def collect_dll_dirs(build_dir: Path) -> list[Path]:
    """Collect DLL search directories needed by the Python bindings."""

    dll_dirs: list[Path] = []

    def add_dir(path: Path) -> None:
        if path.exists() and path not in dll_dirs:
            dll_dirs.append(path)

    add_dir(build_dir / "bin")
    add_dir(build_dir / "lib")

    cache = read_cmake_cache(build_dir / "CMakeCache.txt")
    trt_root = cache.get("TRT_ROOT", "")
    if trt_root:
        add_dir(Path(trt_root) / "bin")
        add_dir(Path(trt_root) / "lib")

    opencv_dir = cache.get("OpenCV_DIR", "")
    if not opencv_dir:
        details = cache.get("FIND_PACKAGE_MESSAGE_DETAILS_OpenCV", "")
        match = re.search(r"\[([A-Za-z]:/.+?)\]", details)
        if match:
            opencv_dir = match.group(1)

    if opencv_dir:
        opencv_root = Path(opencv_dir)
        add_dir(opencv_root / "bin")
        add_dir(opencv_root / "x64" / "vc16" / "bin")
        add_dir(opencv_root.parent / "bin")

    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        add_dir(Path(cuda_path) / "bin")

    extra_dirs = os.environ.get("INFERRT_DLL_DIRS", "")
    for raw in extra_dirs.split(";"):
        raw = raw.strip()
        if raw:
            add_dir(Path(raw))

    return dll_dirs


def ensure_module_path(build_dir: Path) -> None:
    """Add Python module and runtime DLL directories for InferRT bindings."""

    build_bin = build_dir / "bin"
    build_lib = build_dir / "lib"
    path_entries = os.environ.get("PATH", "").split(os.pathsep)

    if str(build_lib) not in sys.path:
        sys.path.insert(0, str(build_lib))

    if str(build_bin) not in sys.path:
        sys.path.insert(0, str(build_bin))

    for dll_dir in collect_dll_dirs(build_dir):
        dll_dir_str = str(dll_dir)
        if dll_dir_str not in path_entries:
            path_entries.insert(0, dll_dir_str)
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(dll_dir_str))

    os.environ["PATH"] = os.pathsep.join(path_entries)


def preprocess_image(image_path: Path) -> np.ndarray:
    """Preprocess an image using the repo's ImageNet classification convention."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    chw = np.transpose(image, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32, copy=False)


def load_labels(label_path: Path) -> list[str]:
    """Load ImageNet labels from a text file."""

    with label_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def split_csv_names(csv: str) -> list[str]:
    """Split a comma-separated name list and drop empty entries."""

    return [token.strip() for token in csv.split(",") if token.strip()]


def sanitize_file_stem(name: str) -> str:
    """Convert a tensor name into a filesystem-safe file stem."""

    sanitized = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            sanitized.append(ch)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_") or "tensor"


def dims_to_csv(shape: tuple[int, ...] | list[int]) -> str:
    """Encode a tensor shape as a comma-separated string."""

    return ",".join(str(dim) for dim in shape)


def numpy_dtype_name(array: np.ndarray) -> str:
    """Return the manifest-friendly NumPy dtype name."""

    return str(array.dtype)
