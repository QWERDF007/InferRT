"""Python 集成测试使用的图像预处理、输出分配和运行时路径工具。"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_DLL_DIRECTORY_HANDLES: list[object] = []


def _read_cmake_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line or ":" not in line:
            continue
        key_type, value = line.split("=", 1)
        values[key_type.split(":", 1)[0].strip()] = value.strip()
    return values


def _collect_dll_dirs(build_dir: Path) -> list[Path]:
    dll_dirs: list[Path] = []

    def add_dir(path: Path) -> None:
        if path.exists() and path not in dll_dirs:
            dll_dirs.append(path)

    add_dir(build_dir / "bin")
    add_dir(build_dir / "lib")

    cache = _read_cmake_cache(build_dir / "CMakeCache.txt")
    trt_root = cache.get("TRT_ROOT", "")
    if trt_root:
        add_dir(Path(trt_root) / "bin")
        add_dir(Path(trt_root) / "lib")

    onnxruntime_root = cache.get("ONNXRUNTIME_ROOT", "")
    if onnxruntime_root:
        add_dir(Path(onnxruntime_root) / "bin")
        add_dir(Path(onnxruntime_root) / "lib")

    openvino_root = cache.get("INFERRT_OPENVINO_ROOT", "")
    if openvino_root:
        for candidate in (
            Path(openvino_root) / "runtime" / "bin" / "intel64" / "Release",
            Path(openvino_root) / "runtime" / "bin" / "intel64",
            Path(openvino_root) / "runtime" / "bin",
            Path(openvino_root) / "runtime" / "3rdparty" / "tbb" / "bin",
        ):
            add_dir(candidate)

    faiss_root = cache.get("Faiss_HOME", "")
    if faiss_root:
        add_dir(Path(faiss_root) / "bin")

    mkl_root = cache.get("MKL_ROOT", "")
    if mkl_root:
        for candidate in (
            Path(mkl_root) / "bin",
            Path(mkl_root) / "compiler" / "bin",
            Path(mkl_root) / "mkl" / "bin",
            Path(mkl_root).parent / "compiler" / "bin",
        ):
            add_dir(candidate)

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

    for raw in os.environ.get("INFERRT_DLL_DIRS", "").split(";"):
        if raw.strip():
            add_dir(Path(raw.strip()))
    return dll_dirs


def ensure_module_path(build_dir: Path) -> None:
    """将构建出的 Python 模块和 InferRT 依赖 DLL 加入搜索路径。"""

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for module_dir in (build_dir / "lib", build_dir / "bin"):
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))

    for dll_dir in _collect_dll_dirs(build_dir):
        dll_text = str(dll_dir)
        if dll_text not in path_entries:
            path_entries.insert(0, dll_text)
        if hasattr(os, "add_dll_directory"):
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(dll_text))
    os.environ["PATH"] = os.pathsep.join(path_entries)


def preprocess_image(image_path: Path, image_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """按 ImageNet 约定将一张图片转换为 ``NCHW`` FP32。"""

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = cv2.subtract(image, (0.485, 0.456, 0.406, 0.0))
    image = cv2.divide(image, (0.229, 0.224, 0.225, 1.0))
    return np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0).astype(np.float32, copy=False)


def allocate_output_tensors(model: object, output_names: list[str]) -> dict[str, np.ndarray]:
    """按 InferRT 模型的输出元数据分配 NumPy 输出缓冲区。"""

    return {
        name: np.empty(model.tensor_shape(name), dtype=np.dtype(model.tensor_dtype(name)))  # type: ignore[attr-defined]
        for name in output_names
    }
