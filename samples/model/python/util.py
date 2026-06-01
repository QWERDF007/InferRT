"""InferRT Python 示例脚本共用的辅助函数。"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_DLL_DIRECTORY_HANDLES: list[object] = []


def resolve_project_root() -> Path:
    """根据当前文件位置解析仓库根目录。

    Returns:
        Path: 当前仓库的根目录路径。
    """

    return Path(__file__).resolve().parents[3]


def read_cmake_cache(cache_path: Path) -> dict[str, str]:
    """读取 `CMakeCache.txt` 并返回键值映射。

    Args:
        cache_path: `CMakeCache.txt` 文件路径。

    Returns:
        dict[str, str]: 从缓存文件中解析出的配置项；若文件不存在则返回空字典。
    """

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
    """收集 Python 绑定运行时需要加入搜索路径的 DLL 目录。

    Args:
        build_dir: CMake 构建目录。

    Returns:
        list[Path]: 需要加入 DLL 搜索路径的目录列表。
    """

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
    """配置 InferRT Python 绑定所需的模块路径和 DLL 搜索路径。

    该函数会把构建产物目录加入 `sys.path`，并在 Windows 上尽量通过
    `os.add_dll_directory` 注册依赖 DLL 所在目录。

    Args:
        build_dir: CMake 构建目录。
    """

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


def preprocess_image(image_path: Path, image_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """按仓库中的 ImageNet 分类预处理约定处理输入图片。

    处理流程包括读取图片、BGR 转 RGB、缩放到目标尺寸、归一化，
    再转换为 `NCHW` 布局并补齐 batch 维度。

    Args:
        image_path: 输入图片路径。
        image_size: OpenCV resize 使用的 `(width, height)` 目标尺寸。

    Returns:
        np.ndarray: 预处理后的四维浮点张量，形状为 `(1, 3, H, W)`。

    Raises:
        FileNotFoundError: 图片读取失败时抛出。
    """

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) * (1.0 / 255.0)

    image = cv2.subtract(image, (0.485, 0.456, 0.406, 0.0))
    image = cv2.divide(image, (0.229, 0.224, 0.225, 1.0))

    chw = np.transpose(image, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32, copy=False)


def preprocess_images(image_paths: list[Path], image_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """将多张图片预处理并拼接为一个 NCHW batch。

    Args:
        image_paths: 输入图片路径列表。
        image_size: OpenCV resize 使用的 `(width, height)` 目标尺寸。

    Returns:
        np.ndarray: 形状为 `(N, 3, H, W)` 的 float32 batch。
    """

    if not image_paths:
        raise ValueError("At least one image path is required")
    tensors = [preprocess_image(path, image_size=image_size) for path in image_paths]
    return np.ascontiguousarray(np.concatenate(tensors, axis=0), dtype=np.float32)


def split_path_list(value: str) -> list[str]:
    """拆分逗号或分号分隔的路径列表。"""

    return [token.strip() for token in re.split(r"[,;]", value) if token.strip()]


def load_labels(label_path: Path) -> list[str]:
    """从文本文件加载标签列表。

    Args:
        label_path: 标签文件路径。

    Returns:
        list[str]: 去除空行后的标签字符串列表。
    """

    with label_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def allocate_output_tensors(model: object, output_names: list[str]) -> dict[str, np.ndarray]:
    """根据模型运行时张量信息分配输出数组。

    Args:
        model: 已加载好的模型对象，需要提供 `tensor_shape` 和 `tensor_dtype`
            接口。
        output_names: 待分配的输出张量名称列表。

    Returns:
        dict[str, np.ndarray]: 以张量名称为键、预分配 NumPy 数组为值的字典。
    """

    output_tensors: dict[str, np.ndarray] = {}
    for output_name in output_names:
        output_shape = model.tensor_shape(output_name)
        output_dtype = np.dtype(model.tensor_dtype(output_name))
        output_tensors[output_name] = np.empty(output_shape, dtype=output_dtype)
    return output_tensors


def split_csv_names(csv: str) -> list[str]:
    """拆分逗号分隔的名称字符串，并移除空项。

    Args:
        csv: 逗号分隔的名称字符串。

    Returns:
        list[str]: 去除首尾空白和空项后的名称列表。
    """

    return [token.strip() for token in csv.split(",") if token.strip()]


def sanitize_file_stem(name: str) -> str:
    """将张量名称转换为适合文件名的安全 stem。

    仅保留字母、数字、连字符和下划线，其余字符会被替换为下划线。

    Args:
        name: 原始张量名称。

    Returns:
        str: 可安全用于文件名的 stem；若结果为空则返回 `"tensor"`。
    """

    sanitized = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            sanitized.append(ch)
        else:
            sanitized.append("_")
    return "".join(sanitized).strip("_") or "tensor"


def dims_to_csv(shape: tuple[int, ...] | list[int]) -> str:
    """将张量形状编码为逗号分隔字符串。

    Args:
        shape: 张量形状，可为元组或列表。

    Returns:
        str: 逗号分隔的维度字符串，例如 `"1,3,224,224"`。
    """

    return ",".join(str(dim) for dim in shape)


def numpy_dtype_name(array: np.ndarray) -> str:
    """返回适合写入 manifest 的 NumPy 数据类型名称。

    Args:
        array: 输入 NumPy 数组。

    Returns:
        str: 数组的 dtype 名称字符串。
    """

    return str(array.dtype)
