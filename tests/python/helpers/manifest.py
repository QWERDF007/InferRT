"""C++/Python sample 张量 dump 的 manifest 解析与数值断言工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    """manifest 中一条 ``tensor|...`` 记录对应的二进制文件描述。

    Attributes:
        name: 张量逻辑名，与 Python/C++ 输出字典键一致。
        dtype: manifest 中的类型字符串，如 ``float32``。
        dims: 张量形状元组；空元组表示一维展开存储。
        file_name: dump 目录下的二进制文件名。
    """

    name: str
    dtype: str
    dims: tuple[int, ...]
    file_name: str


def parse_dims(value: str) -> tuple[int, ...]:
    """将 manifest 中的维度字符串解析为元组。

    Args:
        value: 逗号分隔的维度，如 ``1,3,224,224``；空字符串表示标量/一维。

    Returns:
        tuple[int, ...]: 各维大小。
    """

    if not value:
        return ()
    return tuple(int(part) for part in value.split(","))


def dtype_to_numpy(dtype: str) -> np.dtype:
    """将 manifest 中的类型名映射为 ``numpy.dtype``。

    Args:
        dtype: 类型名字符串。

    Returns:
        np.dtype: 对应的 NumPy  dtype。

    Raises:
        ValueError: 不支持的类型名。
    """

    mapping = {
        "float32": np.float32,
        "float16": np.float16,
        "int8": np.int8,
        "uint8": np.uint8,
        "int32": np.int32,
        "int64": np.int64,
        "bool": np.bool_,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype in manifest: {dtype}")
    return mapping[dtype]


def parse_manifest(manifest_path: Path) -> tuple[dict[str, str], dict[str, TensorSpec]]:
    """解析 C++ sample 写出的 ``manifest.txt``。

    Args:
        manifest_path: manifest 文件路径。

    Returns:
        tuple[dict[str, str], dict[str, TensorSpec]]:
            - 元数据：``key=value`` 行（如 ``model_name``、``backend``）。
            - 张量表：张量名 -> ``TensorSpec``；张量行为 ``tensor|name|dtype|dims|file``。
    """

    metadata: dict[str, str] = {}
    tensors: dict[str, TensorSpec] = {}

    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("tensor|"):
            _, name, dtype, dims, file_name = line.split("|")
            tensors[name] = TensorSpec(name=name, dtype=dtype, dims=parse_dims(dims), file_name=file_name)
            continue
        key, value = line.split("=", 1)
        metadata[key] = value

    return metadata, tensors


def load_tensor_from_dump(dump_dir: Path, spec: TensorSpec) -> np.ndarray:
    """从 dump 目录按描述读取张量。

    Args:
        dump_dir: 含 ``.bin`` 与 ``manifest.txt`` 的目录。
        spec: 该张量对应的 ``TensorSpec``。

    Returns:
        np.ndarray: 按 ``spec.dims`` reshape 后的数组。
    """

    values = np.fromfile(dump_dir / spec.file_name, dtype=dtype_to_numpy(spec.dtype))
    if spec.dims:
        values = values.reshape(spec.dims)
    return values


def write_manifest(
    dump_dir: Path,
    metadata: dict[str, str],
    tensors: dict[str, np.ndarray],
) -> None:
    """写出与 C++ sample 相同格式的 manifest 及 ``.bin`` 文件（测试/调试辅助）。

    Args:
        dump_dir: 输出目录，不存在则创建。
        metadata: 元数据键值对，每行 ``key=value``。
        tensors: 张量名 -> NumPy 数组；文件名由张量名消毒后生成。
    """

    dump_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key, value in metadata.items():
        lines.append(f"{key}={value}")
    for name, array in tensors.items():
        file_name = f"{name.replace('/', '_').replace('.', '_')}.bin"
        dims = ",".join(str(dim) for dim in array.shape)
        lines.append(f"tensor|{name}|{array.dtype.name}|{dims}|{file_name}")
        np.ascontiguousarray(array).tofile(dump_dir / file_name)
    (dump_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_tensors_close(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    rtol: float,
    atol: float,
    name: str,
) -> None:
    """断言两路张量在 ``numpy.allclose`` 意义下一致（统一转为 float32 比较）。

    Args:
        reference: 参考张量（通常为 C++ dump 或 ``infer()`` 输出）。
        actual: 待验证张量。
        rtol: 相对容差，见 ``numpy.allclose``。
        atol: 绝对容差，见 ``numpy.allclose``。
        name: 张量名，仅用于错误信息。

    Raises:
        AssertionError: 形状不一致或超出容差时，附带 ``max_abs`` / ``mean_abs``。
    """

    ref = np.asarray(reference).astype(np.float32, copy=False)
    act = np.asarray(actual).astype(np.float32, copy=False)
    if ref.shape != act.shape:
        raise AssertionError(f"Shape mismatch for {name}: reference={ref.shape}, actual={act.shape}")
    if not np.allclose(ref, act, rtol=rtol, atol=atol):
        diff = np.abs(act - ref)
        raise AssertionError(
            f"Tensor '{name}' mismatch: max_abs={diff.max():.6g}, mean_abs={diff.mean():.6g}, "
            f"ref_abs_max={np.abs(ref).max():.6g}"
        )
