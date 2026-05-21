"""Manifest parsing and tensor dump helpers for InferRT Python API tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    dims: tuple[int, ...]
    file_name: str


def parse_dims(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in value.split(","))


def dtype_to_numpy(dtype: str) -> np.dtype:
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
    values = np.fromfile(dump_dir / spec.file_name, dtype=dtype_to_numpy(spec.dtype))
    if spec.dims:
        values = values.reshape(spec.dims)
    return values


def write_manifest(
    dump_dir: Path,
    metadata: dict[str, str],
    tensors: dict[str, np.ndarray],
) -> None:
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
