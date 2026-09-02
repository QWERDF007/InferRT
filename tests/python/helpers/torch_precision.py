"""Deterministic precision scope for PyTorch parity references."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def strict_fp32_reference(torch_module: Any) -> Iterator[Callable[[], None]]:
    """Run a reference path in strict FP32 and restore process-global state.

    Some upstream model packages change PyTorch TF32 settings while importing.
    The yielded callable reapplies the reference policy after such imports.
    """

    previous_precision = torch_module.get_float32_matmul_precision()
    previous_matmul_tf32 = bool(torch_module.backends.cuda.matmul.allow_tf32)
    previous_cudnn_tf32 = bool(torch_module.backends.cudnn.allow_tf32)

    def apply() -> None:
        torch_module.set_float32_matmul_precision("highest")
        torch_module.backends.cuda.matmul.allow_tf32 = False
        torch_module.backends.cudnn.allow_tf32 = False

    apply()
    try:
        yield apply
    finally:
        torch_module.set_float32_matmul_precision(previous_precision)
        torch_module.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch_module.backends.cudnn.allow_tf32 = previous_cudnn_tf32
