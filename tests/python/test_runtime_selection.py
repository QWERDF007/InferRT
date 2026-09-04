"""Tests for CMake-aware Python runtime selection."""

from types import SimpleNamespace

import pytest

from helpers.runtime import available_runtime_device_pairs, runtime_is_enabled


def _module(*, tensorrt: bool = True, onnx: bool = False, openvino: bool = False) -> object:
    return SimpleNamespace(
        tensorrt_enabled=tensorrt,
        onnxruntime_enabled=onnx,
        openvino_enabled=openvino,
    )


def test_available_pairs_ignore_disabled_optional_backends() -> None:
    assert available_runtime_device_pairs(
        _module(),
        ["TENSORRT", "ONNXRUNTIME", "OPENVINO"],
        ["cpu", "gpu"],
    ) == [("TENSORRT", "gpu")]


def test_available_pairs_keep_enabled_graph_backends_on_requested_devices() -> None:
    assert available_runtime_device_pairs(
        _module(onnx=True, openvino=True),
        ["TENSORRT", "ONNXRUNTIME", "OPENVINO"],
        ["cpu", "gpu"],
    ) == [
        ("TENSORRT", "gpu"),
        ("ONNXRUNTIME", "cpu"),
        ("ONNXRUNTIME", "gpu"),
        ("OPENVINO", "cpu"),
        ("OPENVINO", "gpu"),
    ]


def test_runtime_is_enabled_rejects_unknown_runtime() -> None:
    with pytest.raises(ValueError, match="Unsupported runtime attribute"):
        runtime_is_enabled(_module(), "UNKNOWN")
