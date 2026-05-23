"""@brief YOLO 检测权重导出工具的单元测试。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
YOLO_MODEL_ZOO_PATH = ROOT / "samples" / "model" / "detection" / "yolo_model_zoo.py"

spec = importlib.util.spec_from_file_location("inferrt_detection_yolo_model_zoo", YOLO_MODEL_ZOO_PATH)
assert spec is not None and spec.loader is not None
yolo_model_zoo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yolo_model_zoo)


def test_list_supported_models_includes_yolov5_and_yolov8() -> None:
    """@brief 检测导出器应列出 InferRT 原生注册的 YOLOv5/YOLOv8 key。"""

    names = yolo_model_zoo.list_supported_models()

    assert "yolov5n" in names
    assert "yolov5x" in names
    assert "yolov8n" in names
    assert "yolov8x" in names


def test_resolve_model_weights_uses_alias_defaults() -> None:
    """@brief 未显式传入权重时，兼容别名应解析到常用的 nano/small 权重名。"""

    assert yolo_model_zoo.resolve_model_weights("yolov5", None) == "yolov5su.pt"
    assert yolo_model_zoo.resolve_model_weights("yolov5n", None) == "yolov5nu.pt"
    assert yolo_model_zoo.resolve_model_weights("yolov5x", None) == "yolov5xu.pt"
    assert yolo_model_zoo.resolve_model_weights("yolov8", None) == "yolov8n.pt"
    assert yolo_model_zoo.resolve_model_weights("yolov8s", None) == "yolov8s.pt"


def test_resolve_model_weights_prefers_explicit_weights() -> None:
    """@brief 用户显式传入的本地权重路径应优先于模型 key 默认值。"""

    assert yolo_model_zoo.resolve_model_weights("yolov8n", "D:/models/custom.pt") == "D:/models/custom.pt"


def test_load_ultralytics_model_uses_yolo_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """@brief 模型加载应通过 ``ultralytics.YOLO`` 并返回 FP32 eval 模型。"""

    captured: dict[str, object] = {}

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.float_called = False
            self.eval_called = False

        def float(self) -> "FakeModel":
            self.float_called = True
            return self

        def eval(self) -> "FakeModel":
            self.eval_called = True
            return self

    fake_model = FakeModel()

    class FakeYOLO:
        def __init__(self, source: str) -> None:
            captured["source"] = source
            self.model = fake_model

    repo = ROOT / "build" / "python_test_artifacts" / "mock_ultralytics_repo"
    repo.mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))

    model = yolo_model_zoo.load_ultralytics_model("yolov8n", weights="local.pt", repo=repo)

    assert captured["source"] == "local.pt"
    assert model is fake_model
    assert fake_model.float_called is True
    assert fake_model.eval_called is True
    assert str(repo.resolve()) in sys.path


def test_write_wts_exports_big_endian_float_hex() -> None:
    """@brief ``.wts`` 写出格式应与项目中其它权重导出脚本保持一致。"""

    class FakeModel:
        def state_dict(self) -> dict[str, torch.Tensor]:
            return {"model.0.conv.weight": torch.tensor([1.0, -2.0], dtype=torch.float32)}

    artifact_dir = ROOT / "build" / "python_test_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output = artifact_dir / f"yolov8n_mock_{os.getpid()}.wts"
    yolo_model_zoo.write_wts(FakeModel(), output, verbose=False)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "1"
    assert lines[1] == "model.0.conv.weight 2 3f800000 c0000000"
    output.unlink(missing_ok=True)
