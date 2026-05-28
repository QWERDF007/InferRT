"""ONNX 导出脚本的参数传递和输入尺寸解析单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import export_onnx  # noqa: E402


class FakeModel(torch.nn.Module):
    """用于导出脚本测试的最小 PyTorch 模型。"""

    def __init__(self) -> None:
        """设置默认输入尺寸配置。"""

        super().__init__()
        self.default_cfg = {"input_size": (3, 32, 48)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回固定类别数的零张量。

        Args:
            x: 输入图像张量。

        Returns:
            形状为 ``(batch, 2)`` 的输出张量。
        """

        return torch.zeros((x.shape[0], 2), dtype=x.dtype)


def test_export_onnx_uses_shared_model_zoo_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证导出脚本会把 model_zoo 相关参数传给 ``create_model``。"""

    captured: dict[str, object] = {}

    def fake_create_model(model_name: str, backend: str, **kwargs: object) -> torch.nn.Module:
        """捕获模型创建参数并返回伪造模型。

        Args:
            model_name: 导出脚本传入的模型名。
            backend: 导出脚本传入的模型来源后端。
            **kwargs: 传给共享 model_zoo 的额外参数。

        Returns:
            伪造的 PyTorch 模型。
        """

        captured["model_name"] = model_name
        captured["backend"] = backend
        captured["create_kwargs"] = kwargs
        return FakeModel()

    def fake_export(model: torch.nn.Module, dummy_input: torch.Tensor, output_path: str, **kwargs: object) -> None:
        """捕获 ``torch.onnx.export`` 参数并写出占位文件。

        Args:
            model: 待导出的模型。
            dummy_input: 导出示例输入。
            output_path: ONNX 输出路径。
            **kwargs: ONNX 导出参数。
        """

        captured["dummy_shape"] = tuple(dummy_input.shape)
        captured["output_path"] = output_path
        captured["export_kwargs"] = kwargs
        Path(output_path).write_text("fake onnx", encoding="utf-8")

    monkeypatch.setattr(export_onnx, "create_model", fake_create_model)
    monkeypatch.setattr(export_onnx.torch.onnx, "export", fake_export)

    output_path = tmp_path / "dinov2_vits14.onnx"
    args = export_onnx.build_arg_parser().parse_args(
        [
            "-m",
            "dinov2_vits14",
            "-b",
            "torchhub",
            "--hub-repo",
            "local_dinov2",
            "--hub-source",
            "local",
            "--hub-weights",
            "dinov2_vits14.pth",
            "-o",
            str(output_path),
            "--exporter",
            "legacy",
        ]
    )

    export_onnx.main(args)

    assert captured["model_name"] == "dinov2_vits14"
    assert captured["backend"] == "torchhub"
    assert captured["create_kwargs"] == {
        "hub_repo": "local_dinov2",
        "hub_source": "local",
        "pretrained": True,
        "hub_weights": "dinov2_vits14.pth",
        "hf_model_id": None,
        "local_files_only": False,
    }
    assert captured["dummy_shape"] == (1, 3, 32, 48)
    assert captured["output_path"] == str(output_path)
    assert "dynamo" not in captured["export_kwargs"]
    assert captured["export_kwargs"]["opset_version"] == 17
    assert output_path.read_text(encoding="utf-8") == "fake onnx"


def test_export_onnx_parser_accepts_model_zoo_image_size_formats() -> None:
    """验证导出脚本参数解析复用 model_zoo 的输入尺寸格式。"""

    args = export_onnx.build_arg_parser().parse_args(["--input-size", "1x3x16x32"])

    assert args.input_size == (16, 32)
