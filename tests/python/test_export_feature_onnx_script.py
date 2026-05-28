"""特征 ONNX/OpenVINO 导出脚本的 wrapper、参数和输出契约测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import export_feature_onnx  # noqa: E402


class FakeFeatureModel(torch.nn.Module):
    """用于特征导出脚本测试的最小 PyTorch 模型。"""

    def __init__(self) -> None:
        """设置默认输入尺寸配置。"""

        super().__init__()
        self.default_cfg = {"input_size": (3, 16, 20)}

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """返回向量特征和 patch token 特征。

        Args:
            x: 输入图像张量。

        Returns:
            以特征名为 key 的张量字典。
        """

        return {
            "cls": x.mean(dim=(2, 3)),
            "patch": x.flatten(2).transpose(1, 2),
        }


def test_feature_output_wrapper_returns_requested_features_in_order() -> None:
    """验证 FeatureOutputWrapper 按用户请求顺序返回特征。"""

    model = FakeFeatureModel()
    wrapper = export_feature_onnx.FeatureOutputWrapper(model, ["patch", "cls"])
    x = torch.ones((2, 3, 4, 5), dtype=torch.float32)

    patch, cls = wrapper(x)

    assert patch.shape == (2, 20, 3)
    assert cls.shape == (2, 3)
    assert torch.equal(cls, torch.ones((2, 3), dtype=torch.float32))


def test_feature_output_wrapper_rejects_missing_features() -> None:
    """验证请求不存在特征时 wrapper 会抛出明确错误。"""

    wrapper = export_feature_onnx.FeatureOutputWrapper(FakeFeatureModel(), ["missing"])

    with pytest.raises(KeyError, match="missing"):
        wrapper(torch.ones((1, 3, 4, 5), dtype=torch.float32))


def test_export_feature_parser_accepts_feature_csv_and_image_size() -> None:
    """验证特征 CSV 和输入尺寸命令行参数解析。"""

    args = export_feature_onnx.build_arg_parser().parse_args(
        ["--features", "cls, patch", "--input-size", "1x3x16x20"]
    )

    assert args.features == ["cls", "patch"]
    assert args.input_size == (16, 20)


def test_export_feature_onnx_uses_features_as_graph_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证特征导出脚本会把特征名作为 ONNX 输出名。"""

    captured: dict[str, object] = {}

    def fake_create_model(model_name: str, backend: str, **kwargs: object) -> torch.nn.Module:
        """捕获模型创建参数并返回伪造特征模型。

        Args:
            model_name: 导出脚本传入的模型名。
            backend: 导出脚本传入的模型来源后端。
            **kwargs: 传给共享 model_zoo 的额外参数。

        Returns:
            伪造的特征模型。
        """

        captured["model_name"] = model_name
        captured["backend"] = backend
        captured["create_kwargs"] = kwargs
        return FakeFeatureModel()

    def fake_export(model: torch.nn.Module, dummy_input: torch.Tensor, output_path: str, **kwargs: object) -> None:
        """捕获 ONNX 导出参数和 wrapper 实际输出形状。

        Args:
            model: 已包装的待导出模型。
            dummy_input: 导出示例输入。
            output_path: ONNX 输出路径。
            **kwargs: ONNX 导出参数。
        """

        captured["dummy_shape"] = tuple(dummy_input.shape)
        captured["output_path"] = output_path
        captured["export_kwargs"] = kwargs
        captured["wrapped_output_shapes"] = [tuple(output.shape) for output in model(dummy_input)]
        Path(output_path).write_text("fake feature onnx", encoding="utf-8")

    monkeypatch.setattr(export_feature_onnx, "create_model", fake_create_model)
    monkeypatch.setattr(export_feature_onnx.torch.onnx, "export", fake_export)

    output_path = tmp_path / "dinov2_features.onnx"
    args = export_feature_onnx.build_arg_parser().parse_args(
        [
            "-m",
            "dinov2_vits14",
            "-b",
            "torchhub",
            "-f",
            "cls,patch",
            "--hub-repo",
            "local_dinov2",
            "--hub-source",
            "local",
            "--hub-weights",
            "dinov2_vits14.pth",
            "-o",
            str(output_path),
            "--dynamic-batch",
            "--exporter",
            "legacy",
        ]
    )

    export_feature_onnx.main(args)

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
    assert captured["dummy_shape"] == (1, 3, 16, 20)
    assert captured["output_path"] == str(output_path)
    assert captured["wrapped_output_shapes"] == [(1, 3), (1, 320, 3)]

    export_kwargs = captured["export_kwargs"]
    assert isinstance(export_kwargs, dict)
    assert export_kwargs["input_names"] == ["input"]
    assert export_kwargs["output_names"] == ["cls", "patch"]
    assert export_kwargs["dynamic_axes"] == {
        "input": {0: "batch"},
        "cls": {0: "batch"},
        "patch": {0: "batch"},
    }
    assert output_path.read_text(encoding="utf-8") == "fake feature onnx"


def test_export_feature_main_can_emit_openvino_ir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证开启 OpenVINO 输出时会调用 ONNX 到 IR 转换流程。"""

    captured: dict[str, object] = {}

    def fake_create_model(model_name: str, backend: str, **kwargs: object) -> torch.nn.Module:
        """返回伪造特征模型供导出主流程使用。

        Args:
            model_name: 导出脚本传入的模型名。
            backend: 导出脚本传入的模型来源后端。
            **kwargs: 传给共享 model_zoo 的额外参数。

        Returns:
            伪造的特征模型。
        """

        return FakeFeatureModel()

    def fake_export(model: torch.nn.Module, dummy_input: torch.Tensor, output_path: str, **kwargs: object) -> None:
        """写出占位 ONNX 文件。

        Args:
            model: 待导出的模型。
            dummy_input: 导出示例输入。
            output_path: ONNX 输出路径。
            **kwargs: ONNX 导出参数。
        """

        Path(output_path).write_text("fake feature onnx", encoding="utf-8")

    def fake_convert(onnx_path: Path, xml_path: Path, openvino_root: str | None) -> None:
        """模拟 OpenVINO IR 转换并记录路径参数。

        Args:
            onnx_path: 输入 ONNX 路径。
            xml_path: 输出 XML 路径。
            openvino_root: OpenVINO 根目录配置。
        """

        captured["onnx_path"] = onnx_path
        captured["xml_path"] = xml_path
        captured["openvino_root"] = openvino_root
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        xml_path.write_text("fake openvino xml", encoding="utf-8")

    monkeypatch.setattr(export_feature_onnx, "create_model", fake_create_model)
    monkeypatch.setattr(export_feature_onnx.torch.onnx, "export", fake_export)
    monkeypatch.setattr(export_feature_onnx, "convert_to_openvino_ir", fake_convert)

    onnx_path = tmp_path / "features.onnx"
    openvino_dir = tmp_path / "openvino_ir"
    args = export_feature_onnx.build_arg_parser().parse_args(
        [
            "-f",
            "cls",
            "-o",
            str(onnx_path),
            "--openvino-output",
            str(openvino_dir),
            "--openvino-root",
            "D:/Software/openvino_toolkit",
            "--exporter",
            "legacy",
        ]
    )

    export_feature_onnx.main(args)

    assert captured["onnx_path"] == onnx_path
    assert captured["xml_path"] == openvino_dir / "features.xml"
    assert captured["openvino_root"] == "D:/Software/openvino_toolkit"
