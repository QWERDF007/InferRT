"""ONNX Runtime/OpenVINO 后端的 Python 绑定构建、推理与特征输出测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

from helpers.manifest import assert_tensors_close


FEATURE_NAMES = ["tiny_cls", "tiny_patch"]
FEATURE_INPUT_SHAPE = (1, 3, 8, 10)
FEATURE_RTOL = 1.0e-4
FEATURE_ATOL = 1.0e-4


@pytest.fixture(params=["ONNXRUNTIME", "OPENVINO"])
def graph_backend_attr(request: pytest.FixtureRequest, compare_runtimes: list[str]) -> str:
    """按 ``--inferrt-compare-runtime`` 选择当前要执行的图后端。

    Args:
        request: pytest 参数化请求，提供候选后端枚举名。
        compare_runtimes: 用户通过命令行选择的后端列表。

    Returns:
        当前测试用例启用的后端枚举名。
    """

    backend_attr = str(request.param)
    if backend_attr not in compare_runtimes:
        pytest.skip(f"{backend_attr} is disabled; pass --inferrt-compare-runtime=onnx,openvino")
    return backend_attr


@pytest.fixture
def tiny_onnx_path(tmp_path: Path) -> Path:
    """导出一个最小卷积 ONNX 模型供后端 smoke 测试使用。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        已生成的 ONNX 文件路径。
    """

    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")

    class TinyONNXModel(torch.nn.Module):
        """用于验证图后端元数据和基础推理的固定权重小模型。"""

        def __init__(self) -> None:
            """构造卷积和池化层，并写入确定性权重。"""

            super().__init__()
            self.conv = torch.nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=True)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

            with torch.no_grad():
                weight = torch.arange(self.conv.weight.numel(), dtype=torch.float32).reshape_as(self.conv.weight)
                self.conv.weight.copy_(((weight % 13) - 6.0) / 32.0)
                self.conv.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """执行卷积、ReLU、池化并返回二维输出。

            Args:
                x: 形状为 ``NCHW`` 的输入张量。

            Returns:
                形状为 ``(N, 2)`` 的输出张量。
            """

            return self.pool(torch.relu(self.conv(x))).flatten(1)

    path = tmp_path / "tiny_backend.onnx"
    model = TinyONNXModel().eval()
    dummy = torch.full((1, 3, 8, 10), 0.01, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    return path


def _run_backend(irt_module: object, onnx_path: Path, backend: object) -> tuple[object, dict[str, np.ndarray]]:
    """使用指定图后端构建 ONNX 模型并执行一次推理。

    Args:
        irt_module: 已导入的 ``inferrt_model_py`` 模块。
        onnx_path: 待加载的 ONNX 文件路径。
        backend: InferRT 后端枚举值。

    Returns:
        已构建模型对象和输出张量字典。
    """

    model = irt_module.create_model("onnx", backend=backend, device=irt_module.ModelDevice.CPU)
    model.build_or_load(str(onnx_path))

    input_names = model.input_tensor_names()
    output_names = model.output_tensor_names()
    assert input_names == ["input"]
    assert output_names == ["output"]

    input_shape = tuple(model.tensor_shape(input_names[0]))
    assert input_shape == (1, 3, 8, 10)
    input_tensor = np.full(input_shape, 0.01, dtype=np.float32)

    result = model.infer({input_names[0]: input_tensor}, None)
    if isinstance(result, dict):
        outputs = {name: np.asarray(value) for name, value in result.items()}
    else:
        outputs = {output_names[0]: np.asarray(result)}

    output = outputs[output_names[0]]
    assert output.shape == (1, 2)
    assert np.isfinite(output).all()
    assert np.any(np.abs(output) > 1.0e-7)
    return model, outputs


def test_onnx_graph_backends_build_and_infer_cpu(
    graph_backend_attr: str,
    irt_module: object,
    tiny_onnx_path: Path,
) -> None:
    """验证图后端能在 CPU 上构建 ONNX 模型并完成基础推理。"""

    backend = getattr(irt_module.ModelBackend, graph_backend_attr)
    model, _outputs = _run_backend(irt_module, tiny_onnx_path, backend)

    assert model.backend() == backend
    assert model.device() == irt_module.ModelDevice.CPU


def test_onnx_graph_backends_forward_features_uses_graph_outputs(
    graph_backend_attr: str,
    irt_module: object,
    tiny_onnx_path: Path,
) -> None:
    """验证图后端的 ``forward_features`` 使用配置的图输出。"""

    backend = getattr(irt_module.ModelBackend, graph_backend_attr)
    config = irt_module.ModelConfig()
    config.backend = backend
    config.device = irt_module.ModelDevice.CPU
    config.feature_only = True
    config.feature_tensor_names = ["output"]
    config.output_tensor_names = ["output"]

    model = irt_module.create_model("onnx", config=config)
    model.build_or_load(str(tiny_onnx_path))

    input_shape = tuple(model.tensor_shape("input"))
    input_tensor = np.full(input_shape, 0.01, dtype=np.float32)
    features = model.forward_features({"input": input_tensor})

    assert tuple(features.shape) == (1, 2)
    assert np.isfinite(features).all()
    assert np.any(np.abs(features) > 1.0e-7)


def test_onnx_graph_backends_reject_missing_output_name(
    graph_backend_attr: str,
    irt_module: object,
    tiny_onnx_path: Path,
) -> None:
    """验证指定不存在的输出张量名时，构建阶段会报错。"""

    config = irt_module.ModelConfig()
    config.backend = getattr(irt_module.ModelBackend, graph_backend_attr)
    config.device = irt_module.ModelDevice.CPU
    config.output_tensor_names = ["missing_output"]

    model = irt_module.create_model("onnx", config=config)
    with pytest.raises(irt_module.InferRTError, match="missing_output"):
        model.build_or_load(str(tiny_onnx_path))


def _make_feature_input() -> np.ndarray:
    """生成固定形状和数值范围的特征导出测试输入。

    Returns:
        形状为 ``FEATURE_INPUT_SHAPE`` 的 ``float32`` 输入数组。
    """

    values = np.linspace(-1.0, 1.0, num=np.prod(FEATURE_INPUT_SHAPE), dtype=np.float32)
    return values.reshape(FEATURE_INPUT_SHAPE)


def _make_tiny_feature_model():
    """构造带 ``forward_features`` 的确定性小模型。

    Returns:
        处于 eval 模式的 PyTorch 模型。
    """

    torch = pytest.importorskip("torch")

    class TinyFeatureModel(torch.nn.Module):
        """同时输出向量特征和空间特征的小型卷积模型。"""

        def __init__(self) -> None:
            """构造卷积/投影层，并写入固定权重。"""

            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
            self.proj = torch.nn.Linear(4, 2, bias=True)

            with torch.no_grad():
                conv_weight = torch.arange(self.conv.weight.numel(), dtype=torch.float32).reshape_as(self.conv.weight)
                self.conv.weight.copy_(((conv_weight % 19) - 9.0) / 64.0)
                self.conv.bias.copy_(torch.tensor([-0.04, -0.01, 0.02, 0.05], dtype=torch.float32))
                proj_weight = torch.tensor(
                    [[0.25, -0.125, 0.0625, 0.5], [-0.375, 0.1875, 0.3125, -0.25]],
                    dtype=torch.float32,
                )
                self.proj.weight.copy_(proj_weight)
                self.proj.bias.copy_(torch.tensor([0.01, -0.02], dtype=torch.float32))

        def forward_features(self, x):
            """计算测试所需的两个特征输出。

            Args:
                x: 形状为 ``NCHW`` 的输入张量。

            Returns:
                以特征名为 key 的张量字典。
            """

            y = torch.relu(self.conv(x))
            pooled = y.mean(dim=(2, 3))
            return {
                "tiny_cls": self.proj(pooled),
                "tiny_patch": y[:, :2, ::2, ::2],
            }

    model = TinyFeatureModel()
    model.eval()
    return model


def _reference_features(model, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
    """用 PyTorch 计算特征输出作为数值参考。

    Args:
        model: 提供 ``forward_features`` 的 PyTorch 模型。
        input_tensor: NumPy 格式输入张量。

    Returns:
        特征名到 NumPy 输出数组的映射。
    """

    torch = pytest.importorskip("torch")
    from export_feature_onnx import FeatureOutputWrapper

    wrapper = FeatureOutputWrapper(model, FEATURE_NAMES)
    wrapper.eval()

    batch = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))
    with torch.inference_mode():
        outputs = wrapper(batch)
    return {name: output.detach().cpu().numpy() for name, output in zip(FEATURE_NAMES, outputs)}


def _export_feature_onnx(model, input_tensor: np.ndarray, output_path: Path) -> None:
    """将测试特征模型导出为多输出 ONNX 图。

    Args:
        model: 待导出的 PyTorch 模型。
        input_tensor: 导出时使用的示例输入。
        output_path: ONNX 输出路径。
    """

    pytest.importorskip("onnx")
    torch = pytest.importorskip("torch")
    from export_feature_onnx import FeatureOutputWrapper, export_features_with_onnx

    args = SimpleNamespace(
        input_name="input",
        features=FEATURE_NAMES,
        opset=17,
        dynamic_batch=False,
    )
    wrapper = FeatureOutputWrapper(model, FEATURE_NAMES)
    wrapper.eval()
    dummy_input = torch.from_numpy(np.ascontiguousarray(input_tensor, dtype=np.float32))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_features_with_onnx(wrapper, dummy_input, output_path, args, use_dynamo=False)


def _run_feature_backend_outputs_or_skip(
    irt_module: object,
    *,
    model_path: Path,
    backend_attr: str,
    input_tensor: np.ndarray,
) -> dict[str, np.ndarray]:
    """通过图后端加载特征 ONNX/IR 并返回 ``forward_features`` 输出。

    Args:
        irt_module: 已导入的 ``inferrt_model_py`` 模块。
        model_path: ONNX 或 OpenVINO IR 模型路径。
        backend_attr: 后端枚举属性名。
        input_tensor: NumPy 格式输入张量。

    Returns:
        特征名到 NumPy 输出数组的映射。
    """

    config = irt_module.ModelConfig()
    config.backend = getattr(irt_module.ModelBackend, backend_attr)
    config.device = irt_module.ModelDevice.CPU
    config.feature_only = True
    config.feature_tensor_names = FEATURE_NAMES
    config.output_tensor_names = FEATURE_NAMES

    model = irt_module.create_model("onnx", config=config)
    try:
        model.build_or_load(str(model_path))
    except irt_module.InferRTError as exc:
        message = str(exc)
        unavailable_markers = (
            "backend is not enabled",
            "Failed to load ONNX Runtime DLL",
            "OpenVINO backend is not enabled",
            "OpenVINO error while loading",
        )
        if any(marker in message for marker in unavailable_markers):
            pytest.skip(f"{backend_attr} backend unavailable: {message}")
        raise

    assert model.input_tensor_names() == ["input"]
    assert model.output_tensor_names() == FEATURE_NAMES

    outputs = model.forward_features({"input": np.ascontiguousarray(input_tensor, dtype=np.float32)})
    if isinstance(outputs, np.ndarray):
        return {FEATURE_NAMES[0]: outputs}
    return {str(name): np.asarray(value) for name, value in dict(outputs).items()}


def _assert_feature_dicts_close(reference: dict[str, np.ndarray], actual: dict[str, np.ndarray], label: str) -> None:
    """按统一容差比较两组特征输出。

    Args:
        reference: 参考特征输出。
        actual: 待验证特征输出。
        label: 断言失败时使用的标签前缀。
    """

    assert sorted(actual) == sorted(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert_tensors_close(
            reference[name],
            actual[name],
            rtol=FEATURE_RTOL,
            atol=FEATURE_ATOL,
            name=f"{label}.{name}",
        )


@pytest.mark.integration
@pytest.mark.slow
def test_feature_graph_backend_matches_pytorch(
    graph_backend_attr: str,
    irt_module: object,
    tmp_path: Path,
) -> None:
    """验证单个图后端的特征输出与 PyTorch 参考一致。"""

    model = _make_tiny_feature_model()
    input_tensor = _make_feature_input()
    reference = _reference_features(model, input_tensor)

    onnx_path = tmp_path / "tiny_feature.onnx"
    _export_feature_onnx(model, input_tensor, onnx_path)

    outputs = _run_feature_backend_outputs_or_skip(
        irt_module,
        model_path=onnx_path,
        backend_attr=graph_backend_attr,
        input_tensor=input_tensor,
    )
    label = "onnx" if graph_backend_attr == "ONNXRUNTIME" else "openvino"
    _assert_feature_dicts_close(reference, outputs, label)


@pytest.mark.integration
@pytest.mark.slow
def test_feature_graph_backends_match_each_other(
    compare_runtimes: list[str],
    irt_module: object,
    tmp_path: Path,
) -> None:
    """同时启用 ONNX Runtime 和 OpenVINO 时，比较二者特征输出一致性。"""

    if not {"ONNXRUNTIME", "OPENVINO"}.issubset(compare_runtimes):
        pytest.skip("ONNX/OpenVINO cross-check requires --inferrt-compare-runtime=onnx,openvino")

    model = _make_tiny_feature_model()
    input_tensor = _make_feature_input()
    onnx_path = tmp_path / "tiny_feature.onnx"
    _export_feature_onnx(model, input_tensor, onnx_path)

    onnx_outputs = _run_feature_backend_outputs_or_skip(
        irt_module,
        model_path=onnx_path,
        backend_attr="ONNXRUNTIME",
        input_tensor=input_tensor,
    )
    openvino_outputs = _run_feature_backend_outputs_or_skip(
        irt_module,
        model_path=onnx_path,
        backend_attr="OPENVINO",
        input_tensor=input_tensor,
    )
    _assert_feature_dicts_close(onnx_outputs, openvino_outputs, "openvino_vs_onnx")


@pytest.mark.integration
@pytest.mark.slow
def test_openvino_ir_feature_model_matches_pytorch(
    compare_runtimes: list[str],
    irt_module: object,
    tmp_path: Path,
) -> None:
    """验证导出的 OpenVINO IR 特征模型输出与 PyTorch 参考一致。"""

    if "OPENVINO" not in compare_runtimes:
        pytest.skip("OpenVINO IR parity is disabled; pass --inferrt-compare-runtime=openvino")

    model = _make_tiny_feature_model()
    input_tensor = _make_feature_input()
    reference = _reference_features(model, input_tensor)

    onnx_path = tmp_path / "tiny_feature.onnx"
    xml_path = tmp_path / "openvino_ir" / "tiny_feature.xml"
    _export_feature_onnx(model, input_tensor, onnx_path)

    from export_feature_onnx import convert_to_openvino_ir

    try:
        convert_to_openvino_ir(onnx_path, xml_path, openvino_root="D:/Software/openvino_toolkit")
    except RuntimeError as exc:
        if "OpenVINO Python API is required" in str(exc):
            pytest.skip(str(exc))
        raise

    outputs = _run_feature_backend_outputs_or_skip(
        irt_module,
        model_path=xml_path,
        backend_attr="OPENVINO",
        input_tensor=input_tensor,
    )
    _assert_feature_dicts_close(reference, outputs, "openvino_ir")
