"""``infer()``（NumPy 主机缓冲）与 ``infer_v2()``（CUDA DLPack）输出一致性测试。"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from helpers.manifest import assert_tensors_close
from helpers.runtime import require_weights
from util import allocate_output_tensors, preprocess_image

INFER_V2_CASES = [
    pytest.param("resnet18", "samples/model/classification/resnet18.wts", id="resnet18"),
    pytest.param("alexnet", "samples/model/classification/alexnet.wts", id="alexnet"),
]


def _require_torch_cuda() -> tuple[object, object]:
    """确保已安装 PyTorch 且 CUDA 可用，否则跳过测试。

    Returns:
        tuple[module, module]: ``(torch, torch)``，便于测试内解构为 ``torch, _``。
    """

    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for infer_v2 parity tests")
    return torch, torch


@pytest.mark.parametrize("model_name,weights_rel", INFER_V2_CASES)
def test_infer_v2_matches_numpy_infer(
    model_name: str,
    weights_rel: str,
    repo_root,
    irt_module: object,
    default_image,
    tolerances: tuple[float, float],
) -> None:
    """同一模型、同一输入下，``infer_v2`` 的 GPU 输出应与 ``infer`` 的 NumPy 输出一致。

    Args:
        model_name: 内置模型名，须为单输入模型（否则 skip）。
        weights_rel: 相对 ``repo_root`` 的权重路径；缺失时 skip。
        repo_root: 仓库根目录，用于 ``require_weights``。
        irt_module: ``inferrt_model_py`` 模块。
        default_image: 默认测试图路径；在本测试中现场 ``preprocess_image``，
            与 ``input_tensor`` fixture 语义相同但未复用 session 缓存。
        tolerances: ``(rtol, atol)``；``infer_v2`` 与 ``infer`` 同引擎，宜配合较严容差。

    需要 CUDA 与 PyTorch；无 GPU 时整例 skip。
    """

    rtol, atol = tolerances
    torch, _ = _require_torch_cuda()

    try:
        weights = require_weights(repo_root, weights_rel)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    input_array = preprocess_image(default_image)
    model = irt_module.create_model(model_name)
    model.build_or_load(str(weights))

    input_names = model.input_tensor_names()
    output_names = model.output_tensor_names()
    if len(input_names) != 1:
        pytest.skip(f"infer_v2 test expects one input tensor, got: {input_names}")

    # 参考路径：NumPy 缓冲区 + infer()
    numpy_inputs = {input_names[0]: input_array}
    numpy_outputs = allocate_output_tensors(model, output_names)
    model.infer(numpy_inputs, numpy_outputs)

    # 待测路径：CUDA 张量 + infer_v2()，需连续内存以便 DLPack 导出
    input_tensor = torch.from_numpy(np.ascontiguousarray(input_array)).cuda().contiguous()
    output_tensors = {}
    for output_name in output_names:
        shape = model.tensor_shape(output_name)
        dtype_name = model.tensor_dtype(output_name)
        torch_dtype = getattr(torch, dtype_name)
        output_tensors[output_name] = torch.empty(shape, dtype=torch_dtype, device="cuda")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        model.infer_v2(
            {input_names[0]: input_tensor},
            output_tensors,
            stream_ptr=stream.cuda_stream,
            non_blocking=True,
        )
    stream.synchronize()

    for output_name in output_names:
        assert_tensors_close(
            numpy_outputs[output_name],
            output_tensors[output_name].detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            name=output_name,
        )
