"""Parity tests between NumPy infer() and DLPack infer_v2() Python bindings."""

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

    numpy_inputs = {input_names[0]: input_array}
    numpy_outputs = allocate_output_tensors(model, output_names)
    model.infer(numpy_inputs, numpy_outputs)

    input_tensor = torch.from_numpy(numpy.ascontiguousarray(input_array)).cuda().contiguous()
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
