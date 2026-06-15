"""RoIAlign Python binding parity tests against torchvision."""

from __future__ import annotations

import numpy as np
import pytest


def _torchvision_roi_align(input_array, rois_array, output_size, spatial_scale, sampling_ratio, aligned):
    torch = pytest.importorskip("torch")
    roi_align = pytest.importorskip("torchvision.ops").roi_align

    input_tensor = torch.from_numpy(np.ascontiguousarray(input_array, dtype=np.float32))
    rois_tensor = torch.from_numpy(np.ascontiguousarray(rois_array, dtype=np.float32))
    with torch.inference_mode():
        output = roi_align(
            input_tensor,
            rois_tensor,
            output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
        )
    return output.cpu().numpy()


@pytest.mark.parametrize(
    ("output_size", "spatial_scale", "sampling_ratio", "aligned"),
    [
        ((2, 2), 1.0, 1, False),
        ((2, 3), 0.5, 2, False),
        ((3, 2), 1.0, 2, True),
        ((2, 2), 1.0, -1, False),
        (2, 1.0, -1, True),
    ],
)
def test_roi_align_matches_torchvision(ops_module, output_size, spatial_scale, sampling_ratio, aligned) -> None:
    input_array = np.arange(2 * 3 * 5 * 6, dtype=np.float32).reshape(2, 3, 5, 6) / 10.0
    rois_array = np.array(
        [
            [0.0, 0.0, 0.0, 4.0, 4.0],
            [1.0, 1.0, 0.5, 5.0, 4.5],
            [0.0, -0.25, 1.0, 3.25, 5.0],
        ],
        dtype=np.float32,
    )

    expected = _torchvision_roi_align(
        input_array,
        rois_array,
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned,
    )
    actual = ops_module.roi_align(
        input_array,
        rois_array,
        output_size,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_roi_align_class_matches_function(ops_module) -> None:
    input_array = np.linspace(0.0, 1.0, num=1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
    rois_array = np.array([[0.0, 0.5, 0.5, 3.0, 3.0]], dtype=np.float32)

    op = ops_module.RoIAlign((2, 2), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    class_output = op.forward(input_array, rois_array)
    function_output = ops_module.roi_align(
        input_array,
        rois_array,
        (2, 2),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )

    assert op.output_size == [2, 2]
    assert op.sampling_ratio == 2
    assert op.aligned is True
    np.testing.assert_allclose(class_output, function_output, rtol=0, atol=0)


def test_roi_align_rejects_invalid_shapes(ops_module) -> None:
    input_array = np.zeros((1, 1, 4, 4), dtype=np.float32)
    bad_rois = np.zeros((1, 4), dtype=np.float32)

    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.roi_align(input_array, bad_rois, (2, 2))

    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.roi_align(np.zeros((1, 4, 4), dtype=np.float32), np.zeros((1, 5), dtype=np.float32), (2, 2))
