"""DLPack tensor parity tests for ops v2 bindings."""

from __future__ import annotations

import pytest


def _device_or_skip(device: str):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_nms_v2_matches_torchvision_tensor(ops_module, device: str) -> None:
    torch = _device_or_skip(device)
    tv_nms = pytest.importorskip("torchvision.ops").nms

    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [20.0, 20.0, 30.0, 30.0],
            [21.0, 21.0, 31.0, 31.0],
            [40.0, 40.0, 45.0, 45.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    scores = torch.tensor([0.95, 0.7, 0.9, 0.6, 0.2], dtype=torch.float32, device=device)

    expected = tv_nms(boxes, scores, 0.5)
    actual = ops_module.nms_v2(boxes, scores, 0.5)

    assert actual.device.type == device
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_roi_align_v2_matches_torchvision_tensor(ops_module, device: str) -> None:
    torch = _device_or_skip(device)
    roi_align = pytest.importorskip("torchvision.ops").roi_align

    input_tensor = torch.arange(2 * 3 * 5 * 6, dtype=torch.float32, device=device).reshape(2, 3, 5, 6) / 10.0
    rois = torch.tensor(
        [
            [0.0, 0.0, 0.0, 4.0, 4.0],
            [1.0, 1.0, 0.5, 5.0, 4.5],
            [0.0, -0.25, 1.0, 3.25, 5.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    expected = roi_align(
        input_tensor,
        rois,
        (2, 3),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )
    actual = ops_module.roi_align_v2(
        input_tensor,
        rois,
        (2, 3),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )

    assert actual.device.type == device
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
