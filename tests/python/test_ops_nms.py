"""NMS Python binding parity tests against torchvision."""

from __future__ import annotations

import numpy as np
import pytest


def _torchvision_nms(boxes_array, scores_array, iou_threshold):
    torch = pytest.importorskip("torch")
    nms = pytest.importorskip("torchvision.ops").nms

    boxes_tensor = torch.from_numpy(np.ascontiguousarray(boxes_array, dtype=np.float32))
    scores_tensor = torch.from_numpy(np.ascontiguousarray(scores_array, dtype=np.float32))
    with torch.inference_mode():
        keep = nms(boxes_tensor, scores_tensor, iou_threshold)
    return keep.cpu().numpy()


@pytest.mark.parametrize(
    ("boxes", "scores", "iou_threshold"),
    [
        (
            np.array(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [1.0, 1.0, 11.0, 11.0],
                    [20.0, 20.0, 30.0, 30.0],
                ],
                dtype=np.float32,
            ),
            np.array([0.9, 0.8, 0.7], dtype=np.float32),
            0.5,
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 2.0, 2.0],
                    [2.0, 0.0, 4.0, 2.0],
                    [0.5, 0.5, 2.5, 2.5],
                ],
                dtype=np.float32,
            ),
            np.array([0.95, 0.8, 0.7], dtype=np.float32),
            0.0,
        ),
        (
            np.array(
                [
                    [-1.0, -1.0, 1.0, 1.0],
                    [-0.5, -0.5, 1.5, 1.5],
                    [10.0, 10.0, 12.0, 12.0],
                    [20.0, 20.0, 21.0, 21.0],
                ],
                dtype=np.float32,
            ),
            np.array([0.3, 0.6, 0.9, 0.1], dtype=np.float32),
            0.3,
        ),
        (
            np.array(
                [
                    [0.0, 0.0, 5.0, 5.0],
                    [1.0, 1.0, 6.0, 6.0],
                    [2.0, 2.0, 7.0, 7.0],
                ],
                dtype=np.float32,
            ),
            np.array([0.4, 0.9, 0.7], dtype=np.float32),
            1.0,
        ),
    ],
)
def test_nms_matches_torchvision(ops_module, boxes, scores, iou_threshold) -> None:
    expected = _torchvision_nms(boxes, scores, iou_threshold)
    actual = ops_module.nms(boxes, scores, iou_threshold)

    np.testing.assert_array_equal(actual, expected)


def test_nms_empty_matches_torchvision(ops_module) -> None:
    boxes = np.empty((0, 4), dtype=np.float32)
    scores = np.empty((0,), dtype=np.float32)

    expected = _torchvision_nms(boxes, scores, 0.5)
    actual = ops_module.nms(boxes, scores, 0.5)

    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, expected)


def test_nms_rejects_invalid_shapes(ops_module) -> None:
    scores = np.array([0.5], dtype=np.float32)

    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.nms(np.zeros((1, 5), dtype=np.float32), scores, 0.5)

    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.nms(np.zeros((1, 4), dtype=np.float32), np.zeros((2,), dtype=np.float32), 0.5)

    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.nms(np.zeros((1, 4), dtype=np.float32), np.zeros((1, 1), dtype=np.float32), 0.5)
