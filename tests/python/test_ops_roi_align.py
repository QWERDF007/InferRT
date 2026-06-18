"""RoIAlign Python 绑定与 torchvision 的一致性测试。"""

from __future__ import annotations

import numpy as np
import pytest


def _torchvision_roi_align(input_array, rois_array, output_size, spatial_scale, sampling_ratio, aligned):
    """使用 torchvision 计算 RoIAlign 参考输出。

    Args:
        input_array: ``[N, C, H, W]`` 的输入特征图数组。
        rois_array: ``[K, 5]`` 的 RoI 数组，每行为 batch index 与 xyxy 坐标。
        output_size: 输出池化尺寸，支持整数或 ``(height, width)``。
        spatial_scale: RoI 坐标到特征图坐标的缩放比例。
        sampling_ratio: 每个 bin 的采样点数量，负数表示自适应采样。
        aligned: 是否启用 torchvision aligned 坐标规则。

    Returns:
        torchvision 返回的 RoIAlign 输出数组。
    """
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
    """验证 NumPy 路径 RoIAlign 与 torchvision 的数值结果一致。

    Args:
        ops_module: pytest fixture 提供的 InferRT ops Python 绑定模块。
        output_size: 当前参数组合的输出池化尺寸。
        spatial_scale: 当前参数组合的坐标缩放比例。
        sampling_ratio: 当前参数组合的采样点数量。
        aligned: 当前参数组合是否启用 aligned 坐标规则。

    Returns:
        None。
    """
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
    """验证 RoIAlign 类封装与函数式接口输出一致。

    Args:
        ops_module: pytest fixture 提供的 InferRT ops Python 绑定模块。

    Returns:
        None。
    """
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
    """验证 RoIAlign Python 绑定会拒绝非法输入形状。

    Args:
        ops_module: pytest fixture 提供的 InferRT ops Python 绑定模块。

    Returns:
        None。
    """
    input_array = np.zeros((1, 1, 4, 4), dtype=np.float32)
    bad_rois = np.zeros((1, 4), dtype=np.float32)

    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.roi_align(input_array, bad_rois, (2, 2))

    with pytest.raises(ops_module.InferRTOpsError):
        ops_module.roi_align(np.zeros((1, 4, 4), dtype=np.float32), np.zeros((1, 5), dtype=np.float32), (2, 2))
