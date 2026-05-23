"""@brief 分类样例 ``model_zoo`` 工具函数的单元测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("torchvision")

from model_zoo import (  # noqa: E402
    normalize_image_size,
    parse_image_size,
    preprocess,
    resolve_input_size,
)


class DummyPatchEmbedModel:
    """@brief 模拟 timm ViT，通过 ``patch_embed.img_size`` 暴露真实输入尺寸。"""

    patch_embed = SimpleNamespace(img_size=(384, 384))
    default_cfg = {"input_size": (3, 224, 224)}


class DummyConfigModel:
    """@brief 模拟仅通过配置暴露输入尺寸的 timm/torchvision 模型。"""

    pretrained_cfg = {"input_size": (3, 320, 256)}


def test_resolve_input_size_prefers_patch_embed_size() -> None:
    """@brief ViT 的实际 patch 输入尺寸应优先于默认配置。"""

    assert resolve_input_size(DummyPatchEmbedModel()) == (384, 384)


def test_resolve_input_size_reads_model_config() -> None:
    """@brief 当模型没有 ``patch_embed`` 时，应从配置中的 ``input_size`` 解析 H/W。"""

    assert resolve_input_size(DummyConfigModel()) == (320, 256)


def test_resolve_input_size_falls_back_to_default() -> None:
    """@brief 模型未提供尺寸元信息时应回退到 ImageNet 默认 224。"""

    assert resolve_input_size(object()) == (224, 224)


def test_normalize_image_size_rejects_invalid_shape() -> None:
    """@brief 非法尺寸应尽早报错，避免 OpenCV resize 阶段出现难读异常。"""

    with pytest.raises(ValueError):
        normalize_image_size((1, 3, 224, 224, 1))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("384", (384, 384)),
        ("384x256", (384, 256)),
        ("3x320x256", (320, 256)),
        ("1,3,640,480", (640, 480)),
    ],
)
def test_parse_image_size_accepts_cli_formats(raw: str, expected: tuple[int, int]) -> None:
    """@brief 命令行尺寸解析应兼容常见 H/W、CHW 和 NCHW 写法。"""

    assert parse_image_size(raw) == expected


def test_parse_image_size_rejects_invalid_text() -> None:
    """@brief 非数字尺寸字符串应在进入预处理前报错。"""

    with pytest.raises(ValueError):
        parse_image_size("abc")


def test_preprocess_uses_custom_hw_size() -> None:
    """@brief 预处理应按调用方传入的 H/W 输出 NCHW 张量。"""

    image = np.zeros((12, 20, 3), dtype=np.uint8)
    tensor = preprocess(image, image_size=(384, 256))

    assert tuple(tensor.shape) == (1, 3, 384, 256)
    assert str(tensor.dtype) == "torch.float32"
