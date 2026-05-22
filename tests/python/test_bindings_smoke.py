"""``inferrt_model_py`` 轻量冒烟测试：无需权重文件，不执行 CUDA 推理。"""

from __future__ import annotations


def test_registered_models(irt_module: object) -> None:
    """内置模型注册表应包含常用分类网络，且名称解析大小写不敏感。

    Args:
        irt_module: ``inferrt_model_py`` 模块（``conftest.irt_module``）。
    """

    names = irt_module.get_registered_model_names()
    assert "resnet18" in names
    assert "alexnet" in names
    assert "googlenet" in names
    assert irt_module.is_supported_model("ResNet18")
    assert irt_module.is_supported_model("GoogLeNet")
    assert not irt_module.is_supported_model("not_a_model")


def test_model_config_defaults(irt_module: object) -> None:
    """``ModelConfig`` 默认值应与 ImageNet 分类示例一致。

    Args:
        irt_module: ``inferrt_model_py`` 模块，用于构造 ``ModelConfig()``。
    """

    config = irt_module.ModelConfig()
    assert config.num_classes == 1000
    assert config.feature_only is False
    assert config.input_shape == [1, 3, 224, 224]


def test_create_model_without_build(irt_module: object) -> None:
    """未 ``build_or_load`` 时也应能查询张量名与规范化模型名。

    Args:
        irt_module: ``inferrt_model_py`` 模块；本测试仅调用 ``create_model("vgg11")``，
            不加载权重、不推理。
    """

    model = irt_module.create_model("vgg11")
    assert model.name() == "VGG11"
    assert model.input_tensor_names()
    assert model.output_tensor_names()


def test_create_googlenet_without_build(irt_module: object) -> None:
    """GoogLeNet 应可通过 Python 绑定按大小写混合名称创建，且不依赖权重文件。

    Args:
        irt_module: ``inferrt_model_py`` 模块。
    """

    model = irt_module.create_model("GoogLeNet")
    assert model.name() == "GoogLeNet"
    assert model.input_tensor_names()
    assert model.output_tensor_names()
