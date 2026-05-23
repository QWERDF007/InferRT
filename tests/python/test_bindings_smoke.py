"""``inferrt_model_py`` 轻量冒烟测试：无需权重文件，不执行 CUDA 推理。"""

from __future__ import annotations

import pytest


def test_registered_models(irt_module: object) -> None:
    """内置模型注册表应包含常用分类网络，且名称解析大小写不敏感。

    Args:
        irt_module: ``inferrt_model_py`` 模块（``conftest.irt_module``）。
    """

    names = irt_module.get_registered_model_names()
    assert "resnet18" in names
    assert "alexnet" in names
    assert "googlenet" in names
    assert "vit_base_patch16_224" in names
    assert "vit_base_patch16_384" in names
    assert irt_module.is_supported_model("ResNet18")
    assert irt_module.is_supported_model("GoogLeNet")
    assert irt_module.is_supported_model("ViT_Base_Patch16_224")
    assert irt_module.is_supported_model("ViT_Base_Patch16_384")
    assert irt_module.is_supported_model("ViT")
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
    assert config.input_shapes == [[1, 3, 224, 224]]
    assert config.input_tensor_names == ["input"]
    assert config.output_tensor_names == ["output"]
    assert config.feature_tensor_names == []


def test_model_config_setters_round_trip(irt_module: object) -> None:
    """``ModelConfig`` setter 应完整保留 Python 侧写入的配置。

    Args:
        irt_module: ``inferrt_model_py`` 模块，用于构造配置对象。
    """

    config = irt_module.ModelConfig()
    config.num_classes = 7
    config.input_shape = [2, 3, 32, 32]
    config.input_shapes = [[2, 3, 32, 32], [1, 1, 16, 16]]
    config.input_tensor_names = ["image", "mask"]
    config.output_tensor_names = ["logits", "aux"]
    config.feature_tensor_names = ["layer1", "layer4"]
    config.feature_only = True

    assert config.num_classes == 7
    assert config.input_shape == [2, 3, 32, 32]
    assert config.input_shapes == [[2, 3, 32, 32], [1, 1, 16, 16]]
    assert config.input_tensor_names == ["image", "mask"]
    assert config.output_tensor_names == ["logits", "aux"]
    assert config.feature_tensor_names == ["layer1", "layer4"]
    assert config.feature_only is True


def test_model_config_rejects_invalid_shape_rank(irt_module: object) -> None:
    """Python 配置绑定应拒绝非 4D 输入形状，避免把非法 NCHW 配置传给 C++。

    Args:
        irt_module: ``inferrt_model_py`` 模块，用于访问 ``InferRTError``。

    Raises:
        irt_module.InferRTError: 当 ``input_shape`` 或 ``input_shapes`` 不是 4D 时抛出。
    """

    config = irt_module.ModelConfig()

    with pytest.raises(irt_module.InferRTError):
        config.input_shape = [1, 3, 224]

    with pytest.raises(irt_module.InferRTError):
        config.input_shapes = [[1, 3, 224, 224], [1, 3, 224]]


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


def test_create_vit_without_build(irt_module: object) -> None:
    """ViT 标准变体和兼容别名应可通过 Python 绑定创建，且不依赖权重文件。

    Args:
        irt_module: ``inferrt_model_py`` 模块。
    """

    model = irt_module.create_model("ViT_Base_Patch16_224")
    assert model.name() == "ViTBasePatch16_224"
    assert model.input_tensor_names() == ["input"]
    assert model.output_tensor_names() == ["output"]

    alias_model = irt_module.create_model("vit")
    assert alias_model.name() == "ViTBasePatch16_224"

    high_res_model = irt_module.create_model("ViT_Base_Patch16_384")
    assert high_res_model.name() == "ViTBasePatch16_384"
    assert high_res_model.input_tensor_names() == ["input"]
    assert high_res_model.output_tensor_names() == ["output"]


def test_create_model_rejects_unknown_name(irt_module: object) -> None:
    """未知模型名应通过 Python 异常暴露，便于调用方统一处理输入错误。

    Args:
        irt_module: ``inferrt_model_py`` 模块，用于调用 ``create_model``。

    Raises:
        irt_module.InferRTError: 模型工厂无法创建实例时抛出。
    """

    with pytest.raises(irt_module.InferRTError):
        irt_module.create_model("not_a_model")


def test_create_model_clones_custom_config(irt_module: object) -> None:
    """``create_model`` 应克隆传入配置，模型侧张量名不受后续 Python 配置修改影响。

    Args:
        irt_module: ``inferrt_model_py`` 模块。
    """

    config = irt_module.ModelConfig()
    config.input_tensor_names = ["image"]
    config.output_tensor_names = ["scores"]

    model = irt_module.create_model("alexnet", config)
    config.input_tensor_names = ["changed"]
    config.output_tensor_names = ["changed"]

    assert model.input_tensor_names() == ["image"]
    assert model.output_tensor_names() == ["scores"]


def test_model_log_level_round_trip_without_build(irt_module: object) -> None:
    """模型日志级别应能在未构建 engine 时读写，覆盖纯配置路径。

    Args:
        irt_module: ``inferrt_model_py`` 模块。
    """

    model = irt_module.create_model("alexnet")

    assert model.log_level == irt_module.LogLevel.WARNING
    model.log_level = irt_module.LogLevel.VERBOSE
    assert model.log_level == irt_module.LogLevel.VERBOSE
