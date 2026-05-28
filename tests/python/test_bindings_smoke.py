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
    assert "dinov2_vits14" in names
    assert "vit_small_patch14_dinov2" in names
    assert "dinov3_vitb16" in names
    assert "vit_base_patch16_dinov3" in names
    assert "yolov5n" in names
    assert "yolov8n" in names
    assert "sam" in names
    assert "sam2" in names
    assert "sam3" in names
    assert irt_module.is_supported_model("ResNet18")
    assert irt_module.is_supported_model("GoogLeNet")
    assert irt_module.is_supported_model("ViT_Base_Patch16_224")
    assert irt_module.is_supported_model("ViT_Base_Patch16_384")
    assert irt_module.is_supported_model("ViT")
    assert irt_module.is_supported_model("DINOv2_ViTS14")
    assert irt_module.is_supported_model("ViT_Small_Patch14_DINOv2")
    assert irt_module.is_supported_model("DINOv3_ViTB16")
    assert irt_module.is_supported_model("ViT_Base_Patch16_DINOv3")
    assert irt_module.is_supported_model("YOLOv5N")
    assert irt_module.is_supported_model("YOLOv8N")
    assert irt_module.is_supported_model("SAM")
    assert irt_module.is_supported_model("SAM2_Hiera_Tiny")
    assert irt_module.is_supported_model("SAM3_Image")
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
    assert config.backend == irt_module.ModelBackend.TENSORRT
    assert config.device == irt_module.ModelDevice.GPU


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
    config.backend = irt_module.ModelBackend.ONNXRUNTIME
    config.device = irt_module.ModelDevice.CPU

    assert config.num_classes == 7
    assert config.input_shape == [2, 3, 32, 32]
    assert config.input_shapes == [[2, 3, 32, 32], [1, 1, 16, 16]]
    assert config.input_tensor_names == ["image", "mask"]
    assert config.output_tensor_names == ["logits", "aux"]
    assert config.feature_tensor_names == ["layer1", "layer4"]
    assert config.feature_only is True
    assert config.backend == irt_module.ModelBackend.ONNXRUNTIME
    assert config.device == irt_module.ModelDevice.CPU


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


def test_create_model_accepts_backend_and_device_options(irt_module: object) -> None:
    """验证 ``create_model`` 支持字符串和枚举形式的后端/设备参数。

    Args:
        irt_module: 已导入的 ``inferrt_model_py`` 模块。
    """

    model = irt_module.create_model("onnx", backend="onnxruntime", device="cpu")
    assert model.backend() == irt_module.ModelBackend.ONNXRUNTIME
    assert model.device() == irt_module.ModelDevice.CPU

    enum_model = irt_module.create_model(
        "onnx",
        backend=irt_module.ModelBackend.ONNXRUNTIME,
        device=irt_module.ModelDevice.GPU,
    )
    assert enum_model.backend() == irt_module.ModelBackend.ONNXRUNTIME
    assert enum_model.device() == irt_module.ModelDevice.GPU

    openvino_model = irt_module.create_model("onnx", backend="openvino", device="cpu")
    assert openvino_model.backend() == irt_module.ModelBackend.OPENVINO
    assert openvino_model.device() == irt_module.ModelDevice.CPU


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


def test_create_dino_without_build(irt_module: object) -> None:
    """DINOv2/DINOv3 官方 key 和 timm 别名应可通过 Python 绑定创建。

    Args:
        irt_module: ``inferrt_model_py`` 模块。
    """

    dinov2 = irt_module.create_model("dinov2_vits14")
    assert dinov2.name() == "DINOv2ViTS14"
    assert dinov2.input_tensor_names() == ["input"]
    assert dinov2.output_tensor_names() == ["output"]

    dinov2_alias = irt_module.create_model("vit_small_patch14_reg4_dinov2")
    assert dinov2_alias.name() == "DINOv2ViTS14Reg4"

    dinov3 = irt_module.create_model("dinov3_vitb16")
    assert dinov3.name() == "DINOv3ViTB16"

    dinov3_alias = irt_module.create_model("vit_base_patch16_dinov3_qkvb")
    assert dinov3_alias.name() == "DINOv3ViTB16"


def test_create_yolo_without_build(irt_module: object) -> None:
    """YOLOv5/YOLOv8 检测模型应可通过 Python 绑定创建，并自动使用检测默认配置。
    Args:
        irt_module: ``inferrt_model_py`` 模块。
    """

    yolov5 = irt_module.create_model("yolov5n")
    assert yolov5.name() == "YOLOv5n"
    assert yolov5.input_tensor_names() == ["input"]
    assert yolov5.output_tensor_names() == ["output0", "output1", "output2"]

    yolov8 = irt_module.create_model("YOLOv8")
    assert yolov8.name() == "YOLOv8n"
    assert yolov8.input_tensor_names() == ["input"]
    assert yolov8.output_tensor_names() == ["output0", "output1", "output2"]


def test_create_sam_without_build(irt_module: object) -> None:
    """SAM/SAM2/SAM3 应通过 Python 绑定创建，并暴露统一的 prompt 分割契约。

    Args:
        irt_module: ``inferrt_model_py`` 模块。
    """

    expected_inputs = ["image", "point_coords", "point_labels", "mask_input", "has_mask_input"]
    expected_outputs = ["masks", "iou_predictions", "low_res_masks"]

    sam = irt_module.create_model("sam")
    assert sam.name() == "SAMViTH"
    assert sam.input_tensor_names() == expected_inputs
    assert sam.output_tensor_names() == expected_outputs

    sam2 = irt_module.create_model("SAM2_Hiera_Tiny")
    assert sam2.name() == "SAM2HieraTiny"
    assert sam2.input_tensor_names() == expected_inputs
    assert sam2.output_tensor_names() == expected_outputs

    sam3 = irt_module.create_model("sam3_image")
    assert sam3.name() == "SAM3Image"
    assert sam3.input_tensor_names() == expected_inputs
    assert sam3.output_tensor_names() == expected_outputs


def test_sam_build_rejects_placeholder_weights(irt_module: object, tmp_path) -> None:
    """SAM/SAM2 官方 TensorRT 图应拒绝占位权重，避免误以为空权重可推理。

    Args:
        irt_module: ``inferrt_model_py`` 模块。
        tmp_path: pytest 临时目录 fixture，用于写入最小占位 ``.wts``。

    Raises:
        irt_module.InferRTError: 缺少官方 SAM 权重时抛出。
    """

    weights = tmp_path / "sam_placeholder.wts"
    weights.write_text("1\nplaceholder.weight 1 00000000\n", encoding="utf-8")

    for model_name in ("sam_vit_b", "sam2_hiera_tiny"):
        model = irt_module.create_model(model_name)
        with pytest.raises(irt_module.InferRTError, match="Missing SAM official weight"):
            model.build(str(weights))


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
