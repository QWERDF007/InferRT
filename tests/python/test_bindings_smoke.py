"""Lightweight smoke tests for inferrt_model_py that do not require weights or CUDA inference."""

from __future__ import annotations


def test_registered_models(irt_module: object) -> None:
    names = irt_module.get_registered_model_names()
    assert "resnet18" in names
    assert "alexnet" in names
    assert irt_module.is_supported_model("ResNet18")
    assert not irt_module.is_supported_model("not_a_model")


def test_model_config_defaults(irt_module: object) -> None:
    config = irt_module.ModelConfig()
    assert config.num_classes == 1000
    assert config.feature_only is False
    assert config.input_shape == [1, 3, 224, 224]


def test_create_model_without_build(irt_module: object) -> None:
    model = irt_module.create_model("vgg11")
    assert model.name() == "VGG11"
    assert model.input_tensor_names()
    assert model.output_tensor_names()
