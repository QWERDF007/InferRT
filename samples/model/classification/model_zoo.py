import cv2
import numpy as np
import torch
from torchvision.models import (
    AlexNet_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    VGG11_Weights,
    VGG13_Weights,
    VGG16_Weights,
    VGG19_Weights,
    Wide_ResNet50_2_Weights,
    Wide_ResNet101_2_Weights,
    alexnet,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    vgg11,
    vgg13,
    vgg16,
    vgg19,
    wide_resnet50_2,
    wide_resnet101_2,
)


TORCHVISION_MODEL_ZOO = {
    "alexnet": (alexnet, AlexNet_Weights.IMAGENET1K_V1),
    "mobilenet_v2": (mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V2),
    "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "mobilenet_v3_small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "vgg11": (vgg11, VGG11_Weights.IMAGENET1K_V1),
    "vgg13": (vgg13, VGG13_Weights.IMAGENET1K_V1),
    "vgg16": (vgg16, VGG16_Weights.IMAGENET1K_V1),
    "vgg19": (vgg19, VGG19_Weights.IMAGENET1K_V1),
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2),
    "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V2),
    "wide_resnet50_2": (wide_resnet50_2, Wide_ResNet50_2_Weights.IMAGENET1K_V2),
    "wide_resnet101_2": (wide_resnet101_2, Wide_ResNet101_2_Weights.IMAGENET1K_V2),
}


def read_imagenet_labels(labels_path: str) -> dict[int, str]:
    clsid2label = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            k, v = line.split(": ")
            clsid2label.setdefault(int(k), v[1:-3])
    return clsid2label


def preprocess(img: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) * (1.0 / 255.0)
    img = cv2.subtract(img, (0.485, 0.456, 0.406, 0.0))
    img = cv2.divide(img, (0.229, 0.224, 0.225, 1.0))
    img = img.transpose(2, 0, 1)[None, ...]
    return torch.from_numpy(img)


def list_supported_models(backend: str) -> list[str]:
    if backend == "torchvision":
        return list(TORCHVISION_MODEL_ZOO.keys())
    if backend == "timm":
        from timm import list_models

        return list_models("resnet*") + list_models("wide_resnet*")
    raise ValueError(f"Unsupported backend: {backend}")


def create_model(model_name: str, backend: str) -> torch.nn.Module:
    if backend == "torchvision":
        if model_name not in TORCHVISION_MODEL_ZOO:
            available = ", ".join(TORCHVISION_MODEL_ZOO.keys())
            raise ValueError(f"Unsupported torchvision model: {model_name}. Available: {available}")

        model_fn, weights = TORCHVISION_MODEL_ZOO[model_name]
        return model_fn(weights=weights)

    if backend == "timm":
        from timm import create_model

        return create_model(model_name, pretrained=True)

    raise ValueError(f"Unsupported backend: {backend}")
