import os
import struct
import argparse

import cv2
import numpy as np

import torch
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from timm import create_model, list_models


def read_imagenet_labels() -> dict[int, str]:
    """
    read ImageNet 1000 labels

    Returns:
        dict[int, str]: labels dict
    """
    clsid2label = {}
    with open("../../../assets/imagenet1000_clsidx_to_labels.txt", "r") as f:
        for i in f.readlines():
            k, v = i.split(": ")
            clsid2label.setdefault(int(k), v[1:-3])
    return clsid2label

def preprocess(img: np.array) -> torch.Tensor:
    """
    a preprocess method align with ImageNet dataset

    Args:
        img (np.array): input image

    Returns:
        torch.Tensor: preprocessed image in `NCHW` layout
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)[None, ...]
    return torch.from_numpy(img)

@torch.no_grad()
def main(args):
    if args.list_model:
        models = list_models('resnet*')
        print('ResNet:', len(models))
        print(models)
        models = list_models('wide_resnet*')
        print('Wide ResNet:', len(models))
        print(models)
        exit(0)

    img_path = args.img_path
    print('Loading ', img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Img {img_path} Not Found")
    img = preprocess(img)

    # download url: https://download.pytorch.org/models/
    # model = create_model(args.model, pretrained=True)
    model_zoo = {
        "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1), # https://download.pytorch.org/models/resnet18-f37072fd.pth
        "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
        "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V2),
        "resnet152": (resnet152, ResNet152_Weights.IMAGENET1K_V2),
    }

    if args.model not in model_zoo:
        raise ValueError(f"Unsupported model: {args.model}. Available: {', '.join(model_zoo.keys())}")

    model_fn, weights = model_zoo[args.model]
    model = model_fn(weights=weights)
    model.eval()

    print(model)

    output = model(img)
    labels = read_imagenet_labels()
    print("\nPyTorch inference results:")
    for batch in torch.topk(output, k=3).indices:
        for i, j in enumerate(batch, 1):
            print(f"top: {i:<2}, confidence: {float(output[0, j]):.4f}, label[{j}]: {labels[int(j)]}")

    if not args.predict_only:
        # 生成权重文件
        if args.output is None:
            outpath = f'{args.model}.wts'
        else:
            outpath = args.output
        print(f"\nGenerating weights file to {outpath}...")
        with open(outpath, "w") as f:
            f.write("{}\n".format(len(model.state_dict().keys())))
            for k, v in model.state_dict().items():
                print(f"key: {k}\tvalue: {v.shape}")
                vr = v.reshape(-1).cpu().numpy()
                f.write("{} {}".format(k, len(vr)))
                for vv in vr:
                    f.write(" ")
                    f.write(struct.pack(">f", float(vv)).hex())
                f.write("\n")
        
        print(f"\nWeights file '{outpath}' generated successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate ResNet weights for TensorRT")
    parser.add_argument("-i", "--img_path", type=str, default=os.path.abspath("../../../assets/pics/dog.jpg"), help="Path to the input image")
    parser.add_argument("-m", "--model", type=str, default="resnet18", help="Model name")
    parser.add_argument("-o", "--output", type=str, default=None, help="Path to wts output file")
    parser.add_argument("-l", "--list_model", action='store_true')
    parser.add_argument("-p", '--predict_only', action='store_true')
    args = parser.parse_args()

    main(args)
    