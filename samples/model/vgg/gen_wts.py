import argparse
import os
import struct

import cv2
import numpy as np

import torch
from torchvision.models import (
    VGG11_Weights,
    VGG13_Weights,
    VGG16_Weights,
    VGG19_Weights,
    vgg11,
    vgg13,
    vgg16,
    vgg19,
)


def read_imagenet_labels() -> dict[int, str]:
    clsid2label = {}
    with open("../../../assets/imagenet1000_clsidx_to_labels.txt", "r") as f:
        for i in f.readlines():
            k, v = i.split(": ")
            clsid2label.setdefault(int(k), v[1:-3])
    return clsid2label


def preprocess(img: np.array) -> torch.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)[None, ...]
    return torch.from_numpy(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate VGG weights for TensorRT")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="vgg11",
        choices=["vgg11", "vgg13", "vgg16", "vgg19"],
        help="VGG model name",
    )
    parser.add_argument(
        "-i",
        "--img_path",
        type=str,
        default=os.path.abspath("../../../assets/pics/dog.jpg"),
        help="Path to the input image",
    )
    parser.add_argument("-o", "--output", type=str, default="", help="Path to wts output file")
    parser.add_argument("-p", "--predict_only", action="store_true")
    args = parser.parse_args()

    img_path = args.img_path
    print("Loading ", img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Img {img_path} Not Found")
    img = preprocess(img)

    builders = {
        "vgg11": (vgg11, VGG11_Weights.IMAGENET1K_V1),
        "vgg13": (vgg13, VGG13_Weights.IMAGENET1K_V1),
        "vgg16": (vgg16, VGG16_Weights.IMAGENET1K_V1),
        "vgg19": (vgg19, VGG19_Weights.IMAGENET1K_V1),
    }
    builder, weights = builders[args.model_name]
    output_path = args.output if args.output else f"{args.model_name}.wts"

    model = builder(weights=weights)
    model.eval()

    print(model)

    output = model(img)
    labels = read_imagenet_labels()
    print("\nPyTorch inference results:")
    for batch in torch.topk(output, k=3).indices:
        for i, j in enumerate(batch, 1):
            print(f"top: {i:<2}, confidence: {float(output[0, j]):.4f}, label[{j}]: {labels[int(j)]}")

    if not args.predict_only:
        print(f"\nGenerating weights file to {output_path}...")
        with open(output_path, "w") as f:
            f.write("{}\n".format(len(model.state_dict().keys())))
            for k, v in model.state_dict().items():
                print(f"key: {k}\tvalue: {v.shape}")
                vr = v.reshape(-1).cpu().numpy()
                f.write("{} {}".format(k, len(vr)))
                for vv in vr:
                    f.write(" ")
                    f.write(struct.pack(">f", float(vv)).hex())
                f.write("\n")

        print(f"\nWeights file '{output_path}' generated successfully!")
