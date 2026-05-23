import argparse
import os
import struct
from pathlib import Path

import cv2

import torch
from model_zoo import (
    create_model,
    list_supported_models,
    parse_image_size,
    preprocess,
    read_imagenet_labels,
    resolve_input_size,
)


def parse_cli_image_size(value: str) -> tuple[int, int]:
    """@brief 解析命令行输入尺寸，并转换为 argparse 可展示的错误。

    @param value 用户传入的尺寸字符串。
    @return ``(height, width)`` 格式的输入尺寸。
    @exception argparse.ArgumentTypeError 尺寸格式非法时抛出。
    """

    try:
        return parse_image_size(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def write_wts(model: torch.nn.Module, output_path: str) -> None:
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


@torch.no_grad()
def main(args):
    if args.list_model:
        models = list_supported_models(args.backend)
        print(f"{args.backend} models:", len(models))
        print(models)
        return

    img_path = args.img_path
    print("Loading ", img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Img {img_path} Not Found")

    model = create_model(args.model, args.backend)
    model.eval()

    image_size = args.input_size if args.input_size else resolve_input_size(model)
    size_source = "manual" if args.input_size else "model"
    print(f"Using input size: {image_size[0]}x{image_size[1]} ({size_source})")
    img = preprocess(img, image_size=image_size)

    print(model)

    output = model(img)
    label_path = Path(__file__).resolve().parents[3] / "assets" / "imagenet1000_clsidx_to_labels.txt"
    labels = read_imagenet_labels(label_path)
    print("\nPyTorch inference results:")
    for batch in torch.topk(output, k=3).indices:
        for i, j in enumerate(batch, 1):
            print(f"top: {i:<2}, confidence: {float(output[0, j]):.4f}, label[{j}]: {labels[int(j)]}")

    if not args.predict_only:
        output_path = args.output if args.output else f"{args.model}.wts"
        write_wts(model, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate classification model weights for TensorRT")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="alexnet",
        help="Model name",
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=("timm", "torchvision"),
        default="torchvision",
        help="Model provider backend",
    )
    parser.add_argument(
        "-i",
        "--img_path",
        type=str,
        default=os.path.abspath("../../../assets/pics/dog.jpg"),
        help="Path to the input image",
    )
    parser.add_argument(
        "--input-size",
        "--image-size",
        dest="input_size",
        type=parse_cli_image_size,
        default=None,
        metavar="SIZE",
        help="Override input image size, e.g. 384, 384x384, 3x384x384, or 1x3x384x384",
    )
    parser.add_argument("-o", "--output", type=str, default="", help="Path to wts output file")
    parser.add_argument("-l", "--list_model", action="store_true")
    parser.add_argument("-p", "--predict_only", action="store_true")
    main(parser.parse_args())
