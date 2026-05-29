import argparse
import struct
import time
from pathlib import Path

import cv2

import torch
from classification_model_zoo import (
    create_model,
    export_model_state_dict,
    list_supported_models,
    parse_image_size,
    preprocess,
    read_imagenet_labels,
    resolve_input_size,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "assets" / "pics" / "dog.jpg"


def elapsed_ms(start: float, end: float) -> float:
    """将 ``perf_counter`` 时间差转换为毫秒。

    Args:
        start: 起始时间戳。
        end: 结束时间戳。

    Returns:
        毫秒单位的耗时。
    """

    return (end - start) * 1000.0


def parse_cli_image_size(value: str) -> tuple[int, int]:
    """解析命令行输入尺寸，并转换为 argparse 可展示的错误。

    Args:
        value: 用户传入的尺寸字符串。

    Returns:
        ``(height, width)`` 格式的输入尺寸。

    Raises:
        argparse.ArgumentTypeError: 尺寸格式非法时抛出。
    """

    try:
        return parse_image_size(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def write_wts(model: torch.nn.Module, output_path: str, *, verbose: bool = True) -> None:
    """将 PyTorch 模型 ``state_dict`` 导出为 InferRT ``.wts`` 文本格式。

    Args:
        model: 待导出的 PyTorch 模型。
        output_path: 输出 ``.wts`` 文件路径。
        verbose: 为 true 时打印每个权重 key 和 shape；测试中可关闭以减少日志噪声。
    """

    print(f"\nGenerating weights file to {output_path}...")
    state_dict = export_model_state_dict(model)
    with open(output_path, "w") as f:
        f.write("{}\n".format(len(state_dict.keys())))
        for k, v in state_dict.items():
            if verbose:
                print(f"key: {k}\tvalue: {v.shape}")
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {}".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")

    print(f"\nWeights file '{output_path}' generated successfully!")


def print_inference_results(output: torch.Tensor, labels: dict[int, str]) -> None:
    """打印 PyTorch 前向结果，兼容分类 logits 和 DINO 特征向量。

    Args:
        output: 模型前向输出。
        labels: ImageNet 标签表；仅当输出维度等于 1000 时用于显示类别名。
    """

    output = output.reshape(output.shape[0], -1)
    print("\nPyTorch inference results:")

    if output.shape[1] == len(labels):
        for batch in torch.topk(output, k=3).indices:
            for i, j in enumerate(batch, 1):
                print(f"top: {i:<2}, confidence: {float(output[0, j]):.4f}, label[{j}]: {labels[int(j)]}")
        return

    print(f"feature shape: {tuple(output.shape)}")
    for batch in torch.topk(output, k=min(3, output.shape[1])).indices:
        for i, j in enumerate(batch, 1):
            print(f"top: {i:<2}, value: {float(output[0, j]):.4f}, feature[{int(j)}]")


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

    if args.backend == "torchhub" and args.hub_source == "github" and args.pretrained:
        print(
            "Creating torch.hub model from GitHub; this may download the repository and pretrained weights. "
            "Use --hub-repo <local path> --hub-source local for offline export.",
            flush=True,
        )
    elif args.backend == "transformers":
        print(
            "Creating DINOv3 model with transformers.pipeline(task='image-feature-extraction'); "
            "this may download Hugging Face model files unless they are cached.",
            flush=True,
        )
    else:
        print(f"Creating model: backend={args.backend}, model={args.model}", flush=True)

    model_load_start = time.perf_counter()
    model = create_model(
        args.model,
        args.backend,
        hub_repo=args.hub_repo,
        hub_source=args.hub_source,
        pretrained=args.pretrained,
        hub_weights=args.hub_weights,
        hf_model_id=args.hf_model_id,
        local_files_only=args.local_files_only,
    )
    model_load_end = time.perf_counter()
    print(f"Model created in {elapsed_ms(model_load_start, model_load_end):.3f} ms", flush=True)
    model.eval()

    image_size = args.input_size if args.input_size else resolve_input_size(model)
    size_source = "manual" if args.input_size else "model"
    print(f"Using input size: {image_size[0]}x{image_size[1]} ({size_source})")

    preprocess_start = time.perf_counter()
    img = preprocess(img, image_size=image_size)
    preprocess_end = time.perf_counter()

    print(model)

    inference_start = time.perf_counter()
    output = model(img)
    inference_end = time.perf_counter()

    postprocess_start = time.perf_counter()
    label_path = PROJECT_ROOT / "assets" / "imagenet1000_clsidx_to_labels.txt"
    labels = read_imagenet_labels(label_path)
    print_inference_results(output, labels)
    postprocess_end = time.perf_counter()

    print(
        "Timing: "
        f"preprocess={elapsed_ms(preprocess_start, preprocess_end):.3f} ms, "
        f"inference={elapsed_ms(inference_start, inference_end):.3f} ms, "
        f"postprocess={elapsed_ms(postprocess_start, postprocess_end):.3f} ms"
    )

    if not args.predict_only:
        output_path = args.output if args.output else f"{args.model}.wts"
        write_wts(model, output_path)


def build_arg_parser() -> argparse.ArgumentParser:
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
        choices=("timm", "torchhub", "torchvision", "transformers"),
        default="torchvision",
        help="Model provider backend",
    )
    parser.add_argument(
        "-i",
        "--img_path",
        type=str,
        default=str(DEFAULT_IMAGE_PATH),
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
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="torch.hub repo or local directory; default is inferred from the DINO model name",
    )
    parser.add_argument(
        "--hub-source",
        type=str,
        choices=("github", "local"),
        default="github",
        help="torch.hub source, used when --backend torchhub",
    )
    parser.add_argument(
        "--hub-weights",
        type=str,
        default=None,
        help="Optional DINO torch.hub weights path or URL, used when --backend torchhub",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=None,
        help="Optional Hugging Face model id, used when --backend transformers",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Hugging Face models from the local cache only, used when --backend transformers",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Create model without pretrained weights, useful for offline structural checks",
    )
    parser.set_defaults(pretrained=True)
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
