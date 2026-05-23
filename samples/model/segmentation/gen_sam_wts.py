"""导出官方 Segment Anything v1 权重为 InferRT `.wts` 格式。"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import torch


DEFAULT_IMAGE_PATH = Path(__file__).resolve().parents[3] / "assets" / "pics" / "dog.jpg"
SAM_IMAGE_SIZE = 1024
SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
SAM_PIXEL_STD = (58.395, 57.12, 57.375)


def elapsed_ms(start: float, end: float) -> float:
    """@brief 将 ``perf_counter`` 的时间差转换为毫秒。

    @param start 起始时间戳。
    @param end 结束时间戳。
    @return 毫秒单位耗时。
    """

    return (end - start) * 1000.0


def import_segment_anything(sam_root: str | None) -> Any:
    """@brief 导入官方 ``segment_anything`` 包。

    @param sam_root 官方 ``segment-anything`` 仓库根目录；为空时使用当前 Python 环境。
    @return 已导入的 ``sam_model_registry``。
    """

    if sam_root:
        root = Path(sam_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"SAM root does not exist: {root}")
        sys.path.insert(0, str(root))

    from segment_anything import sam_model_registry  # type: ignore

    return sam_model_registry


def write_wts(state_dict: dict[str, torch.Tensor], output_path: Path, *, verbose: bool = False) -> None:
    """@brief 将官方 ``state_dict`` 写成 InferRT 文本权重格式。

    @param state_dict 官方 SAM v1 权重。
    @param output_path 输出 ``.wts`` 路径。
    @param verbose 为 true 时打印每个权重 key 和 shape。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"{len(state_dict)}\n")
        for key, tensor in state_dict.items():
            if verbose:
                print(f"key: {key}\tvalue: {tuple(tensor.shape)}")
            values = tensor.detach().float().reshape(-1).cpu().numpy()
            file.write(f"{key} {len(values)}")
            for value in values:
                file.write(" ")
                file.write(struct.pack(">f", float(value)).hex())
            file.write("\n")


def compute_resize_shape(original_h: int, original_w: int) -> tuple[int, int]:
    """@brief 复现官方 ``ResizeLongestSide.get_preprocess_shape``。

    @param original_h 原图高度。
    @param original_w 原图宽度。
    @return padding 前的 ``(height, width)``。
    """

    scale = SAM_IMAGE_SIZE / max(original_h, original_w)
    return int(math.floor(original_h * scale + 0.5)), int(math.floor(original_w * scale + 0.5))


def preprocess_image(image_path: Path, device: torch.device) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """@brief 按 InferRT sample 使用的 SAM v1 均值方差预处理图像。

    @param image_path 输入图片路径。
    @param device 输出张量所在设备。
    @return ``(image_tensor, original_size, resized_size)``，其中张量形状为 ``1x3x1024x1024``。
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_size = (image.shape[0], image.shape[1])
    resized_size = compute_resize_shape(*original_size)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (resized_size[1], resized_size[0]), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor(SAM_PIXEL_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(SAM_PIXEL_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    normalized = (tensor - mean) / std
    padded = torch.zeros((1, 3, SAM_IMAGE_SIZE, SAM_IMAGE_SIZE), device=device, dtype=torch.float32)
    padded[:, :, : resized_size[0], : resized_size[1]] = normalized
    return padded, original_size, resized_size


def make_default_prompts(
    device: torch.device,
    resized_size: tuple[int, int],
) -> tuple[tuple[torch.Tensor, torch.Tensor], None, None]:
    """@brief 构造中心正点提示，用于导出时的官方 PyTorch 前向冒烟验证。

    @param device prompt 张量所在设备。
    @param resized_size padding 前图像尺寸。
    @return ``PromptEncoder`` 需要的 ``points, boxes, masks`` 三元组。
    """

    resized_h, resized_w = resized_size
    point_coords = torch.tensor([[[resized_w / 2.0, resized_h / 2.0]]], device=device, dtype=torch.float32)
    point_labels = torch.tensor([[1]], device=device, dtype=torch.int64)
    return (point_coords, point_labels), None, None


@torch.no_grad()
def run_reference_forward(
    model: torch.nn.Module,
    image: torch.Tensor,
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> None:
    """@brief 跑一遍官方 image encoder / prompt encoder / mask decoder。

    @param model 官方 ``Sam`` 模型。
    @param image 预处理后的 ``1x3x1024x1024`` 图像张量。
    @param original_size 原始图片 ``(height, width)``。
    @param resized_size padding 前图像尺寸。
    """

    encoder_start = time.perf_counter()
    image_embeddings = model.image_encoder(image)
    encoder_end = time.perf_counter()

    prompt_start = time.perf_counter()
    points, boxes, masks = make_default_prompts(image.device, resized_size)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(points=points, boxes=boxes, masks=masks)
    prompt_end = time.perf_counter()

    decoder_start = time.perf_counter()
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
    )
    decoder_end = time.perf_counter()

    post_start = time.perf_counter()
    masks_out = model.postprocess_masks(
        low_res_masks,
        input_size=resized_size,
        original_size=original_size,
    )
    post_end = time.perf_counter()

    print(f"image_embeddings: {tuple(image_embeddings.shape)}")
    print(f"sparse_embeddings: {tuple(sparse_embeddings.shape)}")
    print(f"dense_embeddings: {tuple(dense_embeddings.shape)}")
    print(f"low_res_masks: {tuple(low_res_masks.shape)}")
    print(f"iou_predictions: {tuple(iou_predictions.shape)}")
    print(f"postprocessed_masks: {tuple(masks_out.shape)}")
    print(
        "Timing: "
        f"image_encoder={elapsed_ms(encoder_start, encoder_end):.3f} ms, "
        f"prompt_encoder={elapsed_ms(prompt_start, prompt_end):.3f} ms, "
        f"mask_decoder={elapsed_ms(decoder_start, decoder_end):.3f} ms, "
        f"postprocess={elapsed_ms(post_start, post_end):.3f} ms"
    )


def parse_args() -> argparse.Namespace:
    """@brief 解析命令行参数。

    @return 解析后的参数对象。
    """

    parser = argparse.ArgumentParser(description="Export official SAM v1 checkpoint to InferRT .wts")
    parser.add_argument(
        "-m",
        "--model",
        choices=("vit_b", "vit_l", "vit_h", "default"),
        default="vit_b",
        help="Official SAM v1 model type",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        type=Path,
        help="Official SAM v1 checkpoint path, e.g. sam_vit_b_01ec64.pth",
    )
    parser.add_argument(
        "--sam-root",
        type=str,
        default="",
        help="Path to official segment-anything repository root; optional when installed as a package",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .wts path")
    parser.add_argument("-i", "--img-path", type=Path, default=DEFAULT_IMAGE_PATH, help="Image for forward smoke")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-forward", action="store_true", help="Only export weights, skip PyTorch forward smoke")
    parser.add_argument("--verbose", action="store_true", help="Print every exported weight key")
    return parser.parse_args()


def main() -> None:
    """@brief 加载官方 SAM、执行可选前向验证并导出权重。"""

    args = parse_args()
    output_path = args.output if args.output is not None else Path(f"sam_{args.model}.wts")
    device = torch.device(args.device)

    load_start = time.perf_counter()
    registry = import_segment_anything(args.sam_root or None)
    model = registry[args.model](checkpoint=str(args.checkpoint)).to(device)
    model.eval()
    load_end = time.perf_counter()
    print(f"Loaded official SAM {args.model} in {elapsed_ms(load_start, load_end):.3f} ms on {device}")

    if not args.skip_forward:
        preprocess_start = time.perf_counter()
        image, original_size, resized_size = preprocess_image(args.img_path, device)
        preprocess_end = time.perf_counter()
        print(
            f"Preprocess: {elapsed_ms(preprocess_start, preprocess_end):.3f} ms, "
            f"original={original_size}, resized={resized_size}, padded={(SAM_IMAGE_SIZE, SAM_IMAGE_SIZE)}"
        )
        run_reference_forward(model, image, original_size, resized_size)

    export_start = time.perf_counter()
    write_wts(model.state_dict(), output_path, verbose=args.verbose)
    export_end = time.perf_counter()
    print(f"Exported {len(model.state_dict())} tensors to {output_path}")
    print(f"Weight export: {elapsed_ms(export_start, export_end):.3f} ms")


if __name__ == "__main__":
    main()
