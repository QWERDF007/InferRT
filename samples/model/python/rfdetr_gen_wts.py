"""RF-DETR 权重导出脚本：将上游 PyTorch checkpoint 转成 InferRT `.wts`。"""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path
from typing import Mapping

import torch

from rfdetr_compat import configure_rfdetr_import

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RFDETR_ROOT = Path("F:/Github/CV/rf-detr")
DEFAULT_MODEL_DIR = Path("F:/models/rfdetr")

MODEL_CHECKPOINTS = {
    "rfdetr_nano": "rf-detr-nano.pth",
    "rfdetr-nano": "rf-detr-nano.pth",
    "rfdetr_small": "rf-detr-small.pth",
    "rfdetr-small": "rf-detr-small.pth",
}


def elapsed_ms(start: float, end: float) -> float:
    """把 ``perf_counter`` 时间差转换成毫秒。"""

    return (end - start) * 1000.0


def resolve_checkpoint(model: str, checkpoint: Path | None, model_dir: Path) -> Path:
    """解析 RF-DETR checkpoint 路径。

    Args:
        model: InferRT RF-DETR 模型 key。
        checkpoint: 用户显式传入的 checkpoint；为空时从 ``model_dir`` 推断。
        model_dir: 默认 checkpoint 目录。

    Returns:
        已解析的 checkpoint 路径。

    Raises:
        ValueError: 模型 key 暂无默认 checkpoint 映射时抛出。
    """

    if checkpoint is not None:
        return checkpoint.expanduser().resolve()
    try:
        filename = MODEL_CHECKPOINTS[model]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_CHECKPOINTS))
        raise ValueError(f"No default RF-DETR checkpoint for {model!r}. Available: {available}") from exc
    return (model_dir.expanduser() / filename).resolve()


def infer_checkpoint_num_classes(checkpoint: Path) -> int | None:
    """从 checkpoint 检测头权重推断前景类别数。

    Args:
        checkpoint: 上游 RF-DETR ``.pth`` checkpoint。

    Returns:
        检测头输出维度减去背景类；无法推断时返回 ``None``。
    """

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model")
    if not isinstance(model_state, Mapping):
        return None
    class_bias = model_state.get("class_embed.bias")
    if not isinstance(class_bias, torch.Tensor) or class_bias.ndim < 1:
        return None
    return int(class_bias.shape[0]) - 1


def load_export_model(
    checkpoint: Path,
    *,
    rfdetr_root: Path | None,
    device: str,
    num_classes: int | None = None,
) -> torch.nn.Module:
    """加载上游 RF-DETR checkpoint 并切换到 export 前向。

    Args:
        checkpoint: 上游 RF-DETR ``.pth`` checkpoint。
        rfdetr_root: 上游源码目录，用于离线导入。
        device: PyTorch 设备，通常为 ``cpu``。
        num_classes: 前景类别数；为空时使用上游 checkpoint 中的 ``args``。

    Returns:
        已加载、``eval`` 且调用过 ``export()`` 的 PyTorch 模型。
    """

    configure_rfdetr_import(rfdetr_root)
    from rfdetr import from_checkpoint  # type: ignore

    kwargs: dict[str, object] = {"device": device}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    wrapper = from_checkpoint(checkpoint, **kwargs)
    model = wrapper.model.model
    model.eval()
    model.export()
    return model


def export_state_dict(model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
    """返回 RF-DETR 原生 TensorRT 构建器可读取的权重表。

    上游 RF-DETR 训练权重按 ``num_queries * group_detr`` 保存 query embedding；
    export 前向只使用第一组 ``num_queries``。这里写出前同步裁剪，避免 C++ 建图读到训练期
    group-DETR 权重尺寸。
    """

    state_dict = dict(model.state_dict())
    num_queries = int(getattr(model, "num_queries", 0))
    if num_queries > 0:
        for key in ("query_feat.weight", "refpoint_embed.weight"):
            tensor = state_dict.get(key)
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 and tensor.shape[0] > num_queries:
                state_dict[key] = tensor[:num_queries].contiguous()
    return state_dict


def write_wts(state_dict: Mapping[str, torch.Tensor], output_path: Path, *, verbose: bool = True) -> None:
    """写出 InferRT 文本 `.wts` 权重文件。

    Args:
        state_dict: PyTorch 权重表。
        output_path: 输出 `.wts` 文件路径。
        verbose: 为 ``True`` 时打印每个权重 key 和 shape。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(state_dict)}\n")
        for key, tensor in state_dict.items():
            value = tensor.detach().float().reshape(-1).cpu().numpy()
            if verbose:
                print(f"key: {key}\tvalue: {tuple(tensor.shape)}")
            handle.write(f"{key} {len(value)}")
            for item in value:
                handle.write(" ")
                handle.write(struct.pack(">f", float(item)).hex())
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    """解析 RF-DETR `.wts` 导出命令行参数。"""

    parser = argparse.ArgumentParser(description="Generate RF-DETR weights for InferRT native TensorRT graph")
    parser.add_argument("-m", "--model", default="rfdetr_nano", choices=sorted(MODEL_CHECKPOINTS), help="InferRT key")
    parser.add_argument("-c", "--checkpoint", type=Path, default=None, help="RF-DETR .pth checkpoint path")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory used to resolve default checkpoints, e.g. F:/models/rfdetr",
    )
    parser.add_argument(
        "--rfdetr-root",
        type=Path,
        default=DEFAULT_RFDETR_ROOT,
        help="Local RF-DETR repository root; pass an installed-package setup if omitted",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .wts path")
    parser.add_argument("--device", default="cpu", help="PyTorch device used while loading the checkpoint")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Foreground class count; default infers from checkpoint class_embed.bias",
    )
    parser.add_argument("-l", "--list-model", action="store_true", help="List supported RF-DETR model keys")
    parser.add_argument("--quiet", action="store_true", help="Do not print every exported weight key")
    return parser.parse_args()


def main() -> None:
    """加载 RF-DETR checkpoint 并导出 InferRT `.wts`。"""

    args = parse_args()
    if args.list_model:
        models = sorted(MODEL_CHECKPOINTS)
        print(f"RF-DETR models: {len(models)}")
        print(models)
        return

    checkpoint = resolve_checkpoint(args.model, args.checkpoint, args.model_dir)
    output = args.output if args.output is not None else PROJECT_ROOT / f"{args.model}.wts"
    num_classes = args.num_classes if args.num_classes is not None else infer_checkpoint_num_classes(checkpoint)

    print(f"Loading RF-DETR checkpoint: {checkpoint}")
    if num_classes is not None:
        print(f"Using RF-DETR num_classes={num_classes}")
    load_start = time.perf_counter()
    model = load_export_model(checkpoint, rfdetr_root=args.rfdetr_root, device=args.device, num_classes=num_classes)
    load_end = time.perf_counter()

    print(f"Writing weights: {output}")
    export_start = time.perf_counter()
    write_wts(export_state_dict(model), output, verbose=not args.quiet)
    export_end = time.perf_counter()

    print(
        "Timing: "
        f"load={elapsed_ms(load_start, load_end):.3f} ms, "
        f"export={elapsed_ms(export_start, export_end):.3f} ms"
    )


if __name__ == "__main__":
    main()
