"""导出 DINOv2、DINOv3 和 LingBot-Vision 的 InferRT ``.wts`` 权重。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dino_model_zoo import (
    DEFAULT_LINGBOT_MODEL_DIR,
    DEFAULT_LINGBOT_ROOT,
    DINO_BACKENDS,
    create_model,
    export_model_state_dict,
    list_supported_models,
)
from wts_utils import write_wts as write_state_dict


def write_wts(model: torch.nn.Module, output_path: str | Path, *, verbose: bool = True) -> None:
    """将 DINO/LingBot-Vision 模型写为 InferRT ``.wts``。"""

    write_state_dict(export_model_state_dict(model), output_path, verbose=verbose)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate InferRT .wts for DINO and LingBot-Vision models")
    parser.add_argument("-m", "--model", default="dinov2_vits14", help="DINO model name")
    parser.add_argument("-b", "--backend", choices=DINO_BACKENDS, default="torchhub")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .wts path")
    parser.add_argument("-c", "--checkpoint", type=Path, default=None, help="DINO/LingBot checkpoint path")
    parser.add_argument("--lingbot-root", type=Path, default=DEFAULT_LINGBOT_ROOT)
    parser.add_argument("--lingbot-model-dir", type=Path, default=DEFAULT_LINGBOT_MODEL_DIR)
    parser.add_argument("--device", default="cpu", help="PyTorch loading device")
    parser.add_argument("--hub-repo", default=None, help="Local DINOv2 torch.hub repository")
    parser.add_argument("--hub-source", choices=("github", "local"), default="github")
    parser.add_argument("--hub-weights", default=None, help="Optional DINOv2 torch.hub checkpoint")
    parser.add_argument("--hf-model-id", default=None, help="Optional DINOv3 Hugging Face model id")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("-l", "--list-model", "--list_model", dest="list_model", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Do not print every weight key")
    parser.set_defaults(pretrained=True)
    return parser


def main(args: argparse.Namespace) -> None:
    if args.list_model:
        models = list_supported_models(args.backend)
        print(f"{args.backend} models: {len(models)}")
        print(models)
        return

    model = create_model(
        args.model,
        args.backend,
        checkpoint=args.checkpoint,
        model_dir=args.lingbot_model_dir,
        lingbot_root=args.lingbot_root,
        device=args.device,
        hub_repo=args.hub_repo,
        hub_source=args.hub_source,
        pretrained=args.pretrained,
        hub_weights=args.hub_weights,
        hf_model_id=args.hf_model_id,
        local_files_only=args.local_files_only,
    )
    model.eval()
    output_path = args.output if args.output is not None else Path(f"{args.model}.wts")
    print(f"Exporting {args.backend}/{args.model} weights to {output_path}")
    write_wts(model, output_path, verbose=not args.quiet)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
