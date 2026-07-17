"""导出分类模型的 InferRT ``.wts`` 权重。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from classification_model_zoo import (
    create_model,
    export_model_state_dict,
    list_supported_models,
)
from wts_utils import write_wts as write_state_dict


def list_supported_classification_models(backend: str) -> list[str]:
    """列出分类脚本可用模型，过滤 DINO 专用模型。"""

    models = list_supported_models(backend)
    if backend == "timm":
        return [name for name in models if "dino" not in name.lower()]
    return models


def write_wts(model: torch.nn.Module, output_path: str | Path, *, verbose: bool = True) -> None:
    """导出分类模型权重。"""

    write_state_dict(export_model_state_dict(model), output_path, verbose=verbose)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate InferRT .wts for classification models")
    parser.add_argument("-m", "--model", default="alexnet", help="Model name")
    parser.add_argument(
        "-b",
        "--backend",
        choices=("torchvision", "timm"),
        default="torchvision",
        help="Model provider: torchvision or timm",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .wts path")
    parser.add_argument("-l", "--list-model", "--list_model", dest="list_model", action="store_true")
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Do not load pretrained weights; useful for structural checks",
    )
    parser.set_defaults(pretrained=True)
    return parser


def main(args: argparse.Namespace) -> None:
    if args.list_model:
        models = list_supported_classification_models(args.backend)
        print(f"{args.backend} models: {len(models)}")
        print(models)
        return

    model = create_model(
        args.model,
        args.backend,
        pretrained=args.pretrained,
    )
    model.eval()
    output_path = args.output if args.output is not None else Path(f"{args.model}.wts")
    print(f"Exporting {args.backend}/{args.model} weights to {output_path}")
    write_wts(model, output_path)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
