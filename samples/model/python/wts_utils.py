"""InferRT ``.wts`` 文件的公共写出逻辑。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import struct

import torch


def write_wts(
    state_dict: Mapping[str, torch.Tensor],
    output_path: str | Path,
    *,
    verbose: bool = True,
) -> None:
    """将 PyTorch 权重表写成 InferRT 文本 ``.wts`` 文件。

    InferRT 的 ``.wts`` 格式以大端序 FP32 十六进制保存每个张量元素。
    不同模型只需要负责准备适合 C++ 构建器的 ``state_dict``，文件编码统一在这里处理。
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
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
