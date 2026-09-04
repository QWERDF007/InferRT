from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")


SAMPLES_PYTHON = Path(__file__).resolve().parents[2] / "samples" / "model" / "python"
if str(SAMPLES_PYTHON) not in sys.path:
    sys.path.insert(0, str(SAMPLES_PYTHON))

from sam_export_onnx import SAMV1OnnxWrapper, export_graph


class _PositionEncoder(torch.nn.Module):
    def forward_with_coords(self, coords: torch.Tensor, _image_size: tuple[int, int]) -> torch.Tensor:
        return coords.sum(dim=-1, keepdim=True).repeat(1, 1, 2)


class _PromptEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 2
        self.input_image_size = (4, 4)
        self.image_embedding_size = (1, 1)
        self.pe_layer = _PositionEncoder()
        self.not_a_point_embed = torch.nn.Embedding(1, self.embed_dim)
        self.point_embeddings = torch.nn.ModuleList(
            torch.nn.Embedding(1, self.embed_dim) for _ in range(4)
        )
        self.no_mask_embed = torch.nn.Embedding(1, self.embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return torch.zeros((1, self.embed_dim, 1, 1))

    def forward(self, *, points, boxes, masks):
        del boxes, masks
        coords, labels = points
        coords = torch.cat([coords + 0.5, torch.zeros((coords.shape[0], 1, 2))], dim=1)
        labels = torch.cat([labels, -torch.ones((labels.shape[0], 1))], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        dense_embedding = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            coords.shape[0], -1, *self.image_embedding_size
        )
        return point_embedding, dense_embedding


class _ImageEncoder(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pooled = image.mean(dim=(1, 2, 3), keepdim=True)
        return pooled.reshape(image.shape[0], 1, 1, 1).repeat(1, 2, 1, 1)


class _MaskDecoder(torch.nn.Module):
    def predict_masks(
        self,
        *,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = sparse_prompt_embeddings.mean(dim=(1, 2), keepdim=True).reshape(-1, 1, 1, 1)
        dense = dense_prompt_embeddings.mean(dim=(1, 2, 3), keepdim=True)
        masks = (image_embeddings[:, :1] + image_pe[:, :1] + prompt + dense).repeat(1, 3, 2, 2)
        iou = sparse_prompt_embeddings[:, :3].mean(dim=2)
        return masks, iou


class _SAM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = _ImageEncoder()
        self.prompt_encoder = _PromptEncoder()
        self.mask_decoder = _MaskDecoder()


def _inputs() -> dict[str, np.ndarray]:
    point_coords = np.zeros((1, 16, 2, 1), dtype=np.float32)
    point_labels = np.full((1, 16, 1, 1), -1.0, dtype=np.float32)
    point_labels[0, 0, 0, 0] = 1.0
    return {
        "image": np.zeros((1, 3, 4, 4), dtype=np.float32),
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": np.zeros((1, 1, 4, 4), dtype=np.float32),
        "has_mask_input": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }


@pytest.mark.optional_backend
def test_sam_v1_wrapper_exports_onnxruntime_loadable_graph(tmp_path: Path, irt_module: object) -> None:
    inputs = _inputs()
    output_path = tmp_path / "sam_v1.onnx"

    export_graph(SAMV1OnnxWrapper(_SAM()).eval(), inputs, output_path, torch.device("cpu"), 17)

    if not irt_module.onnxruntime_enabled:
        pytest.skip("InferRT ONNX Runtime backend is disabled")

    config = irt_module.ModelConfig()
    config.runtime = "onnxruntime:cpu"
    model = irt_module.create_model("onnx", config=config)
    model.build_or_load(str(output_path))
    assert model.input_tensor_names() == list(inputs)


def test_export_graph_validates_before_replacing_existing_output(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "sam_v1.onnx"
    output_path.write_bytes(b"existing-model")

    def reject_graph(*_args, **_kwargs) -> None:
        raise RuntimeError("invalid exported graph")

    monkeypatch.setattr(onnx.checker, "check_model", reject_graph)

    with pytest.raises(RuntimeError, match="invalid exported graph"):
        export_graph(SAMV1OnnxWrapper(_SAM()).eval(), _inputs(), output_path, torch.device("cpu"), 17)

    assert output_path.read_bytes() == b"existing-model"
