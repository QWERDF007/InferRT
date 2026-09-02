"""Unit tests for real-model pytest helper paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from helpers.model_integration import conversion_artifact_dir, resolve_model_file


def test_conversion_artifact_dir_uses_checkpoint_stem_under_model_root(tmp_path: Path) -> None:
    checkpoint = tmp_path / "yolov8" / "yolov8n.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")

    output_dir = conversion_artifact_dir(tmp_path, checkpoint)

    assert output_dir == tmp_path / "yolov8" / "yolov8n"
    assert output_dir.is_dir()


def test_conversion_artifact_dir_uses_directory_checkpoint_itself(tmp_path: Path) -> None:
    checkpoint = tmp_path / "dinov3" / "dinov3_vitb16"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")

    output_dir = conversion_artifact_dir(tmp_path, checkpoint)

    assert output_dir == checkpoint


def test_conversion_artifact_dir_rejects_checkpoint_outside_model_root(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    outside_checkpoint = tmp_path / "outside" / "model.pt"
    model_root.mkdir()
    outside_checkpoint.parent.mkdir()
    outside_checkpoint.write_bytes(b"fake")

    with pytest.raises(ValueError, match="Checkpoint must be under model root"):
        conversion_artifact_dir(model_root, outside_checkpoint)


def test_resolve_model_file_uses_first_existing_candidate(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    fallback = model_root / "resnet" / "resnet18-f37072fd" / "resnet18.wts"
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes(b"weights")

    resolved = resolve_model_file(
        model_root,
        ("resnet/resnet18.wts", "resnet/resnet18-f37072fd/resnet18.wts"),
        "ResNet18 weights",
    )

    assert resolved == fallback


def test_resolve_model_file_prefers_canonical_candidate(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    canonical = model_root / "sam1" / "sam_vit_b_01ec64.pth"
    legacy = model_root / "sam" / "sam_vit_b_01ec64.pth"
    canonical.parent.mkdir(parents=True)
    legacy.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical")
    legacy.write_bytes(b"legacy")

    resolved = resolve_model_file(
        model_root,
        ("sam1/sam_vit_b_01ec64.pth", "sam/sam_vit_b_01ec64.pth"),
        "SAM ViT-B checkpoint",
    )

    assert resolved == canonical
