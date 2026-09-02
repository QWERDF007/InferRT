"""PyTorch reference precision isolation contract."""

from __future__ import annotations

import pytest

from helpers.torch_precision import strict_fp32_reference


def test_strict_fp32_reference_reapplies_and_restores_global_state() -> None:
    """Upstream imports may mutate TF32 settings without leaking across parity tests."""

    torch = pytest.importorskip("torch")
    original = (
        torch.get_float32_matmul_precision(),
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
    )
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        expected_restored = (
            torch.get_float32_matmul_precision(),
            bool(torch.backends.cuda.matmul.allow_tf32),
            bool(torch.backends.cudnn.allow_tf32),
        )

        with strict_fp32_reference(torch) as reapply:
            assert torch.get_float32_matmul_precision() == "highest"
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32

            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32 = True
            reapply()

            assert torch.get_float32_matmul_precision() == "highest"
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32

        assert (
            torch.get_float32_matmul_precision(),
            bool(torch.backends.cuda.matmul.allow_tf32),
            bool(torch.backends.cudnn.allow_tf32),
        ) == expected_restored
    finally:
        torch.set_float32_matmul_precision(original[0])
        torch.backends.cuda.matmul.allow_tf32 = original[1]
        torch.backends.cudnn.allow_tf32 = original[2]
