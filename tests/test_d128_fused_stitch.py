from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tk_fa4 import interface


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
CUDA_SOURCE = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"


def test_d128_fused_stitch_delegates_explicit_field_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    output = object()

    def stitch(*args: object) -> object:
        calls.append(args)
        return output

    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        SimpleNamespace(stitch_gqa_d128_inverse_rope_grad=stitch),
    )
    operands = tuple(object() for _ in range(5))

    result = interface.b300_stitch_gqa_d128_inverse_rope_gradient(
        *operands,
        q_gradient_scale=1,
        k_gradient_scale=2,
        v_gradient_scale=3,
    )

    assert result is output
    assert calls == [(*operands, 1.0, 2.0, 3.0)]


def test_d128_training_uses_exact_standalone_weight_gradient_stitch() -> None:
    source = RUNTIME.read_text()
    backward = source.split("class _LowpAttentionFunction", 1)[1].split(
        "class LowpAttention", 1
    )[0]

    assert "b300_stitch_gqa_d128_inverse_rope_gradient(" in backward
    assert 'runtime.projection_dgrad == "nvfp4"' in backward
    assert "return_combined_gradient=True" not in backward
    assert "runtime.backward_v_weight_gain" in backward
    assert "weight_gradient_lhs = weight_gradient_input.T" in backward
    assert "if not fused_d128_weight_gradient:" in backward
    assert (
        "grad_output.reshape(c.sequence, c.hidden).T.contiguous()"
        not in backward
    )


def test_d128_fused_stitch_is_exported_by_cuda_extension() -> None:
    source = CUDA_SOURCE.read_text()

    assert "stitch_gqa_d128_inverse_rope_grad_kernel" in source
    assert '"stitch_gqa_d128_inverse_rope_grad"' in source
    assert "&stitch_gqa_d128_inverse_rope_grad" in source


def test_d128_stitch_has_exact_h32_kv8_warp_row_specialization() -> None:
    source = CUDA_SOURCE.read_text()

    assert "stitch_gqa_d128_h32_kv8_inverse_rope_grad_kernel" in source
    assert "constexpr int kQHeads = 32" in source
    assert "constexpr int kKvHeads = 8" in source
    assert "dq.size(0) <= 2 && dq.size(1) == 4096" in source
    assert "dq.size(1) == 4096" in source
    assert "q_heads == 32 && kv_heads == 8" in source
    assert "if (exact_h32_kv8)" in source
    assert "} else {" in source
    assert "stitch_gqa_d128_inverse_rope_grad_kernel<<<" in source
