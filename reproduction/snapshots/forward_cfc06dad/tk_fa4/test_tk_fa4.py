from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest
import torch

_FLASH_ROOT = Path(__file__).resolve().parent.parent / "flash-attention"
_CUTLASS_PY = _FLASH_ROOT / "csrc" / "cutlass" / "python" / "CuTeDSL"
for _path in (Path(__file__).resolve().parent.parent, _FLASH_ROOT, _CUTLASS_PY):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.append(_path_str)

from flash_attn.cute.interface import _flash_attn_bwd as cute_flash_attn_bwd
from flash_attn.cute.interface import flash_attn_func as cute_flash_attn_func
from tk_fa4 import (
    b300_flash_attn_func,
    b300_flash_attn_func_experimental,
    b300_mha_bwd,
    b300_mha_bwd_experimental,
    b300_mha_fwd,
    mha_fwd as deprecated_mha_fwd,
)


DEFAULT_TEST_SEQLENS = "2048"
FULL_TEST_SEQLENS = "2048,4096,8192,16384"
TEST_SEQLENS = [
    int(x)
    for x in os.getenv(
        "TK_FA4_TEST_SEQLENS",
        FULL_TEST_SEQLENS if os.getenv("TK_FA4_TEST_FULL") == "1" else DEFAULT_TEST_SEQLENS,
    ).split(",")
    if x
]
HOT_STAGE1_SEQLENS = [2048]


def _make_inputs(seqlen: int, batch: int = 1, heads: int = 16, seed: int = 0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    q = torch.randn(batch, seqlen, heads, 192, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(batch, seqlen, heads, 192, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(batch, seqlen, heads, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    dout = torch.randn(batch, seqlen, heads, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    return q, k, v, dout


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol), (
        (actual.float() - expected.float()).abs().max().item(),
        actual.shape,
    )


def _normalize_ref_lse(lse: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if lse.shape == q.shape[:3]:
        return lse
    if lse.shape == (q.shape[0], q.shape[2], q.shape[1]):
        return lse.permute(0, 2, 1).contiguous()
    raise AssertionError(f"unexpected lse shape {tuple(lse.shape)}")


def _hot_stage1_supported(seqlen: int, causal: bool, deterministic: bool) -> bool:
    return seqlen == 2048 and causal and not deterministic


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seqlen", TEST_SEQLENS)
def test_b300_forward_matches_cute(causal: bool, seqlen: int) -> None:
    q, k, v, _ = _make_inputs(seqlen, seed=seqlen + int(causal))

    out_tk, lse_tk = b300_mha_fwd(q, k, v, causal=causal, return_lse=True)
    out_ref, lse_ref_raw = cute_flash_attn_func(q, k, v, causal=causal, return_lse=True)
    lse_ref = _normalize_ref_lse(lse_ref_raw, q)

    _assert_close(out_tk, out_ref, atol=3e-1, rtol=3e-1)
    _assert_close(lse_tk, lse_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_b300_backward_matches_cute(causal: bool, deterministic: bool) -> None:
    q, k, v, dout = _make_inputs(2048, seed=11 + int(causal) + int(deterministic))

    out_ref, lse_ref_raw = cute_flash_attn_func(q, k, v, causal=causal, return_lse=True)
    lse_ref = _normalize_ref_lse(lse_ref_raw, q)
    dq_tk, dk_tk, dv_tk = b300_mha_bwd(
        q,
        k,
        v,
        out_ref,
        lse_ref,
        dout,
        causal=causal,
        deterministic=deterministic,
    )
    dq_ref, dk_ref, dv_ref = cute_flash_attn_bwd(
        q,
        k,
        v,
        out_ref,
        dout,
        lse_ref_raw,
        causal=causal,
        deterministic=deterministic,
    )

    assert dq_tk.dtype == torch.float32
    assert dk_tk.dtype == torch.float32
    assert dv_tk.dtype == torch.float32
    _assert_close(dq_tk, dq_ref, atol=4e-1, rtol=4e-1)
    _assert_close(dk_tk, dk_ref, atol=4e-1, rtol=4e-1)
    _assert_close(dv_tk, dv_ref, atol=4e-1, rtol=4e-1)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("seqlen", TEST_SEQLENS)
def test_b300_backward_experimental_matches_cute(causal: bool, deterministic: bool, seqlen: int) -> None:
    q, k, v, dout = _make_inputs(seqlen, seed=101 + seqlen + 10 * int(causal) + int(deterministic))

    out_ref, lse_ref_raw = cute_flash_attn_func(q, k, v, causal=causal, return_lse=True)
    lse_ref = _normalize_ref_lse(lse_ref_raw, q)
    dq_tk, dk_tk, dv_tk = b300_mha_bwd_experimental(
        q,
        k,
        v,
        out_ref,
        lse_ref,
        dout,
        causal=causal,
        deterministic=deterministic,
    )
    dq_ref, dk_ref, dv_ref = cute_flash_attn_bwd(
        q,
        k,
        v,
        out_ref,
        dout,
        lse_ref_raw,
        causal=causal,
        deterministic=deterministic,
    )

    assert dq_tk.dtype == torch.float32
    assert dk_tk.dtype == torch.float32
    assert dv_tk.dtype == torch.float32
    _assert_close(dq_tk, dq_ref, atol=4e-1, rtol=4e-1)
    _assert_close(dk_tk, dk_ref, atol=4e-1, rtol=4e-1)
    _assert_close(dv_tk, dv_ref, atol=4e-1, rtol=4e-1)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("seqlen", HOT_STAGE1_SEQLENS)
def test_b300_backward_experimental_hot_matches_ref_and_cute(
    seqlen: int,
    causal: bool,
    deterministic: bool,
) -> None:
    q, k, v, dout = _make_inputs(seqlen, seed=151 + seqlen + 10 * int(causal) + int(deterministic))

    out_ref, lse_ref_raw = cute_flash_attn_func(q, k, v, causal=causal, return_lse=True)
    lse_ref = _normalize_ref_lse(lse_ref_raw, q)
    if not _hot_stage1_supported(seqlen, causal, deterministic):
        with pytest.raises(ValueError, match="CuTe16 hot mode not implemented yet"):
            b300_mha_bwd_experimental(
                q,
                k,
                v,
                out_ref,
                lse_ref,
                dout,
                causal=causal,
                deterministic=deterministic,
                implementation="hot",
            )
        return

    dq_hot, dk_hot, dv_hot = b300_mha_bwd_experimental(
        q,
        k,
        v,
        out_ref,
        lse_ref,
        dout,
        causal=causal,
        deterministic=deterministic,
        implementation="hot",
    )
    dq_ref_impl, dk_ref_impl, dv_ref_impl = b300_mha_bwd_experimental(
        q,
        k,
        v,
        out_ref,
        lse_ref,
        dout,
        causal=causal,
        deterministic=deterministic,
        implementation="ref",
    )
    dq_ref, dk_ref, dv_ref = cute_flash_attn_bwd(
        q,
        k,
        v,
        out_ref,
        dout,
        lse_ref_raw,
        causal=causal,
        deterministic=deterministic,
    )

    assert dq_hot.dtype == torch.float32
    assert dk_hot.dtype == torch.float32
    assert dv_hot.dtype == torch.float32
    assert torch.isfinite(dq_hot).all()
    assert torch.isfinite(dk_hot).all()
    assert torch.isfinite(dv_hot).all()
    assert dq_hot.shape == dq_ref_impl.shape == dq_ref.shape
    assert dk_hot.shape == dk_ref_impl.shape == dk_ref.shape
    assert dv_hot.shape == dv_ref_impl.shape == dv_ref.shape
    assert (dq_hot - dq_ref).abs().max().item() < 32.0
    assert (dk_hot - dk_ref).abs().max().item() < 32.0
    assert (dv_hot - dv_ref).abs().max().item() < 32.0


def test_b300_flash_attn_autograd_smoke() -> None:
    q, k, v, dout = _make_inputs(2048, seed=23)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out, lse = b300_flash_attn_func(q, k, v, causal=False, deterministic=True, return_lse=True)
    assert lse.shape == (q.shape[0], q.shape[1], q.shape[2])
    out.backward(dout)

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@pytest.mark.parametrize("causal", [False, True])
def test_b300_flash_attn_experimental_autograd_smoke(causal: bool) -> None:
    q, k, v, dout = _make_inputs(2048, seed=53 + int(causal))
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out, lse = b300_flash_attn_func_experimental(q, k, v, causal=causal, deterministic=True, return_lse=True)
    assert lse.shape == (q.shape[0], q.shape[1], q.shape[2])
    out.backward(dout)

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_b300_flash_attn_experimental_hot_autograd_smoke(causal: bool, deterministic: bool) -> None:
    q, k, v, dout = _make_inputs(2048, seed=63 + 10 * int(causal) + int(deterministic))
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out, lse = b300_flash_attn_func_experimental(
        q,
        k,
        v,
        causal=causal,
        deterministic=deterministic,
        return_lse=True,
        implementation="hot",
    )
    assert lse.shape == (q.shape[0], q.shape[1], q.shape[2])
    if not _hot_stage1_supported(2048, causal, deterministic):
        with pytest.raises(ValueError, match="CuTe16 hot mode not implemented yet"):
            out.backward(dout)
        return

    out.backward(dout)

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@pytest.mark.parametrize("causal", [False, True])
def test_b300_backward_experimental_hot_is_reproducible(causal: bool) -> None:
    q, k, v, dout = _make_inputs(2048, seed=71 + int(causal))
    out_ref, lse_ref_raw = cute_flash_attn_func(q, k, v, causal=causal, return_lse=True)
    lse_ref = _normalize_ref_lse(lse_ref_raw, q)

    with pytest.raises(ValueError, match="CuTe16 hot mode not implemented yet"):
        b300_mha_bwd_experimental(
            q, k, v, out_ref, lse_ref, dout, causal=causal, deterministic=True, implementation="hot"
        )


def test_b300_backward_experimental_auto_routing_matches_hot_and_ref() -> None:
    q, k, v, dout = _make_inputs(2048, seed=81)
    out_ref, lse_ref_raw = cute_flash_attn_func(q, k, v, causal=False, return_lse=True)
    lse_ref = _normalize_ref_lse(lse_ref_raw, q)

    grads_auto = b300_mha_bwd_experimental(
        q, k, v, out_ref, lse_ref, dout, causal=False, deterministic=False
    )
    grads_force_ref = b300_mha_bwd_experimental(
        q, k, v, out_ref, lse_ref, dout, causal=False, deterministic=False, implementation="ref"
    )

    for grad_auto, grad_forced in zip(grads_auto, grads_force_ref):
        _assert_close(grad_auto, grad_forced, atol=1e-3, rtol=1e-3)


def test_b300_backward_experimental_hot_rejects_unsupported_shapes() -> None:
    q_pad, k_pad, v_pad, dout_pad = _make_inputs(2176, seed=99)
    out_pad, lse_pad_raw = cute_flash_attn_func(q_pad, k_pad, v_pad, causal=True, return_lse=True)
    lse_pad = _normalize_ref_lse(lse_pad_raw, q_pad)
    with pytest.raises(ValueError, match="current stage only supports"):
        b300_mha_bwd_experimental(
            q_pad,
            k_pad,
            v_pad,
            out_pad,
            lse_pad,
            dout_pad,
            causal=True,
            deterministic=False,
            implementation="hot",
        )


def test_short_sequence_rejected() -> None:
    q = torch.randn(1, 1024, 16, 192, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1024, 16, 192, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 1024, 16, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="seqlen >="):
        b300_mha_fwd(q, k, v)


def test_wrong_head_dims_rejected() -> None:
    q = torch.randn(1, 2048, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2048, 16, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2048, 16, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head_dim"):
        b300_mha_fwd(q, k, v)


def test_deprecated_shim_warns() -> None:
    q = torch.randn(1, 2048, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2048, 16, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 2048, 16, 128, device="cuda", dtype=torch.bfloat16)

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            deprecated_mha_fwd(q, k, v, causal=False)
        except ImportError:
            pytest.skip("legacy deprecated extension is not built")
    assert any(item.category is DeprecationWarning for item in seen)
