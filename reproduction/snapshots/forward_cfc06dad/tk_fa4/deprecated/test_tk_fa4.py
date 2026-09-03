from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FLASH_ATTN_ROOT = ROOT / "flash-attention"
if str(FLASH_ATTN_ROOT) not in sys.path:
    sys.path.insert(0, str(FLASH_ATTN_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TK_FA4_FWD_MODE", "ref")
os.environ.setdefault("TK_FA4_BWD_MODE", "ref")

from flash_attn.cute.interface import _flash_attn_bwd as cute_flash_attn_bwd  # noqa: E402
from flash_attn.cute.interface import flash_attn_func as cute_flash_attn_func  # noqa: E402
from tk_fa4 import _C as tk_extension  # noqa: E402
from tk_fa4 import flash_attn_func as tk_flash_attn_func  # noqa: E402


def _rand(shape: tuple[int, ...], *, device: str = "cuda") -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=torch.bfloat16)


def _align_ref_lse(lse: torch.Tensor) -> torch.Tensor:
    return lse.permute(0, 2, 1).contiguous() if lse.ndim == 3 else lse


def _pad_bshd(x: torch.Tensor, multiple: int = 128) -> torch.Tensor:
    pad = (-x.shape[1]) % multiple
    if pad == 0:
        return x.contiguous()
    pad_tensor = torch.zeros((x.shape[0], pad, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
    return torch.cat((x, pad_tensor), dim=1).contiguous()


def _pad_bsh(x: torch.Tensor, multiple: int = 128) -> torch.Tensor:
    pad = (-x.shape[1]) % multiple
    if pad == 0:
        return x.contiguous()
    pad_tensor = torch.zeros((x.shape[0], pad, x.shape[2]), device=x.device, dtype=x.dtype)
    return torch.cat((x, pad_tensor), dim=1).contiguous()


def _to_bhsd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def _from_bhsd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def _enable_dense_hot(monkeypatch) -> None:
    monkeypatch.setenv("TK_FA4_FWD_MODE", "ref")
    monkeypatch.setenv("TK_FA4_BWD_MODE", "hot")
    monkeypatch.setenv("TK_FA4_BWD_DENSE_HOT", "1")
    monkeypatch.delenv("TK_FA4_BWD_WG_HOT", raising=False)


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("seqlen", [128, 512, 2048])
@pytest.mark.parametrize("heads, heads_kv", [(32, 32), (32, 8), (32, 1)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_dense_parity(batch, seqlen, heads, heads_kv, head_dim, causal, deterministic):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("GB200 / SM100 is required")

    q = _rand((batch, seqlen, heads, head_dim)).requires_grad_(True)
    k = _rand((batch, seqlen, heads_kv, head_dim)).requires_grad_(True)
    v = _rand((batch, seqlen, heads_kv, head_dim)).requires_grad_(True)
    dout = _rand((batch, seqlen, heads, head_dim))

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_tk, lse_tk = tk_flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        deterministic=deterministic,
        return_lse=True,
    )
    out_ref, lse_ref = cute_flash_attn_func(
        q_ref,
        k_ref,
        v_ref,
        causal=causal,
        deterministic=deterministic,
        return_lse=True,
    )
    lse_ref = _align_ref_lse(lse_ref)

    out_tk.backward(dout)
    out_ref.backward(dout)

    torch.testing.assert_close(out_tk.float(), out_ref.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(lse_tk.float(), lse_ref.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(q.grad.float(), q_ref.grad.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(k.grad.float(), k_ref.grad.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(v.grad.float(), v_ref.grad.float(), rtol=7e-2, atol=7e-2)


def test_dense_hot_backward_smoke(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("GB200 / SM100 is required")

    _enable_dense_hot(monkeypatch)
    torch.manual_seed(0)

    q = _rand((1, 512, 32, 128)).requires_grad_(True)
    k = _rand((1, 512, 32, 128)).requires_grad_(True)
    v = _rand((1, 512, 32, 128)).requires_grad_(True)
    dout = _rand((1, 512, 32, 128))

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_tk, lse_tk = tk_flash_attn_func(q, k, v, causal=False, deterministic=False, return_lse=True)
    out_ref, lse_ref = cute_flash_attn_func(
        q_ref,
        k_ref,
        v_ref,
        causal=False,
        deterministic=False,
        return_lse=True,
    )
    lse_ref = _align_ref_lse(lse_ref)

    out_tk.backward(dout)
    out_ref.backward(dout)

    torch.testing.assert_close(out_tk.float(), out_ref.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(lse_tk.float(), lse_ref.float(), rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(q.grad.float(), q_ref.grad.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(k.grad.float(), k_ref.grad.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(v.grad.float(), v_ref.grad.float(), rtol=7e-2, atol=7e-2)


@pytest.mark.parametrize("seqlen", [512, 2048, 8192])
def test_dense_raw_backward_spot(monkeypatch, seqlen):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("GB200 / SM100 is required")

    _enable_dense_hot(monkeypatch)
    torch.manual_seed(0)

    batch, heads, head_dim = 1, 32, 128
    scale = head_dim ** -0.5
    q = _rand((batch, seqlen, heads, head_dim))
    k = _rand((batch, seqlen, heads, head_dim))
    v = _rand((batch, seqlen, heads, head_dim))
    dout = _rand((batch, seqlen, heads, head_dim))

    out_ref, lse_ref_raw = cute_flash_attn_func(q, k, v, causal=False, deterministic=False, return_lse=True)
    lse_ref_bsh = _align_ref_lse(lse_ref_raw)
    l_aux_bhs1 = _pad_bsh((-lse_ref_bsh / scale).contiguous()).permute(0, 2, 1).contiguous().unsqueeze(2)

    dq_tk, dk_tk, dv_tk = tk_extension.mha_bwd(
        _to_bhsd(_pad_bshd(q)),
        _to_bhsd(_pad_bshd(k)),
        _to_bhsd(_pad_bshd(v)),
        _to_bhsd(_pad_bshd(out_ref)),
        l_aux_bhs1,
        _to_bhsd(_pad_bshd(dout)),
        False,
        scale,
        seqlen,
        False,
    )
    torch.cuda.synchronize()
    dq_tk = _from_bhsd(dq_tk[:, :, :seqlen, :])
    dk_tk = _from_bhsd(dk_tk[:, :, :seqlen, :])
    dv_tk = _from_bhsd(dv_tk[:, :, :seqlen, :])

    dq_ref, dk_ref, dv_ref = cute_flash_attn_bwd(
        q,
        k,
        v,
        out_ref,
        dout,
        lse_ref_raw,
        softmax_scale=scale,
        causal=False,
    )

    torch.testing.assert_close(dq_tk.float(), dq_ref.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(dk_tk.float(), dk_ref.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(dv_tk.float(), dv_ref.float(), rtol=7e-2, atol=7e-2)


@pytest.mark.parametrize("seqlen", [512, 2048])
def test_dense_hot_matches_ref_backward_spot(monkeypatch, seqlen):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("GB200 / SM100 is required")

    monkeypatch.setenv("TK_FA4_FWD_MODE", "ref")
    torch.manual_seed(0)

    batch, heads, head_dim = 1, 32, 128
    scale = head_dim ** -0.5
    q = _rand((batch, seqlen, heads, head_dim))
    k = _rand((batch, seqlen, heads, head_dim))
    v = _rand((batch, seqlen, heads, head_dim))
    dout = _rand((batch, seqlen, heads, head_dim))

    out_ref, lse_ref_raw = cute_flash_attn_func(q, k, v, causal=False, deterministic=False, return_lse=True)
    lse_ref_bsh = _align_ref_lse(lse_ref_raw)
    l_aux_bhs1 = _pad_bsh((-lse_ref_bsh / scale).contiguous()).permute(0, 2, 1).contiguous().unsqueeze(2)

    monkeypatch.setenv("TK_FA4_BWD_MODE", "ref")
    dq_split, dk_split, dv_split = tk_extension.mha_bwd(
        _to_bhsd(_pad_bshd(q)),
        _to_bhsd(_pad_bshd(k)),
        _to_bhsd(_pad_bshd(v)),
        _to_bhsd(_pad_bshd(out_ref)),
        l_aux_bhs1,
        _to_bhsd(_pad_bshd(dout)),
        False,
        scale,
        seqlen,
        False,
    )
    torch.cuda.synchronize()

    monkeypatch.setenv("TK_FA4_BWD_MODE", "hot")
    monkeypatch.delenv("TK_FA4_BWD_DENSE_HOT", raising=False)
    monkeypatch.delenv("TK_FA4_BWD_WG_HOT", raising=False)
    dq_hot, dk_hot, dv_hot = tk_extension.mha_bwd(
        _to_bhsd(_pad_bshd(q)),
        _to_bhsd(_pad_bshd(k)),
        _to_bhsd(_pad_bshd(v)),
        _to_bhsd(_pad_bshd(out_ref)),
        l_aux_bhs1,
        _to_bhsd(_pad_bshd(dout)),
        False,
        scale,
        seqlen,
        False,
    )
    torch.cuda.synchronize()

    dq_split = _from_bhsd(dq_split[:, :, :seqlen, :])
    dk_split = _from_bhsd(dk_split[:, :, :seqlen, :])
    dv_split = _from_bhsd(dv_split[:, :, :seqlen, :])
    dq_hot = _from_bhsd(dq_hot[:, :, :seqlen, :])
    dk_hot = _from_bhsd(dk_hot[:, :, :seqlen, :])
    dv_hot = _from_bhsd(dv_hot[:, :, :seqlen, :])

    torch.testing.assert_close(dq_hot.float(), dq_split.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(dk_hot.float(), dk_split.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(dv_hot.float(), dv_split.float(), rtol=7e-2, atol=7e-2)


@pytest.mark.parametrize("seqlen", [512, 2048])
def test_dense_clustered_hot_matches_ref_backward_spot(monkeypatch, seqlen):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("GB200 / SM100 is required")

    monkeypatch.setenv("TK_FA4_FWD_MODE", "ref")
    torch.manual_seed(0)

    batch, heads, head_dim = 1, 32, 128
    scale = head_dim ** -0.5
    q = _rand((batch, seqlen, heads, head_dim))
    k = _rand((batch, seqlen, heads, head_dim))
    v = _rand((batch, seqlen, heads, head_dim))
    dout = _rand((batch, seqlen, heads, head_dim))

    out_ref, lse_ref_raw = cute_flash_attn_func(q, k, v, causal=False, deterministic=False, return_lse=True)
    lse_ref_bsh = _align_ref_lse(lse_ref_raw)
    l_aux_bhs1 = _pad_bsh((-lse_ref_bsh / scale).contiguous()).permute(0, 2, 1).contiguous().unsqueeze(2)

    monkeypatch.setenv("TK_FA4_BWD_MODE", "ref")
    dq_ref_mode, dk_ref_mode, dv_ref_mode = tk_extension.mha_bwd(
        _to_bhsd(_pad_bshd(q)),
        _to_bhsd(_pad_bshd(k)),
        _to_bhsd(_pad_bshd(v)),
        _to_bhsd(_pad_bshd(out_ref)),
        l_aux_bhs1,
        _to_bhsd(_pad_bshd(dout)),
        False,
        scale,
        seqlen,
        False,
    )
    torch.cuda.synchronize()

    _enable_dense_hot(monkeypatch)
    dq_hot, dk_hot, dv_hot = tk_extension.mha_bwd(
        _to_bhsd(_pad_bshd(q)),
        _to_bhsd(_pad_bshd(k)),
        _to_bhsd(_pad_bshd(v)),
        _to_bhsd(_pad_bshd(out_ref)),
        l_aux_bhs1,
        _to_bhsd(_pad_bshd(dout)),
        False,
        scale,
        seqlen,
        False,
    )
    torch.cuda.synchronize()

    dq_ref_mode = _from_bhsd(dq_ref_mode[:, :, :seqlen, :])
    dk_ref_mode = _from_bhsd(dk_ref_mode[:, :, :seqlen, :])
    dv_ref_mode = _from_bhsd(dv_ref_mode[:, :, :seqlen, :])
    dq_hot = _from_bhsd(dq_hot[:, :, :seqlen, :])
    dk_hot = _from_bhsd(dk_hot[:, :, :seqlen, :])
    dv_hot = _from_bhsd(dv_hot[:, :, :seqlen, :])

    torch.testing.assert_close(dq_hot.float(), dq_ref_mode.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(dk_hot.float(), dk_ref_mode.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(dv_hot.float(), dv_ref_mode.float(), rtol=7e-2, atol=7e-2)


@pytest.mark.skipif(
    os.getenv("TK_FA4_LONG_TESTS") != "1",
    reason="long dense backward spot is opt-in",
)
def test_dense_hot_8192_matches_cute_backward_spot(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("GB200 / SM100 is required")

    _enable_dense_hot(monkeypatch)
    torch.manual_seed(0)

    batch, seqlen, heads, head_dim = 1, 8192, 32, 128
    scale = head_dim ** -0.5
    q = _rand((batch, seqlen, heads, head_dim))
    k = _rand((batch, seqlen, heads, head_dim))
    v = _rand((batch, seqlen, heads, head_dim))
    dout = _rand((batch, seqlen, heads, head_dim))

    out_ref, lse_ref_raw = cute_flash_attn_func(q, k, v, causal=False, deterministic=False, return_lse=True)
    lse_ref_bsh = _align_ref_lse(lse_ref_raw)
    l_aux_bhs1 = _pad_bsh((-lse_ref_bsh / scale).contiguous()).permute(0, 2, 1).contiguous().unsqueeze(2)

    dq_hot, dk_hot, dv_hot = tk_extension.mha_bwd(
        _to_bhsd(_pad_bshd(q)),
        _to_bhsd(_pad_bshd(k)),
        _to_bhsd(_pad_bshd(v)),
        _to_bhsd(_pad_bshd(out_ref)),
        l_aux_bhs1,
        _to_bhsd(_pad_bshd(dout)),
        False,
        scale,
        seqlen,
        False,
    )
    torch.cuda.synchronize()

    dq_hot = _from_bhsd(dq_hot[:, :, :seqlen, :])
    dk_hot = _from_bhsd(dk_hot[:, :, :seqlen, :])
    dv_hot = _from_bhsd(dv_hot[:, :, :seqlen, :])

    dq_ref, dk_ref, dv_ref = cute_flash_attn_bwd(
        q,
        k,
        v,
        out_ref,
        dout,
        lse_ref_raw,
        softmax_scale=scale,
        causal=False,
    )

    torch.testing.assert_close(dq_hot.float(), dq_ref.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(dk_hot.float(), dk_ref.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(dv_hot.float(), dv_ref.float(), rtol=7e-2, atol=7e-2)


@pytest.mark.parametrize("seqlen", [512, 2048])
def test_dense_hot_deterministic_matches_ref_backward_spot(monkeypatch, seqlen):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("GB200 / SM100 is required")

    monkeypatch.setenv("TK_FA4_FWD_MODE", "ref")
    torch.manual_seed(0)

    batch, heads, head_dim = 1, 32, 128
    scale = head_dim ** -0.5
    q = _rand((batch, seqlen, heads, head_dim))
    k = _rand((batch, seqlen, heads, head_dim))
    v = _rand((batch, seqlen, heads, head_dim))
    dout = _rand((batch, seqlen, heads, head_dim))

    out_ref, lse_ref_raw = cute_flash_attn_func(q, k, v, causal=False, deterministic=True, return_lse=True)
    lse_ref_bsh = _align_ref_lse(lse_ref_raw)
    l_aux_bhs1 = _pad_bsh((-lse_ref_bsh / scale).contiguous()).permute(0, 2, 1).contiguous().unsqueeze(2)

    monkeypatch.setenv("TK_FA4_BWD_MODE", "ref")
    dq_ref_mode, dk_ref_mode, dv_ref_mode = tk_extension.mha_bwd(
        _to_bhsd(_pad_bshd(q)),
        _to_bhsd(_pad_bshd(k)),
        _to_bhsd(_pad_bshd(v)),
        _to_bhsd(_pad_bshd(out_ref)),
        l_aux_bhs1,
        _to_bhsd(_pad_bshd(dout)),
        False,
        scale,
        seqlen,
        True,
    )
    torch.cuda.synchronize()

    monkeypatch.setenv("TK_FA4_BWD_MODE", "hot")
    monkeypatch.delenv("TK_FA4_BWD_DENSE_HOT", raising=False)
    monkeypatch.delenv("TK_FA4_BWD_WG_HOT", raising=False)
    dq_hot, dk_hot, dv_hot = tk_extension.mha_bwd(
        _to_bhsd(_pad_bshd(q)),
        _to_bhsd(_pad_bshd(k)),
        _to_bhsd(_pad_bshd(v)),
        _to_bhsd(_pad_bshd(out_ref)),
        l_aux_bhs1,
        _to_bhsd(_pad_bshd(dout)),
        False,
        scale,
        seqlen,
        True,
    )
    torch.cuda.synchronize()

    dq_ref_mode = _from_bhsd(dq_ref_mode[:, :, :seqlen, :])
    dk_ref_mode = _from_bhsd(dk_ref_mode[:, :, :seqlen, :])
    dv_ref_mode = _from_bhsd(dv_ref_mode[:, :, :seqlen, :])
    dq_hot = _from_bhsd(dq_hot[:, :, :seqlen, :])
    dk_hot = _from_bhsd(dk_hot[:, :, :seqlen, :])
    dv_hot = _from_bhsd(dv_hot[:, :, :seqlen, :])

    torch.testing.assert_close(dq_hot.float(), dq_ref_mode.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(dk_hot.float(), dk_ref_mode.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(dv_hot.float(), dv_ref_mode.float(), rtol=7e-2, atol=7e-2)


def test_padded_tail_backward_spot(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("GB200 / SM100 is required")

    monkeypatch.setenv("TK_FA4_FWD_MODE", "ref")
    monkeypatch.setenv("TK_FA4_BWD_MODE", "hot")
    torch.manual_seed(0)

    batch, seqlen, heads, head_dim = 1, 320, 32, 64
    scale = head_dim ** -0.5
    q = _rand((batch, seqlen, heads, head_dim)).requires_grad_(True)
    k = _rand((batch, seqlen, heads, head_dim)).requires_grad_(True)
    v = _rand((batch, seqlen, heads, head_dim)).requires_grad_(True)
    dout = _rand((batch, seqlen, heads, head_dim))

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_tk, _ = tk_flash_attn_func(
        q,
        k,
        v,
        causal=False,
        softmax_scale=scale,
        deterministic=True,
        return_lse=True,
    )
    out_ref, _ = cute_flash_attn_func(
        q_ref,
        k_ref,
        v_ref,
        causal=False,
        softmax_scale=scale,
        deterministic=True,
        return_lse=True,
    )

    out_tk.backward(dout)
    out_ref.backward(dout)

    torch.testing.assert_close(q.grad.float(), q_ref.grad.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(k.grad.float(), k_ref.grad.float(), rtol=7e-2, atol=7e-2)
    torch.testing.assert_close(v.grad.float(), v_ref.grad.float(), rtol=7e-2, atol=7e-2)
