from __future__ import annotations

import math
from pathlib import Path

import torch

try:
    from . import _C_legacy as _C
except ImportError as exc:  # pragma: no cover - import error is surfaced to the user.
    _C = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


_FORWARD_PAD_MULTIPLE = 128
_BACKWARD_PAD_MULTIPLE = 128


def _ensure_extension() -> None:
    if _C is None:
        build_hint = Path(__file__).resolve().parent
        raise ImportError(
            f"tk_fa4 deprecated extension is not built. Run `make` in {build_hint}."
        ) from _IMPORT_ERROR


def _check_cuda_bf16(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16")
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape (batch, seqlen, heads, head_dim)")


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    _check_cuda_bf16(q, "q")
    _check_cuda_bf16(k, "k")
    _check_cuda_bf16(v, "v")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("batch dimensions must match")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("sequence lengths must match")
    if k.shape[2] != v.shape[2]:
        raise ValueError("k and v must have the same number of heads")
    if q.shape[2] < k.shape[2] or q.shape[2] % k.shape[2] != 0:
        raise ValueError("q heads must be a multiple of kv heads")
    if q.shape[3] != k.shape[3] or q.shape[3] != v.shape[3]:
        raise ValueError("q, k, and v must have the same head_dim")
    if q.shape[3] not in (64, 128):
        raise ValueError("head_dim must be 64 or 128")
    major, minor = torch.cuda.get_device_capability(q.device)
    if (major, minor) != (10, 0):
        raise RuntimeError(
            f"tk_fa4 is currently targeted at GB200 / SM100, got compute capability {(major, minor)}"
        )


def _pad_bshd(x: torch.Tensor, multiple: int) -> torch.Tensor:
    pad = (-x.shape[1]) % multiple
    if pad == 0:
        return x.contiguous()
    pad_tensor = torch.zeros(
        (x.shape[0], pad, x.shape[2], x.shape[3]),
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat((x, pad_tensor), dim=1).contiguous()


def _pad_bsh(x: torch.Tensor, multiple: int) -> torch.Tensor:
    pad = (-x.shape[1]) % multiple
    if pad == 0:
        return x.contiguous()
    pad_tensor = torch.zeros(
        (x.shape[0], pad, x.shape[2]),
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat((x, pad_tensor), dim=1).contiguous()


def _to_bhsd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def _from_bhsd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def _resolve_scale(q: torch.Tensor, softmax_scale: float | None) -> float:
    return float(softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5)


def mha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    return_lse: bool = False,
):
    _ensure_extension()
    _check_inputs(q, k, v)
    softmax_scale = _resolve_scale(q, softmax_scale)
    seqlen = q.shape[1]

    q_pad = _pad_bshd(q, _FORWARD_PAD_MULTIPLE)
    k_pad = _pad_bshd(k, _FORWARD_PAD_MULTIPLE)
    v_pad = _pad_bshd(v, _FORWARD_PAD_MULTIPLE)

    out_bhsd, l_aux_bhs1 = _C.mha_fwd(
        _to_bhsd(q_pad),
        _to_bhsd(k_pad),
        _to_bhsd(v_pad),
        causal,
        softmax_scale,
        seqlen,
    )
    out = _from_bhsd(out_bhsd[:, :, :seqlen, :])
    l_aux = l_aux_bhs1[:, :, 0, :seqlen].permute(0, 2, 1).contiguous()
    if return_lse:
        return out, (-l_aux) * softmax_scale
    return out


def mha_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    l_aux: torch.Tensor,
    dout: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    deterministic: bool = False,
):
    _ensure_extension()
    _check_inputs(q, k, v)
    _check_cuda_bf16(out, "out")
    _check_cuda_bf16(dout, "dout")
    if out.shape != q.shape or dout.shape != q.shape:
        raise ValueError("out and dout must match q shape")
    if l_aux.shape != q.shape[:3]:
        raise ValueError("l_aux must have shape (batch, seqlen, heads)")

    softmax_scale = _resolve_scale(q, softmax_scale)
    seqlen = q.shape[1]

    q_pad = _pad_bshd(q, _BACKWARD_PAD_MULTIPLE)
    k_pad = _pad_bshd(k, _BACKWARD_PAD_MULTIPLE)
    v_pad = _pad_bshd(v, _BACKWARD_PAD_MULTIPLE)
    out_pad = _pad_bshd(out, _BACKWARD_PAD_MULTIPLE)
    dout_pad = _pad_bshd(dout, _BACKWARD_PAD_MULTIPLE)
    l_aux_pad = _pad_bsh(l_aux, _BACKWARD_PAD_MULTIPLE).permute(0, 2, 1).contiguous().unsqueeze(2)

    dq, dk, dv = _C.mha_bwd(
        _to_bhsd(q_pad),
        _to_bhsd(k_pad),
        _to_bhsd(v_pad),
        _to_bhsd(out_pad),
        l_aux_pad,
        _to_bhsd(dout_pad),
        causal,
        softmax_scale,
        seqlen,
        deterministic,
    )
    dq = _from_bhsd(dq[:, :, :seqlen, :]).to(dtype=q.dtype)
    dk = _from_bhsd(dk[:, :, :seqlen, :]).to(dtype=k.dtype)
    dv = _from_bhsd(dv[:, :, :seqlen, :]).to(dtype=v.dtype)
    return dq, dk, dv


class _FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
        deterministic: bool,
    ):
        _ensure_extension()
        _check_inputs(q, k, v)
        softmax_scale = _resolve_scale(q, softmax_scale)
        seqlen = q.shape[1]

        q_pad = _pad_bshd(q, _FORWARD_PAD_MULTIPLE)
        k_pad = _pad_bshd(k, _FORWARD_PAD_MULTIPLE)
        v_pad = _pad_bshd(v, _FORWARD_PAD_MULTIPLE)

        out_bhsd, l_aux_bhs1 = _C.mha_fwd(
            _to_bhsd(q_pad),
            _to_bhsd(k_pad),
            _to_bhsd(v_pad),
            causal,
            softmax_scale,
            seqlen,
        )
        out = _from_bhsd(out_bhsd[:, :, :seqlen, :])
        l_aux = l_aux_bhs1[:, :, 0, :seqlen].permute(0, 2, 1).contiguous()
        lse = (-l_aux) * softmax_scale
        ctx.save_for_backward(q, k, v, out, l_aux)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, dlse: torch.Tensor | None):
        q, k, v, out, l_aux = ctx.saved_tensors
        dq, dk, dv = mha_bwd(
            q,
            k,
            v,
            out,
            l_aux,
            dout.contiguous(),
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
):
    out, lse = _FlashAttnFunc.apply(q, k, v, softmax_scale, causal, deterministic)
    return (out, lse) if return_lse else out
