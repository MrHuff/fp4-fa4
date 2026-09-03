from __future__ import annotations

import math
import warnings
from pathlib import Path

import torch

try:
    from . import _C
except ImportError as exc:  # pragma: no cover - surfaced when the user calls the API.
    _C = None
    _BWD_IMPORT_ERROR = exc
else:
    _BWD_IMPORT_ERROR = None

try:
    from . import _C_b300_causal
except ImportError as exc:  # pragma: no cover - surfaced when the user calls the API.
    _C_b300_causal = None
    _CAUSAL_IMPORT_ERROR = exc
else:
    _CAUSAL_IMPORT_ERROR = None

try:
    from . import _C_b300_causal_bf16_baseline
except ImportError as exc:  # pragma: no cover - surfaced when the user calls the API.
    _C_b300_causal_bf16_baseline = None
    _CAUSAL_BF16_IMPORT_ERROR = exc
else:
    _CAUSAL_BF16_IMPORT_ERROR = None

try:
    from . import _C_b300_noncausal
except ImportError as exc:  # pragma: no cover - surfaced when the user calls the API.
    _C_b300_noncausal = None
    _NONCAUSAL_IMPORT_ERROR = exc
else:
    _NONCAUSAL_IMPORT_ERROR = None


_PAD_MULTIPLE = 128
_EXPERIMENTAL_PAD_MULTIPLE = 256
_QK_HEAD_DIM = 192
_V_HEAD_DIM = 128
_MIN_SEQ_LEN = 2048
_CAUSAL_PERSISTENT_MAX_SEQ = 4096


def _ensure_backward_extension() -> None:
    if _C is not None:
        return
    build_hint = Path(__file__).resolve().parent
    raise ImportError(
        f"tk_fa4 backward extension is not built. Run `make` in {build_hint}."
    ) from _BWD_IMPORT_ERROR


def _ensure_forward_extensions() -> None:
    if _C_b300_causal_bf16_baseline is None or _C_b300_noncausal is None:
        build_hint = Path(__file__).resolve().parent
        missing = []
        if _C_b300_causal_bf16_baseline is None:
            missing.append("causal_bf16_baseline")
        if _C_b300_noncausal is None:
            missing.append("noncausal")
        raise ImportError(
            f"tk_fa4 forward extension(s) missing for {', '.join(missing)}. Run `make` in {build_hint}."
        ) from (_CAUSAL_BF16_IMPORT_ERROR or _NONCAUSAL_IMPORT_ERROR)


def _check_cuda_bf16_bshd(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16")
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape (batch, seqlen, heads, head_dim)")


def _check_cuda_bf16_bhsd(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16")
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape (batch, heads, seqlen, head_dim)")


def _check_sm100(device: torch.device) -> None:
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        raise RuntimeError(
            f"tk_fa4 exact B300 path requires GB200 / SM100, got compute capability {(major, minor)}"
        )


def _check_exact_qkv_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    _check_cuda_bf16_bshd(q, "q")
    _check_cuda_bf16_bshd(k, "k")
    _check_cuda_bf16_bshd(v, "v")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("batch dimensions must match")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("sequence lengths must match")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise ValueError("exact B300 path requires equal q, k, and v head counts")
    if q.shape[3] != _QK_HEAD_DIM or k.shape[3] != _QK_HEAD_DIM:
        raise ValueError(f"q and k head_dim must be {_QK_HEAD_DIM}")
    if v.shape[3] != _V_HEAD_DIM:
        raise ValueError(f"v head_dim must be {_V_HEAD_DIM}")
    if q.shape[1] < _MIN_SEQ_LEN:
        raise ValueError(f"exact B300 path requires seqlen >= {_MIN_SEQ_LEN}")
    _check_sm100(q.device)


def _check_exact_out(x: torch.Tensor, reference: torch.Tensor, name: str) -> None:
    _check_cuda_bf16_bshd(x, name)
    if x.shape != reference.shape:
        raise ValueError(f"{name} must match v/out shape")
    if x.device != reference.device:
        raise ValueError(f"{name} must be on the same CUDA device as q")


def _normalize_lse_bsh(lse: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if not lse.is_cuda:
        raise ValueError("lse must be a CUDA tensor")
    if lse.dtype != torch.float32:
        raise ValueError("lse must be float32")
    if lse.device != q.device:
        raise ValueError("lse must be on the same CUDA device as q")
    if lse.shape == q.shape[:3]:
        return lse.contiguous()
    if lse.ndim == 3 and lse.shape == (q.shape[0], q.shape[2], q.shape[1]):
        return lse.permute(0, 2, 1).contiguous()
    raise ValueError("lse must have shape (batch, seqlen, heads)")


def _pad_bshd(x: torch.Tensor, multiple: int = _PAD_MULTIPLE) -> torch.Tensor:
    pad = (-x.shape[1]) % multiple
    if pad == 0:
        return x.contiguous()
    pad_tensor = torch.zeros(
        (x.shape[0], pad, x.shape[2], x.shape[3]),
        dtype=x.dtype,
        device=x.device,
    )
    return torch.cat((x, pad_tensor), dim=1).contiguous()


def _pad_bsh(x: torch.Tensor, value: float = 0.0, multiple: int = _PAD_MULTIPLE) -> torch.Tensor:
    pad = (-x.shape[1]) % multiple
    if pad == 0:
        return x.contiguous()
    pad_tensor = torch.full(
        (x.shape[0], pad, x.shape[2]),
        fill_value=value,
        dtype=x.dtype,
        device=x.device,
    )
    return torch.cat((x, pad_tensor), dim=1).contiguous()


def _to_bhsd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def _from_bhsd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def _maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _try_hot_fastpath(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not causal or deterministic:
        return None
    if not (q.is_cuda and k.is_cuda and v.is_cuda and out.is_cuda and lse.is_cuda and dout.is_cuda):
        return None
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
        or dout.dtype != torch.bfloat16
        or lse.dtype != torch.float32
    ):
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or out.ndim != 4 or dout.ndim != 4 or lse.ndim != 3:
        return None
    if q.device != k.device or q.device != v.device or q.device != out.device or q.device != lse.device or q.device != dout.device:
        return None
    q_shape = q.shape
    v_shape = v.shape
    if q_shape != k.shape or out.shape != v_shape or dout.shape != v_shape:
        return None
    batch, seqlen, heads, q_dim = q_shape
    if (v_shape[0], v_shape[1], v_shape[2]) != (batch, seqlen, heads):
        return None
    if q_dim != _QK_HEAD_DIM or v_shape[3] != _V_HEAD_DIM:
        return None
    if seqlen < _MIN_SEQ_LEN or seqlen % _EXPERIMENTAL_PAD_MULTIPLE != 0:
        return None
    if lse.shape != q_shape[:3]:
        return None
    default_scale = q_dim ** -0.5
    if softmax_scale is not None and not math.isclose(float(softmax_scale), float(default_scale), rel_tol=0.0, abs_tol=1e-7):
        return None
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous() and lse.is_contiguous() and dout.is_contiguous()):
        return None
    return _C.b300_mha_bwd_hot_cute16_candidate_internal(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        float(default_scale if softmax_scale is None else softmax_scale),
        seqlen,
        deterministic,
    )


def _resolve_scale(q: torch.Tensor, softmax_scale: float | None) -> float:
    default_scale = q.shape[-1] ** -0.5
    if softmax_scale is None:
        return float(default_scale)
    if not math.isclose(float(softmax_scale), float(default_scale), rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(
            f"exact B300 path only supports softmax_scale={default_scale} for head_dim={q.shape[-1]}"
        )
    return float(softmax_scale)


def _select_forward_kernel(causal: bool, seqlen: int):
    if not causal:
        return _C_b300_noncausal.forward
    if seqlen <= _CAUSAL_PERSISTENT_MAX_SEQ:
        return _C_b300_causal_bf16_baseline.forward_persistent
    return _C_b300_causal_bf16_baseline.forward


def _experimental_hot_supported(seqlen: int, causal: bool, deterministic: bool) -> bool:
    return causal and (not deterministic) and seqlen % _EXPERIMENTAL_PAD_MULTIPLE == 0


def _resolve_experimental_impl(
    implementation: str,
    seqlen: int,
    causal: bool,
    deterministic: bool,
) -> str:
    if implementation not in {"auto", "ref", "hot"}:
        raise ValueError("implementation must be one of 'auto', 'ref', or 'hot'")
    if implementation == "ref":
        return "ref"
    if implementation == "hot":
        if not _experimental_hot_supported(seqlen, causal, deterministic):
            raise ValueError(
                "CuTe16 hot mode not implemented yet; current stage only supports causal=True, deterministic=False with seqlen divisible by 256"
            )
        return "hot"
    return "ref"


def _lse_to_l_aux(lse: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    l_aux = (-lse) / softmax_scale
    l_aux_pad = _pad_bsh(l_aux)
    return l_aux_pad.permute(0, 2, 1).contiguous().unsqueeze(2)


def _lse_to_bh1s(lse: torch.Tensor, multiple: int) -> torch.Tensor:
    return _pad_bsh(lse, multiple=multiple).permute(0, 2, 1).contiguous().unsqueeze(2)


def b300_mha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    return_lse: bool = False,
):
    _ensure_forward_extensions()
    _check_exact_qkv_inputs(q, k, v)
    _resolve_scale(q, softmax_scale)

    seqlen = q.shape[1]
    q_pad = _pad_bshd(q)
    k_pad = _pad_bshd(k)
    v_pad = _pad_bshd(v)

    out_pad = torch.empty_like(v_pad)
    lse_pad = torch.empty(
        (q_pad.shape[0], q_pad.shape[2], 1, q_pad.shape[1]),
        dtype=torch.float32,
        device=q.device,
    )
    _select_forward_kernel(causal, seqlen)(q_pad, k_pad, v_pad, out_pad, lse_pad)

    out = out_pad[:, :seqlen].contiguous()
    lse = lse_pad[:, :, 0, :seqlen].permute(0, 2, 1).contiguous()
    return (out, lse) if return_lse else out


def b300_mha_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    deterministic: bool = False,
):
    _ensure_backward_extension()
    _check_exact_qkv_inputs(q, k, v)
    _check_exact_out(out, v, "out")
    _check_exact_out(dout, v, "dout")
    lse = _normalize_lse_bsh(lse, q)

    softmax_scale = _resolve_scale(q, softmax_scale)
    seqlen = q.shape[1]

    q_pad = _to_bhsd(_pad_bshd(q))
    k_pad = _to_bhsd(_pad_bshd(k))
    v_pad = _to_bhsd(_pad_bshd(v))
    out_pad = _to_bhsd(_pad_bshd(out))
    dout_pad = _to_bhsd(_pad_bshd(dout))
    l_aux_pad = _lse_to_l_aux(lse, softmax_scale)

    dq, dk, dv = _C.b300_mha_bwd(
        q_pad,
        k_pad,
        v_pad,
        out_pad,
        l_aux_pad,
        dout_pad,
        causal,
        softmax_scale,
        seqlen,
        deterministic,
    )
    dq = _from_bhsd(dq[:, :, :seqlen, :])
    dk = _from_bhsd(dk[:, :, :seqlen, :])
    dv = _from_bhsd(dv[:, :, :seqlen, :])
    return dq, dk, dv


def b300_mha_bwd_dv_only(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    deterministic: bool = False,
):
    _ensure_backward_extension()
    _check_cuda_bf16_bshd(q, "q")
    _check_cuda_bf16_bshd(k, "k")
    _check_cuda_bf16_bshd(dout, "dout")
    if q.device != k.device or q.device != dout.device:
        raise ValueError("q, k, and dout must be on the same CUDA device")
    if q.shape[0] != k.shape[0] or q.shape[0] != dout.shape[0]:
        raise ValueError("batch dimensions must match")
    if q.shape[1] != k.shape[1] or q.shape[1] != dout.shape[1]:
        raise ValueError("sequence lengths must match")
    if q.shape[2] != k.shape[2] or q.shape[2] != dout.shape[2]:
        raise ValueError("head counts must match")
    if q.shape[3] != _QK_HEAD_DIM or k.shape[3] != _QK_HEAD_DIM:
        raise ValueError(f"q and k head_dim must be {_QK_HEAD_DIM}")
    if dout.shape[3] != _V_HEAD_DIM:
        raise ValueError(f"dout head_dim must be {_V_HEAD_DIM}")
    if q.shape[1] < _MIN_SEQ_LEN:
        raise ValueError(f"exact B300 path requires seqlen >= {_MIN_SEQ_LEN}")
    _check_sm100(q.device)
    lse = _normalize_lse_bsh(lse, q)

    softmax_scale = _resolve_scale(q, softmax_scale)
    seqlen = q.shape[1]

    q_pad = _to_bhsd(_pad_bshd(q))
    k_pad = _to_bhsd(_pad_bshd(k))
    dout_pad = _to_bhsd(_pad_bshd(dout))
    l_aux_pad = _lse_to_l_aux(lse, softmax_scale)

    dv = _C.b300_mha_bwd_dv_only_internal(
        q_pad,
        k_pad,
        dout_pad,
        l_aux_pad,
        causal,
        softmax_scale,
        seqlen,
        deterministic,
    )
    return _from_bhsd(dv[:, :, :seqlen, :])


def b300_mha_bwd_experimental(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    implementation: str = "auto",
):
    _ensure_backward_extension()
    if implementation == "hot":
        hot_fastpath = _try_hot_fastpath(
            q,
            k,
            v,
            out,
            lse,
            dout,
            causal=causal,
            softmax_scale=softmax_scale,
            deterministic=deterministic,
        )
        if hot_fastpath is not None:
            return hot_fastpath
    _check_exact_qkv_inputs(q, k, v)
    _check_exact_out(out, v, "out")
    _check_exact_out(dout, v, "dout")
    lse = _normalize_lse_bsh(lse, q)

    softmax_scale = _resolve_scale(q, softmax_scale)
    seqlen = q.shape[1]
    implementation = _resolve_experimental_impl(implementation, seqlen, causal, deterministic)

    if implementation == "hot":
        dq, dk, dv = _C.b300_mha_bwd_hot_cute16_candidate_internal(
            _maybe_contiguous(q),
            _maybe_contiguous(k),
            _maybe_contiguous(v),
            _maybe_contiguous(out),
            _maybe_contiguous(lse),
            _maybe_contiguous(dout),
            causal,
            softmax_scale,
            seqlen,
            deterministic,
        )
        return dq[:, :seqlen, :, :], dk[:, :seqlen, :, :], dv[:, :seqlen, :, :]

    q_pad = _to_bhsd(_pad_bshd(q, multiple=_EXPERIMENTAL_PAD_MULTIPLE))
    k_pad = _to_bhsd(_pad_bshd(k, multiple=_EXPERIMENTAL_PAD_MULTIPLE))
    v_pad = _to_bhsd(_pad_bshd(v, multiple=_EXPERIMENTAL_PAD_MULTIPLE))
    out_pad = _to_bhsd(_pad_bshd(out, multiple=_EXPERIMENTAL_PAD_MULTIPLE))
    dout_pad = _to_bhsd(_pad_bshd(dout, multiple=_EXPERIMENTAL_PAD_MULTIPLE))
    lse_pad = _lse_to_bh1s(lse, _EXPERIMENTAL_PAD_MULTIPLE)

    if implementation == "hot":
        raise AssertionError("unreachable")
    else:
        dq, dk, dv = _C.b300_mha_bwd_fa4_style_ref(
            q_pad,
            k_pad,
            v_pad,
            out_pad,
            lse_pad,
            dout_pad,
            causal,
            softmax_scale,
            seqlen,
            deterministic,
        )
    dq = _from_bhsd(dq[:, :, :seqlen, :])
    dk = _from_bhsd(dk[:, :, :seqlen, :])
    dv = _from_bhsd(dv[:, :, :seqlen, :])
    return dq, dk, dv


class _B300FlashAttnFunc(torch.autograd.Function):
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
        out, lse = b300_mha_fwd(
            q,
            k,
            v,
            causal=causal,
            softmax_scale=softmax_scale,
            return_lse=True,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = _resolve_scale(q, softmax_scale)
        ctx.causal = causal
        ctx.deterministic = deterministic
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, dlse: torch.Tensor | None):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = b300_mha_bwd(
            q,
            k,
            v,
            out,
            lse,
            dout.contiguous(),
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None


class _B300FlashAttnFuncExperimental(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
        deterministic: bool,
        implementation: str,
    ):
        out, lse = b300_mha_fwd(
            q,
            k,
            v,
            causal=causal,
            softmax_scale=softmax_scale,
            return_lse=True,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = _resolve_scale(q, softmax_scale)
        ctx.causal = causal
        ctx.deterministic = deterministic
        ctx.implementation = implementation
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, dlse: torch.Tensor | None):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = b300_mha_bwd_experimental(
            q,
            k,
            v,
            out,
            lse,
            dout.contiguous(),
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
            deterministic=ctx.deterministic,
            implementation=ctx.implementation,
        )
        return dq, dk, dv, None, None, None, None


def b300_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
):
    out, lse = _B300FlashAttnFunc.apply(q, k, v, softmax_scale, causal, deterministic)
    return (out, lse) if return_lse else out


def b300_flash_attn_func_experimental(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
    implementation: str = "auto",
):
    out, lse = _B300FlashAttnFuncExperimental.apply(
        q, k, v, softmax_scale, causal, deterministic, implementation
    )
    return (out, lse) if return_lse else out


def _warn_deprecated(name: str) -> None:
    warnings.warn(
        f"`tk_fa4.{name}` is deprecated and still routes through the legacy broad-shape implementation in `tk_fa4.deprecated`. "
        "Use `b300_mha_fwd`, `b300_mha_bwd`, or `b300_flash_attn_func` for the exact B300 fast path.",
        DeprecationWarning,
        stacklevel=2,
    )


def mha_fwd(*args, **kwargs):
    _warn_deprecated("mha_fwd")
    from .deprecated.interface import mha_fwd as _legacy_mha_fwd

    return _legacy_mha_fwd(*args, **kwargs)


def mha_bwd(*args, **kwargs):
    _warn_deprecated("mha_bwd")
    from .deprecated.interface import mha_bwd as _legacy_mha_bwd

    return _legacy_mha_bwd(*args, **kwargs)


def flash_attn_func(*args, **kwargs):
    _warn_deprecated("flash_attn_func")
    from .deprecated.interface import flash_attn_func as _legacy_flash_attn_func

    return _legacy_flash_attn_func(*args, **kwargs)
