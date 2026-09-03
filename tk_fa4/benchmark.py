from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FLASH_ROOT = _REPO_ROOT / "flash-attention"
# Use vendored flash-attention, but keep the installed CUTLASS DSL runtime.
# The vendored CuTeDSL Python tree does not include the built `_mlir` extension.
for _path in (_FLASH_ROOT, _REPO_ROOT):
    _path_str = str(_path)
    if _path_str in sys.path:
        sys.path.remove(_path_str)
    sys.path.insert(0, _path_str)

from tk_fa4 import (
    b300_mha_bwd,
    b300_mha_bwd_experimental,
    b300_mha_fwd,
)
from tk_fa4.interface import _C, _try_hot_fastpath

try:
    import flash_attn

    _FLASH_ATTN_PATH = Path(flash_attn.__file__).resolve()
    if not _FLASH_ATTN_PATH.is_relative_to(_FLASH_ROOT):
        raise RuntimeError(
            "benchmark.py must import vendored flash_attn from "
            f"{_FLASH_ROOT}, got {_FLASH_ATTN_PATH}"
        )
    from flash_attn.cute.interface import _flash_attn_bwd as cute_flash_attn_bwd
    from flash_attn.cute.interface import flash_attn_func as cute_flash_attn_func
    _CUTE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment-dependent import
    cute_flash_attn_bwd = None
    cute_flash_attn_func = None
    _CUTE_IMPORT_ERROR = exc

_CUTE_DEPS_HINT = (
    "Install the CUTE FA4 runtime dependencies for this Python, e.g. "
    "python -m pip install 'nvidia-cutlass-dsl>=4.4.2' "
    "'quack-kernels>=0.3.0' 'apache-tvm-ffi>=0.1.5,<0.2' "
    "torch-c-dlpack-ext cuda-python"
)


QK_DIM = 192
V_DIM = 128
_BENCH_DEVICE_INDEX = 0


@dataclass(frozen=True)
class Shape:
    batch: int
    seqlen: int
    heads: int
    causal: bool


@dataclass(frozen=True)
class BackwardInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor
    dout: torch.Tensor


def _clone_backward_inputs(inputs: BackwardInputs) -> BackwardInputs:
    return BackwardInputs(
        q=inputs.q.clone(),
        k=inputs.k.clone(),
        v=inputs.v.clone(),
        out=inputs.out.clone(),
        lse=inputs.lse.clone(),
        dout=inputs.dout.clone(),
    )


def _restore_backward_inputs(dst: BackwardInputs, src: BackwardInputs) -> None:
    with torch.no_grad():
        dst.q.copy_(src.q)
        dst.k.copy_(src.k)
        dst.v.copy_(src.v)
        dst.out.copy_(src.out)
        dst.lse.copy_(src.lse)
        dst.dout.copy_(src.dout)


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x]


def _parse_csv_bools(raw: str) -> list[bool]:
    return [bool(int(x)) for x in raw.split(",") if x]


def _parse_csv_strings(raw: str) -> list[str]:
    return [x for x in raw.split(",") if x]


def _same_input_uses_cute_forward(backend: str) -> bool:
    return backend == "cute"


def _make_inputs(shape: Shape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = f"cuda:{_BENCH_DEVICE_INDEX}"
    q = torch.randn(shape.batch, shape.seqlen, shape.heads, QK_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape.batch, shape.seqlen, shape.heads, QK_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape.batch, shape.seqlen, shape.heads, V_DIM, device=device, dtype=torch.bfloat16)
    dout = torch.randn(shape.batch, shape.seqlen, shape.heads, V_DIM, device=device, dtype=torch.bfloat16)
    return q, k, v, dout


def _require_cute() -> None:
    if _CUTE_IMPORT_ERROR is not None:
        raise RuntimeError(
            f"CuTe backend unavailable: {_CUTE_IMPORT_ERROR}. {_CUTE_DEPS_HINT}"
        ) from _CUTE_IMPORT_ERROR


def _make_forward_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cute_flash_attn_func is not None:
        return cute_flash_attn_func(q, k, v, causal=causal, return_lse=True)
    return b300_mha_fwd(q, k, v, causal=causal, return_lse=True)


def _make_backward_inputs(shape: Shape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v, dout = _make_inputs(shape)
    out, lse = _make_forward_with_lse(q, k, v, causal=shape.causal)
    return q, k, v, out, lse, dout


def _make_same_input_backward_inputs(shape: Shape, seed: int, backend: str) -> BackwardInputs:
    devices = [torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        q, k, v, dout = _make_inputs(shape)
        if _same_input_uses_cute_forward(backend):
            _require_cute()
            out, lse = cute_flash_attn_func(q, k, v, causal=shape.causal, return_lse=True)
        else:
            out, lse = b300_mha_fwd(q, k, v, causal=shape.causal, return_lse=True)
    if not _same_input_uses_cute_forward(backend) and lse.ndim == 3 and lse.shape == (shape.batch, shape.heads, shape.seqlen):
        lse = lse.permute(0, 2, 1).contiguous()
    return BackwardInputs(q=q, k=k, v=v, out=out, lse=lse, dout=dout)


def _experimental_hot_supported(shape: Shape, deterministic: bool) -> bool:
    return not deterministic and shape.seqlen % 256 == 0


def _hot_scale(q: torch.Tensor) -> float:
    return float(q.shape[-1] ** -0.5)


def _run_same_input_backward_backend(
    backend: str,
    inputs: BackwardInputs,
    shape: Shape,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = _hot_scale(inputs.q)
    if backend == "tk_experimental_hot":
        return b300_mha_bwd_experimental(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.out,
            inputs.lse,
            inputs.dout,
            causal=shape.causal,
            deterministic=deterministic,
            implementation="hot",
        )
    if backend == "tk_experimental_hot_trusted":
        return _C.b300_mha_bwd_hot_trusted_internal(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.out,
            inputs.lse,
            inputs.dout,
            shape.causal,
            scale,
            inputs.q.shape[1],
            deterministic,
        )
    if backend == "tk_experimental_hot_candidate":
        return _C.b300_mha_bwd_hot_cute16_candidate_internal(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.out,
            inputs.lse,
            inputs.dout,
            shape.causal,
            scale,
            inputs.q.shape[1],
            deterministic,
        )
    if backend == "tk_experimental_hot_candidate2":
        return _C.b300_mha_bwd_hot_cute16_candidate2_internal(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.out,
            inputs.lse,
            inputs.dout,
            shape.causal,
            scale,
            inputs.q.shape[1],
            deterministic,
        )
    if backend == "tk_experimental_ref":
        return b300_mha_bwd_experimental(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.out,
            inputs.lse,
            inputs.dout,
            causal=shape.causal,
            deterministic=deterministic,
            implementation="ref",
        )
    if backend == "cute":
        _require_cute()
        return cute_flash_attn_bwd(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.out,
            inputs.dout,
            inputs.lse,
            causal=shape.causal,
            deterministic=deterministic,
        )
    raise ValueError(f"unsupported same-input backward backend: {backend}")


def _make_backward_output_tensors(
    inputs: BackwardInputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(inputs.q, dtype=inputs.lse.dtype),
        torch.empty_like(inputs.k, dtype=inputs.lse.dtype),
        torch.empty_like(inputs.v, dtype=inputs.lse.dtype),
    )


def _run_same_input_backward_backend_fixedout(
    backend: str,
    inputs: BackwardInputs,
    shape: Shape,
    deterministic: bool,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = _hot_scale(inputs.q)
    if backend == "tk_experimental_hot_candidate":
        return _C.b300_mha_bwd_hot_cute16_candidate_out_internal(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.out,
            inputs.lse,
            inputs.dout,
            dq,
            dk,
            dv,
            shape.causal,
            scale,
            inputs.q.shape[1],
            deterministic,
        )
    if backend == "tk_experimental_hot_candidate2":
        return _C.b300_mha_bwd_hot_cute16_candidate2_out_internal(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.out,
            inputs.lse,
            inputs.dout,
            dq,
            dk,
            dv,
            shape.causal,
            scale,
            inputs.q.shape[1],
            deterministic,
        )
    raise ValueError(f"unsupported fixed-output backward backend: {backend}")


def _same_input_backward_time(
    backend: str,
    inputs: BackwardInputs,
    shape: Shape,
    warmup: int,
    iters: int,
    deterministic: bool,
) -> float:
    return _time_cuda_op(
        lambda: _run_same_input_backward_backend(backend, inputs, shape, deterministic),
        warmup,
        iters,
    )


def _same_input_backward_time_fixedout(
    backend: str,
    inputs: BackwardInputs,
    shape: Shape,
    warmup: int,
    iters: int,
    deterministic: bool,
) -> float:
    dq, dk, dv = _make_backward_output_tensors(inputs)

    def run() -> None:
        with torch.no_grad():
            dq.zero_()
            dk.zero_()
            dv.zero_()
        _run_same_input_backward_backend_fixedout(
            backend,
            inputs,
            shape,
            deterministic,
            dq,
            dk,
            dv,
        )

    return _time_cuda_op(run, warmup, iters)


def _same_input_backward_parity(
    backend: str,
    inputs: BackwardInputs,
    workspace: BackwardInputs,
    shape: Shape,
    deterministic: bool,
) -> tuple[bool, float, float, float]:
    _restore_backward_inputs(workspace, inputs)
    out = _run_same_input_backward_backend(backend, workspace, shape, deterministic)
    torch.cuda.synchronize()
    _restore_backward_inputs(workspace, inputs)
    ref = _run_same_input_backward_backend("tk_experimental_ref", workspace, shape, deterministic)
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all() and torch.isfinite(out[2]).all())
    return (
        finite,
        (out[0] - ref[0]).abs().max().item(),
        (out[1] - ref[1]).abs().max().item(),
        (out[2] - ref[2]).abs().max().item(),
    )


def _same_input_hot_fastpath_taken(
    inputs: BackwardInputs,
    shape: Shape,
    deterministic: bool,
) -> bool:
    return _try_hot_fastpath(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.out,
        inputs.lse,
        inputs.dout,
        causal=shape.causal,
        softmax_scale=None,
        deterministic=deterministic,
    ) is not None


def _same_input_backward_repeat(
    backend: str,
    inputs: BackwardInputs,
    workspace: BackwardInputs,
    shape: Shape,
    deterministic: bool,
) -> tuple[bool, float, float, float]:
    _restore_backward_inputs(workspace, inputs)
    out0 = _run_same_input_backward_backend(backend, workspace, shape, deterministic)
    torch.cuda.synchronize()
    _restore_backward_inputs(workspace, inputs)
    out1 = _run_same_input_backward_backend(backend, workspace, shape, deterministic)
    torch.cuda.synchronize()
    finite = bool(
        torch.isfinite(out0[0]).all() and torch.isfinite(out0[1]).all() and torch.isfinite(out0[2]).all() and
        torch.isfinite(out1[0]).all() and torch.isfinite(out1[1]).all() and torch.isfinite(out1[2]).all()
    )
    return (
        finite,
        (out0[0] - out1[0]).abs().max().item(),
        (out0[1] - out1[1]).abs().max().item(),
        (out0[2] - out1[2]).abs().max().item(),
    )


def _same_input_backward_repeat_fixedout(
    backend: str,
    inputs: BackwardInputs,
    workspace: BackwardInputs,
    shape: Shape,
    deterministic: bool,
) -> tuple[bool, float, float, float]:
    dq, dk, dv = _make_backward_output_tensors(inputs)
    with torch.no_grad():
        dq.zero_()
        dk.zero_()
        dv.zero_()
    _restore_backward_inputs(workspace, inputs)
    out0 = _run_same_input_backward_backend_fixedout(backend, workspace, shape, deterministic, dq, dk, dv)
    out0 = (out0[0].clone(), out0[1].clone(), out0[2].clone())
    torch.cuda.synchronize()
    with torch.no_grad():
        dq.zero_()
        dk.zero_()
        dv.zero_()
    _restore_backward_inputs(workspace, inputs)
    out1 = _run_same_input_backward_backend_fixedout(backend, workspace, shape, deterministic, dq, dk, dv)
    out1 = (out1[0].clone(), out1[1].clone(), out1[2].clone())
    torch.cuda.synchronize()
    finite = bool(
        torch.isfinite(out0[0]).all() and torch.isfinite(out0[1]).all() and torch.isfinite(out0[2]).all() and
        torch.isfinite(out1[0]).all() and torch.isfinite(out1[1]).all() and torch.isfinite(out1[2]).all()
    )
    return (
        finite,
        (out0[0] - out1[0]).abs().max().item(),
        (out0[1] - out1[1]).abs().max().item(),
        (out0[2] - out1[2]).abs().max().item(),
    )


def _time_cuda_op(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _bench_forward_tk(shape: Shape, warmup: int, iters: int) -> float:
    q, k, v, _ = _make_inputs(shape)
    return _time_cuda_op(
        lambda: b300_mha_fwd(q, k, v, causal=shape.causal),
        warmup,
        iters,
    )


def _bench_forward_cute(shape: Shape, warmup: int, iters: int) -> float:
    _require_cute()
    q, k, v, _ = _make_inputs(shape)
    return _time_cuda_op(
        lambda: cute_flash_attn_func(q, k, v, causal=shape.causal),
        warmup,
        iters,
    )


def _bench_backward_tk(shape: Shape, warmup: int, iters: int, deterministic: bool) -> float:
    q, k, v, out, lse, dout = _make_backward_inputs(shape)
    return _time_cuda_op(
        lambda: b300_mha_bwd(
            q,
            k,
            v,
            out,
            lse,
            dout,
            causal=shape.causal,
            deterministic=deterministic,
        ),
        warmup,
        iters,
    )


def _bench_backward_tk_experimental(shape: Shape, warmup: int, iters: int, deterministic: bool) -> float:
    return _bench_backward_tk_experimental_impl(shape, warmup, iters, deterministic, implementation="auto")


def _bench_backward_tk_experimental_impl(
    shape: Shape,
    warmup: int,
    iters: int,
    deterministic: bool,
    implementation: str,
) -> float:
    q, k, v, out, lse, dout = _make_backward_inputs(shape)
    return _time_cuda_op(
        lambda: b300_mha_bwd_experimental(
            q,
            k,
            v,
            out,
            lse,
            dout,
            causal=shape.causal,
            deterministic=deterministic,
            implementation=implementation,
        ),
        warmup,
        iters,
    )


def _bench_backward_tk_experimental_candidate(shape: Shape, warmup: int, iters: int, deterministic: bool) -> float:
    q, k, v, out, lse, dout = _make_backward_inputs(shape)
    if lse.ndim == 3 and lse.shape == (shape.batch, shape.heads, shape.seqlen):
        lse = lse.permute(0, 2, 1).contiguous()
    scale = float(q.shape[-1] ** -0.5)
    return _time_cuda_op(
        lambda: _C.b300_mha_bwd_hot_cute16_candidate_internal(
            q,
            k,
            v,
            out,
            lse,
            dout,
            shape.causal,
            scale,
            q.shape[1],
            deterministic,
        ),
        warmup,
        iters,
    )


def _bench_backward_tk_experimental_candidate2(shape: Shape, warmup: int, iters: int, deterministic: bool) -> float:
    q, k, v, out, lse, dout = _make_backward_inputs(shape)
    if lse.ndim == 3 and lse.shape == (shape.batch, shape.heads, shape.seqlen):
        lse = lse.permute(0, 2, 1).contiguous()
    scale = float(q.shape[-1] ** -0.5)
    return _time_cuda_op(
        lambda: _C.b300_mha_bwd_hot_cute16_candidate2_internal(
            q,
            k,
            v,
            out,
            lse,
            dout,
            shape.causal,
            scale,
            q.shape[1],
            deterministic,
        ),
        warmup,
        iters,
    )


def _candidate_backward_parity(shape: Shape, deterministic: bool) -> tuple[bool, float, float, float]:
    q, k, v, out, lse, dout = _make_backward_inputs(shape)
    if lse.ndim == 3 and lse.shape == (shape.batch, shape.heads, shape.seqlen):
        lse = lse.permute(0, 2, 1).contiguous()
    scale = float(q.shape[-1] ** -0.5)
    cand = _C.b300_mha_bwd_hot_cute16_candidate_internal(
        q,
        k,
        v,
        out,
        lse,
        dout,
        shape.causal,
        scale,
        q.shape[1],
        deterministic,
    )
    ref = b300_mha_bwd_experimental(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal=shape.causal,
        deterministic=deterministic,
        implementation="ref",
    )
    finite = bool(torch.isfinite(cand[0]).all() and torch.isfinite(cand[1]).all() and torch.isfinite(cand[2]).all())
    return (
        finite,
        (cand[0] - ref[0]).abs().max().item(),
        (cand[1] - ref[1]).abs().max().item(),
        (cand[2] - ref[2]).abs().max().item(),
    )


def _candidate2_backward_parity(shape: Shape, deterministic: bool) -> tuple[bool, float, float, float]:
    q, k, v, out, lse, dout = _make_backward_inputs(shape)
    if lse.ndim == 3 and lse.shape == (shape.batch, shape.heads, shape.seqlen):
        lse = lse.permute(0, 2, 1).contiguous()
    scale = float(q.shape[-1] ** -0.5)
    cand = _C.b300_mha_bwd_hot_cute16_candidate2_internal(
        q,
        k,
        v,
        out,
        lse,
        dout,
        shape.causal,
        scale,
        q.shape[1],
        deterministic,
    )
    ref = b300_mha_bwd_experimental(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal=shape.causal,
        deterministic=deterministic,
        implementation="ref",
    )
    finite = bool(torch.isfinite(cand[0]).all() and torch.isfinite(cand[1]).all() and torch.isfinite(cand[2]).all())
    return (
        finite,
        (cand[0] - ref[0]).abs().max().item(),
        (cand[1] - ref[1]).abs().max().item(),
        (cand[2] - ref[2]).abs().max().item(),
    )


def _hot_backward_parity(shape: Shape, deterministic: bool) -> tuple[bool, float, float, float]:
    q, k, v, out, lse, dout = _make_backward_inputs(shape)
    hot = b300_mha_bwd_experimental(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal=shape.causal,
        deterministic=deterministic,
        implementation="hot",
    )
    ref = b300_mha_bwd_experimental(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal=shape.causal,
        deterministic=deterministic,
        implementation="ref",
    )
    finite = bool(torch.isfinite(hot[0]).all() and torch.isfinite(hot[1]).all() and torch.isfinite(hot[2]).all())
    return (
        finite,
        (hot[0] - ref[0]).abs().max().item(),
        (hot[1] - ref[1]).abs().max().item(),
        (hot[2] - ref[2]).abs().max().item(),
    )


def _bench_backward_cute(shape: Shape, warmup: int, iters: int, deterministic: bool) -> float:
    _require_cute()
    q, k, v, dout = _make_inputs(shape)
    out, lse = cute_flash_attn_func(q, k, v, causal=shape.causal, return_lse=True)
    return _time_cuda_op(
        lambda: cute_flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            causal=shape.causal,
            deterministic=deterministic,
        ),
        warmup,
        iters,
    )


def _bench_full_tk(shape: Shape, warmup: int, iters: int, deterministic: bool) -> float:
    q, k, v, dout = _make_inputs(shape)
    return _time_cuda_op(
        lambda: b300_mha_bwd(
            q,
            k,
            v,
            *b300_mha_fwd(q, k, v, causal=shape.causal, return_lse=True),
            dout,
            causal=shape.causal,
            deterministic=deterministic,
        ),
        warmup,
        iters,
    )


def _bench_full_tk_experimental(shape: Shape, warmup: int, iters: int, deterministic: bool) -> float:
    return _bench_full_tk_experimental_impl(shape, warmup, iters, deterministic, implementation="auto")


def _bench_full_tk_experimental_impl(
    shape: Shape,
    warmup: int,
    iters: int,
    deterministic: bool,
    implementation: str,
) -> float:
    q, k, v, dout = _make_inputs(shape)
    return _time_cuda_op(
        lambda: b300_mha_bwd_experimental(
            q,
            k,
            v,
            *b300_mha_fwd(q, k, v, causal=shape.causal, return_lse=True),
            dout,
            causal=shape.causal,
            deterministic=deterministic,
            implementation=implementation,
        ),
        warmup,
        iters,
    )


def _bench_full_tk_experimental_candidate(shape: Shape, warmup: int, iters: int, deterministic: bool) -> float:
    q, k, v, dout = _make_inputs(shape)
    scale = float(q.shape[-1] ** -0.5)
    def run() -> None:
        out, lse = b300_mha_fwd(q, k, v, causal=shape.causal, return_lse=True)
        _C.b300_mha_bwd_hot_cute16_candidate_internal(
            q,
            k,
            v,
            out,
            lse,
            dout,
            shape.causal,
            scale,
            q.shape[1],
            deterministic,
        )

    return _time_cuda_op(run, warmup, iters)


def _bench_full_tk_experimental_candidate2(shape: Shape, warmup: int, iters: int, deterministic: bool) -> float:
    q, k, v, dout = _make_inputs(shape)
    scale = float(q.shape[-1] ** -0.5)

    def run() -> None:
        out, lse = b300_mha_fwd(q, k, v, causal=shape.causal, return_lse=True)
        _C.b300_mha_bwd_hot_cute16_candidate2_internal(
            q,
            k,
            v,
            out,
            lse,
            dout,
            shape.causal,
            scale,
            q.shape[1],
            deterministic,
        )

    return _time_cuda_op(run, warmup, iters)


def _bench_full_cute(shape: Shape, warmup: int, iters: int, deterministic: bool) -> float:
    _require_cute()
    q, k, v, dout = _make_inputs(shape)

    def run() -> None:
        out, lse = cute_flash_attn_func(q, k, v, causal=shape.causal, return_lse=True)
        cute_flash_attn_bwd(q, k, v, out, dout, lse, causal=shape.causal, deterministic=deterministic)

    return _time_cuda_op(run, warmup, iters)


def _benchmark_backend(
    backend: str,
    pass_name: str,
    shape: Shape,
    warmup: int,
    iters: int,
    deterministic: bool,
) -> float:
    if backend in {"tk", "tk_current"}:
        if pass_name == "forward":
            return _bench_forward_tk(shape, warmup, iters)
        if pass_name == "backward":
            return _bench_backward_tk(shape, warmup, iters, deterministic)
        return _bench_full_tk(shape, warmup, iters, deterministic)

    if backend == "tk_experimental":
        if pass_name == "forward":
            return _bench_forward_tk(shape, warmup, iters)
        if pass_name == "backward":
            return _bench_backward_tk_experimental(shape, warmup, iters, deterministic)
        return _bench_full_tk_experimental(shape, warmup, iters, deterministic)

    if backend == "tk_experimental_ref":
        if pass_name == "forward":
            return _bench_forward_tk(shape, warmup, iters)
        if pass_name == "backward":
            return _bench_backward_tk_experimental_impl(shape, warmup, iters, deterministic, implementation="ref")
        return _bench_full_tk_experimental_impl(shape, warmup, iters, deterministic, implementation="ref")

    if backend == "tk_experimental_hot":
        if not _experimental_hot_supported(shape, deterministic):
            return float("nan")
        if pass_name == "forward":
            return _bench_forward_tk(shape, warmup, iters)
        if pass_name == "backward":
            return _bench_backward_tk_experimental_impl(shape, warmup, iters, deterministic, implementation="hot")
        return _bench_full_tk_experimental_impl(shape, warmup, iters, deterministic, implementation="hot")

    if backend == "tk_experimental_hot_candidate":
        if not _experimental_hot_supported(shape, deterministic):
            return float("nan")
        if pass_name == "forward":
            return _bench_forward_tk(shape, warmup, iters)
        if pass_name == "backward":
            return _bench_backward_tk_experimental_candidate(shape, warmup, iters, deterministic)
        return _bench_full_tk_experimental_candidate(shape, warmup, iters, deterministic)

    if backend == "tk_experimental_hot_candidate2":
        if not _experimental_hot_supported(shape, deterministic):
            return float("nan")
        if pass_name == "forward":
            return _bench_forward_tk(shape, warmup, iters)
        if pass_name == "backward":
            return _bench_backward_tk_experimental_candidate2(shape, warmup, iters, deterministic)
        return _bench_full_tk_experimental_candidate2(shape, warmup, iters, deterministic)

    if pass_name == "forward":
        return _bench_forward_cute(shape, warmup, iters)
    if pass_name == "backward":
        return _bench_backward_cute(shape, warmup, iters, deterministic)
    return _bench_full_cute(shape, warmup, iters, deterministic)


def _print_same_input_backward_report(
    shape: Shape,
    *,
    deterministic: bool,
    warmup: int,
    iters: int,
    seeds: list[int],
    selected_backends: list[str] | None = None,
) -> None:
    default_timing_backends = (
        "tk_experimental_hot",
        "tk_experimental_hot_trusted",
        "tk_experimental_hot_candidate",
        "tk_experimental_hot_candidate2",
        "tk_experimental_ref",
    )
    default_parity_backends = (
        "tk_experimental_hot",
        "tk_experimental_hot_trusted",
        "tk_experimental_hot_candidate",
        "tk_experimental_hot_candidate2",
    )
    default_fixedout_backends = (
        "tk_experimental_hot_candidate",
        "tk_experimental_hot_candidate2",
    )
    if selected_backends is None:
        timing_backends = default_timing_backends
        parity_backends = default_parity_backends
        fixedout_timing_backends = default_fixedout_backends
        fixedout_backends = default_fixedout_backends
    else:
        selected = tuple(selected_backends)
        timing_backends = tuple(backend for backend in default_timing_backends if backend in selected)
        parity_backends = tuple(backend for backend in default_parity_backends if backend in selected)
        fixedout_timing_backends = tuple(backend for backend in default_fixedout_backends if backend in selected)
        fixedout_backends = tuple(backend for backend in default_fixedout_backends if backend in selected)
        if "cute" in selected:
            timing_backends = timing_backends + ("cute",)
            parity_backends = parity_backends + ("cute",)
    timing_seed = seeds[0]
    for backend in timing_backends:
        timing_inputs = _make_same_input_backward_inputs(shape, timing_seed, backend)
        timing_workspace = _clone_backward_inputs(timing_inputs)
        _restore_backward_inputs(timing_workspace, timing_inputs)
        time_us = _same_input_backward_time(
            backend,
            timing_workspace,
            shape,
            warmup,
            iters,
            deterministic,
        )
        print(
            "same_input_time\t"
            f"{backend}\t{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
            f"{timing_seed}\t{time_us:.2f}"
        )
    for backend in fixedout_timing_backends:
        timing_inputs = _make_same_input_backward_inputs(shape, timing_seed, backend)
        fixedout_time_us = _same_input_backward_time_fixedout(
            backend,
            timing_inputs,
            shape,
            warmup,
            iters,
            deterministic,
        )
        print(
            "same_input_fixedout_time\t"
            f"{backend}\t{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
            f"{timing_seed}\t{fixedout_time_us:.2f}"
        )
    if "tk_experimental_hot" in timing_backends or "tk_experimental_hot" in parity_backends:
        timing_inputs = _make_same_input_backward_inputs(shape, timing_seed, "tk_experimental_hot")
        timing_workspace = _clone_backward_inputs(timing_inputs)
        _restore_backward_inputs(timing_workspace, timing_inputs)
        fastpath_taken = _same_input_hot_fastpath_taken(timing_workspace, shape, deterministic)
        print(
            "same_input_fastpath\t"
            f"tk_experimental_hot\t{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
            f"{timing_seed}\t{int(fastpath_taken)}"
        )

    parity_max: dict[str, tuple[bool, float, float, float]] = {}
    repeat_max: dict[str, tuple[bool, float, float, float]] = {}
    fixedout_repeat_max: dict[str, tuple[bool, float, float, float]] = {}
    for seed in seeds:
        for backend in parity_backends:
            inputs = _make_same_input_backward_inputs(shape, seed, backend)
            workspace = _clone_backward_inputs(inputs)
            finite, max_dq, max_dk, max_dv = _same_input_backward_parity(
                backend,
                inputs,
                workspace,
                shape,
                deterministic,
            )
            print(
                "same_input_parity\t"
                f"{backend}\t{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
                f"{seed}\t{int(finite)}\t{max_dq:.6f}\t{max_dk:.6f}\t{max_dv:.6f}"
            )
            prev = parity_max.get(backend)
            if prev is None:
                parity_max[backend] = (finite, max_dq, max_dk, max_dv)
            else:
                parity_max[backend] = (
                    prev[0] and finite,
                    max(prev[1], max_dq),
                    max(prev[2], max_dk),
                    max(prev[3], max_dv),
                )

            repeat_finite, repeat_dq, repeat_dk, repeat_dv = _same_input_backward_repeat(
                backend,
                inputs,
                workspace,
                shape,
                deterministic,
            )
            print(
                "same_input_repeat\t"
                f"{backend}\t{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
                f"{seed}\t{int(repeat_finite)}\t{repeat_dq:.6f}\t{repeat_dk:.6f}\t{repeat_dv:.6f}"
            )
            repeat_prev = repeat_max.get(backend)
            if repeat_prev is None:
                repeat_max[backend] = (repeat_finite, repeat_dq, repeat_dk, repeat_dv)
            else:
                repeat_max[backend] = (
                    repeat_prev[0] and repeat_finite,
                    max(repeat_prev[1], repeat_dq),
                    max(repeat_prev[2], repeat_dk),
                    max(repeat_prev[3], repeat_dv),
                )
        for backend in fixedout_backends:
            inputs = _make_same_input_backward_inputs(shape, seed, backend)
            workspace = _clone_backward_inputs(inputs)
            fixedout_finite, fixedout_dq, fixedout_dk, fixedout_dv = _same_input_backward_repeat_fixedout(
                backend,
                inputs,
                workspace,
                shape,
                deterministic,
            )
            print(
                "same_input_fixedout_repeat\t"
                f"{backend}\t{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
                f"{seed}\t{int(fixedout_finite)}\t{fixedout_dq:.6f}\t{fixedout_dk:.6f}\t{fixedout_dv:.6f}"
            )
            fixedout_prev = fixedout_repeat_max.get(backend)
            if fixedout_prev is None:
                fixedout_repeat_max[backend] = (fixedout_finite, fixedout_dq, fixedout_dk, fixedout_dv)
            else:
                fixedout_repeat_max[backend] = (
                    fixedout_prev[0] and fixedout_finite,
                    max(fixedout_prev[1], fixedout_dq),
                    max(fixedout_prev[2], fixedout_dk),
                    max(fixedout_prev[3], fixedout_dv),
                )

    for backend in parity_backends:
        finite, max_dq, max_dk, max_dv = parity_max[backend]
        print(
            "same_input_parity_max\t"
            f"{backend}\t{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
            f"{int(finite)}\t{max_dq:.6f}\t{max_dk:.6f}\t{max_dv:.6f}"
        )
    for backend in parity_backends:
        finite, max_dq, max_dk, max_dv = repeat_max[backend]
        print(
            "same_input_repeat_max\t"
            f"{backend}\t{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
            f"{int(finite)}\t{max_dq:.6f}\t{max_dk:.6f}\t{max_dv:.6f}"
        )
    for backend in fixedout_backends:
        finite, max_dq, max_dk, max_dv = fixedout_repeat_max[backend]
        print(
            "same_input_fixedout_repeat_max\t"
            f"{backend}\t{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
            f"{int(finite)}\t{max_dq:.6f}\t{max_dk:.6f}\t{max_dv:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exact B300 TK attention against regular CuTe FA4.")
    parser.add_argument(
        "--backend",
        choices=(
            "tk",
            "tk_current",
            "tk_experimental",
            "tk_experimental_ref",
            "tk_experimental_hot",
            "tk_experimental_hot_candidate",
            "tk_experimental_hot_candidate2",
            "cute",
            "both",
            "all",
        ),
        default="both",
    )
    parser.add_argument("--pass", dest="pass_name", choices=("forward", "backward", "full"), default="forward")
    parser.add_argument("--batches", default="1")
    parser.add_argument("--seqlens", default="2048,4096,8192,16384")
    parser.add_argument("--heads", default="16")
    parser.add_argument("--causal-values", default="0,1")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--same-input", action="store_true")
    parser.add_argument("--same-input-backends", default=None)
    parser.add_argument("--same-input-seeds", default="0")
    parser.add_argument("--same-input-warmup", type=int, default=None)
    parser.add_argument("--same-input-iters", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    global _BENCH_DEVICE_INDEX
    _BENCH_DEVICE_INDEX = args.device
    torch.cuda.set_device(args.device)
    same_input_backends = None if args.same_input_backends is None else _parse_csv_strings(args.same_input_backends)
    emit_debug_parity = (not args.skip_parity) and same_input_backends is None

    if args.backend == "both":
        backends = ("tk_current", "cute")
    elif args.backend == "all":
        backends = ("tk_current", "tk_experimental_ref", "tk_experimental_hot", "tk_experimental_hot_candidate", "tk_experimental_hot_candidate2", "cute")
    else:
        backends = ("tk_current",) if args.backend == "tk" else (args.backend,)
    shapes = [
        Shape(batch=batch, seqlen=seqlen, heads=heads, causal=causal)
        for batch in _parse_csv_ints(args.batches)
        for seqlen in _parse_csv_ints(args.seqlens)
        for heads in _parse_csv_ints(args.heads)
        for causal in _parse_csv_bools(args.causal_values)
    ]

    print("backend\tpass\tbatch\tseqlen\theads\tcausal\ttime_us")
    summaries: list[tuple[Shape, float, float, float]] = []
    if args.same_input and args.pass_name == "backward":
        print(
            "# same_input_* rows reuse identical q/k/v/out/lse/dout tensors across backends; "
            "hot_parity/candidate_parity/candidate2_parity rows below remain debug-only because they use independent random inputs"
        )
    for shape in shapes:
        shape_results: dict[str, float] = {}
        for backend in backends:
            time_us = _benchmark_backend(
                backend,
                args.pass_name,
                shape,
                args.warmup,
                args.iters,
                args.deterministic,
            )
            shape_results[backend] = time_us
            print(
                f"{backend}\t{args.pass_name}\t{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t{time_us:.2f}"
            )
        if {"tk_experimental_hot", "tk_experimental_ref", "cute"} <= shape_results.keys():
            summaries.append(
                (
                    shape,
                    shape_results["tk_experimental_hot"],
                    shape_results["tk_experimental_ref"],
                    shape_results["cute"],
                )
            )
        if emit_debug_parity and args.pass_name == "backward" and "tk_experimental_hot_candidate" in shape_results:
            finite, max_dq, max_dk, max_dv = _candidate_backward_parity(shape, args.deterministic)
            print(
                "candidate_parity\t"
                f"{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
                f"{int(finite)}\t{max_dq:.6f}\t{max_dk:.6f}\t{max_dv:.6f}"
            )
        if emit_debug_parity and args.pass_name == "backward" and "tk_experimental_hot_candidate2" in shape_results:
            finite, max_dq, max_dk, max_dv = _candidate2_backward_parity(shape, args.deterministic)
            print(
                "candidate2_parity\t"
                f"{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
                f"{int(finite)}\t{max_dq:.6f}\t{max_dk:.6f}\t{max_dv:.6f}"
            )
        if emit_debug_parity and args.pass_name == "backward" and "tk_experimental_hot" in shape_results:
            finite, max_dq, max_dk, max_dv = _hot_backward_parity(shape, args.deterministic)
            print(
                "hot_parity\t"
                f"{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
                f"{int(finite)}\t{max_dq:.6f}\t{max_dk:.6f}\t{max_dv:.6f}"
            )
        if args.same_input and args.pass_name == "backward":
            _print_same_input_backward_report(
                shape,
                deterministic=args.deterministic,
                warmup=args.warmup if args.same_input_warmup is None else args.same_input_warmup,
                iters=args.iters if args.same_input_iters is None else args.same_input_iters,
                seeds=_parse_csv_ints(args.same_input_seeds),
                selected_backends=same_input_backends,
            )

    if summaries:
        print()
        print("summary\tpass\tbatch\tseqlen\theads\tcausal\thot_us\tref_us\tcute_us\thot_over_cute")
        for shape, hot_us, ref_us, cute_us in summaries:
            hot_over_cute = hot_us / cute_us if cute_us and not math.isnan(cute_us) else float("nan")
            print(
                f"summary\t{args.pass_name}\t{shape.batch}\t{shape.seqlen}\t{shape.heads}\t{int(shape.causal)}\t"
                f"{hot_us:.2f}\t{ref_us:.2f}\t{cute_us:.2f}\t{hot_over_cute:.3f}"
            )


if __name__ == "__main__":
    main()
