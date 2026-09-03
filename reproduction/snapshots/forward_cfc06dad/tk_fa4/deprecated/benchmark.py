from __future__ import annotations

import argparse
import gc
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FLASH_ATTN_ROOT = ROOT / "flash-attention"
if str(FLASH_ATTN_ROOT) not in sys.path:
    sys.path.insert(0, str(FLASH_ATTN_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cute_flash_attn_func = None
cute_flash_attn_bwd = None
_CUTE_IMPORT_ERROR = None
tk_flash_attn_func = None
tk_extension = None
_TK_IMPORT_ERROR = None


def _load_tk_flash() -> None:
    global tk_flash_attn_func, tk_extension, _TK_IMPORT_ERROR
    if (tk_flash_attn_func is not None and tk_extension is not None) or _TK_IMPORT_ERROR is not None:
        return
    try:
        from tk_fa4 import _C as _tk_extension  # noqa: E402
        from tk_fa4 import flash_attn_func as _tk_flash_attn_func  # noqa: E402
    except Exception as exc:  # pragma: no cover - import errors are environment-specific.
        _TK_IMPORT_ERROR = exc
    else:
        tk_flash_attn_func = _tk_flash_attn_func
        tk_extension = _tk_extension


def _load_cute_flash() -> None:
    global cute_flash_attn_func, cute_flash_attn_bwd, _CUTE_IMPORT_ERROR
    if (cute_flash_attn_func is not None and cute_flash_attn_bwd is not None) or _CUTE_IMPORT_ERROR is not None:
        return
    try:
        from flash_attn.cute.interface import _flash_attn_bwd as _cute_flash_attn_bwd  # noqa: E402
        from flash_attn.cute.interface import flash_attn_func as _cute_flash_attn_func  # noqa: E402
    except Exception as exc:  # pragma: no cover - import errors are environment-specific.
        _CUTE_IMPORT_ERROR = exc
    else:
        cute_flash_attn_func = _cute_flash_attn_func
        cute_flash_attn_bwd = _cute_flash_attn_bwd


DEFAULT_BATCHES = [1, 4]
DEFAULT_SEQLENS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
DEFAULT_HEAD_PAIRS = [(32, 32), (32, 8), (32, 1)]
DEFAULT_HEAD_DIMS = [64, 128]
DEFAULT_CAUSAL = [False, True]
FLAGSHIP_SHAPES = {
    (1, 2048, 32, 32, 128, False),
    (1, 8192, 32, 32, 128, False),
}


@dataclass(frozen=True)
class Shape:
    batch: int
    seqlen: int
    heads: int
    heads_kv: int
    head_dim: int
    causal: bool

    @property
    def as_tuple(self) -> tuple[int, int, int, int, int, bool]:
        return (
            self.batch,
            self.seqlen,
            self.heads,
            self.heads_kv,
            self.head_dim,
            self.causal,
        )


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _parse_head_pairs(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        if not item:
            continue
        if "x" in item:
            heads, heads_kv = item.split("x", maxsplit=1)
        elif ":" in item:
            heads, heads_kv = item.split(":", maxsplit=1)
        else:
            raise ValueError(
                f"Unsupported head-pair selector: {item!r}. Use 'HxK' or 'H:K', for example '32x8'."
            )
        pairs.append((int(heads), int(heads_kv)))
    return pairs


def _parse_causal_values(value: str) -> list[bool]:
    mapping = {
        "0": False,
        "1": True,
        "false": False,
        "true": True,
        "noncausal": False,
        "causal": True,
    }
    values: list[bool] = []
    for item in value.split(","):
        lowered = item.strip().lower()
        if lowered not in mapping:
            raise ValueError(f"Unsupported causal selector: {item!r}")
        values.append(mapping[lowered])
    return values


def _forward_flops(shape: Shape) -> float:
    avg_seqlen = shape.seqlen / 2 if shape.causal else shape.seqlen
    return (
        shape.batch
        * shape.heads
        * 4.0
        * shape.seqlen
        * avg_seqlen
        * shape.head_dim
    )


def _backward_flops(shape: Shape) -> float:
    avg_seqlen = shape.seqlen / 2 if shape.causal else shape.seqlen
    return (
        shape.batch
        * shape.heads
        * 10.0
        * shape.seqlen
        * avg_seqlen
        * shape.head_dim
    )


def _attention_flops(shape: Shape, pass_name: str) -> float:
    if pass_name == "forward":
        return _forward_flops(shape)
    if pass_name == "backward":
        return _backward_flops(shape)
    if pass_name == "full":
        return _forward_flops(shape) + _backward_flops(shape)
    raise ValueError(pass_name)


def _shape_label(shape: Shape) -> str:
    mode = "causal" if shape.causal else "dense"
    return (
        f"b={shape.batch:>2} s={shape.seqlen:>5} "
        f"h={shape.heads:>2}/{shape.heads_kv:<2} "
        f"d={shape.head_dim:>3} {mode}"
    )


def _time_cuda_forward(fn, *args, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _time_cuda_backward(build_forward, grad: torch.Tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _, out = build_forward()
        out.backward(grad)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _, out = build_forward()
        out.backward(grad)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _time_cuda_callable(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _unwrap_out(x):
    return x[0] if isinstance(x, tuple) else x


def _align_lse_bsh(lse: torch.Tensor) -> torch.Tensor:
    return lse.permute(0, 2, 1).contiguous() if lse.ndim == 3 else lse


def _pad_bshd(x: torch.Tensor, multiple: int = 128) -> torch.Tensor:
    pad = (-x.shape[1]) % multiple
    if pad == 0:
        return x.contiguous()
    pad_tensor = torch.zeros(
        (x.shape[0], pad, x.shape[2], x.shape[3]),
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat((x, pad_tensor), dim=1).contiguous()


def _pad_bsh(x: torch.Tensor, multiple: int = 128) -> torch.Tensor:
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


def _prepare_raw_backward_state(shape: Shape) -> dict[str, torch.Tensor | float | bool | int]:
    q, k, v = _alloc_inputs(shape)
    scale = shape.head_dim ** -0.5
    if cute_flash_attn_func is not None:
        out, lse = cute_flash_attn_func(q, k, v, causal=shape.causal, return_lse=True)
        lse_bsh = _align_lse_bsh(lse)
        lse_raw = lse
    elif tk_flash_attn_func is not None:
        out, lse_bsh = tk_flash_attn_func(q, k, v, causal=shape.causal, return_lse=True)
        lse_raw = lse_bsh.permute(0, 2, 1).contiguous()
    else:  # pragma: no cover - guarded by _ensure_imports / benchmark selection.
        raise RuntimeError("Backward benchmark requires at least one attention backend to build reference state")

    dout = torch.randn_like(out)
    l_aux_bsh = (-lse_bsh / scale).contiguous()
    q_pad = _pad_bshd(q)
    k_pad = _pad_bshd(k)
    v_pad = _pad_bshd(v)
    out_pad = _pad_bshd(out)
    dout_pad = _pad_bshd(dout)
    l_aux_bhs1 = _pad_bsh(l_aux_bsh).permute(0, 2, 1).contiguous().unsqueeze(2)

    return {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "dout": dout,
        "lse_raw": lse_raw,
        "q_bhsd": _to_bhsd(q_pad),
        "k_bhsd": _to_bhsd(k_pad),
        "v_bhsd": _to_bhsd(v_pad),
        "out_bhsd": _to_bhsd(out_pad),
        "dout_bhsd": _to_bhsd(dout_pad),
        "l_aux_bhs1": l_aux_bhs1,
        "scale": scale,
        "causal": shape.causal,
        "seqlen": shape.seqlen,
    }


def _alloc_inputs(shape: Shape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(
        shape.batch,
        shape.seqlen,
        shape.heads,
        shape.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        shape.batch,
        shape.seqlen,
        shape.heads_kv,
        shape.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        shape.batch,
        shape.seqlen,
        shape.heads_kv,
        shape.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return q, k, v


def _benchmark_shape(
    shape: Shape,
    warmup: int,
    iters: int,
    backend: str,
    pass_name: str,
) -> dict[str, float | str]:
    tk_ms = math.nan
    cute_ms = math.nan

    if pass_name == "forward":
        q, k, v = _alloc_inputs(shape)
        if backend in ("tk", "both"):
            tk_ms = _time_cuda_forward(
                lambda a, b, c: tk_flash_attn_func(a, b, c, causal=shape.causal),
                q,
                k,
                v,
                warmup=warmup,
                iters=iters,
            )
        if backend in ("cute", "both"):
            q_ref = q.detach().clone()
            k_ref = k.detach().clone()
            v_ref = v.detach().clone()
            cute_ms = _time_cuda_forward(
                lambda a, b, c: cute_flash_attn_func(a, b, c, causal=shape.causal),
                q_ref,
                k_ref,
                v_ref,
                warmup=warmup,
                iters=iters,
            )
    elif pass_name == "backward":
        state = _prepare_raw_backward_state(shape)
        if backend in ("tk", "both"):
            tk_ms = _time_cuda_callable(
                lambda: tk_extension.mha_bwd(
                    state["q_bhsd"],
                    state["k_bhsd"],
                    state["v_bhsd"],
                    state["out_bhsd"],
                    state["l_aux_bhs1"],
                    state["dout_bhsd"],
                    state["causal"],
                    state["scale"],
                    state["seqlen"],
                    False,
                ),
                warmup=warmup,
                iters=iters,
            )
        if backend in ("cute", "both"):
            cute_ms = _time_cuda_callable(
                lambda: cute_flash_attn_bwd(
                    state["q"],
                    state["k"],
                    state["v"],
                    state["out"],
                    state["dout"],
                    state["lse_raw"],
                    softmax_scale=state["scale"],
                    causal=state["causal"],
                ),
                warmup=warmup,
                iters=iters,
            )
    else:
        q, k, v = _alloc_inputs(shape)
        grad = torch.randn_like(q)
        if backend in ("tk", "both"):
            def build_tk():
                qg = q.detach().clone().requires_grad_(True)
                kg = k.detach().clone().requires_grad_(True)
                vg = v.detach().clone().requires_grad_(True)
                out = _unwrap_out(tk_flash_attn_func(qg, kg, vg, causal=shape.causal))
                return (qg, kg, vg), out

            tk_ms = _time_cuda_backward(build_tk, grad, warmup=warmup, iters=iters)
        if backend in ("cute", "both"):
            q_ref = q.detach().clone()
            k_ref = k.detach().clone()
            v_ref = v.detach().clone()
            def build_cute():
                qg = q_ref.detach().clone().requires_grad_(True)
                kg = k_ref.detach().clone().requires_grad_(True)
                vg = v_ref.detach().clone().requires_grad_(True)
                out = _unwrap_out(cute_flash_attn_func(qg, kg, vg, causal=shape.causal))
                return (qg, kg, vg), out

            cute_ms = _time_cuda_backward(build_cute, grad, warmup=warmup, iters=iters)

    flops = _attention_flops(shape, pass_name)
    tk_tflops = flops / (tk_ms * 1e-3) * 1e-12 if not math.isnan(tk_ms) else math.nan
    cute_tflops = flops / (cute_ms * 1e-3) * 1e-12 if not math.isnan(cute_ms) else math.nan
    return {
        "shape": _shape_label(shape),
        "tk_ms": tk_ms,
        "cute_ms": cute_ms,
        "ratio": tk_ms / cute_ms if backend == "both" else math.nan,
        "tk_tflops": tk_tflops,
        "cute_tflops": cute_tflops,
        "status": "ok",
        "error": "",
    }


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _error_row(shape: Shape, backend: str, exc: Exception) -> dict[str, float | str]:
    return {
        "shape": _shape_label(shape),
        "tk_ms": math.nan,
        "cute_ms": math.nan,
        "ratio": math.nan if backend == "both" else math.nan,
        "tk_tflops": math.nan,
        "cute_tflops": math.nan,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
    }


def _progress_line(index: int, total: int, backend: str, row: dict[str, float | str]) -> str:
    prefix = f"progress [{index}/{total}] {row['status']} {row['shape']}"
    if row["status"] != "ok":
        return f"{prefix} :: {row['error']}"
    if backend == "both":
        return (
            f"{prefix} :: tk_ms={float(row['tk_ms']):.3f} "
            f"cute_ms={float(row['cute_ms']):.3f} ratio={float(row['ratio']):.3f}"
        )
    if backend == "tk":
        return f"{prefix} :: tk_ms={float(row['tk_ms']):.3f}"
    return f"{prefix} :: cute_ms={float(row['cute_ms']):.3f}"


def _iter_shapes(args: argparse.Namespace) -> list[Shape]:
    if args.sweep_defaults:
        batches = DEFAULT_BATCHES
        seqlens = DEFAULT_SEQLENS
        head_pairs = DEFAULT_HEAD_PAIRS
        head_dims = DEFAULT_HEAD_DIMS
        causal_values = DEFAULT_CAUSAL
    else:
        batches = _parse_int_list(args.batches)
        seqlens = _parse_int_list(args.seqlens)
        head_pairs = _parse_head_pairs(args.head_pairs)
        head_dims = _parse_int_list(args.head_dims)
        causal_values = _parse_causal_values(args.causal_values)

    return [
        Shape(batch, seqlen, heads, heads_kv, head_dim, causal)
        for batch, seqlen, (heads, heads_kv), head_dim, causal in itertools.product(
            batches,
            seqlens,
            head_pairs,
            head_dims,
            causal_values,
        )
    ]


def _apply_compat_shape_args(args: argparse.Namespace) -> None:
    if args.batch is not None:
        args.batches = str(args.batch)
    if args.seqlen is not None:
        args.seqlens = str(args.seqlen)
    if args.head_dim is not None:
        args.head_dims = str(args.head_dim)
    if args.causal is not None:
        args.causal_values = args.causal

    if args.heads is None and args.heads_kv is None:
        return
    if args.heads is None or args.heads_kv is None:
        raise ValueError("--heads and --heads-kv must be provided together")
    args.head_pairs = f"{args.heads}x{args.heads_kv}"


def _ensure_imports(backend: str) -> None:
    if backend in ("tk", "both"):
        _load_tk_flash()
    if backend in ("cute", "both"):
        _load_cute_flash()

    if backend in ("cute", "both") and cute_flash_attn_func is None:
        raise ImportError(
            "flash_attn.cute is not importable in this Python environment. "
            "Run the benchmark from the CuTe-enabled GPU environment."
        ) from _CUTE_IMPORT_ERROR
    if backend in ("tk", "both") and tk_flash_attn_func is None:
        raise ImportError(
            "tk_fa4 is not importable in this Python environment. "
            "Build the extension and run the benchmark from the matching Python environment."
        ) from _TK_IMPORT_ERROR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--backend", choices=("tk", "cute", "both"), default="both")
    parser.add_argument("--pass", dest="pass_name", choices=("forward", "backward", "full"), default="forward")
    parser.add_argument("--sweep-defaults", action="store_true")
    parser.add_argument("--batches", type=str, default="1")
    parser.add_argument("--seqlens", type=str, default="2048")
    parser.add_argument("--head-pairs", type=str, default="32x32")
    parser.add_argument("--head-dims", type=str, default="128")
    parser.add_argument("--causal-values", type=str, default="0")
    parser.add_argument("--batch", type=int, help="Compatibility alias for --batches with a single shape")
    parser.add_argument("--seqlen", type=int, help="Compatibility alias for --seqlens with a single shape")
    parser.add_argument("--heads", type=int, help="Compatibility alias for --head-pairs with a single shape")
    parser.add_argument("--heads-kv", dest="heads_kv", type=int, help="Compatibility alias for --head-pairs")
    parser.add_argument("--head-dim", dest="head_dim", type=int, help="Compatibility alias for --head-dims")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--causal",
        nargs="?",
        const="1",
        type=str,
        help="Compatibility alias for --causal-values; accepts '--causal' or '--causal 0/1'",
    )
    args = parser.parse_args()
    _apply_compat_shape_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run tk_fa4 benchmarks")
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("tk_fa4 benchmarks require GB200 / SM100")
    _ensure_imports(args.backend)

    shapes = _iter_shapes(args)
    total_shapes = len(shapes)
    results = []
    for index, shape in enumerate(shapes, start=1):
        print(f"progress [{index}/{total_shapes}] start {_shape_label(shape)}", file=sys.stderr, flush=True)
        try:
            row = _benchmark_shape(shape, args.warmup, args.iters, args.backend, args.pass_name)
        except Exception as exc:
            row = _error_row(shape, args.backend, exc)
            if args.fail_fast:
                raise
        finally:
            _release_cuda_memory()
        results.append(row)
        print(_progress_line(index, total_shapes, args.backend, row), file=sys.stderr, flush=True)

    if args.backend == "both":
        print(
            f"{'shape':<34} {'tk_ms':>9} {'cute_ms':>9} {'ratio':>8} "
            f"{'tk_tflops':>11} {'cute_tflops':>12} {'status':>8}"
        )
        for row in results:
            print(
                f"{row['shape']:<34} "
                f"{row['tk_ms']:>9.3f} "
                f"{row['cute_ms']:>9.3f} "
                f"{row['ratio']:>8.3f} "
                f"{row['tk_tflops']:>11.1f} "
                f"{row['cute_tflops']:>12.1f} "
                f"{row['status']:>8}"
            )

        finite_ratios = [float(row["ratio"]) for row in results if math.isfinite(float(row["ratio"]))]
        flagship_rows = [
            row
            for shape, row in zip(shapes, results, strict=True)
            if shape.as_tuple in FLAGSHIP_SHAPES and math.isfinite(float(row["ratio"]))
        ]
        worst_ratio = max(finite_ratios, default=float("nan"))
        flagship_ratio = max((float(row["ratio"]) for row in flagship_rows), default=float("nan"))
        print()
        print(f"pass={args.pass_name}")
        print(f"worst_ratio={worst_ratio:.3f}")
        if flagship_rows:
            print(f"flagship_worst_ratio={flagship_ratio:.3f}")
    elif args.backend == "tk":
        print(f"{'shape':<34} {'tk_ms':>9} {'tk_tflops':>11} {'status':>8}")
        for row in results:
            print(
                f"{row['shape']:<34} "
                f"{row['tk_ms']:>9.3f} "
                f"{row['tk_tflops']:>11.1f} "
                f"{row['status']:>8}"
            )
    else:
        print(f"{'shape':<34} {'cute_ms':>9} {'cute_tflops':>12} {'status':>8}")
        for row in results:
            print(
                f"{row['shape']:<34} "
                f"{row['cute_ms']:>9.3f} "
                f"{row['cute_tflops']:>12.1f} "
                f"{row['status']:>8}"
            )

    failed_rows = [row for row in results if row["status"] != "ok"]
    if failed_rows:
        print(f"failed_shapes={len(failed_rows)}")


if __name__ == "__main__":
    main()
