#!/usr/bin/env python3
"""Compare current MXFP4 forward against SageAttention3 Blackwell NVFP4.

This is an experimental forward-only benchmark helper. On GB200/SM100,
SageAttention3's upstream native attention kernel currently targets SM120
block-scale MMA, so this script records the reported SageAttention3 backend
status and compares that path against our MXFP4-vs-CuTe BF16 numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Any

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tk_fa4.fp4_pv_experiments import (  # noqa: E402
    _make_live_bf16_source_inputs,
    benchmark_forward_streaming_live_mxfp4_vs_bf16,
    run_cute_dsl_fa4_bf16_baseline,
)


def _event_time_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return float(values[len(values) // 2])


def _output_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    out_f = out.float()
    ref_f = ref.float()
    diff = out_f - ref_f
    abs_diff = diff.abs().flatten()
    ref_abs = ref_f.abs().flatten()
    mse = float(diff.square().mean().item())
    rmse = math.sqrt(mse)
    ref_rms = math.sqrt(float(ref_f.square().mean().item()))
    return {
        "max_abs_diff": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs_diff": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "p99_abs_diff": float(torch.quantile(abs_diff, 0.99).item()) if abs_diff.numel() else 0.0,
        "rmse": rmse,
        "normalized_rmse": float(rmse / max(ref_rms, 1.0e-6)),
        "max_rel_diff": float((abs_diff / ref_abs.clamp_min(1.0e-3)).max().item()) if abs_diff.numel() else 0.0,
        "nonzero_diff_fraction": float((abs_diff > 0).float().mean().item()) if abs_diff.numel() else 0.0,
        "output_has_nan": bool(torch.isnan(out).any().item()),
        "output_has_nonfinite": bool((~torch.isfinite(out)).any().item()),
    }


def _load_sageattention3_blackwell(sage_root: pathlib.Path):
    sys.path.insert(0, str(sage_root))
    import sageattn3  # type: ignore

    status_fn = getattr(sageattn3, "sageattn3_backend_status", lambda: {})
    return sageattn3.sageattn3_blackwell, status_fn


def _run_sageattention3(
    q_bf16: torch.Tensor,
    k_bf16: torch.Tensor,
    v_bf16: torch.Tensor,
    *,
    sage_root: pathlib.Path,
    warmup: int,
    iters: int,
    per_block_mean: bool,
    time_fallback: bool,
) -> dict[str, Any]:
    try:
        sageattn3_blackwell, backend_status = _load_sageattention3_blackwell(sage_root)
    except Exception as exc:  # noqa: BLE001 - benchmark should degrade cleanly
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "sage_root": str(sage_root),
        }

    status_before = backend_status()
    native_available = bool(status_before.get("fp4attn_cuda_available"))
    if not native_available and not time_fallback:
        return {
            "available": False,
            "native_available": False,
            "timed": False,
            "sage_root": str(sage_root),
            "backend_status_before": status_before,
            "skip_reason": (
                "native fp4attn_cuda is unavailable for sm_100; use "
                "--time-sage-fallback to time the labeled fallback wrapper"
            ),
        }

    # SageAttention3 expects B,H,S,D and mutates K during preprocessing.
    q_h = q_bf16.permute(0, 2, 1, 3).contiguous()
    k_h = k_bf16.permute(0, 2, 1, 3).contiguous()
    v_h = v_bf16.permute(0, 2, 1, 3).contiguous()

    def run_once() -> torch.Tensor:
        out = sageattn3_blackwell(
            q_h,
            k_h.clone(),
            v_h,
            is_causal=True,
            per_block_mean=per_block_mean,
        )
        return out.permute(0, 2, 1, 3).contiguous()

    try:
        for _ in range(warmup):
            run_once()
        samples = []
        out = None
        for _ in range(iters):
            out, ms = _event_time_ms(run_once)
            samples.append(ms)
        status_after = backend_status()
        assert out is not None
        return {
            "available": True,
            "native_available": native_available,
            "timed": True,
            "sage_root": str(sage_root),
            "backend_status_before": status_before,
            "backend_status_after": status_after,
            "ms": _median(samples),
            "samples_ms": samples,
            "out": out,
        }
    except Exception as exc:  # noqa: BLE001 - keep comparator usable on SM100
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "sage_root": str(sage_root),
        }


def _collect_cute_bf16(
    q_bf16: torch.Tensor,
    k_bf16: torch.Tensor,
    v_bf16: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        run_cute_dsl_fa4_bf16_baseline(q_bf16, k_bf16, v_bf16)
    samples = []
    record = None
    for _ in range(iters):
        record = run_cute_dsl_fa4_bf16_baseline(q_bf16, k_bf16, v_bf16)
        samples.append(float(record["timing_ms"]))
    assert record is not None
    return {
        "record": record,
        "ms": _median(samples),
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mxfp4-config", default=None)
    parser.add_argument("--sage-root", type=pathlib.Path, default=REPO_ROOT / "SageAttention" / "sageattention3_blackwell")
    parser.add_argument("--no-per-block-mean", action="store_true")
    parser.add_argument(
        "--time-sage-fallback",
        action="store_true",
        help="Time SageAttention3's labeled non-native fallback path when native fp4attn_cuda is unavailable.",
    )
    parser.add_argument("--json", type=pathlib.Path, default=None)
    args = parser.parse_args()

    q_bf16, k_bf16, v_bf16 = _make_live_bf16_source_inputs(
        args.seqlen,
        seed=args.seed,
        batch=args.batch,
        heads=args.heads,
        device=args.device,
    )
    bf16_cute = _collect_cute_bf16(
        q_bf16,
        k_bf16,
        v_bf16,
        warmup=args.warmup,
        iters=args.iters,
    )
    bf16_record = bf16_cute["record"]
    mxfp4_record = benchmark_forward_streaming_live_mxfp4_vs_bf16(
        seqlen=args.seqlen,
        seed=args.seed,
        batch=args.batch,
        heads=args.heads,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        mxfp4_fwd_config=args.mxfp4_config,
        launch_mode="fullgrid",
        bf16_baseline="cute",
        include_output_only=False,
    )
    sage_record = _run_sageattention3(
        q_bf16,
        k_bf16,
        v_bf16,
        sage_root=args.sage_root,
        warmup=args.warmup,
        iters=args.iters,
        per_block_mean=not args.no_per_block_mean,
        time_fallback=args.time_sage_fallback,
    )
    sage_out = sage_record.pop("out", None)
    if sage_out is not None:
        sage_record["comparison_vs_cute_bf16"] = _output_metrics(sage_out, bf16_record["out"])
        sage_ms = float(sage_record["ms"])
        if sage_ms > 0:
            sage_record["speedup_over_cute_bf16"] = float(bf16_cute["ms"] / sage_ms)

    result = {
        "seqlen": args.seqlen,
        "heads": args.heads,
        "batch": args.batch,
        "seed": args.seed,
        "device": str(torch.device(args.device)),
        "bf16_cute_ms": bf16_cute["ms"],
        "bf16_cute_samples_ms": bf16_cute["samples_ms"],
        "mxfp4": {
            "config": mxfp4_record["mxfp4_fwd_config"],
            "ms": mxfp4_record["mxfp4_ms"],
            "samples_ms": mxfp4_record["mxfp4_samples_ms"],
            "speedup_over_cute_bf16": mxfp4_record["speedup_mxfp4_over_bf16_cute_dsl_fa4"],
            "comparison_vs_cute_bf16": mxfp4_record["comparison_vs_bf16_cute_dsl_fa4"],
        },
        "sageattention3_nvfp4": sage_record,
    }

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
