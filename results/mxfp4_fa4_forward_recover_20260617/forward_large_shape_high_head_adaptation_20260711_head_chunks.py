#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tk_fa4.fp4_pv_experiments import (  # noqa: E402
    _D_VO,
    _fp4_qk_mxfp4_v_inputs_from_bf16_source,
    _load_bf16_causal_baseline_ext,
    _load_forward_experiments_ext,
    _make_live_bf16_source_inputs,
    _mxfp4_quant_mode_to_int,
)


E16PC_CONFIG = (
    "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_"
    "arrivereuse_pscreusefold_skippscarrive_ex2e16pc_vtma_vstma_pstage2_q200_p112_o56_qkscfix"
)


def timing_summary(samples: list[float]) -> dict[str, float | int]:
    return {
        "count": len(samples),
        "p50": float(statistics.median(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
    }


def delta(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    diff = actual.float() - expected.float()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "finite": bool(torch.isfinite(actual.float()).all().item()),
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(diff).item()
            / max(torch.linalg.vector_norm(expected.float()).item(), 1.0e-30)
        ),
    }


def time_default_stream(launch: Callable[[], None]) -> float:
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    launch()
    stop.record(stream)
    stop.synchronize()
    return float(start.elapsed_time(stop))


class ConcurrentLauncher:
    def __init__(self, launches: list[Callable[[], None]], device: torch.device) -> None:
        self.launches = launches
        self.streams = [torch.cuda.Stream(device=device) for _ in launches]
        self.timing_stream = torch.cuda.Stream(device=device)
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)
        self.done = [torch.cuda.Event() for _ in launches]

    def time_once(self) -> float:
        self.start.record(self.timing_stream)
        for launch, stream, done in zip(self.launches, self.streams, self.done):
            stream.wait_event(self.start)
            with torch.cuda.stream(stream):
                launch()
                done.record(stream)
        for done in self.done:
            self.timing_stream.wait_event(done)
        self.stop.record(self.timing_stream)
        self.stop.synchronize()
        return float(self.start.elapsed_time(self.stop))


def make_fp4_launch(
    ext: Any,
    tensors: tuple[torch.Tensor, ...],
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    persistent: bool,
) -> Callable[[], None]:
    q, q_sc, q_sg, k, k_sc, k_sg, v, v_sc = tensors

    def launch() -> None:
        ext.forward_streaming_live_mxfp4(
            q,
            q_sc,
            q_sg,
            k,
            k_sc,
            k_sg,
            v,
            v_sc,
            out,
            lse,
            _mxfp4_quant_mode_to_int(None),
            persistent,
        )

    return launch


def load_cute() -> Any:
    flash_root = ROOT / "flash-attention"
    if str(flash_root) not in sys.path:
        sys.path.insert(0, str(flash_root))
    import flash_attn.cute.interface as cute_interface

    return cute_interface


def run_case(
    *,
    device: torch.device,
    seqlen: int,
    heads: int,
    chunk_heads: int,
    seed: int,
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    if heads % chunk_heads:
        raise ValueError("heads must be divisible by chunk_heads")
    chunks = heads // chunk_heads
    q_bf16, k_bf16, v_bf16 = _make_live_bf16_source_inputs(
        seqlen,
        seed=seed,
        batch=1,
        heads=heads,
        device=device,
    )
    ext = _load_forward_experiments_ext()
    os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = E16PC_CONFIG

    full_inputs = _fp4_qk_mxfp4_v_inputs_from_bf16_source(
        q_bf16,
        k_bf16,
        v_bf16,
        qk_quant_backend="v5",
    )
    full_out = torch.empty((1, seqlen, heads, _D_VO), dtype=torch.bfloat16, device=device)
    full_lse = torch.empty((1, heads, 1, seqlen), dtype=torch.float32, device=device)
    single_persistent = seqlen <= 4096
    single_launch = make_fp4_launch(
        ext,
        full_inputs,
        full_out,
        full_lse,
        persistent=single_persistent,
    )

    chunk_launches: list[Callable[[], None]] = []
    chunk_outs: list[torch.Tensor] = []
    chunk_lses: list[torch.Tensor] = []
    for chunk in range(chunks):
        start = chunk * chunk_heads
        stop = start + chunk_heads
        q_chunk = q_bf16[:, :, start:stop, :].contiguous()
        k_chunk = k_bf16[:, :, start:stop, :].contiguous()
        v_chunk = v_bf16[:, :, start:stop, :].contiguous()
        chunk_inputs = _fp4_qk_mxfp4_v_inputs_from_bf16_source(
            q_chunk,
            k_chunk,
            v_chunk,
            qk_quant_backend="v5",
        )
        out = torch.empty(
            (1, seqlen, chunk_heads, _D_VO),
            dtype=torch.bfloat16,
            device=device,
        )
        lse = torch.empty(
            (1, chunk_heads, 1, seqlen),
            dtype=torch.float32,
            device=device,
        )
        chunk_outs.append(out)
        chunk_lses.append(lse)
        chunk_launches.append(
            make_fp4_launch(ext, chunk_inputs, out, lse, persistent=False)
        )

    def launch_chunks_sequential() -> None:
        for launch in chunk_launches:
            launch()

    concurrent = ConcurrentLauncher(chunk_launches, device)

    bf16_ext = _load_bf16_causal_baseline_ext()
    bf16_out = torch.empty_like(full_out)
    bf16_lse = torch.empty_like(full_lse)

    def launch_tk_bf16() -> None:
        bf16_ext.forward(q_bf16, k_bf16, v_bf16, bf16_out, bf16_lse)

    cute = load_cute()
    cute_out = torch.empty_like(full_out)
    cute_lse = torch.empty((1, heads, seqlen), dtype=torch.float32, device=device)

    def launch_cute_bf16() -> None:
        cute._flash_attn_fwd(
            q_bf16,
            k_bf16,
            v_bf16,
            causal=True,
            return_lse=True,
            out=cute_out,
            lse=cute_lse,
        )

    serial_routes = {
        "single_e16pc": single_launch,
        "chunks_sequential_fullgrid": launch_chunks_sequential,
        "tk_bf16_fullgrid": launch_tk_bf16,
        "cute_bf16": launch_cute_bf16,
    }
    for _ in range(warmup):
        for launch in serial_routes.values():
            launch()
        concurrent.time_once()
    torch.cuda.synchronize(device)

    route_order = [*serial_routes, "chunks_concurrent_fullgrid"]
    timings: dict[str, list[float]] = {name: [] for name in route_order}
    for sample in range(samples):
        offset = sample % len(route_order)
        for name in route_order[offset:] + route_order[:offset]:
            if name == "chunks_concurrent_fullgrid":
                elapsed = concurrent.time_once()
            else:
                elapsed = time_default_stream(serial_routes[name])
            timings[name].append(elapsed)

    single_launch()
    launch_chunks_sequential()
    torch.cuda.synchronize(device)
    chunk_out = torch.cat(chunk_outs, dim=2)
    chunk_lse = torch.cat(chunk_lses, dim=1)
    correctness = {
        "chunk_vs_single_output": delta(chunk_out, full_out),
        "chunk_vs_single_lse": delta(chunk_lse, full_lse),
        "single_vs_tk_bf16_output": delta(full_out, bf16_out),
        "single_vs_tk_bf16_lse": delta(full_lse, bf16_lse),
    }
    summarized = {name: timing_summary(values) for name, values in timings.items()}
    bf16_best = min(
        summarized["tk_bf16_fullgrid"]["p50"],
        summarized["cute_bf16"]["p50"],
    )
    summarized["chunks_concurrent_fullgrid"]["speedup_vs_single"] = (
        summarized["single_e16pc"]["p50"]
        / summarized["chunks_concurrent_fullgrid"]["p50"]
    )
    summarized["chunks_concurrent_fullgrid"]["speedup_vs_bf16_best"] = (
        bf16_best / summarized["chunks_concurrent_fullgrid"]["p50"]
    )
    return {
        "shape": {"batch": 1, "seqlen": seqlen, "heads": heads},
        "decomposition": {"chunks": chunks, "chunk_heads": chunk_heads},
        "config": E16PC_CONFIG,
        "single_launch_mode": "persistent" if single_persistent else "fullgrid",
        "chunk_launch_mode": "fullgrid",
        "warmup": warmup,
        "samples": samples,
        "timing_ms": summarized,
        "correctness": correctness,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cases = [(4096, 16, 8), (4096, 32, 8), (4096, 32, 16), (8192, 8, 4), (8192, 16, 4)]
    payload: dict[str, Any] = {
        "task": "large-shape high-head host head-chunk concurrency probe",
        "created_unix": time.time(),
        "device": str(device),
        "results": [],
    }
    for index, (seqlen, heads, chunk_heads) in enumerate(cases):
        result = run_case(
            device=device,
            seqlen=seqlen,
            heads=heads,
            chunk_heads=chunk_heads,
            seed=771100 + index,
            warmup=args.warmup,
            samples=args.samples,
        )
        payload["results"].append(result)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"shape": result["shape"], "decomposition": result["decomposition"], "timing_ms": result["timing_ms"]}), flush=True)
    payload["completed_unix"] = time.time()
    args.json.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
