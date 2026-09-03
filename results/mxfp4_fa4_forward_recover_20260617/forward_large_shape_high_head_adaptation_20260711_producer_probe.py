#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tk_fa4.fp4_pv_experiments import (  # noqa: E402
    _benchmark_cuda_preflight,
    _fp4_qk_mxfp4_v_inputs_from_bf16_source,
    _make_live_bf16_source_inputs,
    _prepare_mxfp4_fwd_inputs_for_config,
    _run_streaming_live_qk_only_lse_only_chunked_by_head_timed,
    dump_live_quant_p_from_scores,
)


STAGE2 = (
    "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_"
    "earlyreuse_arrivereuse_pscreusefold_skippscarrive_pchainc_vtma_vstma_"
    "pstage2_q200_p112_o56_qkscfix"
)


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def prepare(seqlen: int, heads: int, device_arg: str):
    device = _benchmark_cuda_preflight(device_arg, what="materialized-P producer probe")
    torch.cuda.set_device(device)
    q_bf16, k_bf16, v_bf16 = _make_live_bf16_source_inputs(
        seqlen, seed=94601, batch=1, heads=heads, device=device, zero_qk=False
    )
    raw = _fp4_qk_mxfp4_v_inputs_from_bf16_source(
        q_bf16, k_bf16, v_bf16, qk_quant_backend="v5"
    )
    return _prepare_mxfp4_fwd_inputs_for_config(raw, seqlen=seqlen, config=STAGE2)


def run_worker(kind: str, seqlen: int, heads: int, device_arg: str) -> None:
    q, q_sc, q_sg, k, k_sc, k_sg, _, _ = prepare(seqlen, heads, device_arg)
    if kind == "qk_lse":
        lse, elapsed = _run_streaming_live_qk_only_lse_only_chunked_by_head_timed(
            q, q_sc, q_sg, k, k_sc, k_sg,
            head_chunk_size=1,
            timeout_ms=5_000.0,
        )
        print(json.dumps({"finite": bool(torch.isfinite(lse).all()), "timing_ms": elapsed}))
    elif kind == "live_dump":
        p_fp4, p_sc, lse, debug = dump_live_quant_p_from_scores(
            q, q_sc, q_sg, k, k_sc, k_sg,
            return_debug=True,
            allow_unsafe_live=True,
        )
        torch.cuda.synchronize(q.device)
        print(json.dumps({
            "finite": bool(torch.isfinite(lse).all()),
            "p_shape": list(p_fp4.shape),
            "p_scale_shape": list(p_sc.shape),
            "debug_path": debug["path"],
        }))
    else:
        raise ValueError(f"unknown worker kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("qk_lse", "live_dump"))
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        run_worker(args.worker, args.seqlen, args.heads, args.device)
        return
    if args.output is None:
        raise ValueError("--output is required in parent mode")

    cases = (
        ("qk_lse", 4096, 1),
        ("qk_lse", 4096, 16),
        ("live_dump", 1024, 2),
        ("live_dump", 4096, 16),
        ("live_dump", 8192, 16),
        ("live_dump", 16384, 4),
        ("live_dump", 16384, 16),
    )
    records: list[dict[str, object]] = []
    for kind, seqlen, heads in cases:
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker", kind,
            "--seqlen", str(seqlen),
            "--heads", str(heads),
            "--device", args.device,
        ]
        started = time.time()
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
            record = {
                "kind": kind,
                "seqlen": seqlen,
                "heads": heads,
                "timed_out": False,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-8000:],
                "stderr": completed.stderr[-8000:],
            }
        except subprocess.TimeoutExpired as exc:
            record = {
                "kind": kind,
                "seqlen": seqlen,
                "heads": heads,
                "timed_out": True,
                "returncode": None,
                "stdout": (exc.stdout or b"").decode(errors="replace")[-8000:]
                if isinstance(exc.stdout, bytes) else (exc.stdout or "")[-8000:],
                "stderr": (exc.stderr or b"").decode(errors="replace")[-8000:]
                if isinstance(exc.stderr, bytes) else (exc.stderr or "")[-8000:],
            }
        record["wall_seconds"] = time.time() - started
        records.append(record)
        write_json(args.output, {
            "task": "bounded materialized-P producer liveness probes",
            "device": args.device,
            "timeout_seconds": 30.0,
            "records": records,
        })


if __name__ == "__main__":
    main()
