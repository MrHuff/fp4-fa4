#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tk_fa4.fp4_pv_experiments import (  # noqa: E402
    _D_VO,
    _benchmark_cuda_preflight,
    _fp4_qk_mxfp4_v_inputs_from_bf16_source,
    _load_forward_experiments_ext,
    _make_live_bf16_source_inputs,
    _mxfp4_quant_mode_to_int,
    _prepare_mxfp4_fwd_inputs_for_config,
    _wait_for_event,
)


STAGE2 = (
    "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_"
    "earlyreuse_arrivereuse_pscreusefold_skippscarrive_pchainc_vtma_vstma_"
    "pstage2_q200_p112_o56_qkscfix"
)
CONFIGS = {
    "split_full": "scorederived_ex2e16pc_split2wg_full_pstage2_q152_p112_o48",
    "split_k64": "scorederived_ex2e16pc_split2wg_k64_pstage2_q152_p112_o48",
}
SHAPES = (
    (128, 4), (256, 4), (384, 4), (1024, 4), (4096, 16),
    (4096, 32), (8192, 4), (8192, 8), (16384, 1), (16384, 4),
)


def tensor_delta(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    delta = actual.float() - expected.float()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.abs().max().item()),
        "finite": bool(torch.isfinite(actual.float()).all().item()),
    }


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--launches-per-route", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = _benchmark_cuda_preflight(args.device, what="split-P mixed stress")
    torch.cuda.set_device(device)
    ext = _load_forward_experiments_ext()
    records: list[dict[str, object]] = []
    total_launches = 0

    for shape_index, (seqlen, heads) in enumerate(SHAPES):
        q_bf16, k_bf16, v_bf16 = _make_live_bf16_source_inputs(
            seqlen,
            seed=94601 + shape_index,
            batch=1,
            heads=heads,
            device=device,
            zero_qk=False,
        )
        raw = _fp4_qk_mxfp4_v_inputs_from_bf16_source(
            q_bf16, k_bf16, v_bf16, qk_quant_backend="v5"
        )
        q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc = _prepare_mxfp4_fwd_inputs_for_config(
            raw, seqlen=seqlen, config=STAGE2
        )
        for name, config in CONFIGS.items():
            out = torch.empty((1, seqlen, heads, _D_VO), dtype=torch.bfloat16, device=device)
            lse = torch.empty((1, heads, 1, seqlen), dtype=torch.float32, device=device)
            os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = config
            out_reference = None
            lse_reference = None
            max_out_delta = 0.0
            max_lse_delta = 0.0
            finite = True
            stream = torch.cuda.current_stream(device=device)
            for launch in range(args.launches_per_route):
                end = torch.cuda.Event()
                ext.forward_streaming_live_mxfp4(
                    q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc, out, lse,
                    _mxfp4_quant_mode_to_int(None), False,
                )
                end.record(stream)
                _wait_for_event(end, timeout_ms=30_000.0, what="split-P mixed stress")
                if launch == 0:
                    out_reference = out.clone()
                    lse_reference = lse.clone()
                else:
                    out_delta = tensor_delta(out, out_reference)
                    lse_delta = tensor_delta(lse, lse_reference)
                    max_out_delta = max(max_out_delta, float(out_delta["max_abs"]))
                    max_lse_delta = max(max_lse_delta, float(lse_delta["max_abs"]))
                    finite = finite and bool(out_delta["finite"]) and bool(lse_delta["finite"])
                total_launches += 1
            records.append({
                "seqlen": seqlen,
                "heads": heads,
                "route": name,
                "launches": args.launches_per_route,
                "finite": finite,
                "max_run_to_run_output_abs": max_out_delta,
                "max_run_to_run_lse_abs": max_lse_delta,
            })
        del q_bf16, k_bf16, v_bf16, raw, q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc
        torch.cuda.empty_cache()
        write_json(args.output, {
            "task": "mixed-shape split-P lifecycle stress",
            "device": str(device),
            "launches_per_route_per_shape": args.launches_per_route,
            "total_launches": total_launches,
            "records": records,
        })


if __name__ == "__main__":
    main()
