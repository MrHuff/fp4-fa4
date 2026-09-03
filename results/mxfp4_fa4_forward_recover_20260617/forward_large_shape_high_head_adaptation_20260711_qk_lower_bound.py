#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
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
)


STAGE2 = (
    "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_"
    "earlyreuse_arrivereuse_pscreusefold_skippscarrive_pchainc_vtma_vstma_"
    "pstage2_q200_p112_o56_qkscfix"
)
CELLS = ((1, 4096, 16), (1, 8192, 16), (1, 16384, 4), (1, 16384, 16))


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = _benchmark_cuda_preflight(args.device, what="QK/LSE lower bound")
    torch.cuda.set_device(device)
    result: dict[str, object] = {
        "task": "optimistic fused QK plus online-LSE producer lower bound",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "warmup": args.warmup,
        "samples": args.samples,
        "compact_p_write_excluded": True,
        "e16pc_exp_pack_excluded": True,
        "rows": [],
    }
    for index, (batch, seqlen, heads) in enumerate(CELLS):
        q_bf16, k_bf16, v_bf16 = _make_live_bf16_source_inputs(
            seqlen,
            seed=94601 + index,
            batch=batch,
            heads=heads,
            device=device,
            zero_qk=False,
        )
        raw = _fp4_qk_mxfp4_v_inputs_from_bf16_source(
            q_bf16, k_bf16, v_bf16, qk_quant_backend="v5"
        )
        q, q_sc, q_sg, k, k_sc, k_sg, _, _ = _prepare_mxfp4_fwd_inputs_for_config(
            raw, seqlen=seqlen, config=STAGE2
        )

        def run() -> tuple[torch.Tensor, float]:
            return _run_streaming_live_qk_only_lse_only_chunked_by_head_timed(
                q,
                q_sc,
                q_sg,
                k,
                k_sc,
                k_sg,
                head_chunk_size=1,
                timeout_ms=5_000.0,
            )

        last, _ = run()
        for _ in range(args.warmup - 1):
            last, _ = run()
        samples: list[float] = []
        for _ in range(args.samples):
            last, elapsed = run()
            samples.append(float(elapsed))
        row = {
            "batch": batch,
            "seqlen": seqlen,
            "heads": heads,
            "timing_ms_p50": float(statistics.median(samples)),
            "timing_ms_min": float(min(samples)),
            "samples_ms": samples,
            "lse_finite": bool(torch.isfinite(last).all().item()),
            "lse_bytes": int(last.numel() * last.element_size()),
            "head_chunk_size": 1,
            "head_slice_and_lse_concat_excluded": True,
        }
        result["rows"].append(row)
        write_json(args.output, result)
        del q_bf16, k_bf16, v_bf16, raw, q, q_sc, q_sg, k, k_sc, k_sg, last
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
