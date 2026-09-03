#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tk_fa4.fp4_pv_experiments import (  # noqa: E402
    _CUDA_EVENT_TIMEOUT_MS,
    _D_VO,
    _benchmark_cuda_preflight,
    _compare_outputs,
    _fp4_qk_mxfp4_v_inputs_from_bf16_source,
    _load_bf16_causal_baseline_ext,
    _load_forward_experiments_ext,
    _make_live_bf16_source_inputs,
    _mxfp4_quant_mode_to_int,
    _prepare_mxfp4_fwd_inputs_for_config,
    _resolve_mxfp4_fwd_launch_mode,
    _temporary_environ,
    _wait_for_event,
)


CONFIGS = {
    "stage2": (
        "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_"
        "arrivereuse_pscreusefold_skippscarrive_vtma_vstma_pstage2_q200_p112_o56_qkscfix"
    ),
    "c": (
        "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_"
        "arrivereuse_pscreusefold_skippscarrive_pchainc_vtma_vstma_pstage2_q200_p112_o56_qkscfix"
    ),
    "e16pc": (
        "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_"
        "arrivereuse_pscreusefold_skippscarrive_ex2e16pc_vtma_vstma_pstage2_q200_p112_o56_qkscfix"
    ),
    "split_full": "scorederived_ex2e16pc_split2wg_full_pstage2_q152_p112_o48",
    "split_k64": "scorederived_ex2e16pc_split2wg_k64_pstage2_q152_p112_o48",
}

STAMP_NAMES = {
    0: "score_tmem_load_begin",
    1: "score_tmem_load_complete",
    2: "p_stage_reuse_complete",
    3: "causal_mask_complete",
    4: "block_row_max_complete",
    5: "p_scale_derived",
    6: "p_scale_store_issued",
    7: "p_scale_store_wait_complete",
    8: "exp_pack_complete",
    9: "payload_stores_complete",
    10: "payload_proxy_publish_complete",
    11: "producer_p_ready_signal",
    12: "row_sum_corr_complete",
    13: "issue_loop_entry",
    14: "output_rescale_ready",
    15: "v_ready_staged",
    16: "p_descriptor_acquired",
    17: "p_scale_ready_observed",
    18: "issue_side_p_ready_observed",
    19: "pv_tcgen_issue",
    20: "owner_identity",
}

STAMP_INTERVALS = {
    "score_tmem_load": ((0,), 1),
    "score_load_to_p_stage_reuse": ((1,), 2),
    "p_stage_reuse_to_causal_mask": ((2,), 3),
    "causal_mask_to_block_row_max": ((3,), 4),
    "block_row_max_to_p_scale": ((4,), 5),
    "p_scale_store_issue_to_wait": ((6,), 7),
    "p_scale_to_exp_pack": ((5,), 8),
    "exp_pack_to_payload_stores": ((8,), 9),
    "payload_stores_to_proxy_publish": ((9,), 10),
    "publication_scale_to_producer_p_ready": ((10, 7), 11),
    "producer_chain_total": ((0,), 11),
    "issue_entry_to_output_rescale_ready": ((13,), 14),
    "issue_entry_to_v_ready": ((13,), 15),
    "issue_entry_to_p_descriptor": ((13,), 16),
    "p_descriptor_to_p_scale_ready": ((16,), 17),
    "p_scale_ready_to_issue_p_ready": ((17,), 18),
    "producer_p_ready_to_issue_p_ready": ((11,), 18),
    "issue_p_ready_to_pv_issue": ((18,), 19),
    "producer_p_ready_to_pv_issue": ((11,), 19),
    "output_rescale_ready_to_pv_issue": ((14,), 19),
    "v_ready_to_pv_issue": ((15,), 19),
}

STAMP_OWNER_SLOT = 20
STAMP_REQUIRED_SLOTS = tuple(range(20))
SPLIT_STAMP_NAMES = {
    21: "half0_score_load_complete",
    22: "half1_score_load_complete",
    23: "half0_block_max_published",
    24: "half1_block_max_published",
    25: "half0_exp_pack_complete",
    26: "half1_exp_pack_complete",
    27: "half0_p_ready",
    28: "half1_p_ready",
    29: "half0_stage_reused_by_idx_plus_2",
    30: "half1_stage_reused_by_idx_plus_2",
}
SPLIT_REQUIRED_SLOTS = (0, 1, 2, 6, 7, 8, *range(13, 20), *range(21, 29))
SPLIT_STAMP_INTERVALS = {
    "half0_load_to_max_publish": ((21,), 23),
    "half1_load_to_max_publish": ((22,), 24),
    "max_rendezvous_to_p_scale_issue": ((23, 24), 6),
    "p_scale_issue_to_completion": ((6,), 7),
    "half0_max_to_exp_pack": ((23,), 25),
    "half1_max_to_exp_pack": ((24,), 26),
    "full_payload_ready_to_pv_issue": ((27, 28), 19),
    "first_payload_ready_to_pv_issue": ((27,), 19),
}


def _percentile(values: list[float | int], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "median": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "median": float(statistics.median(values)),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _tensor_delta(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    diff = actual.float() - expected.float()
    abs_diff = diff.abs()
    denom = expected.float().abs().clamp_min(1.0e-6)
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "max_rel": float((abs_diff / denom).max().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(diff).item()
            / max(torch.linalg.vector_norm(expected.float()).item(), 1.0e-30)
        ),
        "finite": bool(torch.isfinite(actual.float()).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=94601)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--launch-mode",
        choices=("auto", "persistent", "fullgrid"),
        default="auto",
    )
    parser.add_argument("--configs", default="stage2,c")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--samples-per-round", type=int, default=5)
    parser.add_argument("--stamp-runs", type=int, default=11)
    parser.add_argument("--stamp-target-idx", type=int, default=0)
    parser.add_argument("--timeout-ms", type=float, default=_CUDA_EVENT_TIMEOUT_MS)
    parser.add_argument("--skip-bf16", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    names = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise ValueError(f"unknown configs: {unknown}")
    if "stage2" not in names:
        names.insert(0, "stage2")

    device = _benchmark_cuda_preflight(args.device, what="Stage2 P-chain benchmark")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    bf16_ext = None if args.skip_bf16 else _load_bf16_causal_baseline_ext()
    ext = _load_forward_experiments_ext()
    if hasattr(ext, "reset_mxfp4_forward_timeline"):
        ext.reset_mxfp4_forward_timeline()
    if hasattr(ext, "reset_mxfp4_forward_policy126_counters"):
        ext.reset_mxfp4_forward_policy126_counters()

    q_bf16, k_bf16, v_bf16 = _make_live_bf16_source_inputs(
        args.seqlen,
        seed=args.seed,
        batch=args.batch,
        heads=args.heads,
        device=device,
        zero_qk=False,
    )
    fp4_inputs = _fp4_qk_mxfp4_v_inputs_from_bf16_source(
        q_bf16,
        k_bf16,
        v_bf16,
        qk_quant_backend="v5",
    )
    fp4_inputs = _prepare_mxfp4_fwd_inputs_for_config(
        fp4_inputs,
        seqlen=args.seqlen,
        config=CONFIGS["stage2"],
    )
    q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc = fp4_inputs
    launch_mode = _resolve_mxfp4_fwd_launch_mode(
        args.seqlen,
        args.heads,
        args.launch_mode,
    )
    persistent_launch = launch_mode != "fullgrid"

    shape = (args.batch, args.seqlen, args.heads, _D_VO)
    lse_shape = (args.batch, args.heads, 1, args.seqlen)
    outputs = {
        name: (
            torch.empty(shape, dtype=torch.bfloat16, device=device),
            torch.empty(lse_shape, dtype=torch.float32, device=device),
        )
        for name in names
    }
    bf16_out = torch.empty(shape, dtype=torch.bfloat16, device=device)
    bf16_lse = torch.empty(lse_shape, dtype=torch.float32, device=device)
    bf16_fn = None
    if bf16_ext is not None:
        bf16_fn = bf16_ext.forward_persistent if args.seqlen <= 8192 else bf16_ext.forward

    def run_bf16() -> None:
        if bf16_fn is not None:
            bf16_fn(q_bf16, k_bf16, v_bf16, bf16_out, bf16_lse)

    def run_config(name: str, out: torch.Tensor, lse: torch.Tensor) -> None:
        with _temporary_environ({
            "TK_FA4_FP4PV_FWD_CONFIG": CONFIGS[name],
            "TK_FA4_PCHAIN_RUNTIME_TARGET_IDX": str(args.stamp_target_idx),
        }):
            ext.forward_streaming_live_mxfp4(
                q,
                q_sc,
                q_sg,
                k,
                k_sc,
                k_sg,
                v_fp4,
                v_sc,
                out,
                lse,
                _mxfp4_quant_mode_to_int(None),
                persistent_launch,
            )

    if not args.skip_bf16:
        run_bf16()
    for name in names:
        out, lse = outputs[name]
        for _ in range(args.warmup):
            run_config(name, out, lse)
    torch.cuda.synchronize(device)

    samples: dict[str, list[float]] = {name: [] for name in names}
    stream = torch.cuda.current_stream(device=device)
    for round_idx in range(args.rounds):
        order = names if (round_idx & 1) == 0 else list(reversed(names))
        for name in order:
            out, lse = outputs[name]
            for _ in range(args.samples_per_round):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                run_config(name, out, lse)
                end.record(stream)
                _wait_for_event(end, timeout_ms=args.timeout_ms, what=f"P-chain timing {name}")
                samples[name].append(float(start.elapsed_time(end)))

    for name in names:
        out, lse = outputs[name]
        run_config(name, out, lse)
    torch.cuda.synchronize(device)
    baseline_out, baseline_lse = outputs["stage2"]
    e16pc_out, e16pc_lse = outputs.get("e16pc", outputs["stage2"])

    records: dict[str, Any] = {}
    for name in names:
        out, lse = outputs[name]
        det_out = torch.empty_like(out)
        det_lse = torch.empty_like(lse)
        run_config(name, det_out, det_lse)
        torch.cuda.synchronize(device)
        records[name] = {
            "config": CONFIGS[name],
            "timing_ms": _summary(samples[name]),
            "samples_ms": samples[name],
            "finite": bool(
                torch.isfinite(out.float()).all().item()
                and torch.isfinite(lse.float()).all().item()
            ),
            "determinism_output": _tensor_delta(det_out, out),
            "determinism_lse": _tensor_delta(det_lse, lse),
            "vs_stage2_output": _tensor_delta(out, baseline_out),
            "vs_stage2_lse": _tensor_delta(lse, baseline_lse),
            "vs_e16pc_output": _tensor_delta(out, e16pc_out),
            "vs_e16pc_lse": _tensor_delta(lse, e16pc_lse),
            "vs_bf16": None
            if args.skip_bf16
            else _compare_outputs(
                out,
                lse[:, :, 0, :],
                bf16_out,
                bf16_lse[:, :, 0, :],
            ),
        }

    stamp_api = hasattr(ext, "reset_mxfp4_forward_pchain_stamps") and hasattr(
        ext,
        "read_mxfp4_forward_pchain_stamps",
    )
    if stamp_api and args.stamp_runs > 0:
        for name in names:
            split_config = name in ("split_full", "split_k64")
            required_slots = SPLIT_REQUIRED_SLOTS if split_config else STAMP_REQUIRED_SLOTS
            interval_specs = SPLIT_STAMP_INTERVALS if split_config else STAMP_INTERVALS
            raw_runs: list[list[int]] = []
            valid_runs: list[dict[str, Any]] = []
            rejected_runs: list[dict[str, Any]] = []
            interval_values: dict[str, list[int]] = {
                interval: [] for interval in interval_specs
            }
            out, lse = outputs[name]
            for _ in range(args.stamp_runs):
                ext.reset_mxfp4_forward_pchain_stamps()
                run_config(name, out, lse)
                torch.cuda.synchronize(device)
                raw = [int(value) for value in ext.read_mxfp4_forward_pchain_stamps()]
                raw_runs.append(raw)
                missing_slots = [
                    slot
                    for slot in required_slots
                    if len(raw) <= slot or raw[slot] == 0
                ]
                owner = raw[STAMP_OWNER_SLOT] if len(raw) > STAMP_OWNER_SLOT else 0
                owner_block = ((owner >> 32) - 1) if owner else None
                owner_task = (((owner & 0xFFFFFFFF) - 1) if owner else None)
                interval_cycles: dict[str, int] = {}
                invalid_intervals: list[str] = []
                if not missing_slots and owner:
                    for interval, (start_slots, end_slot) in interval_specs.items():
                        start_value = max(raw[slot] for slot in start_slots)
                        delta = raw[end_slot] - start_value
                        if delta < 0:
                            invalid_intervals.append(interval)
                        else:
                            interval_cycles[interval] = delta
                valid = not missing_slots and owner != 0 and not invalid_intervals
                run_record = {
                    "valid": valid,
                    "owner": owner,
                    "owner_block": owner_block,
                    "owner_task": owner_task,
                    "target_idx": args.stamp_target_idx,
                    "missing_slots": missing_slots,
                    "invalid_intervals": invalid_intervals,
                    "intervals_cycles": interval_cycles,
                }
                if valid:
                    valid_runs.append(run_record)
                    for interval, value in interval_cycles.items():
                        interval_values[interval].append(value)
                else:
                    rejected_runs.append(run_record)
            records[name]["pchain_stamps"] = {
                "slot_names": {**STAMP_NAMES, **(SPLIT_STAMP_NAMES if split_config else {})},
                "raw_runs": raw_runs,
                "valid_runs": valid_runs,
                "rejected_runs": rejected_runs,
                "valid_count": len(valid_runs),
                "rejected_count": len(rejected_runs),
                "intervals_cycles": {
                    interval: _summary(values)
                    for interval, values in interval_values.items()
                },
            }

    timeline_raw = (
        [int(value) for value in ext.read_mxfp4_forward_timeline()]
        if hasattr(ext, "read_mxfp4_forward_timeline")
        else []
    )
    policy126_counters = (
        [int(value) for value in ext.read_mxfp4_forward_policy126_counters()]
        if hasattr(ext, "read_mxfp4_forward_policy126_counters")
        else []
    )
    result = {
        "shape": f"h{args.heads}_s{args.seqlen}",
        "seqlen": args.seqlen,
        "heads": args.heads,
        "batch": args.batch,
        "seed": args.seed,
        "device": str(device),
        "device_name": props.name,
        "launch_mode": launch_mode,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "samples_per_round": args.samples_per_round,
        "stamp_runs": args.stamp_runs,
        "stamp_target_idx": args.stamp_target_idx,
        "bf16_skipped": args.skip_bf16,
        "stamp_api": stamp_api,
        "timeline_raw": timeline_raw,
        "policy126_counters": policy126_counters,
        "records": records,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    compact = {
        "shape": result["shape"],
        "device_name": result["device_name"],
        "launch_mode": launch_mode,
        "timeline_raw_count": len(timeline_raw),
        "policy126_counter_nonzero": sum(value != 0 for value in policy126_counters),
        "records": {
            name: {
                "finite": records[name]["finite"],
                "timing_ms": records[name]["timing_ms"],
                "vs_stage2_max_abs": records[name]["vs_stage2_output"]["max_abs"],
                "vs_e16pc_max_abs": records[name]["vs_e16pc_output"]["max_abs"],
                "deterministic": records[name]["determinism_output"]["exact"],
                "stamp_valid_count": records[name].get("pchain_stamps", {}).get(
                    "valid_count",
                    0,
                ),
                "stamp_rejected_count": records[name].get("pchain_stamps", {}).get(
                    "rejected_count",
                    0,
                ),
                "stamp_intervals": records[name].get("pchain_stamps", {}).get(
                    "intervals_cycles",
                    {},
                ),
            }
            for name in names
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
