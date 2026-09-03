#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
import re
import statistics
from collections import Counter
from pathlib import Path


RESULTS = Path(__file__).resolve().parent
PREFIX = "forward_large_shape_high_head_adaptation_20260711"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def p50(record: dict | None) -> float | None:
    if not record:
        return None
    value = record.get("timing_ms", {}).get("p50")
    return None if value is None else float(value)


def cell_key(cell: dict) -> str:
    shape = cell["shape"]
    return f"b{shape['batch']}/s{shape['seqlen']}/h{shape['heads']}"


def parse_ncu(path: Path) -> dict[str, float | int | str | None]:
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"'))
    reader = csv.DictReader(io.StringIO("\n".join(lines[header:])))
    next(reader)
    row = next(reader)
    keys = (
        "gpu__time_duration.avg",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block_static",
        "launch__shared_mem_per_block_dynamic",
        "launch__barrier_count",
        "launch__waves_per_multiprocessor",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
        "smsp__issue_active.avg.per_cycle_active",
        "smsp__warps_eligible.avg.per_cycle_active",
        "profiler__replayer_passes",
    )
    parsed: dict[str, float | int | str | None] = {}
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            parsed[key] = None
        else:
            try:
                number = float(value.replace(",", ""))
                parsed[key] = int(number) if number.is_integer() else number
            except ValueError:
                parsed[key] = value
    parsed["kernel_name"] = row.get("Kernel Name")
    return parsed


def sparse_summary() -> dict:
    result: dict[str, object] = {}
    for target in (0, 2, 16, 31):
        data = load(f"{PREFIX}_pchain_idx{target}_s4096_h16.json")
        target_result: dict[str, object] = {}
        for route in ("split_full", "split_k64"):
            stamps = data["records"][route]["pchain_stamps"]
            raw_runs = [
                raw for raw in stamps["raw_runs"]
                if len(raw) >= 31 and all(raw[slot] for slot in range(21, 29))
            ]
            reuse = [
                max(raw[29], raw[30]) - raw[19]
                for raw in raw_runs if raw[19] and raw[29] and raw[30]
            ]
            target_result[route] = {
                "valid_count": stamps["valid_count"],
                "rejected_count": stamps["rejected_count"],
                "interval_medians_cycles": {
                    name: values["median"]
                    for name, values in stamps["intervals_cycles"].items()
                },
                "score_load_completion_skew_cycles": statistics.median(
                    abs(raw[21] - raw[22]) for raw in raw_runs
                ),
                "exp_pack_completion_skew_cycles": statistics.median(
                    abs(raw[25] - raw[26]) for raw in raw_runs
                ),
                "p_ready_skew_cycles": statistics.median(
                    abs(raw[27] - raw[28]) for raw in raw_runs
                ),
                "pv_to_stage_reuse_cycles": statistics.median(reuse) if reuse else None,
                "owner_pairs": sorted({
                    (run["owner_block"], run["owner_task"])
                    for run in stamps["valid_runs"] + stamps["rejected_runs"]
                }),
            }
        if target == 0:
            e16 = data["records"]["e16pc"]["pchain_stamps"]
            target_result["e16pc"] = {
                "valid_count": e16["valid_count"],
                "interval_medians_cycles": {
                    name: values["median"]
                    for name, values in e16["intervals_cycles"].items()
                },
            }
        result[str(target)] = target_result
    return result


def main() -> None:
    regression = load(f"{PREFIX}_regression57.json")
    historical = load("forward_issue_lane_overlap_bf16_matrix_20260711_summary.json")
    required = load(f"{PREFIX}_required_matrix.json")
    required_map = {cell_key(cell): cell for cell in required["cells"]}
    regression_map = {cell_key(cell): cell for cell in regression["cells"]}
    historical_robust = set(historical["lists"]["robust_win"])
    retained_robust = sorted(
        key for key in historical_robust
        if regression_map[key]["measured_comparison"]["status"] == "robust_win"
    )
    reg_status = Counter(
        cell["measured_comparison"]["status"]
        for cell in regression["cells"]
        if cell.get("measured_comparison")
    )

    timing_rows = []
    for key, cell in required_map.items():
        records = cell["records"]
        timing_rows.append({
            "cell": key,
            "stage2_ms": p50(records.get("stage2")),
            "e16pc_ms": p50(records.get("e16pc")),
            "old_auto_ms": p50(records.get("fp4_auto")),
            "split_full_ms": p50(records.get("split_full")),
            "split_k64_ms": p50(records.get("split_k64")),
            "tk_bf16_fullgrid_ms": p50(records.get("tk_bf16_fullgrid")),
            "cute_bf16_ms": p50(records.get("cute_bf16")),
            "best_comparison": cell.get("measured_comparison"),
        })

    finalists = {}
    for seqlen in (16384, 32768):
        data = load(f"{PREFIX}_finalist60_b1_s{seqlen}_h1.json")
        finalists[str(seqlen)] = {
            route: {
                "p50_ms": p50(record),
                "p25_ms": record["timing_ms"].get("p25"),
                "p75_ms": record["timing_ms"].get("p75"),
                "min_ms": record["timing_ms"].get("min"),
                "determinism_output_max_abs": record.get("determinism_output", {}).get("max_abs"),
                "determinism_lse_max_abs": record.get("determinism_lse", {}).get("max_abs"),
            }
            for route, record in data["records"].items()
        }

    ncu = {}
    for shape in ("h16_s4096", "h16_s8192", "h32_s4096"):
        ncu[shape] = {
            route: parse_ncu(RESULTS / f"{PREFIX}_ncu_final_{route}_{shape}_raw.csv")
            for route in ("e16pc", "split_full")
        }

    consumer = load(f"{PREFIX}_fallback_consumer_floor.json")
    producer = load(f"{PREFIX}_producer_liveness.json")
    fallback_rows = []
    for row in consumer["rows"]:
        key = f"b{row['batch']}/s{row['seqlen']}/h{row['heads']}"
        matrix = required_map[key]
        bf16 = min(
            value for value in (
                p50(matrix["records"].get("tk_bf16_fullgrid")),
                p50(matrix["records"].get("cute_bf16")),
            ) if value is not None
        )
        payload = float(row["pv_hbm_full_payload_mib"])
        fallback_rows.append({
            "cell": key,
            "bf16_best_ms": bf16,
            "consumer_only_full_p_ms": row["pv_hbm_full_ms"],
            "producer_budget_to_match_bf16_ms": bf16 - row["pv_hbm_full_ms"],
            "full_materialized_p_mib": payload,
            "minimum_p_write_plus_read_mib": 2.0 * payload,
            "recommended_chunk_cols": row["recommended_chunk_cols"],
            "recommended_chunk_consumer_ms": row["recommended_chunk_pv_ms"],
            "recommended_peak_ring_mib": row["recommended_chunk_ring_payload_mib"],
        })

    resource_text = (RESULTS / f"{PREFIX}_resource_usage.txt").read_text()
    resources = {}
    for route, pattern in {
        "e16pc": r"config_fp4pv_stage2_ex2_alu_pchain_cILi16.*?\n\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        "split_full": r"config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_splitILi16.*?\n\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
        "split_k64": r"config_fp4pv_4wg_stage2_ex2_alu_pchain_c_score_split_k64ILi16.*?\n\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
    }.items():
        match = re.search(pattern, resource_text)
        resources[route] = None if match is None else dict(zip(
            ("registers", "stack_bytes", "static_shared_bytes", "local_bytes"),
            map(int, match.groups()),
        ))

    summary = {
        "task": "large-sequence and high-head MXFP4 adaptation",
        "result": {
            "retained_route": "split_full",
            "dispatch_gate": "batch == 1 and heads == 1 and seqlen >= 16384",
            "batch_aware_persistent_safety": "use batch * heads for S4096/S9984 thresholds",
            "split_k64_retained_as_explicit_default_off_experiment": True,
            "materialized_p_fallback_built": False,
        },
        "required_matrix": timing_rows,
        "finalist_60_sample": finalists,
        "regression57": {
            "cell_count": len(regression["cells"]),
            "worker_errors": sum("worker_error" in cell for cell in regression["cells"]),
            "status_counts": dict(reg_status),
            "historical_robust_count": len(historical_robust),
            "historical_robust_retained_count": len(retained_robust),
            "historical_robust_regressions": sorted(historical_robust - set(retained_robust)),
            "remaining_loss_cells": sorted(
                key for key, cell in regression_map.items()
                if cell.get("measured_comparison", {}).get("status") == "loss"
            ),
            "remaining_no_finite_fp4_cells": sorted(
                key for key, cell in regression_map.items()
                if not any(
                    record.get("family") == "fp4" and record.get("finite")
                    for record in cell.get("records", {}).values()
                )
            ),
        },
        "sparse_pchain": sparse_summary(),
        "ncu": ncu,
        "resources": resources,
        "mixed_stress": load(f"{PREFIX}_mixed_stress.json"),
        "fallback": {
            "consumer_floor": fallback_rows,
            "producer_liveness": producer,
            "decision": "reject: required producer is not finite/correct on representative high-head cells",
        },
        "final_defaults": load(f"{PREFIX}_final_defaults.json"),
        "branch_movement": {
            "observed_task_start": "d0e185e045c67497862b9b0732d5004561275913",
            "observed_final": "b8cc39d782de17464576d5f39ecefa2e035d6d7b",
            "commit_or_push_by_this_task": False,
        },
    }
    (RESULTS / f"{PREFIX}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
