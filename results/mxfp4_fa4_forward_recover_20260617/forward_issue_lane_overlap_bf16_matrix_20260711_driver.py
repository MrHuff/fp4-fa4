#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tk_fa4.fp4_pv_experiments import (  # noqa: E402
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
    _select_mxfp4_fwd_config_for_shape,
    _wait_for_event,
)


PREFIX = "forward_issue_lane_overlap_bf16_matrix_20260711"
EXPLICIT_CONFIGS = {
    "stage2": (
        "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_"
        "arrivereuse_pscreusefold_skippscarrive_pchainc_vtma_vstma_pstage2_q200_p112_o56_qkscfix"
    ),
    "e16pc": (
        "dualaccum_directrescale_scorepack_prescaled_floor_x1sc_fusedmax_diagzero_prepub_earlyreuse_"
        "arrivereuse_pscreusefold_skippscarrive_ex2e16pc_vtma_vstma_pstage2_q200_p112_o56_qkscfix"
    ),
    "split_full": "scorederived_ex2e16pc_split2wg_full_pstage2_q152_p112_o48",
    "split_k64": "scorederived_ex2e16pc_split2wg_k64_pstage2_q152_p112_o48",
    "split_localmax_reference": "dualaccum_directrescale_localmax_split2wg_q152_p112_o48",
}
DEFAULT_FP4_ROUTE_NAMES = ("stage2", "fp4_auto")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "p50": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "p50": float(statistics.median(values)),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def tensor_delta(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    diff = actual.float() - expected.float()
    abs_diff = diff.abs()
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(diff).item()
            / max(torch.linalg.vector_norm(expected.float()).item(), 1.0e-30)
        ),
        "finite": bool(torch.isfinite(actual.float()).all().item()),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def normalize_lse(lse: torch.Tensor) -> torch.Tensor:
    if lse.ndim == 4:
        return lse[:, :, 0, :]
    return lse


def load_cute_interface() -> Any:
    flash_root = ROOT / "flash-attention"
    if str(flash_root) not in sys.path:
        sys.path.insert(0, str(flash_root))
    import flash_attn.cute.interface as cute_interface

    return cute_interface


def make_route(
    *,
    name: str,
    family: str,
    implementation: str,
    launch_mode: str,
    out: torch.Tensor,
    lse: torch.Tensor,
    run: Callable[[], None],
    activate: Callable[[], None] | None = None,
    config: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "implementation": implementation,
        "launch_mode": launch_mode,
        "config": config,
        "out": out,
        "lse": lse,
        "run": run,
        "activate": activate or (lambda: None),
        "samples_ms": [],
        "samples_by_round_ms": [],
        "error": None,
    }


def build_routes(
    *,
    batch: int,
    seqlen: int,
    heads: int,
    device: torch.device,
    q_bf16: torch.Tensor,
    k_bf16: torch.Tensor,
    v_bf16: torch.Tensor,
    fp4_route_names: tuple[str, ...],
    fp4_launch_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    routes: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    ext = _load_forward_experiments_ext()
    raw_fp4_inputs = None
    if fp4_route_names:
        raw_fp4_inputs = _fp4_qk_mxfp4_v_inputs_from_bf16_source(
            q_bf16,
            k_bf16,
            v_bf16,
            qk_quant_backend="v5",
        )

    fp4_configs = dict(EXPLICIT_CONFIGS)
    fp4_configs["fp4_auto"] = _select_mxfp4_fwd_config_for_shape(seqlen, heads)
    for name in fp4_route_names:
        config = fp4_configs[name]
        prepared = _prepare_mxfp4_fwd_inputs_for_config(
            raw_fp4_inputs,
            seqlen=seqlen,
            config=config,
        )
        q, q_sc, q_sg, k, k_sc, k_sg, v_fp4, v_sc = prepared
        out = torch.empty(
            (batch, seqlen, heads, _D_VO),
            dtype=torch.bfloat16,
            device=device,
        )
        lse = torch.empty(
            (batch, heads, 1, seqlen),
            dtype=torch.float32,
            device=device,
        )
        launch_mode = _resolve_mxfp4_fwd_launch_mode(
            seqlen,
            heads,
            fp4_launch_mode,
        )
        persistent_launch = launch_mode != "fullgrid"

        runtime_config = (
            "auto"
            if name == "fp4_auto" and heads == 1 and seqlen >= 16384
            else config
        )

        def activate_fp4(selected: str = runtime_config) -> None:
            os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = selected

        def run_fp4(
            q: torch.Tensor = q,
            q_sc: torch.Tensor = q_sc,
            q_sg: torch.Tensor = q_sg,
            k: torch.Tensor = k,
            k_sc: torch.Tensor = k_sc,
            k_sg: torch.Tensor = k_sg,
            v_fp4: torch.Tensor = v_fp4,
            v_sc: torch.Tensor = v_sc,
            out: torch.Tensor = out,
            lse: torch.Tensor = lse,
            persistent_launch: bool = persistent_launch,
        ) -> None:
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

        routes.append(
            make_route(
                name=name,
                family="fp4",
                implementation="tk_mxfp4_streaming",
                launch_mode=launch_mode,
                config=config,
                out=out,
                lse=lse,
                run=run_fp4,
                activate=activate_fp4,
            )
        )

    if seqlen < 512:
        for mode in ("persistent", "fullgrid"):
            unsupported.append(
                {
                    "name": f"tk_bf16_{mode}",
                    "family": "bf16",
                    "implementation": "tk_bf16_causal_fa4",
                    "launch_mode": mode,
                    "reason": (
                        "source unsupported: bf16_b300_mha_causal.cu get_tile_idx computes "
                        "num_block=(S/128)/(2 softmax WGs * 2 CTA cluster), which is zero for S<512"
                    ),
                }
            )
    else:
        bf16_ext = _load_bf16_causal_baseline_ext()
        tk_modes = ["fullgrid"] if seqlen > 4096 else ["persistent", "fullgrid"]
        for mode in tk_modes:
            out = torch.empty(
                (batch, seqlen, heads, _D_VO),
                dtype=torch.bfloat16,
                device=device,
            )
            lse = torch.empty(
                (batch, heads, 1, seqlen),
                dtype=torch.float32,
                device=device,
            )
            fn = bf16_ext.forward_persistent if mode == "persistent" else bf16_ext.forward

            def run_tk_bf16(
                fn: Callable[..., None] = fn,
                out: torch.Tensor = out,
                lse: torch.Tensor = lse,
            ) -> None:
                fn(q_bf16, k_bf16, v_bf16, out, lse)

            routes.append(
                make_route(
                    name=f"tk_bf16_{mode}",
                    family="bf16",
                    implementation="tk_bf16_causal_fa4",
                    launch_mode=mode,
                    out=out,
                    lse=lse,
                    run=run_tk_bf16,
                )
            )

    try:
        cute_interface = load_cute_interface()
        out = torch.empty(
            (batch, seqlen, heads, _D_VO),
            dtype=torch.bfloat16,
            device=device,
        )
        lse = torch.empty(
            (batch, heads, seqlen),
            dtype=torch.float32,
            device=device,
        )

        def run_cute() -> None:
            cute_interface._flash_attn_fwd(
                q_bf16,
                k_bf16,
                v_bf16,
                causal=True,
                return_lse=True,
                out=out,
                lse=lse,
            )

        routes.append(
            make_route(
                name="cute_bf16",
                family="bf16",
                implementation="cute_dsl_flashattention4",
                launch_mode="native",
                out=out,
                lse=lse,
                run=run_cute,
            )
        )
    except Exception as exc:
        unsupported.append(
            {
                "name": "cute_bf16",
                "family": "bf16",
                "implementation": "cute_dsl_flashattention4",
                "launch_mode": "native",
                "reason": f"initialization failed: {type(exc).__name__}: {exc}",
            }
        )

    return routes, unsupported


def rotate(values: list[Any], offset: int) -> list[Any]:
    if not values:
        return []
    offset %= len(values)
    return values[offset:] + values[:offset]


def timed_worker(args: argparse.Namespace) -> dict[str, Any]:
    device = _benchmark_cuda_preflight(args.device, what="issue-lane BF16 matrix")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    q_bf16, k_bf16, v_bf16 = _make_live_bf16_source_inputs(
        args.seqlen,
        seed=args.seed,
        batch=args.batch,
        heads=args.heads,
        device=device,
        zero_qk=False,
    )
    requested_routes = (
        {name.strip() for name in args.routes.split(",") if name.strip()}
        if args.routes
        else None
    )
    if requested_routes is None:
        fp4_route_names = (
            DEFAULT_FP4_ROUTE_NAMES if args.seqlen < 16384 else ()
        )
    else:
        fp4_route_names = tuple(
            name
            for name in (*EXPLICIT_CONFIGS.keys(), "fp4_auto")
            if name in requested_routes
        )
    routes, unsupported = build_routes(
        batch=args.batch,
        seqlen=args.seqlen,
        heads=args.heads,
        device=device,
        q_bf16=q_bf16,
        k_bf16=k_bf16,
        v_bf16=v_bf16,
        fp4_route_names=fp4_route_names,
        fp4_launch_mode=args.fp4_launch_mode,
    )
    if requested_routes is not None:
        routes = [route for route in routes if route["name"] in requested_routes]
        unsupported = [
            route for route in unsupported if route["name"] in requested_routes
        ]
    route_order = rotate(routes, args.order_offset)

    active: list[dict[str, Any]] = []
    for route in route_order:
        try:
            route["activate"]()
            route["run"]()
            torch.cuda.synchronize(device)
            for _ in range(args.warmup - 1):
                route["run"]()
            torch.cuda.synchronize(device)
            active.append(route)
        except Exception as exc:
            route["error"] = f"warmup failed: {type(exc).__name__}: {exc}"

    samples_per_round = args.samples // 2
    extra_samples = args.samples - 2 * samples_per_round
    stream = torch.cuda.current_stream(device=device)
    for round_idx in range(2):
        order = active if round_idx == 0 else list(reversed(active))
        count = samples_per_round + (extra_samples if round_idx == 1 else 0)
        for route in order:
            round_samples: list[float] = []
            route["samples_by_round_ms"].append(round_samples)
            try:
                route["activate"]()
                for _ in range(count):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    route["run"]()
                    end.record(stream)
                    _wait_for_event(
                        end,
                        timeout_ms=args.timeout_ms,
                        what=f"matrix timing {route['name']}",
                    )
                    elapsed = float(start.elapsed_time(end))
                    route["samples_ms"].append(elapsed)
                    round_samples.append(elapsed)
            except Exception as exc:
                route["error"] = f"timing failed: {type(exc).__name__}: {exc}"

    route_map = {route["name"]: route for route in routes}
    for route in active:
        if route["error"] is not None:
            continue
        try:
            route["activate"]()
            route["run"]()
            torch.cuda.synchronize(device)
            out_snapshot = route["out"].clone()
            lse_snapshot = normalize_lse(route["lse"]).clone()
            route["run"]()
            torch.cuda.synchronize(device)
            route["determinism_output"] = tensor_delta(route["out"], out_snapshot)
            route["determinism_lse"] = tensor_delta(
                normalize_lse(route["lse"]),
                lse_snapshot,
            )
            del out_snapshot, lse_snapshot
        except Exception as exc:
            route["error"] = f"determinism failed: {type(exc).__name__}: {exc}"

    reference = route_map.get("cute_bf16")
    if reference is None or reference.get("error") is not None:
        reference = next(
            (
                route
                for route in routes
                if route["family"] == "bf16" and route.get("error") is None
            ),
            None,
        )

    if reference is not None:
        ref_out = reference["out"]
        ref_lse = normalize_lse(reference["lse"])
        for route in routes:
            if route.get("error") is None:
                route["vs_bf16_reference"] = _compare_outputs(
                    route["out"],
                    normalize_lse(route["lse"]),
                    ref_out,
                    ref_lse,
                )

    records: dict[str, Any] = {}
    for route in routes:
        finite = False
        if route.get("error") is None:
            finite = bool(
                torch.isfinite(route["out"].float()).all().item()
                and torch.isfinite(normalize_lse(route["lse"])).all().item()
            )
        deterministic = bool(
            route.get("determinism_output", {}).get("exact", False)
            and route.get("determinism_lse", {}).get("exact", False)
        )
        comparison = route.get("vs_bf16_reference")
        correct = finite
        if comparison is not None:
            if route["family"] == "fp4":
                correct = correct and comparison["max_abs_diff"] <= 2.0
                correct = correct and comparison["lse_max_abs_diff"] <= 0.1
            else:
                correct = correct and comparison["max_abs_diff"] <= 0.05
                correct = correct and comparison["lse_max_abs_diff"] <= 0.05
        timing = summary(route["samples_ms"])
        timing_rounds = [summary(values) for values in route["samples_by_round_ms"]]
        complete_timing = timing["count"] == args.samples
        record = {
            key: route.get(key)
            for key in (
                "name",
                "family",
                "implementation",
                "launch_mode",
                "config",
                "error",
                "determinism_output",
                "determinism_lse",
                "vs_bf16_reference",
                "vs_e16pc_output",
                "vs_e16pc_lse",
            )
            if route.get(key) is not None
        }
        record.update(
            {
                "samples_ms": route["samples_ms"],
                "samples_by_round_ms": route["samples_by_round_ms"],
                "timing_ms": timing,
                "timing_by_round_ms": timing_rounds,
                "finite": finite,
                "correct": correct,
                "deterministic": deterministic,
                "complete_timing": complete_timing,
                "eligible": bool(
                    finite and correct and deterministic and complete_timing
                ),
            }
        )
        records[route["name"]] = record

    for item in unsupported:
        records[item["name"]] = {
            **item,
            "finite": False,
            "correct": False,
            "deterministic": False,
            "complete_timing": False,
            "eligible": False,
            "unsupported": True,
            "samples_ms": [],
            "timing_ms": summary([]),
        }

    def fastest(family: str, *, eligible_only: bool) -> dict[str, Any] | None:
        candidates = [
            record
            for record in records.values()
            if record.get("family") == family
            and record.get("timing_ms", {}).get("p50") is not None
            and (record.get("eligible", False) if eligible_only else record.get("finite", False))
            and (record.get("correct", False) if not eligible_only else True)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda record: float(record["timing_ms"]["p50"]))

    bf16_best = fastest("bf16", eligible_only=True)
    fp4_best = fastest("fp4", eligible_only=True)
    bf16_measured = fastest("bf16", eligible_only=False)
    fp4_measured = fastest("fp4", eligible_only=False)

    def comparison_payload(
        bf16: dict[str, Any] | None,
        fp4: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if bf16 is None or fp4 is None:
            return None
        bf16_ms = float(bf16["timing_ms"]["p50"])
        fp4_ms = float(fp4["timing_ms"]["p50"])
        speedup = bf16_ms / fp4_ms
        return {
            "bf16_route": bf16["name"],
            "bf16_launch_mode": bf16["launch_mode"],
            "bf16_p50_ms": bf16_ms,
            "fp4_route": fp4["name"],
            "fp4_launch_mode": fp4["launch_mode"],
            "fp4_p50_ms": fp4_ms,
            "speedup": speedup,
            "margin_us": (bf16_ms - fp4_ms) * 1000.0,
            "status": (
                "robust_win"
                if speedup > 1.02
                else "win"
                if speedup > 1.0
                else "parity"
                if speedup >= 0.98
                else "loss"
            ),
        }

    strict_comparison = comparison_payload(bf16_best, fp4_best)
    measured_comparison = comparison_payload(bf16_measured, fp4_measured)
    result = {
        "shape": {
            "batch": args.batch,
            "seqlen": args.seqlen,
            "heads": args.heads,
            "dqk": 192,
            "dvo": 128,
            "causal": True,
        },
        "contract": {
            "quantization_timed": False,
            "inputs_precreated": True,
            "outputs_preallocated": True,
            "kernel_only": True,
        },
        "seed": args.seed,
        "device": str(device),
        "device_name": props.name,
        "warmup": args.warmup,
        "samples": args.samples,
        "order_offset": args.order_offset,
        "route_order": [route["name"] for route in route_order],
        "bf16_reference": reference["name"] if reference is not None else None,
        "records": records,
        "strict_comparison": strict_comparison,
        "measured_comparison": measured_comparison,
    }
    return result


def matrix_cells() -> list[tuple[int, int, int]]:
    cells = [
        (1, seqlen, heads)
        for seqlen in (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
        for heads in (1, 2, 4, 8, 16, 32)
    ]
    cells.extend(
        [
            (1, 32768, 1),
            (1, 32768, 4),
            (1, 32768, 16),
            (1, 4096, 64),
            (1, 8192, 64),
            (2, 1024, 8),
            (2, 4096, 8),
            (4, 1024, 4),
            (4, 4096, 4),
        ]
    )
    return cells


def parse_cells(value: str | None) -> list[tuple[int, int, int]]:
    if not value:
        return matrix_cells()
    parsed: list[tuple[int, int, int]] = []
    for item in value.split(","):
        batch_s, seqlen_s, heads_s = item.split(":")
        parsed.append((int(batch_s), int(seqlen_s), int(heads_s)))
    return parsed


def gpu2_processes() -> str:
    command = [
        "nvidia-smi",
        "-i",
        "2",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.stdout.strip()


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    cells = parse_cells(args.cells)
    aggregate: dict[str, Any] = {
        "task": "issue-lane overlap and broad fastest-BF16 matrix",
        "created_unix": time.time(),
        "driver": str(Path(__file__).resolve()),
        "warmup": args.warmup,
        "samples": args.samples,
        "cells_requested": len(cells),
        "cells": [],
    }
    if args.json.exists() and args.resume:
        aggregate = json.loads(args.json.read_text())
    completed_keys = {
        (
            int(item["shape"]["batch"]),
            int(item["shape"]["seqlen"]),
            int(item["shape"]["heads"]),
        )
        for item in aggregate.get("cells", [])
        if "shape" in item
    }

    for index, (batch, seqlen, heads) in enumerate(cells):
        key = (batch, seqlen, heads)
        if key in completed_keys:
            continue
        busy = gpu2_processes()
        cell_path = RESULTS / (
            f"{PREFIX}_cell_b{batch}_s{seqlen}_h{heads}.json"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--batch",
            str(batch),
            "--seqlen",
            str(seqlen),
            "--heads",
            str(heads),
            "--seed",
            str(args.seed + index),
            "--device",
            args.device,
            "--warmup",
            str(args.warmup),
            "--samples",
            str(args.samples),
            "--order-offset",
            str(index),
            "--timeout-ms",
            str(args.timeout_ms),
            "--json",
            str(cell_path),
        ]
        if args.routes:
            command.extend(["--routes", args.routes])
        command.extend(["--fp4-launch-mode", args.fp4_launch_mode])
        timeout_s = 900 if seqlen >= 16384 or batch > 1 else 420
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if completed.returncode == 0 and cell_path.exists():
                cell = json.loads(cell_path.read_text())
            else:
                cell = {
                    "shape": {"batch": batch, "seqlen": seqlen, "heads": heads},
                    "worker_error": f"exit {completed.returncode}",
                    "worker_stdout_tail": completed.stdout[-4000:],
                    "worker_stderr_tail": completed.stderr[-4000:],
                }
        except subprocess.TimeoutExpired as exc:
            cell = {
                "shape": {"batch": batch, "seqlen": seqlen, "heads": heads},
                "worker_error": f"timeout after {timeout_s}s",
                "worker_stdout_tail": (exc.stdout or "")[-4000:]
                if isinstance(exc.stdout, str)
                else "",
                "worker_stderr_tail": (exc.stderr or "")[-4000:]
                if isinstance(exc.stderr, str)
                else "",
            }
        cell["gpu2_processes_before"] = busy
        aggregate.setdefault("cells", []).append(cell)
        write_json(args.json, aggregate)
        comparison = cell.get("strict_comparison") or cell.get("measured_comparison")
        compact = {
            "cell": f"b{batch}/s{seqlen}/h{heads}",
            "comparison": comparison,
            "worker_error": cell.get("worker_error"),
        }
        print(json.dumps(compact), flush=True)

    aggregate["completed_unix"] = time.time()
    write_json(args.json, aggregate)
    return aggregate


def run_isolated(args: argparse.Namespace) -> dict[str, Any]:
    if not args.routes:
        raise ValueError("--isolate requires --routes")
    cells = parse_cells(args.cells)
    route_names = [name.strip() for name in args.routes.split(",") if name.strip()]
    aggregate: dict[str, Any] = {
        "task": "fresh-process isolated route matrix",
        "created_unix": time.time(),
        "driver": str(Path(__file__).resolve()),
        "warmup": args.warmup,
        "samples": args.samples,
        "fp4_launch_mode": args.fp4_launch_mode,
        "results": [],
    }
    if args.json.exists() and args.resume:
        aggregate = json.loads(args.json.read_text())
    completed = {
        (
            int(item["shape"]["batch"]),
            int(item["shape"]["seqlen"]),
            int(item["shape"]["heads"]),
            item["route"],
            item["fp4_launch_mode"],
        )
        for item in aggregate.get("results", [])
    }

    for cell_index, (batch, seqlen, heads) in enumerate(cells):
        for route_index, route in enumerate(route_names):
            key = (batch, seqlen, heads, route, args.fp4_launch_mode)
            if key in completed:
                continue
            busy = gpu2_processes()
            route_path = RESULTS / (
                f"{PREFIX}_isolated_{route}_{args.fp4_launch_mode}_"
                f"b{batch}_s{seqlen}_h{heads}.json"
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--batch",
                str(batch),
                "--seqlen",
                str(seqlen),
                "--heads",
                str(heads),
                "--seed",
                str(args.seed + cell_index),
                "--device",
                args.device,
                "--warmup",
                str(args.warmup),
                "--samples",
                str(args.samples),
                "--order-offset",
                str(route_index),
                "--routes",
                route,
                "--fp4-launch-mode",
                args.fp4_launch_mode,
                "--timeout-ms",
                str(args.timeout_ms),
                "--json",
                str(route_path),
            ]
            started = time.time()
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=args.process_timeout_s,
                    check=False,
                )
                status = "completed" if result.returncode == 0 else "failed"
                payload = json.loads(route_path.read_text()) if route_path.exists() else None
                item = {
                    "shape": {"batch": batch, "seqlen": seqlen, "heads": heads},
                    "route": route,
                    "fp4_launch_mode": args.fp4_launch_mode,
                    "status": status,
                    "returncode": result.returncode,
                    "elapsed_s": time.time() - started,
                    "artifact": str(route_path),
                    "payload": payload,
                    "stdout_tail": result.stdout[-2000:],
                    "stderr_tail": result.stderr[-2000:],
                    "gpu2_processes_before": busy,
                }
            except subprocess.TimeoutExpired as exc:
                route_path.unlink(missing_ok=True)
                item = {
                    "shape": {"batch": batch, "seqlen": seqlen, "heads": heads},
                    "route": route,
                    "fp4_launch_mode": args.fp4_launch_mode,
                    "status": "timeout",
                    "elapsed_s": time.time() - started,
                    "timeout_s": args.process_timeout_s,
                    "artifact": None,
                    "stdout_tail": (exc.stdout or b"")[-2000:].decode(errors="replace")
                    if isinstance(exc.stdout, bytes)
                    else (exc.stdout or "")[-2000:],
                    "stderr_tail": (exc.stderr or b"")[-2000:].decode(errors="replace")
                    if isinstance(exc.stderr, bytes)
                    else (exc.stderr or "")[-2000:],
                    "gpu2_processes_before": busy,
                }
            aggregate.setdefault("results", []).append(item)
            write_json(args.json, aggregate)
            print(
                json.dumps(
                    {
                        "cell": f"b{batch}/s{seqlen}/h{heads}",
                        "route": route,
                        "mode": args.fp4_launch_mode,
                        "status": item["status"],
                        "elapsed_s": item["elapsed_s"],
                    }
                ),
                flush=True,
            )

    aggregate["completed_unix"] = time.time()
    write_json(args.json, aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--isolate", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=94601)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--order-offset", type=int, default=0)
    parser.add_argument("--routes")
    parser.add_argument(
        "--fp4-launch-mode",
        choices=("auto", "persistent", "fullgrid"),
        default="auto",
    )
    parser.add_argument("--timeout-ms", type=float, default=30000.0)
    parser.add_argument("--cells")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--process-timeout-s", type=float, default=45.0)
    parser.add_argument(
        "--json",
        type=Path,
        default=RESULTS / f"{PREFIX}_matrix.json",
    )
    args = parser.parse_args()

    try:
        if args.worker:
            result = timed_worker(args)
            write_json(args.json, result)
            print(
                json.dumps(
                    {
                        "shape": result["shape"],
                        "strict_comparison": result["strict_comparison"],
                        "measured_comparison": result["measured_comparison"],
                    },
                    indent=2,
                )
            )
        elif args.isolate:
            run_isolated(args)
        else:
            run_matrix(args)
    except Exception as exc:
        failure = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "shape": {
                "batch": args.batch,
                "seqlen": args.seqlen,
                "heads": args.heads,
            },
        }
        write_json(args.json, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
