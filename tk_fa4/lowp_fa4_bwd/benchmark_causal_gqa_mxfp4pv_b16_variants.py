#!/usr/bin/env python3
"""Compare anchored and unanchored B16 MXFP4-PV on fixed operands."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from tk_fa4.lowp_fa4_bwd.authenticate_causal_gqa_mxfp4pv_forward import (
    authenticate_artifact,
    load_canonical_extension,
    require_topology,
)
from tk_fa4.lowp_fa4_bwd.validate_causal_gqa_fp8pv_batch import (
    _authenticate_projection_artifact,
    _byte_equal,
    _decode_nvfp4_qk,
    _load,
    _make_rope,
    _require_exact_forward_topology,
)


BATCH = 16
SEQUENCE = 4096
HIDDEN = 2048
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 64
SEED = 20260825
GIB = 1 << 30
REPO_ROOT = _BOOTSTRAP_ROOT
FLASH_ATTN_ROOT = REPO_ROOT / "flash-attention"

ARTIFACTS = {
    "aggressive": {
        "path": Path(
            "/tmp/_C_cfwd_mx_d4q01_unanchored_splitmix_v6_"
            "b16s4096h32kv8d64_sm100_20260825."
            "cpython-312-aarch64-linux-gnu.so"
        ),
        "module": (
            "_C_cfwd_mx_d4q01_unanchored_splitmix_v6_"
            "b16s4096h32kv8d64_sm100_20260825"
        ),
        "sha256": (
            "93488ece199812bbd001d9e1f662db79a"
            "c39ecc230d91e8f0de2c2e4321976d3"
        ),
        "bytes": 1_958_304,
    },
    "anchored": {
        "path": Path(
            "/tmp/_C_cfwd_mx_d4q01_i1_b16s4096h32kv8d64_20260825."
            "cpython-312-aarch64-linux-gnu.so"
        ),
        "module": "_C_cfwd_mx_d4q01_i1_b16s4096h32kv8d64_20260825",
        "sha256": (
            "cc06fe4337fdc3a7c900f81d68fabc4a"
            "8e0c375ea536fbe6405754237a393717"
        ),
        "bytes": 1_958_000,
    },
    "fp8": {
        "path": Path(
            "/tmp/_C_cfwd_fp8exact0_b16_s4096h32kv8d64_sm100_"
            "topofix_b200_20260825.cpython-312-aarch64-linux-gnu.so"
        ),
        "module": (
            "_C_cfwd_fp8exact0_b16_s4096h32kv8d64_sm100_"
            "topofix_b200_20260825"
        ),
        "sha256": (
            "88d81d3783e5aa80f0e9cf259a2ea7c9"
            "35da4c2a5dc3ba1868e63f802a2c6208"
        ),
        "bytes": 1_817_256,
    },
    "projection": {
        "path": Path(
            "/tmp/fa4-dolma3-d64-assets.QZwFvk/assets/"
            "_C_b300_lowp_bwd.cpython-312-aarch64-linux-gnu.so"
        ),
        "sha256": (
            "bfdec1e43a0a19acec5afbac3fa837e2"
            "f4d1b25be80ae7fb5ff3b5bc5e9e25ce"
        ),
        "bytes": 17_504_688,
    },
}


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing_summary(values: Sequence[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    return {
        "unit": "milliseconds",
        "samples": len(values),
        "p10_ms": _percentile(values, 0.10),
        "p50_ms": statistics.median(values),
        "p90_ms": _percentile(values, 0.90),
        "mean_ms": mean,
        "cv": statistics.stdev(values) / mean if len(values) > 1 else 0.0,
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "samples_ms": list(values),
    }


def _time_rotated(
    runners: dict[str, Callable[[], None]],
    *,
    warmups: int,
    samples: int,
) -> tuple[dict[str, dict[str, Any]], list[list[str]]]:
    import torch

    names = tuple(runners)
    for iteration in range(warmups):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            runners[name]()
        torch.cuda.synchronize()
    values: dict[str, list[float]] = {name: [] for name in names}
    orders = []
    for iteration in range(samples):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        orders.append(list(order))
        events = []
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runners[name]()
            end.record()
            events.append((name, start, end))
        events[-1][2].synchronize()
        for name, start, end in events:
            values[name].append(float(start.elapsed_time(end)))
    return (
        {name: _timing_summary(values[name]) for name in names},
        orders,
    )


def _metrics(reference: Any, actual: Any) -> dict[str, float | bool]:
    import torch

    reference_flat = reference.reshape(-1)
    actual_flat = actual.reshape(-1)
    chunk = 1 << 20
    dot = reference_sq = actual_sq = difference_sq = 0.0
    maximum = 0.0
    absolute_sum = 0.0
    finite = True
    for offset in range(0, reference_flat.numel(), chunk):
        reference_part = reference_flat[offset : offset + chunk].float()
        actual_part = actual_flat[offset : offset + chunk].float()
        difference = actual_part - reference_part
        dot += float(torch.dot(reference_part, actual_part))
        reference_sq += float(torch.dot(reference_part, reference_part))
        actual_sq += float(torch.dot(actual_part, actual_part))
        difference_sq += float(torch.dot(difference, difference))
        absolute_sum += float(difference.abs().sum())
        maximum = max(maximum, float(difference.abs().max()))
        finite = finite and bool(torch.isfinite(actual_part).all())
    reference_norm = math.sqrt(max(reference_sq, 1.0e-40))
    actual_norm = math.sqrt(max(actual_sq, 1.0e-40))
    return {
        "finite": finite,
        "cosine": dot / (reference_norm * actual_norm),
        "relative_l2": math.sqrt(difference_sq) / reference_norm,
        "norm_ratio": actual_norm / reference_norm,
        "mean_abs": absolute_sum / reference_flat.numel(),
        "max_abs": maximum,
    }


def _apply_pair_rope(tensor: Any, cosine: Any, sine: Any) -> Any:
    import torch

    pairs = tensor.float().reshape(*tensor.shape[:-1], HEAD_DIM // 2, 2)
    first = pairs[..., 0]
    second = pairs[..., 1]
    cosine_f = cosine.float().unsqueeze(2)
    sine_f = sine.float().unsqueeze(2)
    return torch.stack(
        (
            first * cosine_f - second * sine_f,
            first * sine_f + second * cosine_f,
        ),
        dim=-1,
    ).flatten(-2).bfloat16().contiguous()


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    values = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(values.mean()),
        "rms": float(values.square().mean().sqrt()),
        "max_abs": float(values.abs().max()),
    }


def _identity(path: Path, sha256: str, byte_count: int) -> dict[str, Any]:
    return authenticate_artifact(
        path,
        expected_sha256=sha256,
        expected_bytes=byte_count,
    )


def _sample_isolation(
    baseline: dict[str, tuple[Any, Any]],
    changed: dict[str, tuple[Any, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"perturbed_sample": BATCH - 1, "routes": {}}
    for route in baseline:
        output, lse = baseline[route]
        changed_output, changed_lse = changed[route]
        untouched = {
            f"sample_{index}": {
                "output_byte_equal": _byte_equal(
                    output[index], changed_output[index]
                ),
                "lse_byte_equal": _byte_equal(lse[index], changed_lse[index]),
            }
            for index in range(BATCH - 1)
        }
        result["routes"][route] = {
            "untouched_samples": untouched,
            "perturbed_output_changed": not _byte_equal(
                output[-1], changed_output[-1]
            ),
            "perturbed_lse_changed": not _byte_equal(lse[-1], changed_lse[-1]),
        }
    result["passed"] = all(
        all(all(fields.values()) for fields in route["untouched_samples"].values())
        and route["perturbed_output_changed"]
        and route["perturbed_lse_changed"]
        for route in result["routes"].values()
    )
    return result


def _leakage_result(baseline: Any, changed: Any, cutoff: int) -> dict[str, Any]:
    prefix_equal = _byte_equal(baseline[:, :cutoff], changed[:, :cutoff])
    suffix_equal = _byte_equal(baseline[:, cutoff:], changed[:, cutoff:])
    return {
        "prefix_output_byte_equal": prefix_equal,
        "suffix_output_changed": not suffix_equal,
        "prefix_metrics": _metrics(baseline[:, :cutoff], changed[:, :cutoff]),
        "passed": prefix_equal and not suffix_equal,
    }


def main() -> None:
    import torch
    import torch.nn.functional as F

    warmups = 10
    samples = 41
    gpu = 0
    projection = ARTIFACTS["projection"]
    projection_path, projection_identity = _authenticate_projection_artifact(
        projection["path"],
        supplied_sha256=projection["sha256"],
        supplied_bytes=projection["bytes"],
    )
    os.environ["TK_FA4_LOWP_BWD_EXTENSION_SOURCE"] = str(projection_path)
    import tk_fa4.interface as tk_interface

    projection_extension = _load(projection_path, "_C_b300_lowp_bwd")
    tk_interface._C_b300_lowp_bwd = projection_extension
    tk_interface._LOWP_BWD_IMPORT_ERROR = None
    identities = {
        name: _identity(spec["path"], spec["sha256"], spec["bytes"])
        for name, spec in ARTIFACTS.items()
        if name != "projection"
    }

    sys.path.insert(0, str(FLASH_ATTN_ROOT))
    try:
        from flash_attn.cute import interface as flash_interface
    finally:
        sys.path.pop(0)
    from tk_fa4 import (
        b300_pack_gqa_d64_paired_rope,
        b300_prepare_e4m3_projection_operand,
        b300_prepare_e4m3_projection_weight,
        b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3,
    )

    torch.cuda.set_device(gpu)
    free_before, total_memory = torch.cuda.mem_get_info(gpu)
    if free_before < 150 * GIB:
        raise RuntimeError(f"GPU0 has only {free_before / GIB:.2f} GiB free")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    aggressive, aggressive_topology, aggressive_identity = (
        load_canonical_extension(
            ARTIFACTS["aggressive"]["path"],
            variant="unanchored-splitmix-v6",
            batch=BATCH,
        )
    )
    anchored = _load(
        ARTIFACTS["anchored"]["path"],
        ARTIFACTS["anchored"]["module"],
    )
    fp8 = _load(ARTIFACTS["fp8"]["path"], ARTIFACTS["fp8"]["module"])
    anchored_topology = dict(anchored.read_hao_direct_topology())
    fp8_topology = dict(fp8.read_hao_direct_topology())
    require_topology(anchored_topology, variant="anchored", batch=BATCH)
    _require_exact_forward_topology(
        fp8_topology,
        batch=BATCH,
        sequence=SEQUENCE,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
    )

    rows = torch.randn(
        BATCH * SEQUENCE,
        HIDDEN,
        device="cuda",
        dtype=torch.float32,
    ).bfloat16()
    total_width = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
    weight = (
        torch.randn(
            total_width,
            HIDDEN,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    ).bfloat16()
    cosine, sine = _make_rope(BATCH, SEQUENCE, "cuda")
    packed_rope = b300_pack_gqa_d64_paired_rope(cosine, sine)
    qk_scales = torch.zeros(
        BATCH,
        Q_HEADS // 2,
        7,
        device="cuda",
        dtype=torch.float32,
    )
    qk_scales[..., 0] = 2.25
    qk_scales[..., 1] = 2.0
    input_operand = tuple(b300_prepare_e4m3_projection_operand(rows))
    weight_operand = tuple(b300_prepare_e4m3_projection_weight(weight))

    def publish(current_rows: Any, *, mx: bool) -> Any:
        current_input = (
            input_operand
            if current_rows.data_ptr() == rows.data_ptr()
            else tuple(b300_prepare_e4m3_projection_operand(current_rows))
        )
        return b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(
            current_input,
            weight_operand,
            qk_scales,
            packed_rope,
            batch=BATCH,
            seqlen=SEQUENCE,
            q_heads=Q_HEADS,
            kv_heads=KV_HEADS,
            publish_mxfp4_v=mx,
            interleave_causal_kv=mx,
            represented_backward=True,
            per_block_qk_scales=True,
            experimental_split_v_backward=mx,
        )

    exact_bundle = publish(rows, mx=False)
    mx_bundle = publish(rows, mx=True)
    q_weight, k_weight, v_weight = torch.split(
        weight,
        (Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM),
    )
    dense_q = _apply_pair_rope(
        F.linear(rows, q_weight).reshape(BATCH, SEQUENCE, Q_HEADS, HEAD_DIM),
        cosine,
        sine,
    )
    dense_k = _apply_pair_rope(
        F.linear(rows, k_weight).reshape(BATCH, SEQUENCE, KV_HEADS, HEAD_DIM),
        cosine,
        sine,
    )
    dense_v = F.linear(rows, v_weight).reshape(
        BATCH, SEQUENCE, KV_HEADS, HEAD_DIM
    ).contiguous()
    represented_q = _decode_nvfp4_qk(
        exact_bundle.backward.score_q_fp4,
        exact_bundle.q_forward_scales,
        exact_bundle.q_forward_global_scale,
        scale_tile_rows=128,
    ).permute(0, 2, 1, 3).contiguous().bfloat16()
    represented_k = _decode_nvfp4_qk(
        exact_bundle.backward.score_k_fp4,
        exact_bundle.k_forward_scales,
        exact_bundle.k_forward_global_scale,
        scale_tile_rows=64,
    ).permute(0, 2, 1, 3).contiguous().bfloat16()
    represented_v = (
        exact_bundle.v_forward_fp8.permute(0, 3, 1, 2)
        .contiguous()
        .float()
        .mul(0.25)
        .bfloat16()
    )

    output_shape = (BATCH, SEQUENCE, Q_HEADS, HEAD_DIM)
    lowp_lse_shape = (BATCH, Q_HEADS, 1, SEQUENCE)
    outputs = {
        name: torch.empty(output_shape, device="cuda", dtype=torch.bfloat16)
        for name in ("aggressive", "anchored", "fp8", "bf16", "represented_bf16")
    }
    lses = {
        name: torch.empty(
            lowp_lse_shape,
            device="cuda",
            dtype=torch.float32,
        )
        for name in ("aggressive", "anchored", "fp8")
    }
    lses["bf16"] = torch.empty(
        BATCH, Q_HEADS, SEQUENCE, device="cuda", dtype=torch.float32
    )
    lses["represented_bf16"] = torch.empty_like(lses["bf16"])

    def run_mx(
        extension: Any,
        name: str,
        bundle: Any = mx_bundle,
        *,
        store_lse: bool = True,
        output: Any | None = None,
        lse: Any | None = None,
        v: Any | None = None,
    ) -> None:
        operands = list(bundle.forward_operands())
        if v is not None:
            operands[6] = v
        extension.forward_hao_direct_fp4pv(
            *operands,
            outputs[name] if output is None else output,
            lses[name] if lse is None else lse,
            0,
            True,
            store_lse,
        )

    def run_fp8(
        bundle: Any = exact_bundle,
        *,
        store_lse: bool = True,
        output: Any | None = None,
        lse: Any | None = None,
        v: Any | None = None,
    ) -> None:
        fp8.forward_hao_direct_fp8pv(
            *bundle.qk_forward_operands(),
            bundle.v_forward_fp8 if v is None else v,
            outputs["fp8"] if output is None else output,
            lses["fp8"] if lse is None else lse,
            0,
            True,
            store_lse,
        )

    def run_bf16(
        name: str,
        q: Any,
        k: Any,
        v: Any,
        *,
        store_lse: bool = True,
        output: Any | None = None,
        lse: Any | None = None,
    ) -> None:
        flash_interface._flash_attn_fwd(
            q,
            k,
            v,
            out=outputs[name] if output is None else output,
            lse=lses[name] if lse is None else lse,
            causal=True,
            return_lse=store_lse,
            num_splits=1,
            pack_gqa=False,
            _arch=100,
        )

    def run_all(*, store_lse: bool = True) -> None:
        run_mx(aggressive, "aggressive", store_lse=store_lse)
        run_mx(anchored, "anchored", store_lse=store_lse)
        run_fp8(store_lse=store_lse)
        run_bf16("bf16", dense_q, dense_k, dense_v, store_lse=store_lse)

    run_all()
    run_bf16(
        "represented_bf16",
        represented_q,
        represented_k,
        represented_v,
    )
    torch.cuda.synchronize()
    aggressive_topology = dict(aggressive.read_hao_direct_topology())
    anchored_topology = dict(anchored.read_hao_direct_topology())
    fp8_topology = dict(fp8.read_hao_direct_topology())
    require_topology(
        aggressive_topology,
        variant="unanchored-splitmix-v6",
        batch=BATCH,
        runtime_populated=True,
    )
    require_topology(
        anchored_topology,
        variant="anchored",
        batch=BATCH,
        runtime_populated=True,
    )
    _require_exact_forward_topology(
        fp8_topology,
        batch=BATCH,
        sequence=SEQUENCE,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        runtime_populated=True,
    )

    timing_runners = {
        "aggressive_unanchored_mx": lambda: run_mx(aggressive, "aggressive"),
        "anchored_mx": lambda: run_mx(anchored, "anchored"),
        "exact_fp8": run_fp8,
        "dense_bf16_fa4": lambda: run_bf16("bf16", dense_q, dense_k, dense_v),
    }
    timings, timing_orders = _time_rotated(
        timing_runners,
        warmups=warmups,
        samples=samples,
    )
    no_lse_runners = {
        "aggressive_unanchored_mx": lambda: run_mx(
            aggressive, "aggressive", store_lse=False
        ),
        "anchored_mx": lambda: run_mx(
            anchored, "anchored", store_lse=False
        ),
        "exact_fp8": lambda: run_fp8(store_lse=False),
        "dense_bf16_fa4": lambda: run_bf16(
            "bf16", dense_q, dense_k, dense_v, store_lse=False
        ),
    }
    no_lse_timings, no_lse_orders = _time_rotated(
        no_lse_runners,
        warmups=warmups,
        samples=samples,
    )

    run_all()
    run_bf16(
        "represented_bf16",
        represented_q,
        represented_k,
        represented_v,
    )
    torch.cuda.synchronize()
    baseline = {
        name: (outputs[name].clone(), lses[name].clone())
        for name in outputs
    }
    correctness = {}
    for reference_name in ("bf16", "represented_bf16", "anchored", "fp8"):
        reference_output, reference_lse = baseline[reference_name]
        if reference_lse.ndim == 3:
            reference_lse = reference_lse.unsqueeze(2)
        for actual_name in ("aggressive",):
            correctness[f"{actual_name}_vs_{reference_name}"] = {
                "output": _metrics(reference_output, baseline[actual_name][0]),
                "lse": _metrics(reference_lse, baseline[actual_name][1]),
            }
    for actual_name in ("anchored", "fp8"):
        correctness[f"{actual_name}_vs_bf16"] = {
            "output": _metrics(baseline["bf16"][0], baseline[actual_name][0]),
            "lse": _metrics(
                baseline["bf16"][1].unsqueeze(2), baseline[actual_name][1]
            ),
        }

    changed_rows = rows.clone()
    changed_rows.view(BATCH, SEQUENCE, HIDDEN)[-1].mul_(0.5).add_(0.25)
    changed_exact = publish(changed_rows, mx=False)
    changed_mx = publish(changed_rows, mx=True)
    changed_q = _apply_pair_rope(
        F.linear(changed_rows, q_weight).reshape(
            BATCH, SEQUENCE, Q_HEADS, HEAD_DIM
        ),
        cosine,
        sine,
    )
    changed_k = _apply_pair_rope(
        F.linear(changed_rows, k_weight).reshape(
            BATCH, SEQUENCE, KV_HEADS, HEAD_DIM
        ),
        cosine,
        sine,
    )
    changed_v = F.linear(changed_rows, v_weight).reshape(
        BATCH, SEQUENCE, KV_HEADS, HEAD_DIM
    ).contiguous()
    changed_outputs = {
        name: torch.empty_like(outputs[name])
        for name in ("aggressive", "anchored", "fp8", "bf16")
    }
    changed_lses = {
        name: torch.empty_like(lses[name])
        for name in ("aggressive", "anchored", "fp8", "bf16")
    }
    run_mx(
        aggressive,
        "aggressive",
        changed_mx,
        output=changed_outputs["aggressive"],
        lse=changed_lses["aggressive"],
    )
    run_mx(
        anchored,
        "anchored",
        changed_mx,
        output=changed_outputs["anchored"],
        lse=changed_lses["anchored"],
    )
    run_fp8(
        changed_exact,
        output=changed_outputs["fp8"],
        lse=changed_lses["fp8"],
    )
    run_bf16(
        "bf16",
        changed_q,
        changed_k,
        changed_v,
        output=changed_outputs["bf16"],
        lse=changed_lses["bf16"],
    )
    torch.cuda.synchronize()
    isolation = _sample_isolation(
        {name: baseline[name] for name in changed_outputs},
        {
            name: (changed_outputs[name], changed_lses[name])
            for name in changed_outputs
        },
    )

    cutoff = SEQUENCE // 2
    future_dense_v = dense_v.clone()
    future_dense_v[:, cutoff:] = 0
    future_fp8_v = exact_bundle.v_forward_fp8.clone()
    future_fp8_v[..., cutoff:] = 0
    future_mx_v = mx_bundle.v_forward_fp4.clone()
    future_mx_v.view(torch.uint8)[..., cutoff // 2 :] = 0
    leakage_outputs = {
        name: torch.empty_like(outputs[name])
        for name in ("aggressive", "anchored", "fp8", "bf16")
    }
    leakage_lses = {
        name: torch.empty_like(lses[name])
        for name in ("aggressive", "anchored", "fp8", "bf16")
    }
    run_mx(
        aggressive,
        "aggressive",
        output=leakage_outputs["aggressive"],
        lse=leakage_lses["aggressive"],
        v=future_mx_v,
    )
    run_mx(
        anchored,
        "anchored",
        output=leakage_outputs["anchored"],
        lse=leakage_lses["anchored"],
        v=future_mx_v,
    )
    run_fp8(
        output=leakage_outputs["fp8"],
        lse=leakage_lses["fp8"],
        v=future_fp8_v,
    )
    run_bf16(
        "bf16",
        dense_q,
        dense_k,
        future_dense_v,
        output=leakage_outputs["bf16"],
        lse=leakage_lses["bf16"],
    )
    torch.cuda.synchronize()
    leakage = {
        "cutoff": cutoff,
        "routes": {
            name: {
                **_leakage_result(
                    baseline[name][0], leakage_outputs[name], cutoff
                ),
                "lse_full_byte_equal": _byte_equal(
                    baseline[name][1], leakage_lses[name]
                ),
            }
            for name in leakage_outputs
        },
    }
    leakage["passed"] = all(
        route["passed"] and route["lse_full_byte_equal"]
        for route in leakage["routes"].values()
    )

    publication_fields = (
        "q_forward_fp4",
        "q_forward_scales",
        "q_forward_global_scale",
    )
    projection_match = {
        field: _byte_equal(
            getattr(exact_bundle, field), getattr(mx_bundle, field)
        )
        for field in publication_fields
    }
    result = {
        "schema": "causal_gqa_mxfp4pv_b16_variant_comparison_v1",
        "shape": {
            "batch": BATCH,
            "sequence": SEQUENCE,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "causal": True,
        },
        "seed": SEED,
        "artifacts": {
            **identities,
            "aggressive": aggressive_identity,
            "projection": projection_identity,
        },
        "topology": {
            "aggressive": aggressive_topology,
            "anchored": anchored_topology,
            "fp8": fp8_topology,
        },
        "fixed_operand_receipt": {
            "same_rows_weights_rope_and_qk_scales": True,
            "same_prepared_input_and_weight_operands": True,
            "q_publications_exact_vs_mx_byte_equal": projection_match,
            "expected_k_and_v_layout_difference": True,
            "rows": _tensor_summary(rows),
            "weight": _tensor_summary(weight),
        },
        "timing_with_lse": timings,
        "timing_without_lse": no_lse_timings,
        "timing_protocol": {
            "warmups_per_route": warmups,
            "samples_per_route": samples,
            "rotating_order_with_lse": timing_orders,
            "rotating_order_without_lse": no_lse_orders,
            "scope": "prepared causal attention forward only",
        },
        "correctness": correctness,
        "sample_isolation": isolation,
        "causal_leakage": leakage,
        "checks": {
            "runtime_topology_authenticated": True,
            "fixed_q_publications": all(projection_match.values()),
            "sample_isolation": isolation["passed"],
            "causal_leakage": leakage["passed"],
            "all_outputs_finite": all(
                comparison["output"]["finite"]
                and comparison["lse"]["finite"]
                for comparison in correctness.values()
            ),
        },
        "device": {
            "name": torch.cuda.get_device_name(gpu),
            "total_memory_bytes": total_memory,
            "free_before_bytes": free_before,
            "free_after_bytes": torch.cuda.mem_get_info(gpu)[0],
        },
    }
    result["passed"] = all(result["checks"].values())
    output_path = Path(
        "/tmp/causal_gqa_mxfp4pv_b16_variants_sm100_20260825.json"
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    output_path.write_text(encoded + "\n")
    print(encoded)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
