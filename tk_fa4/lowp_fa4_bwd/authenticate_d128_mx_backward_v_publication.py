#!/usr/bin/env python3
"""Authenticate and time the D128 MX-only backward-V publication.

The diagnostic covers the two authenticated Llama-8B projection shapes,
B1/B2, at S4096/H4096/Hq32/Hkv8/D128.  It requires an extension selected
through ``TK_FA4_LOWP_BWD_EXTENSION_SOURCE`` before importing :mod:`tk_fa4`.
For each batch it prepares one shared set of NVFP4 activation/weight operands
and packed RoPE, then compares two disjoint caller-owned workspaces:

* retained output-shared dual-V, which publishes E4M3 backward V;
* experimental MX-backward-V, which publishes packed rowwise MXFP4 V and
  E8M0 pages while leaving the inactive E4M3 V owner byte-for-byte untouched.

First-use allocating/checked authentication is excluded from timing. All
measured calls use the binders' unchecked out-parameter symbols and paired
ABBA/BAAB same-stream CUDA events. One mode self-conditions each target with
same-provider replays; a second uses identical cache-scrub predecessors. The
JSON receipt is written create-only and records extension/source identities,
prepared-operand hashes, caller-owned pointer receipts, correctness
comparisons, paired bootstrap intervals, and raw samples.

This file intentionally contains no build logic.  Do not run it until the
selected extension exports the new checked and unchecked symbols.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import stat
import statistics
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SEQUENCE = 4096
HIDDEN = 4096
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
QKV_WIDTH = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM
BATCHES = (1, 2)
BASE_SYMBOL = (
    "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
)
RETAINED_SUFFIX = "_output_shared_dual_v_mx_forward_out"
MX_BACKWARD_SUFFIX = "_mx_backward_v_mx_forward_out"
HASH_CHUNK_BYTES = 16 << 20


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--warmups", type=_nonnegative_int, default=12)
    parser.add_argument("--samples", type=_positive_int, default=40)
    parser.add_argument(
        "--conditioning-replays",
        type=_positive_int,
        default=4,
    )
    parser.add_argument(
        "--cache-scrub-mib",
        type=_nonnegative_int,
        default=256,
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=_positive_int,
        default=10000,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.samples % 2:
        parser.error("--samples must be even for balanced provider order")
    return args


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    selected = _absolute(path)
    observed = selected.lstat()
    if not stat.S_ISREG(observed.st_mode):
        raise RuntimeError(
            f"artifact must be a regular non-symlink file: {selected}"
        )
    resolved = selected.resolve(strict=True)
    resolved_stat = resolved.stat()
    return {
        "selected_path": str(selected),
        "resolved_path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved_stat.st_size,
        "mtime_ns": resolved_stat.st_mtime_ns,
    }


def _git_output(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", "-C", str(REPO_ROOT), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.rstrip("\n")


def _git_receipt() -> dict[str, Any]:
    status = _git_output("status", "--porcelain=v1")
    return {
        "root": str(REPO_ROOT),
        "head": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "status": status,
    }


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    selected_extension = os.environ.get("TK_FA4_LOWP_BWD_EXTENSION_SOURCE")
    return {
        "schema": "d128_mx_backward_v_publication_plan_v2",
        "dry_run": True,
        "touches_cuda": False,
        "shape": {
            "batches": list(BATCHES),
            "sequence": SEQUENCE,
            "hidden": HIDDEN,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
        },
        "parameters": {
            "seed": args.seed,
            "warmups": args.warmups,
            "samples": args.samples,
            "conditioning_replays": args.conditioning_replays,
            "cache_scrub_mib": args.cache_scrub_mib,
            "bootstrap_draws": args.bootstrap_draws,
        },
        "providers": [
            "retained_output_shared_dual_v",
            "experimental_mx_backward_v",
        ],
        "extension_selection": {
            "environment_variable": "TK_FA4_LOWP_BWD_EXTENSION_SOURCE",
            "is_set": bool(selected_extension),
            "selected_path": selected_extension,
        },
        "timing": {
            "method": (
                "self-conditioned and fixed-cache-scrub ABBA/BAAB paired "
                "same-stream CUDA events"
            ),
            "first_use_authentication_excluded": True,
            "warmups_per_batch": args.warmups,
            "samples_per_provider_per_batch": args.samples,
        },
        "output": str(_absolute(args.output)),
    }


def _tensor_bytes(torch: Any, tensor: Any) -> Any:
    return tensor.detach().contiguous().view(torch.uint8).reshape(-1)


def _tensor_sha256(torch: Any, tensor: Any) -> str:
    byte_tensor = _tensor_bytes(torch, tensor)
    digest = hashlib.sha256()
    for begin in range(0, byte_tensor.numel(), HASH_CHUNK_BYTES):
        chunk = byte_tensor[begin : begin + HASH_CHUNK_BYTES].cpu()
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def _tensor_receipt(torch: Any, tensor: Any) -> dict[str, Any]:
    byte_tensor = _tensor_bytes(torch, tensor)
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
        "storage_bytes": int(byte_tensor.numel()),
        "sha256": _tensor_sha256(torch, tensor),
    }


def _tuple_receipt(torch: Any, tensors: Sequence[Any]) -> list[dict[str, Any]]:
    return [_tensor_receipt(torch, tensor) for tensor in tensors]


def _bitwise_comparison(torch: Any, left: Any, right: Any) -> dict[str, Any]:
    metadata_equal = bool(
        left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and left.device == right.device
        and left.is_contiguous()
        and right.is_contiguous()
    )
    if not metadata_equal:
        return {
            "metadata_equal": False,
            "bytes_compared": 0,
            "byte_mismatches": None,
            "passed": False,
        }
    left_bytes = _tensor_bytes(torch, left)
    right_bytes = _tensor_bytes(torch, right)
    mismatches = int((left_bytes != right_bytes).sum().item())
    return {
        "metadata_equal": True,
        "bytes_compared": int(left_bytes.numel()),
        "byte_mismatches": mismatches,
        "passed": mismatches == 0,
    }


def _workspace_owners(workspace: Any) -> dict[str, Any]:
    owners = {
        "q_payload": workspace.q_payload,
        "k_payload": workspace.k_payload,
        "q_scale_pages": workspace.q_scale_pages,
        "q_global_scale": workspace.q_global_scale,
        "k_scale_pages": workspace.k_scale_pages,
        "k_global_scale": workspace.k_global_scale,
        "v_mxfp4_payload": workspace.v_mxfp4_payload,
        "v_mxfp4_scale_pages": workspace.v_mxfp4_scale_pages,
        "v_fp8_payload": workspace.v_fp8_payload,
        "v_backward_fp8": workspace.v_backward_fp8,
        "q_backward_fp8": workspace.q_backward_fp8,
        "k_backward_fp8": workspace.k_backward_fp8,
    }
    if workspace.v_backward_mxfp4 is not None:
        owners["v_backward_mxfp4"] = workspace.v_backward_mxfp4
    if workspace.v_backward_mxfp4_scale_pages is not None:
        owners["v_backward_mxfp4_scale_pages"] = (
            workspace.v_backward_mxfp4_scale_pages
        )
    return owners


def _workspace_pointer_receipt(workspace: Any) -> dict[str, Any]:
    owners = _workspace_owners(workspace)
    pointers = {name: int(tensor.data_ptr()) for name, tensor in owners.items()}
    return {
        "owners": {
            name: {
                "pointer": pointers[name],
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
            for name, tensor in owners.items()
        },
        "owner_count": len(owners),
        "owner_pointers_unique": len(set(pointers.values())) == len(pointers),
        "q_fp4_alias_matches": (
            workspace.q_payload_fp4.data_ptr() == workspace.q_payload.data_ptr()
        ),
        "k_fp4_alias_matches": (
            workspace.k_payload_fp4.data_ptr() == workspace.k_payload.data_ptr()
        ),
    }


def _fill_storage_bytes(torch: Any, tensor: Any, byte: int) -> None:
    tensor.view(torch.uint8).fill_(byte)


def _allocate_workspace(
    torch: Any,
    tk_fa4: Any,
    *,
    batch: int,
    include_mx_backward_v: bool,
) -> Any:
    device = torch.device("cuda")
    q_payload = torch.empty(
        batch,
        Q_HEADS,
        SEQUENCE,
        HEAD_DIM // 2,
        device=device,
        dtype=torch.uint8,
    )
    k_payload = torch.empty(
        batch,
        KV_HEADS,
        SEQUENCE,
        HEAD_DIM // 2,
        device=device,
        dtype=torch.uint8,
    )
    workspace = tk_fa4.B300E4M3QKVForwardWorkspace(
        q_payload=q_payload,
        k_payload=k_payload,
        q_scale_pages=torch.empty(
            batch,
            SEQUENCE // 128,
            Q_HEADS * 2,
            512,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        q_global_scale=torch.empty(
            batch,
            Q_HEADS,
            device=device,
            dtype=torch.float32,
        ),
        k_scale_pages=torch.empty(
            batch,
            SEQUENCE // 64,
            KV_HEADS * 2,
            512,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        k_global_scale=torch.empty(
            batch,
            KV_HEADS,
            device=device,
            dtype=torch.float32,
        ),
        v_mxfp4_payload=torch.empty(
            batch,
            KV_HEADS,
            HEAD_DIM,
            SEQUENCE // 2,
            device=device,
            dtype=torch.float4_e2m1fn_x2,
        ),
        v_mxfp4_scale_pages=torch.empty(
            batch,
            SEQUENCE // 128,
            KV_HEADS,
            512,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        v_fp8_payload=torch.empty(
            batch,
            KV_HEADS,
            HEAD_DIM,
            SEQUENCE,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        v_backward_fp8=torch.empty(
            batch,
            SEQUENCE,
            KV_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        q_backward_fp8=torch.empty(
            batch,
            SEQUENCE,
            Q_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        k_backward_fp8=torch.empty(
            batch,
            SEQUENCE,
            KV_HEADS,
            HEAD_DIM,
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        q_payload_fp4=q_payload.view(torch.float4_e2m1fn_x2),
        k_payload_fp4=k_payload.view(torch.float4_e2m1fn_x2),
        empty_bf16=torch.empty(0, device=device, dtype=torch.bfloat16),
        empty_byte=torch.empty(0, device=device, dtype=torch.uint8),
        empty_fp8=torch.empty(0, device=device, dtype=torch.float8_e4m3fn),
        empty_fp4=torch.empty(
            0,
            device=device,
            dtype=torch.float4_e2m1fn_x2,
        ),
        v_backward_mxfp4=(
            torch.empty(
                batch,
                SEQUENCE,
                KV_HEADS,
                HEAD_DIM // 2,
                device=device,
                dtype=torch.uint8,
            )
            if include_mx_backward_v
            else None
        ),
        v_backward_mxfp4_scale_pages=(
            torch.empty(
                batch,
                SEQUENCE // 128,
                KV_HEADS,
                512,
                device=device,
                dtype=torch.uint8,
            )
            if include_mx_backward_v
            else None
        ),
    )
    for index, tensor in enumerate(_workspace_owners(workspace).values()):
        _fill_storage_bytes(torch, tensor, (37 * index + 11) & 0xFF)
    # Make the inactive owners visually distinctive in the receipt and fail
    # closed if the MX-only specialization touches either representation.
    _fill_storage_bytes(torch, workspace.v_fp8_payload, 0xA5)
    _fill_storage_bytes(torch, workspace.v_backward_fp8, 0x5A)
    return workspace


def _make_rope(torch: Any, tk_fa4: Any, batch: int) -> Any:
    positions = torch.arange(SEQUENCE, device="cuda", dtype=torch.float32)
    frequencies = 1.0 / (
        500_000.0
        ** (
            torch.arange(
                HEAD_DIM // 2,
                device="cuda",
                dtype=torch.float32,
            )
            / (HEAD_DIM // 2)
        )
    )
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos()[None].repeat(batch, 1, 1).bfloat16().contiguous()
    sine = angles.sin()[None].repeat(batch, 1, 1).bfloat16().contiguous()
    return tk_fa4.b300_pack_gqa_d128_rope(cosine, sine)


def _prepare_operands(
    torch: Any,
    tk_fa4: Any,
    *,
    batch: int,
    seed: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...], Any, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    rows = torch.empty(
        batch * SEQUENCE,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=1.0, generator=generator)
    q_weight = torch.empty(
        Q_HEADS * HEAD_DIM,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=0.02, generator=generator)
    k_weight = torch.empty(
        KV_HEADS * HEAD_DIM,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
    ).normal_(mean=0.0, std=0.02, generator=generator)
    v_weight = torch.empty_like(k_weight).normal_(
        mean=0.0,
        std=0.02,
        generator=generator,
    )
    q_weight, k_weight = (
        tk_fa4.b300_pair_interleave_gqa_d128_qk_projection_weights(
            q_weight,
            k_weight,
        )
    )
    qkv_weight = tk_fa4.b300_stack_gqa_d128_qkv_projection_weights(
        q_weight,
        k_weight,
        v_weight,
    )
    input_operand = tuple(
        tk_fa4.b300_prepare_nvfp4_projection_operand(rows)
    )
    weight_operand = tuple(
        tk_fa4.b300_prepare_nvfp4_projection_weight(qkv_weight)
    )
    qk_scales = torch.zeros(
        batch,
        Q_HEADS,
        7,
        device="cuda",
        dtype=torch.float32,
    )
    qk_scales[..., 0] = 2.25
    qk_scales[..., 1] = 2.0
    packed_rope = _make_rope(torch, tk_fa4, batch)
    return input_operand, weight_operand, qk_scales, packed_rope


def _require_bundle_pointer_ownership(
    bundle: Any,
    workspace: Any,
    *,
    mx_backward_v: bool,
) -> dict[str, Any]:
    checks = {
        "q_forward_payload": (
            bundle.q_forward_fp4.data_ptr() == workspace.q_payload.data_ptr()
        ),
        "k_forward_payload": (
            bundle.k_forward_fp4.data_ptr() == workspace.k_payload.data_ptr()
        ),
        "forward_v_payload": (
            bundle.v_forward_fp4.data_ptr()
            == workspace.v_mxfp4_payload.data_ptr()
        ),
        "forward_v_scales": (
            bundle.v_forward_scales.data_ptr()
            == workspace.v_mxfp4_scale_pages.data_ptr()
        ),
        "backward_q": (
            bundle.q_backward_fp8 is not None
            and bundle.q_backward_fp8.data_ptr()
            == workspace.q_backward_fp8.data_ptr()
        ),
        "backward_k": (
            bundle.k_backward_fp8 is not None
            and bundle.k_backward_fp8.data_ptr()
            == workspace.k_backward_fp8.data_ptr()
        ),
    }
    if mx_backward_v:
        checks.update(
            {
                "backward_e4m3_v_absent": bundle.v_backward_fp8 is None,
                "backward_mx_v": (
                    workspace.v_backward_mxfp4 is not None
                    and bundle.v_backward_fp4.data_ptr()
                    == workspace.v_backward_mxfp4.data_ptr()
                ),
                "backward_mx_v_scales": (
                    workspace.v_backward_mxfp4_scale_pages is not None
                    and bundle.v_backward_scales.data_ptr()
                    == workspace.v_backward_mxfp4_scale_pages.data_ptr()
                ),
            }
        )
    else:
        checks["backward_e4m3_v"] = (
            bundle.v_backward_fp8 is not None
            and bundle.v_backward_fp8.data_ptr()
            == workspace.v_backward_fp8.data_ptr()
        )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"bundle escaped caller-owned workspace: {failed}")
    return {"checks": checks, "passed": True}


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
    return {
        "unit": "microseconds",
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "minimum_us": min(values),
        "p10_us": _percentile(values, 0.10),
        "p90_us": _percentile(values, 0.90),
        "maximum_us": max(values),
        "samples_us": list(values),
    }


def _bootstrap_paired_comparison(
    retained_blocks: Sequence[float],
    candidate_blocks: Sequence[float],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    if len(retained_blocks) != len(candidate_blocks) or not retained_blocks:
        raise ValueError("paired timing blocks must be non-empty and aligned")
    block_deltas = [
        candidate - retained
        for retained, candidate in zip(
            retained_blocks,
            candidate_blocks,
            strict=True,
        )
    ]
    block_speedups = [
        retained / candidate
        for retained, candidate in zip(
            retained_blocks,
            candidate_blocks,
            strict=True,
        )
    ]
    generator = random.Random(seed)
    count = len(block_deltas)
    delta_draws = []
    speedup_draws = []
    for _ in range(draws):
        selected = [generator.randrange(count) for _ in range(count)]
        delta_draws.append(
            statistics.fmean(block_deltas[index] for index in selected)
        )
        speedup_draws.append(
            statistics.fmean(block_speedups[index] for index in selected)
        )
    delta_ci = [
        _percentile(delta_draws, 0.025),
        _percentile(delta_draws, 0.975),
    ]
    speedup_ci = [
        _percentile(speedup_draws, 0.025),
        _percentile(speedup_draws, 0.975),
    ]
    return {
        "paired_blocks": count,
        "retained_block_means_us": list(retained_blocks),
        "candidate_block_means_us": list(candidate_blocks),
        "candidate_minus_retained_block_deltas_us": block_deltas,
        "retained_over_candidate_block_speedups": block_speedups,
        "mean_candidate_minus_retained_us": statistics.fmean(block_deltas),
        "mean_retained_over_candidate_speedup": statistics.fmean(
            block_speedups
        ),
        "bootstrap_draws": draws,
        "bootstrap_95pct_delta_us": delta_ci,
        "bootstrap_95pct_speedup": speedup_ci,
        "candidate_faster_at_95pct": (
            delta_ci[1] < 0.0 and speedup_ci[0] > 1.0
        ),
        "retained_faster_at_95pct": (
            delta_ci[0] > 0.0 and speedup_ci[1] < 1.0
        ),
    }


def _measure(
    torch: Any,
    functions: dict[str, Callable[[], Any]],
    *,
    warmups: int,
    samples: int,
    conditioning_replays: int,
    cache_scrub_mib: int,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    names = (
        "retained_output_shared_dual_v",
        "experimental_mx_backward_v",
    )
    if tuple(functions) != names:
        raise ValueError("providers are not in canonical order")
    for iteration in range(warmups):
        order = names if iteration % 2 == 0 else names[::-1]
        retained = [functions[name]() for name in order]
        torch.cuda.synchronize()
        del retained

    cache_scrub = None
    if cache_scrub_mib:
        cache_scrub = torch.zeros(
            (cache_scrub_mib << 20) // 4,
            dtype=torch.float32,
            device="cuda",
        )
        torch.cuda.synchronize()

    def measure_mode(
        mode: str,
        *,
        comparison_seed: int,
    ) -> dict[str, Any]:
        values = {name: [] for name in names}
        block_means = {name: [] for name in names}
        orders = []
        for block in range(samples // 2):
            order = (
                (names[0], names[1], names[1], names[0])
                if block % 2 == 0
                else (names[1], names[0], names[0], names[1])
            )
            orders.append(list(order))
            block_values = {name: [] for name in names}
            for name in order:
                retained = []
                if mode == "self_conditioned_single_launch":
                    retained.extend(
                        functions[name]()
                        for _ in range(conditioning_replays)
                    )
                elif mode == "cache_scrub_conditioned_single_launch":
                    if cache_scrub is None:
                        raise RuntimeError("cache scrub timing is disabled")
                    cache_scrub.add_(1.0)
                    cache_scrub.add_(1.0)
                else:
                    raise ValueError(f"unknown timing mode: {mode}")
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                retained.append(functions[name]())
                end.record()
                end.synchronize()
                elapsed = float(start.elapsed_time(end) * 1000.0)
                values[name].append(elapsed)
                block_values[name].append(elapsed)
                del retained
            for name in names:
                if len(block_values[name]) != 2:
                    raise RuntimeError("ABBA/BAAB block is unbalanced")
                block_means[name].append(
                    statistics.fmean(block_values[name])
                )

        summaries = {name: _timing_summary(values[name]) for name in names}
        return {
            "conditioning": {
                "mode": mode,
                "same_provider_replays": (
                    conditioning_replays
                    if mode == "self_conditioned_single_launch"
                    else 0
                ),
                "cache_scrub_mib": (
                    cache_scrub_mib
                    if mode == "cache_scrub_conditioned_single_launch"
                    else 0
                ),
                "cache_scrub_passes": (
                    2 if mode == "cache_scrub_conditioned_single_launch" else 0
                ),
            },
            "providers": summaries,
            "block_orders": orders,
            "comparison": _bootstrap_paired_comparison(
                block_means[names[0]],
                block_means[names[1]],
                seed=comparison_seed,
                draws=bootstrap_draws,
            ),
        }

    modes = {
        "self_conditioned_single_launch": measure_mode(
            "self_conditioned_single_launch",
            comparison_seed=seed,
        )
    }
    if cache_scrub is not None:
        modes["cache_scrub_conditioned_single_launch"] = measure_mode(
            "cache_scrub_conditioned_single_launch",
            comparison_seed=seed ^ 0x9E3779B9,
        )
    faster = [
        mode["comparison"]["candidate_faster_at_95pct"]
        for mode in modes.values()
    ]
    retained_faster = [
        mode["comparison"]["retained_faster_at_95pct"]
        for mode in modes.values()
    ]
    return {
        "protocol": "paired_abba_baab_v2",
        "modes": modes,
        "claim": {
            "candidate_faster_in_all_modes_at_95pct": all(faster),
            "retained_faster_in_all_modes_at_95pct": all(retained_faster),
            "otherwise_cache_or_state_sensitive": not (
                all(faster) or all(retained_faster)
            ),
        },
    }


def _common_comparisons(
    torch: Any,
    retained_workspace: Any,
    candidate_workspace: Any,
) -> dict[str, Any]:
    names = (
        "q_payload",
        "k_payload",
        "q_scale_pages",
        "q_global_scale",
        "k_scale_pages",
        "k_global_scale",
        "v_mxfp4_payload",
        "v_mxfp4_scale_pages",
        "q_backward_fp8",
        "k_backward_fp8",
    )
    return {
        name: _bitwise_comparison(
            torch,
            getattr(retained_workspace, name),
            getattr(candidate_workspace, name),
        )
        for name in names
    }


def _run_case(
    torch: Any,
    tk_fa4: Any,
    *,
    batch: int,
    seed: int,
    warmups: int,
    samples: int,
    conditioning_replays: int,
    cache_scrub_mib: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    input_operand, weight_operand, qk_scales, packed_rope = _prepare_operands(
        torch,
        tk_fa4,
        batch=batch,
        seed=seed,
    )
    prepared_receipt = {
        "input_operand": _tuple_receipt(torch, input_operand),
        "weight_operand": _tuple_receipt(torch, weight_operand),
        "qk_scales": _tensor_receipt(torch, qk_scales),
        "packed_rope": _tensor_receipt(torch, packed_rope),
    }

    retained_workspace = _allocate_workspace(
        torch,
        tk_fa4,
        batch=batch,
        include_mx_backward_v=True,
    )
    candidate_workspace = _allocate_workspace(
        torch,
        tk_fa4,
        batch=batch,
        include_mx_backward_v=True,
    )
    retained_pointers_before = _workspace_pointer_receipt(retained_workspace)
    candidate_pointers_before = _workspace_pointer_receipt(candidate_workspace)
    if not retained_pointers_before["owner_pointers_unique"]:
        raise RuntimeError("retained workspace owners alias one another")
    if not candidate_pointers_before["owner_pointers_unique"]:
        raise RuntimeError("candidate workspace owners alias one another")

    inactive_e4m3_v_snapshot = candidate_workspace.v_backward_fp8.clone()
    inactive_e4m3_v_hash_before = _tensor_sha256(
        torch,
        candidate_workspace.v_backward_fp8,
    )
    inactive_forward_fp8_v_snapshot = candidate_workspace.v_fp8_payload.clone()

    retained = tk_fa4.b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(
        batch=batch,
        seqlen=SEQUENCE,
        hidden=HIDDEN,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=False,
        per_block_qk_scales=True,
        experimental_output_shared_dual_v=True,
    )
    candidate = tk_fa4.b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(
        batch=batch,
        seqlen=SEQUENCE,
        hidden=HIDDEN,
        q_heads=Q_HEADS,
        kv_heads=KV_HEADS,
        publish_mxfp4_v=True,
        v_mxfp4_scale_2d=False,
        per_block_qk_scales=True,
        experimental_mx_backward_v=True,
    )
    expected_symbols = {
        "retained_checked": BASE_SYMBOL + RETAINED_SUFFIX,
        "retained_unchecked": BASE_SYMBOL + RETAINED_SUFFIX + "_unchecked",
        "candidate_checked": BASE_SYMBOL + MX_BACKWARD_SUFFIX,
        "candidate_unchecked": BASE_SYMBOL + MX_BACKWARD_SUFFIX + "_unchecked",
    }
    observed_symbols = {
        "retained_checked": retained.checked_symbol,
        "retained_unchecked": retained.unchecked_symbol,
        "candidate_checked": candidate.checked_symbol,
        "candidate_unchecked": candidate.unchecked_symbol,
    }
    if observed_symbols != expected_symbols:
        raise RuntimeError(
            "D128 binders selected unexpected symbols: "
            f"{observed_symbols!r} != {expected_symbols!r}"
        )

    def project(projector: Any, workspace: Any) -> Any:
        return projector(
            input_operand,
            weight_operand,
            qk_scales,
            packed_rope,
            forward_workspace=workspace,
        )

    # These first calls execute the allocating references and checked ABIs.
    with torch.no_grad():
        retained_bundle = project(retained, retained_workspace)
        candidate_bundle = project(candidate, candidate_workspace)
        torch.cuda.synchronize()

    for name, projector in (("retained", retained), ("candidate", candidate)):
        if not projector.forward_workspace_abi_validated:
            raise RuntimeError(f"{name} binder did not authenticate workspace")
        if projector.validated_forward_workspace_count != 1:
            raise RuntimeError(
                f"{name} binder authenticated an unexpected workspace count"
            )
    ownership = {
        "retained": _require_bundle_pointer_ownership(
            retained_bundle,
            retained_workspace,
            mx_backward_v=False,
        ),
        "candidate": _require_bundle_pointer_ownership(
            candidate_bundle,
            candidate_workspace,
            mx_backward_v=True,
        ),
    }
    inactive_after_auth = {
        "backward_e4m3_v": _bitwise_comparison(
            torch,
            inactive_e4m3_v_snapshot,
            candidate_workspace.v_backward_fp8,
        ),
        "forward_fp8_v": _bitwise_comparison(
            torch,
            inactive_forward_fp8_v_snapshot,
            candidate_workspace.v_fp8_payload,
        ),
    }
    common_after_auth = _common_comparisons(
        torch,
        retained_workspace,
        candidate_workspace,
    )
    if not all(item["passed"] for item in inactive_after_auth.values()):
        raise RuntimeError("MX backward-V route modified an inactive FP8 owner")
    if not all(item["passed"] for item in common_after_auth.values()):
        raise RuntimeError(
            "MX backward-V route changed a common forward/QK publication"
        )

    # Time the already-authenticated raw unchecked symbols.  Calling the
    # public binder here would place Python bundle/dataclass construction
    # between the kernel enqueue and the end-event record; for a short
    # projection that host gap can become GPU idle time inside the reported
    # interval.  Correctness and ownership remain authenticated above through
    # the public checked binder.
    def raw_unchecked(projector: Any, workspace: Any) -> Callable[[], Any]:
        compact_outputs = (
            workspace.compact_mx_backward_v_outputs()
            if projector.experimental_mx_backward_v
            else workspace.compact_outputs()
        )
        arguments = (
            *input_operand,
            *weight_operand,
            qk_scales,
            packed_rope,
            batch,
            SEQUENCE,
            Q_HEADS,
            KV_HEADS,
            projector.v_mxfp4_scale_2d,
            projector.per_block_qk_scales,
            projector.cluster_cap,
            projector.cache_packed_rope,
            projector.cache_adaptive_qk_scale,
            *compact_outputs,
        )
        raw_project = projector._project_unchecked

        def launch() -> Any:
            return raw_project(*arguments)

        return launch

    functions = {
        "retained_output_shared_dual_v": raw_unchecked(
            retained,
            retained_workspace,
        ),
        "experimental_mx_backward_v": raw_unchecked(
            candidate,
            candidate_workspace,
        ),
    }
    with torch.no_grad():
        # Explicitly cross the checked-to-unchecked boundary before warmup.
        functions["retained_output_shared_dual_v"]()
        functions["experimental_mx_backward_v"]()
        torch.cuda.synchronize()
        timing = _measure(
            torch,
            functions,
            warmups=warmups,
            samples=samples,
            conditioning_replays=conditioning_replays,
            cache_scrub_mib=cache_scrub_mib,
            bootstrap_draws=bootstrap_draws,
            seed=seed,
        )

    retained_pointers_after = _workspace_pointer_receipt(retained_workspace)
    candidate_pointers_after = _workspace_pointer_receipt(candidate_workspace)
    pointers_stable = bool(
        retained_pointers_before == retained_pointers_after
        and candidate_pointers_before == candidate_pointers_after
    )
    if not pointers_stable:
        raise RuntimeError("caller-owned workspace pointers changed")
    inactive_after_timing = {
        "backward_e4m3_v": _bitwise_comparison(
            torch,
            inactive_e4m3_v_snapshot,
            candidate_workspace.v_backward_fp8,
        ),
        "forward_fp8_v": _bitwise_comparison(
            torch,
            inactive_forward_fp8_v_snapshot,
            candidate_workspace.v_fp8_payload,
        ),
    }
    common_after_timing = _common_comparisons(
        torch,
        retained_workspace,
        candidate_workspace,
    )
    if not all(item["passed"] for item in inactive_after_timing.values()):
        raise RuntimeError("timed MX route modified an inactive FP8 owner")
    if not all(item["passed"] for item in common_after_timing.values()):
        raise RuntimeError("timed routes diverged in common publications")

    assert candidate_workspace.v_backward_mxfp4 is not None
    assert candidate_workspace.v_backward_mxfp4_scale_pages is not None
    output_hashes = {
        "candidate_backward_mx_v": _tensor_receipt(
            torch,
            candidate_workspace.v_backward_mxfp4,
        ),
        "candidate_backward_mx_v_scales": _tensor_receipt(
            torch,
            candidate_workspace.v_backward_mxfp4_scale_pages,
        ),
        "candidate_inactive_backward_e4m3_v": _tensor_receipt(
            torch,
            candidate_workspace.v_backward_fp8,
        ),
        "retained_backward_e4m3_v": _tensor_receipt(
            torch,
            retained_workspace.v_backward_fp8,
        ),
    }
    inactive_e4m3_v_hash_after = _tensor_sha256(
        torch,
        candidate_workspace.v_backward_fp8,
    )
    return {
        "batch": batch,
        "shape": {
            "sequence": SEQUENCE,
            "hidden": HIDDEN,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "qkv_width": QKV_WIDTH,
        },
        "prepared_operands": prepared_receipt,
        "symbols": observed_symbols,
        "contracts": {
            "retained_path": retained.output_shared_dual_v_path,
            "candidate_path": candidate.output_shared_dual_v_path,
            "retained_backward_semantics": (
                retained.backward_publication_semantics
            ),
            "candidate_backward_semantics": (
                candidate.backward_publication_semantics
            ),
            "retained_authenticated_workspace_count": (
                retained.validated_forward_workspace_count
            ),
            "candidate_authenticated_workspace_count": (
                candidate.validated_forward_workspace_count
            ),
            "first_use_authentication_excluded_from_timing": True,
            "steady_state_uses_unchecked_symbols": True,
        },
        "pointer_ownership": ownership,
        "workspace_pointers": {
            "retained_before": retained_pointers_before,
            "retained_after": retained_pointers_after,
            "candidate_before": candidate_pointers_before,
            "candidate_after": candidate_pointers_after,
            "stable": pointers_stable,
        },
        "correctness": {
            "common_after_authentication": common_after_auth,
            "common_after_timing": common_after_timing,
            "candidate_inactive_owners_after_authentication": (
                inactive_after_auth
            ),
            "candidate_inactive_owners_after_timing": inactive_after_timing,
            "inactive_backward_e4m3_v_sha256_before": (
                inactive_e4m3_v_hash_before
            ),
            "inactive_backward_e4m3_v_sha256_after": (
                inactive_e4m3_v_hash_after
            ),
            "passed": True,
        },
        "output_hashes": output_hashes,
        "timing": timing,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _write_create_only(path: Path, content: str) -> None:
    destination = _absolute(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "w") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _run(args: argparse.Namespace) -> int:
    output = _absolute(args.output)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    selected_value = os.environ.get("TK_FA4_LOWP_BWD_EXTENSION_SOURCE")
    if not selected_value:
        raise RuntimeError(
            "TK_FA4_LOWP_BWD_EXTENSION_SOURCE must select the built extension"
        )
    extension_before = _file_identity(Path(selected_value))
    source_paths = {
        "diagnostic": Path(__file__),
        "package_init": REPO_ROOT / "tk_fa4" / "__init__.py",
        "interface": REPO_ROOT / "tk_fa4" / "interface.py",
        "projection_translation_unit": HERE / "lowp_fa4_bwd.cu",
        "projection_epilogue": HERE / "projection_fp4_epilogue.cuh",
    }
    sources_before = {
        name: _file_identity(path) for name, path in source_paths.items()
    }

    import torch
    import tk_fa4
    import tk_fa4.interface as interface

    package_identity_before = _file_identity(Path(tk_fa4.__file__))
    interface_identity_before = _file_identity(Path(interface.__file__))
    if package_identity_before != sources_before["package_init"]:
        raise RuntimeError("imported tk_fa4 package is shadowed")
    if interface_identity_before != sources_before["interface"]:
        raise RuntimeError("imported tk_fa4.interface is shadowed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one SM100 GPU to this diagnostic")
    torch.cuda.set_device(0)
    capability = list(torch.cuda.get_device_capability(0))
    if capability != [10, 0]:
        raise RuntimeError("this diagnostic requires an SM100 GPU")
    loaded_extension = Path(
        interface._C_b300_lowp_bwd.__file__
    ).resolve(strict=True)
    if loaded_extension != Path(extension_before["resolved_path"]):
        raise RuntimeError(
            "tk_fa4 loaded a different low-precision extension than the "
            "environment-selected artifact"
        )
    required_symbols = tuple(
        BASE_SYMBOL + suffix + unchecked
        for suffix in (RETAINED_SUFFIX, MX_BACKWARD_SUFFIX)
        for unchecked in ("", "_unchecked")
    )
    missing_symbols = [
        symbol
        for symbol in required_symbols
        if getattr(interface._C_b300_lowp_bwd, symbol, None) is None
    ]
    if missing_symbols:
        raise RuntimeError(
            f"selected extension omits required symbols: {missing_symbols}"
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cases = []
    with torch.no_grad():
        for batch in BATCHES:
            cases.append(
                _run_case(
                    torch,
                    tk_fa4,
                    batch=batch,
                    seed=args.seed + batch * 104729,
                    warmups=args.warmups,
                    samples=args.samples,
                    conditioning_replays=args.conditioning_replays,
                    cache_scrub_mib=args.cache_scrub_mib,
                    bootstrap_draws=args.bootstrap_draws,
                )
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    extension_after = _file_identity(Path(selected_value))
    sources_after = {
        name: _file_identity(path) for name, path in source_paths.items()
    }
    if extension_before != extension_after:
        raise RuntimeError("selected extension changed during the diagnostic")
    if sources_before != sources_after:
        raise RuntimeError("diagnostic source inputs changed during the run")
    package_identity_after = _file_identity(Path(tk_fa4.__file__))
    interface_identity_after = _file_identity(Path(interface.__file__))
    if package_identity_after != sources_after["package_init"]:
        raise RuntimeError("imported tk_fa4 package is shadowed")
    if interface_identity_after != sources_after["interface"]:
        raise RuntimeError("imported tk_fa4.interface is shadowed")

    document = {
        **_plan(args),
        "schema": "d128_mx_backward_v_publication_v2",
        "dry_run": False,
        "touches_cuda": True,
        "created_utc": (
            dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        ),
        "hardware": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": capability,
            "visible_device_count": torch.cuda.device_count(),
            "process_report_after": torch.cuda.list_gpu_processes(0),
        },
        "artifacts": {
            "extension_before": extension_before,
            "extension_after": extension_after,
            "sources_before": sources_before,
            "sources_after": sources_after,
            "loaded_package_before": package_identity_before,
            "loaded_package_after": package_identity_after,
            "loaded_interface_before": interface_identity_before,
            "loaded_interface_after": interface_identity_after,
            "required_symbols": list(required_symbols),
            "extension_and_sources_unchanged": True,
        },
        "git": _git_receipt(),
        "cases": cases,
        "passed": all(case["correctness"]["passed"] for case in cases),
    }
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    _write_create_only(output, rendered)
    print(rendered, end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dry_run:
        print(json.dumps(_plan(args), indent=2, sort_keys=True))
        return 0
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
