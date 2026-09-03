#!/usr/bin/env python3
"""Validate a batched D64 causal NVFP4-QK/E4M3-PV forward build.

The batched result is compared with independent launches of the authenticated
B=1 kernel using identical projection publications.  The validator also
checks sample isolation, represented-input accuracy, shape rejection, and
attention-only latency against both sequential B=1 and BF16 FA4.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import json
import os
import stat
import statistics
from pathlib import Path
from typing import Any, Callable


CANONICAL_FORWARD_ARTIFACTS = {
    1: (
        "e7bb8e69625adf0e545c80d01b194c13af0ea9e12db8765150d2762267716c35",
        1_817_192,
    ),
    2: (
        "4e4c4c9b1afd7a751c3bae9d734f617a04b0b95778370deba9be3f131f5e05d1",
        1_817_192,
    ),
    8: (
        "34114089ab4631093dc2b4dbd38e01a597a6608c9cfb748bd927f8038271db9d",
        1_817_088,
    ),
    16: (
        "88d81d3783e5aa80f0e9cf259a2ea7c935da4c2a5dc3ba1868e63f802a2c6208",
        1_817_256,
    ),
}
CANONICAL_PROJECTION_ARTIFACT = (
    "bfdec1e43a0a19acec5afbac3fa837e2f4d1b25be80ae7fb5ff3b5bc5e9e25ce",
    17_504_688,
)
AUTHENTICATED_BATCHES = (2, 8, 16)
SEQUENTIAL_RELATIVE_L2_LIMIT = 0.01

SIGNED_E2M1_LEVELS = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _require_canonical_forward_identity(
    batch: int,
    supplied_sha256: str,
    supplied_bytes: int,
) -> tuple[str, int]:
    """Reject caller identities that differ from the reviewed artifacts."""
    if batch not in CANONICAL_FORWARD_ARTIFACTS:
        raise ValueError(f"no canonical forward artifact exists for batch {batch}")
    expected_sha256, expected_bytes = CANONICAL_FORWARD_ARTIFACTS[batch]
    normalized_sha256 = supplied_sha256.lower()
    if not hmac.compare_digest(normalized_sha256, expected_sha256):
        raise ValueError(
            f"B{batch} forward SHA-256 is not the canonical identity"
        )
    if supplied_bytes != expected_bytes:
        raise ValueError(
            f"B{batch} forward byte count {supplied_bytes} does not match "
            f"the canonical {expected_bytes}"
        )
    return expected_sha256, expected_bytes


def _authenticate_regular_artifact(
    path: Path,
    expected_sha256: str,
    expected_bytes: int,
) -> tuple[Path, dict[str, Any]]:
    """Hash a regular non-symlink artifact through one opened descriptor."""
    requested = Path(path)
    requested_stat = requested.lstat()
    if not stat.S_ISREG(requested_stat.st_mode):
        raise RuntimeError(
            f"forward artifact must be a regular non-symlink file: {requested}"
        )
    if requested_stat.st_size != expected_bytes:
        raise RuntimeError(
            f"forward artifact byte-count mismatch: expected {expected_bytes}, "
            f"found {requested_stat.st_size}"
        )
    resolved = requested.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as stream:
        opened_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened_stat.st_mode):
            raise RuntimeError("forward artifact stopped being a regular file")
        if opened_stat.st_size != expected_bytes:
            raise RuntimeError(
                "forward artifact changed size while authenticating"
            )
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        raise RuntimeError(
            "forward artifact SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    return resolved, {
        "path": str(resolved),
        "sha256": actual_sha256,
        "bytes": opened_stat.st_size,
    }


def _authenticate_forward_artifact(
    path: Path,
    *,
    batch: int,
    supplied_sha256: str,
    supplied_bytes: int,
) -> tuple[Path, dict[str, Any]]:
    expected_sha256, expected_bytes = _require_canonical_forward_identity(
        batch,
        supplied_sha256,
        supplied_bytes,
    )
    return _authenticate_regular_artifact(
        path,
        expected_sha256,
        expected_bytes,
    )


def _authenticate_projection_artifact(
    path: Path,
    *,
    supplied_sha256: str,
    supplied_bytes: int,
) -> tuple[Path, dict[str, Any]]:
    expected_sha256, expected_bytes = CANONICAL_PROJECTION_ARTIFACT
    if not hmac.compare_digest(supplied_sha256.lower(), expected_sha256):
        raise ValueError(
            "projection extension SHA-256 is not the canonical identity"
        )
    if supplied_bytes != expected_bytes:
        raise ValueError(
            f"projection extension byte count {supplied_bytes} does not "
            f"match the canonical {expected_bytes}"
        )
    return _authenticate_regular_artifact(
        path,
        expected_sha256,
        expected_bytes,
    )


def _require_exact_forward_topology(
    topology: dict[str, Any],
    *,
    batch: int,
    sequence: int,
    q_heads: int,
    kv_heads: int,
    runtime_populated: bool = False,
) -> None:
    expected = {
        "batch": batch,
        "seqlen": sequence,
        "heads": q_heads,
        "kv_heads": kv_heads,
        "dqk": 64,
        "dvo": 64,
        "causal": True,
        "qk_format": "nvfp4_e4m3_block16",
        "pv_format": "e4m3_fp8",
        "shiftless_fp8_mode": 0,
        "route": "real_fwd_tk_hao_direct_causal_gqa_nvfp4_fp8pv",
        "schema": "tk_hao_direct_pipeline_v1",
        "fixed_route_fastpath": True,
        "fixed_p_ceiling": False,
        "score_pack_ceiling": False,
    }
    for key, value in expected.items():
        actual = topology.get(key)
        if actual != value:
            raise RuntimeError(
                f"B{batch} forward topology {key}={actual!r}, expected {value!r}"
            )
    if topology.get("causal_interleaved_kv") is not False:
        raise RuntimeError(
            f"B{batch} forward topology causal_interleaved_kv must be false"
        )
    actual_valid = topology.get("valid")
    valid_matches = (
        actual_valid == 1
        if runtime_populated
        else actual_valid in (0, 1)
    )
    if not valid_matches:
        expected_valid = 1 if runtime_populated else (0, 1)
        raise RuntimeError(
            f"B{batch} forward topology valid={actual_valid!r}, expected "
            f"{expected_valid!r}"
        )


def _load(path: Path, module: str) -> Any:
    spec = importlib.util.spec_from_file_location(module, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _metrics(reference: Any, actual: Any) -> dict[str, float | bool]:
    import torch

    reference_f = reference.float().reshape(-1)
    actual_f = actual.float().reshape(-1)
    difference = actual_f - reference_f
    reference_norm = reference_f.norm().clamp_min(1.0e-30)
    actual_norm = actual_f.norm().clamp_min(1.0e-30)
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(
            torch.dot(reference_f, actual_f) / (reference_norm * actual_norm)
        ),
        "relative_l2": float(difference.norm() / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def _byte_equal(reference: Any, actual: Any) -> bool:
    import torch

    return bool(
        reference.shape == actual.shape
        and reference.dtype == actual.dtype
        and torch.equal(
            reference.contiguous().view(torch.uint8),
            actual.contiguous().view(torch.uint8),
        )
    )


def _make_rope(batch: int, sequence: int, device: str) -> tuple[Any, Any]:
    import torch

    positions = torch.arange(sequence, device=device, dtype=torch.float32)
    frequencies = 1.0 / (
        10_000.0
        ** (torch.arange(32, device=device, dtype=torch.float32) / 32)
    )
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos()[None].repeat(batch, 1, 1).bfloat16().contiguous()
    sine = angles.sin()[None].repeat(batch, 1, 1).bfloat16().contiguous()
    return cosine, sine


def _decode_e2m1(payload: Any) -> Any:
    import torch

    packed = payload.contiguous().view(torch.uint8)
    levels = torch.tensor(
        SIGNED_E2M1_LEVELS,
        device=payload.device,
        dtype=torch.float32,
    )
    return torch.stack(
        (levels[(packed & 0x0F).long()], levels[(packed >> 4).long()]),
        dim=-1,
    ).flatten(-2)


def _decode_nvfp4_qk(
    payload: Any,
    prepared_scale: Any,
    global_scale: Any,
    *,
    scale_tile_rows: int,
) -> Any:
    import torch

    batch, heads, rows, packed_columns = payload.shape
    columns = packed_columns * 2
    row_tiles = rows // scale_tile_rows
    scales = (
        prepared_scale.float()
        .reshape(batch, row_tiles, heads, 32, 16)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )
    decoded = _decode_e2m1(payload).reshape(
        batch, heads, rows, columns // 16, 16
    )
    row = torch.arange(rows, device=payload.device)
    block = torch.arange(columns // 16, device=payload.device)
    tile_index = (row // scale_tile_rows)[:, None]
    row_lane = (row % 32)[:, None]
    scale_slot = (
        ((row % scale_tile_rows) // 32)[:, None] * (columns // 16)
        + block[None, :]
    )
    for batch_index in range(batch):
        for head_index in range(heads):
            local_scale = scales[
                batch_index,
                head_index,
                tile_index,
                row_lane,
                scale_slot,
            ]
            decoded[batch_index, head_index].mul_(
                local_scale[..., None] * global_scale[batch_index, head_index]
            )
    return decoded.reshape(batch, heads, rows, columns)


def _time_rotated(
    runners: dict[str, Callable[[], None]],
    *,
    warmups: int,
    samples: int,
) -> dict[str, dict[str, float | list[float]]]:
    import torch

    names = tuple(runners)
    for iteration in range(warmups):
        for offset in range(len(names)):
            runners[names[(iteration + offset) % len(names)]]()
    torch.cuda.synchronize()
    values: dict[str, list[float]] = {name: [] for name in names}
    for iteration in range(samples):
        for offset in range(len(names)):
            name = names[(iteration + offset) % len(names)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runners[name]()
            end.record()
            end.synchronize()
            values[name].append(float(start.elapsed_time(end)))
    return {
        name: {
            "median_ms": statistics.median(samples_ms),
            "minimum_ms": min(samples_ms),
            "samples_ms": samples_ms,
        }
        for name, samples_ms in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batched-forward", type=Path, required=True)
    parser.add_argument("--batched-forward-sha256", required=True)
    parser.add_argument("--batched-forward-bytes", type=int, required=True)
    parser.add_argument("--batched-module", required=True)
    parser.add_argument("--batch1-forward", type=Path, required=True)
    parser.add_argument("--batch1-forward-sha256", required=True)
    parser.add_argument("--batch1-forward-bytes", type=int, required=True)
    parser.add_argument("--batch1-module", required=True)
    parser.add_argument("--projection-extension", type=Path, required=True)
    parser.add_argument("--projection-extension-sha256", required=True)
    parser.add_argument(
        "--projection-extension-bytes", type=int, required=True
    )
    parser.add_argument(
        "--batch", type=int, choices=AUTHENTICATED_BATCHES, default=2
    )
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sequence % 256 or args.hidden % 128:
        parser.error("sequence must divide 256 and hidden must divide 128")
    if args.q_heads % args.kv_heads or args.q_heads % 2 or args.kv_heads % 2:
        parser.error("paired D64 requires even integral GQA head counts")

    batched_path, batched_identity = _authenticate_forward_artifact(
        args.batched_forward,
        batch=args.batch,
        supplied_sha256=args.batched_forward_sha256,
        supplied_bytes=args.batched_forward_bytes,
    )
    batch1_path, batch1_identity = _authenticate_forward_artifact(
        args.batch1_forward,
        batch=1,
        supplied_sha256=args.batch1_forward_sha256,
        supplied_bytes=args.batch1_forward_bytes,
    )
    projection_path, projection_identity = _authenticate_projection_artifact(
        args.projection_extension,
        supplied_sha256=args.projection_extension_sha256,
        supplied_bytes=args.projection_extension_bytes,
    )
    os.environ["TK_FA4_LOWP_BWD_EXTENSION_SOURCE"] = str(projection_path)

    import torch
    from flash_attn.cute import interface as flash_interface
    from tk_fa4 import (
        b300_pack_gqa_d64_paired_rope,
        b300_prepare_e4m3_projection_operand,
        b300_prepare_e4m3_projection_weight,
        b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3,
    )

    torch.cuda.set_device(args.gpu)
    free_before, total_memory = torch.cuda.mem_get_info(args.gpu)
    allocated_before = torch.cuda.memory_allocated(args.gpu)
    reserved_before = torch.cuda.memory_reserved(args.gpu)
    torch.cuda.reset_peak_memory_stats(args.gpu)
    torch.manual_seed(args.seed)
    batched = _load(batched_path, args.batched_module)
    batch1 = _load(batch1_path, args.batch1_module)
    batched_topology = dict(batched.read_hao_direct_topology())
    batch1_topology = dict(batch1.read_hao_direct_topology())
    for topology, expected_batch in (
        (batched_topology, args.batch),
        (batch1_topology, 1),
    ):
        _require_exact_forward_topology(
            topology,
            batch=expected_batch,
            sequence=args.sequence,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
        )

    rows = torch.randn(
        args.batch * args.sequence,
        args.hidden,
        device="cuda",
        dtype=torch.float32,
    ).bfloat16()
    total_width = (args.q_heads + 2 * args.kv_heads) * 64
    weight = (
        torch.randn(
            total_width,
            args.hidden,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    ).bfloat16()
    weight_operand = tuple(b300_prepare_e4m3_projection_weight(weight))
    rope = b300_pack_gqa_d64_paired_rope(
        *_make_rope(args.batch, args.sequence, "cuda")
    )
    qk_scales = torch.zeros(
        args.batch,
        args.q_heads // 2,
        7,
        device="cuda",
        dtype=torch.float32,
    )
    qk_scales[..., 0] = 2.25
    qk_scales[..., 1] = 2.0

    def publish(current_rows: Any, start: int, count: int) -> Any:
        return b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(
            tuple(b300_prepare_e4m3_projection_operand(current_rows)),
            weight_operand,
            qk_scales[start : start + count],
            rope[start : start + count],
            batch=count,
            seqlen=args.sequence,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            publish_mxfp4_v=False,
        )

    batched_bundle = publish(rows, 0, args.batch)
    row_batches = rows.view(args.batch, args.sequence, args.hidden)
    batch1_bundles = [
        publish(row_batches[index].contiguous(), index, 1)
        for index in range(args.batch)
    ]
    publication_fields = (
        "q_forward_fp4",
        "q_forward_scales",
        "q_forward_global_scale",
        "k_forward_fp4",
        "k_forward_scales",
        "k_forward_global_scale",
        "v_forward_fp8",
        "q_backward_fp8",
        "k_backward_fp8",
        "v_backward_fp8",
    )
    publication_equal = {}
    for field in publication_fields:
        batched_value = getattr(batched_bundle, field)
        sequential_value = torch.cat(
            [getattr(bundle, field) for bundle in batch1_bundles], dim=0
        )
        publication_equal[field] = _byte_equal(batched_value, sequential_value)

    def run_forward(extension: Any, bundle: Any, output: Any, lse: Any) -> None:
        extension.forward_hao_direct_fp8pv(
            *bundle.qk_forward_operands(),
            bundle.v_forward_fp8,
            output,
            lse,
            0,
            True,
            True,
        )

    output = torch.empty(
        args.batch,
        args.sequence,
        args.q_heads,
        64,
        device="cuda",
        dtype=torch.bfloat16,
    )
    lse = torch.empty(
        args.batch,
        args.q_heads,
        1,
        args.sequence,
        device="cuda",
        dtype=torch.float32,
    )
    sequential_outputs = [torch.empty_like(output[:1]) for _ in range(args.batch)]
    sequential_lses = [torch.empty_like(lse[:1]) for _ in range(args.batch)]
    run_forward(batched, batched_bundle, output, lse)
    for index, bundle in enumerate(batch1_bundles):
        run_forward(
            batch1,
            bundle,
            sequential_outputs[index],
            sequential_lses[index],
        )
    torch.cuda.synchronize()
    batched_topology = dict(batched.read_hao_direct_topology())
    batch1_topology = dict(batch1.read_hao_direct_topology())
    _require_exact_forward_topology(
        batched_topology,
        batch=args.batch,
        sequence=args.sequence,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        runtime_populated=True,
    )
    _require_exact_forward_topology(
        batch1_topology,
        batch=1,
        sequence=args.sequence,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        runtime_populated=True,
    )
    sequential_output = torch.cat(sequential_outputs, dim=0)
    sequential_lse = torch.cat(sequential_lses, dim=0)

    q_represented = _decode_nvfp4_qk(
        batched_bundle.backward.score_q_fp4,
        batched_bundle.q_forward_scales,
        batched_bundle.q_forward_global_scale,
        scale_tile_rows=128,
    ).permute(0, 2, 1, 3).contiguous()
    k_represented = _decode_nvfp4_qk(
        batched_bundle.backward.score_k_fp4,
        batched_bundle.k_forward_scales,
        batched_bundle.k_forward_global_scale,
        scale_tile_rows=64,
    ).permute(0, 2, 1, 3).contiguous()
    v_represented = (
        batched_bundle.v_forward_fp8.permute(0, 3, 1, 2)
        .contiguous()
        .float()
        .mul(0.25)
    )
    reference_output, reference_lse = flash_interface.flash_attn_func(
        q_represented.bfloat16(),
        k_represented.bfloat16(),
        v_represented.bfloat16(),
        causal=True,
        return_lse=True,
    )

    perturbed_index = args.batch - 1
    changed_rows = rows.clone()
    changed_rows.view(args.batch, args.sequence, args.hidden)[
        perturbed_index
    ].mul_(0.5).add_(0.25)
    changed_bundle = publish(changed_rows, 0, args.batch)
    changed_output = torch.empty_like(output)
    changed_lse = torch.empty_like(lse)
    run_forward(batched, changed_bundle, changed_output, changed_lse)
    torch.cuda.synchronize()

    wrong_batch_rejected = False
    try:
        run_forward(batched, batch1_bundles[0], output[:1], lse[:1])
        torch.cuda.synchronize()
    except (RuntimeError, ValueError):
        wrong_batch_rejected = True

    def run_batched_attention() -> None:
        batched.forward_hao_direct_fp8pv(
            *batched_bundle.qk_forward_operands(),
            batched_bundle.v_forward_fp8,
            output,
            lse,
            0,
            True,
            False,
        )

    def run_sequential_attention() -> None:
        for index, bundle in enumerate(batch1_bundles):
            batch1.forward_hao_direct_fp8pv(
                *bundle.qk_forward_operands(),
                bundle.v_forward_fp8,
                sequential_outputs[index],
                sequential_lses[index],
                0,
                True,
                False,
            )

    bf16_output = torch.empty_like(output)

    def run_bf16_attention() -> None:
        flash_interface._flash_attn_fwd(
            q_represented.bfloat16(),
            k_represented.bfloat16(),
            v_represented.bfloat16(),
            out=bf16_output,
            causal=True,
            return_lse=False,
            num_splits=1,
            pack_gqa=False,
        )

    timings = _time_rotated(
        {
            "batched_exact": run_batched_attention,
            "sequential_batch1_exact": run_sequential_attention,
            "bf16_fa4": run_bf16_attention,
        },
        warmups=args.warmups,
        samples=args.samples,
    )
    batch_ms = float(timings["batched_exact"]["median_ms"])
    sequential_ms = float(timings["sequential_batch1_exact"]["median_ms"])
    bf16_ms = float(timings["bf16_fa4"]["median_ms"])
    sequential_metrics = {
        "output": _metrics(sequential_output, output),
        "lse": _metrics(sequential_lse, lse),
    }
    reference_metrics = {
        "output": _metrics(reference_output, output),
        "lse": _metrics(reference_lse.unsqueeze(2), lse),
    }
    untouched_isolation = {
        f"sample_{index}": {
            "output_byte_equal": _byte_equal(
                output[index], changed_output[index]
            ),
            "lse_byte_equal": _byte_equal(lse[index], changed_lse[index]),
        }
        for index in range(args.batch - 1)
    }
    isolation = {
        "perturbed_sample": perturbed_index,
        "untouched_samples": untouched_isolation,
        "perturbed_output_changed": not _byte_equal(
            output[perturbed_index], changed_output[perturbed_index]
        ),
    }
    sample_isolation_pass = (
        all(all(values.values()) for values in untouched_isolation.values())
        and isolation["perturbed_output_changed"]
    )
    checks = {
        "all_publications_match_sequential_batch1": all(publication_equal.values()),
        "sequential_output_cosine": sequential_metrics["output"]["cosine"] > 0.999999,
        "sequential_output_relative_l2": (
            sequential_metrics["output"]["relative_l2"]
            <= SEQUENTIAL_RELATIVE_L2_LIMIT
        ),
        "sequential_lse_cosine": sequential_metrics["lse"]["cosine"] > 0.999999,
        "sequential_lse_relative_l2": (
            sequential_metrics["lse"]["relative_l2"]
            <= SEQUENTIAL_RELATIVE_L2_LIMIT
        ),
        "represented_output_cosine": reference_metrics["output"]["cosine"] >= 0.999,
        "represented_output_relative_l2": reference_metrics["output"]["relative_l2"] <= 0.03,
        "sample_isolation": sample_isolation_pass,
        "wrong_batch_rejected": wrong_batch_rejected,
        "batched_faster_than_sequential_batch1": batch_ms < sequential_ms,
    }
    free_after, total_memory_after = torch.cuda.mem_get_info(args.gpu)
    if total_memory_after != total_memory:
        raise RuntimeError("CUDA total memory changed during validation")
    result = {
        "schema": "causal_gqa_fp8pv_batch_validation_v3",
        "shape": {
            "batch": args.batch,
            "sequence": args.sequence,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": 64,
        },
        "topology": {
            "batched": batched_topology,
            "batch1": batch1_topology,
        },
        "artifacts": {
            "batched_forward": batched_identity,
            "batch1_forward": batch1_identity,
            "projection_extension": projection_identity,
        },
        "publication_byte_equal": publication_equal,
        "sequential_batch1_comparison": sequential_metrics,
        "represented_input_comparison": reference_metrics,
        "sample_isolation": isolation,
        "timing": timings,
        "timing_protocol": "per-sample rotated launch order",
        "sequential_relative_l2_limit": SEQUENTIAL_RELATIVE_L2_LIMIT,
        "speedup": {
            "batched_vs_sequential_batch1": sequential_ms / batch_ms,
            "batched_exact_vs_bf16_fa4": bf16_ms / batch_ms,
        },
        "device": {
            "name": torch.cuda.get_device_name(args.gpu),
            "memory": {
                "total_bytes": total_memory,
                "free_before_bytes": free_before,
                "free_after_bytes": free_after,
                "allocated_before_bytes": allocated_before,
                "allocated_after_bytes": torch.cuda.memory_allocated(args.gpu),
                "reserved_before_bytes": reserved_before,
                "reserved_after_bytes": torch.cuda.memory_reserved(args.gpu),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                    args.gpu
                ),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.gpu),
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n")
    print(encoded)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
