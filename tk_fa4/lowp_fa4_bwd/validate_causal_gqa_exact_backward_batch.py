#!/usr/bin/env python3
"""Validate the narrow causal exact D64 batched-backward specialization.

The batched result is compared with independent B1 launches from the same
represented E4M3 state and with BF16 backward controls.  A second batched
launch perturbs only the final sample to detect cross-sample indexing leakage.
The precomposed backward control is authenticated before it is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import CompiledGqaBackward
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control


REPO_ROOT = Path(__file__).resolve().parents[2]
FLASH_ATTN_ROOT = REPO_ROOT / "flash-attention"
DEPTH = 64
DECODE_SCALE = 0.25
REPEATABILITY_RELATIVE_L2_LIMIT = 0.005


@dataclass
class RepresentedState:
    q_fp8: torch.Tensor
    k_fp8: torch.Tensor
    v_fp8: torch.Tensor
    dout_fp8: torch.Tensor
    q_bf16: torch.Tensor
    k_bf16: torch.Tensor
    v_bf16: torch.Tensor
    dout_bf16: torch.Tensor
    output_bf16: torch.Tensor
    lse_bh1s: torch.Tensor
    direct_dpsum: torch.Tensor
    direct_lse_log2: torch.Tensor

    def sample(self, index: int) -> RepresentedState:
        if index not in range(self.q_fp8.shape[0]):
            raise IndexError(index)
        values = {
            # A batch slice can already report contiguous while retaining the
            # parent's storage.  The reusable B1 validator mutates this state,
            # so require a real allocation rather than a contiguous view.
            name: getattr(self, name)[index : index + 1].clone(
                memory_format=torch.contiguous_format
            )
            for name in self.__dataclass_fields__
        }
        return RepresentedState(**values)

    def copy_sample_from_(
        self,
        source: RepresentedState,
        index: int,
    ) -> None:
        """Copy one source sample into this fixed B1 state in place."""
        if self.q_fp8.shape[0] != 1:
            raise ValueError("the reusable reference state must be B1")
        if index not in range(source.q_fp8.shape[0]):
            raise IndexError(index)
        for name in self.__dataclass_fields__:
            destination = getattr(self, name)
            sample = getattr(source, name)[index : index + 1]
            if destination.shape != sample.shape:
                raise ValueError(
                    f"reusable {name} shape mismatch: "
                    f"{tuple(destination.shape)} != {tuple(sample.shape)}"
                )
            destination.copy_(sample)


def _load_flash_attention() -> Callable[..., Any]:
    sys.path.insert(0, str(FLASH_ATTN_ROOT))
    try:
        from flash_attn.cute.interface import flash_attn_func
    finally:
        sys.path.pop(0)
    return flash_attn_func


def _represented_e4m3(
    shape: tuple[int, ...],
    *,
    standard_deviation: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = (
        torch.randn(shape, device="cuda", dtype=torch.float32)
        .mul_(standard_deviation)
        .bfloat16()
    )
    encoded = source.float().mul_(4.0).to(torch.float8_e4m3fn)
    represented = encoded.float().mul_(DECODE_SCALE).bfloat16()
    return encoded, represented


def _make_state(
    *,
    batch: int,
    sequence: int,
    q_heads: int,
    kv_heads: int,
    seed: int,
) -> RepresentedState:
    torch.manual_seed(seed)
    q_shape = (batch, sequence, q_heads, DEPTH)
    kv_shape = (batch, sequence, kv_heads, DEPTH)
    q_fp8, q_bf16 = _represented_e4m3(
        q_shape, standard_deviation=0.25
    )
    k_fp8, k_bf16 = _represented_e4m3(
        kv_shape, standard_deviation=0.25
    )
    v_fp8, v_bf16 = _represented_e4m3(
        kv_shape, standard_deviation=0.25
    )
    dout_fp8, dout_bf16 = _represented_e4m3(
        q_shape, standard_deviation=0.10
    )

    flash_attention = _load_flash_attention()
    output_bf16, lse = flash_attention(
        q_bf16,
        k_bf16,
        v_bf16,
        causal=True,
        return_lse=True,
    )
    if lse.ndim == 3:
        lse_bh1s = lse.unsqueeze(2).contiguous()
    elif lse.ndim == 4 and lse.shape[2] == 1:
        lse_bh1s = lse.contiguous()
    else:
        raise RuntimeError(f"unexpected LSE shape {tuple(lse.shape)}")

    # The projection-native producer stores -16*sum(O*dO).  dout_fp8 carries
    # one factor of four, so this explicit four is the represented O factor.
    direct_dpsum = (
        -4.0
        * (output_bf16.float() * dout_fp8.float())
        .sum(dim=-1)
        .permute(0, 2, 1)
        .unsqueeze(2)
    ).contiguous()
    # D64 pre-lifts P by 2**8 and consumes -LSE*log2(e)+8 directly.
    direct_lse_log2 = (
        -math.log2(math.e) * lse_bh1s + 8.0
    ).contiguous()
    return RepresentedState(
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        dout_fp8=dout_fp8,
        q_bf16=q_bf16,
        k_bf16=k_bf16,
        v_bf16=v_bf16,
        dout_bf16=dout_bf16,
        output_bf16=output_bf16,
        lse_bh1s=lse_bh1s,
        direct_dpsum=direct_dpsum,
        direct_lse_log2=direct_lse_log2,
    )


def _build_lowp(
    control: Any,
    state: RepresentedState,
    *,
    q_heads: int,
    kv_heads: int,
) -> CompiledGqaBackward:
    return CompiledGqaBackward(
        control,
        q=state.q_fp8,
        k=state.k_fp8,
        v=state.v_fp8,
        o_or_sum=state.direct_dpsum,
        dout=state.dout_fp8,
        lse_or_scaled_lse=state.direct_lse_log2,
        q_heads=q_heads,
        kv_heads=kv_heads,
        lowp=True,
        precomputed_stats=True,
        workspace_stats=True,
        scale_softmax=(DEPTH**-0.5) / 16.0,
        exp2_degree=1,
        exp2_period=2,
        reuse_quantized_p=False,
        fp8_ds_lift=16,
        lowp_do_stages=1,
        head_fast_raster=False,
        direct_tma_dkdv=True,
    )


def _build_bf16(
    control: Any,
    state: RepresentedState,
    *,
    q_heads: int,
    kv_heads: int,
) -> CompiledGqaBackward:
    if state.q_bf16.shape[0] != 1:
        raise ValueError("the BF16 control intentionally remains B1")
    return CompiledGqaBackward(
        control,
        q=state.q_bf16,
        k=state.k_bf16,
        v=state.v_bf16,
        o_or_sum=state.output_bf16,
        dout=state.dout_bf16,
        lse_or_scaled_lse=state.lse_bh1s,
        q_heads=q_heads,
        kv_heads=kv_heads,
        lowp=False,
        precomputed_stats=False,
        scale_softmax=DEPTH**-0.5,
    )


def _publish_workspace_statistics(
    backward: CompiledGqaBackward,
    state: RepresentedState,
) -> None:
    stats_numel = state.direct_dpsum.numel()
    pages = backward.workspace_torch[: 2 * stats_numel * 4].view(
        torch.float32
    )
    pages[:stats_numel].copy_(state.direct_dpsum.reshape(-1))
    pages[stats_numel:].copy_(state.direct_lse_log2.reshape(-1))


def _capture(
    backward: CompiledGqaBackward,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    backward.run(reset=True)
    torch.cuda.synchronize()
    return tuple(
        value.clone() for value in (backward.dq, backward.dk, backward.dv)
    )


def _metrics(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, Any]:
    reference_f = reference.float()
    actual_f = actual.float()
    difference = actual_f - reference_f
    reference_norm = reference_f.norm().clamp_min(1.0e-30)
    actual_norm = actual_f.norm().clamp_min(1.0e-30)
    return {
        "reference_finite": bool(torch.isfinite(reference_f).all()),
        "actual_finite": bool(torch.isfinite(actual_f).all()),
        "cosine": float(
            (reference_f * actual_f).sum()
            / (reference_norm * actual_norm)
        ),
        "relative_l2": float(difference.norm() / reference_norm),
        "norm_ratio": float(actual_norm / reference_norm),
        "max_abs": float(difference.abs().max()),
    }


def _aggregate_metrics(
    references: tuple[torch.Tensor, ...],
    actuals: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    dot = torch.zeros((), device="cuda", dtype=torch.float64)
    reference_square = torch.zeros_like(dot)
    actual_square = torch.zeros_like(dot)
    difference_square = torch.zeros_like(dot)
    maximum = torch.zeros((), device="cuda", dtype=torch.float32)
    reference_finite = True
    actual_finite = True
    for reference, actual in zip(references, actuals, strict=True):
        reference_f = reference.float()
        actual_f = actual.float()
        difference = actual_f - reference_f
        dot += (reference_f * actual_f).sum(dtype=torch.float64)
        reference_square += reference_f.square().sum(dtype=torch.float64)
        actual_square += actual_f.square().sum(dtype=torch.float64)
        difference_square += difference.square().sum(dtype=torch.float64)
        maximum = torch.maximum(maximum, difference.abs().max())
        reference_finite &= bool(torch.isfinite(reference_f).all())
        actual_finite &= bool(torch.isfinite(actual_f).all())
    reference_norm = reference_square.sqrt().clamp_min(1.0e-30)
    actual_norm = actual_square.sqrt().clamp_min(1.0e-30)
    return {
        "reference_finite": reference_finite,
        "actual_finite": actual_finite,
        "cosine": float(dot / (reference_norm * actual_norm)),
        "relative_l2": float(difference_square.sqrt() / reference_norm),
        "norm_ratio": float(actual_norm / reference_norm),
        "max_abs": float(maximum),
    }


def _gradient_metrics(
    references: tuple[torch.Tensor, ...],
    actuals: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    result = {
        name: _metrics(reference, actual)
        for name, reference, actual in zip(
            ("dq", "dk", "dv"), references, actuals, strict=True
        )
    }
    result["aggregate"] = _aggregate_metrics(references, actuals)
    return result


def _repeatability_within_limit(
    repeatability: dict[str, Any],
    *,
    limit: float = REPEATABILITY_RELATIVE_L2_LIMIT,
) -> bool:
    """Reject unstable launches before using their noise as leakage slack."""
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError(
            "repeatability relative-L2 limit must be finite and nonnegative"
        )
    if not repeatability:
        return False
    return all(
        sample["aggregate"]["reference_finite"]
        and sample["aggregate"]["actual_finite"]
        and sample["aggregate"]["relative_l2"] <= limit
        for sample in repeatability.values()
    )


def _time_rotated(
    runners: dict[str, Callable[[], None]],
    *,
    warmups: int,
    samples: int,
) -> dict[str, dict[str, Any]]:
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
            values[name].append(float(start.elapsed_time(end) * 1000.0))
    return {
        name: {
            "median_us": statistics.median(samples_us),
            "minimum_us": min(samples_us),
            "samples_us": samples_us,
        }
        for name, samples_us in values.items()
    }


def _authenticate_control(
    source: Path,
    expected_sha256: str,
    expected_bytes: int,
) -> dict[str, Any]:
    resolved = source.resolve(strict=True)
    payload = resolved.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_bytes or actual != expected_sha256:
        raise RuntimeError(
            "precomposed control authentication failed: "
            f"bytes={len(payload)} sha256={actual}"
        )
    return {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": actual,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(2, 8, 16), default=2)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--control-source", required=True, type=Path)
    parser.add_argument("--control-sha256", required=True)
    parser.add_argument("--control-bytes", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to this validator")
    if args.sequence <= 0 or args.sequence % 128:
        raise ValueError("sequence must be a positive multiple of 128")
    if args.q_heads <= 0 or args.kv_heads <= 0:
        raise ValueError("head counts must be positive")
    if args.q_heads % args.kv_heads:
        raise ValueError("q-heads must be divisible by kv-heads")
    if args.warmups < 0 or args.samples <= 0:
        raise ValueError("invalid timing sample counts")

    torch.cuda.set_device(0)
    free_before, total_memory = torch.cuda.mem_get_info()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    control_identity = _authenticate_control(
        args.control_source,
        args.control_sha256,
        args.control_bytes,
    )
    state = _make_state(
        batch=args.batch,
        sequence=args.sequence,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        seed=args.seed,
    )
    reference_state = state.sample(0)
    reference_state_storage_disjoint = all(
        getattr(reference_state, name).untyped_storage().data_ptr()
        != getattr(state, name).untyped_storage().data_ptr()
        for name in state.__dataclass_fields__
    )
    control = _load_control(
        fp8_p_storage="tmem",
        direct_tma_dkdv=True,
        precomposed_control_source=args.control_source,
        precomposed_control_sha256=args.control_sha256,
        precomposed_control_bytes=args.control_bytes,
    )
    # The authenticated direct-TMA module is deliberately lowp-only.  Compile
    # BF16 from the ordinary non-direct control instead of weakening that ABI.
    bf16_control = _load_control()

    batched = _build_lowp(
        control,
        state,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
    )
    sequential = _build_lowp(
        control,
        reference_state,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
    )
    bf16 = _build_bf16(
        bf16_control,
        reference_state,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
    )

    batched_values = _capture(batched)
    repeat_values = _capture(batched)
    sequential_values_list = []
    bf16_values_list = []
    for index in range(args.batch):
        reference_state.copy_sample_from_(state, index)
        _publish_workspace_statistics(sequential, reference_state)
        sequential_values_list.append(_capture(sequential))
        bf16_values_list.append(_capture(bf16))
    sequential_values = tuple(sequential_values_list)
    bf16_values = tuple(bf16_values_list)

    batch_equivalence: dict[str, Any] = {}
    bf16_quality: dict[str, Any] = {}
    for index in range(args.batch):
        batched_sample = tuple(value[index : index + 1] for value in batched_values)
        batch_equivalence[f"sample_{index}"] = _gradient_metrics(
            sequential_values[index], batched_sample
        )
        decoded = tuple(value.float().mul(DECODE_SCALE) for value in batched_sample)
        bf16_quality[f"sample_{index}"] = _gradient_metrics(
            bf16_values[index], decoded
        )

    repeatability = {
        f"sample_{index}": _gradient_metrics(
            tuple(value[index : index + 1] for value in batched_values),
            tuple(value[index : index + 1] for value in repeat_values),
        )
        for index in range(args.batch)
    }
    repeatability_pass = _repeatability_within_limit(repeatability)

    perturbed_index = args.batch - 1
    original_dout = state.dout_fp8[perturbed_index].clone()
    state.dout_fp8[perturbed_index].copy_(
        (-original_dout.float()).to(torch.float8_e4m3fn)
    )
    state.direct_dpsum.copy_(
        -4.0
        * (state.output_bf16.float() * state.dout_fp8.float())
        .sum(dim=-1)
        .permute(0, 2, 1)
        .unsqueeze(2)
    )
    _publish_workspace_statistics(batched, state)
    perturbed_values = _capture(batched)
    cross_sample = {
        f"sample_{index}": _gradient_metrics(
            tuple(value[index : index + 1] for value in batched_values),
            tuple(value[index : index + 1] for value in perturbed_values),
        )
        for index in range(args.batch)
    }
    state.dout_fp8[perturbed_index].copy_(original_dout)
    state.direct_dpsum.copy_(
        -4.0
        * (state.output_bf16.float() * state.dout_fp8.float())
        .sum(dim=-1)
        .permute(0, 2, 1)
        .unsqueeze(2)
    )
    _publish_workspace_statistics(batched, state)

    def run_sequential_b1() -> None:
        # Kernel timing is data-independent for this fixed shape.  Reusing one
        # authenticated B1 controller avoids compiling one generated module
        # per sample; sample-specific copies happen above, outside timing.
        for _ in range(args.batch):
            sequential.run(reset=True)

    timing = _time_rotated(
        {
            "batched_with_clear": lambda: batched.run(reset=True),
            "sequential_b1_with_clear": run_sequential_b1,
        },
        warmups=args.warmups,
        samples=args.samples,
    )
    timing_batched = timing["batched_with_clear"]
    timing_sequential = timing["sequential_b1_with_clear"]
    free_after, total_memory_after = torch.cuda.mem_get_info()
    if total_memory_after != total_memory:
        raise RuntimeError("CUDA total memory changed during validation")
    memory = {
        "total_bytes": total_memory,
        "free_before_bytes": free_before,
        "free_after_bytes": free_after,
        "allocated_before_bytes": allocated_before,
        "allocated_after_bytes": torch.cuda.memory_allocated(),
        "reserved_before_bytes": reserved_before,
        "reserved_after_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }

    equivalence_pass = all(
        sample["aggregate"]["actual_finite"]
        and sample["aggregate"]["cosine"] >= 0.999
        and sample["aggregate"]["relative_l2"] <= 0.02
        for sample in batch_equivalence.values()
    )
    bf16_pass = all(
        sample["aggregate"]["actual_finite"]
        and sample["aggregate"]["cosine"] >= 0.997
        and sample["aggregate"]["relative_l2"] <= 0.10
        and 0.85 <= sample["aggregate"]["norm_ratio"] <= 1.10
        for sample in bf16_quality.values()
    )
    leakage_limits = {
        f"sample_{index}": max(
            0.01,
            5.0
            * repeatability[f"sample_{index}"]["aggregate"]["relative_l2"],
        )
        for index in range(args.batch - 1)
    }
    untouched_isolated = all(
        cross_sample[name]["aggregate"]["actual_finite"]
        and cross_sample[name]["aggregate"]["relative_l2"] <= limit
        for name, limit in leakage_limits.items()
    )
    perturbed_changed = (
        cross_sample[f"sample_{perturbed_index}"]["aggregate"]["relative_l2"]
        >= 0.5
    )
    isolation_pass = untouched_isolated and perturbed_changed
    passed = (
        reference_state_storage_disjoint
        and equivalence_pass
        and bf16_pass
        and repeatability_pass
        and isolation_pass
    )

    document = {
        "schema": "fp4_fa4_causal_exact_backward_batch_v3",
        "status": "passed" if passed else "failed",
        "shape": {
            "batch": args.batch,
            "sequence": args.sequence,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": DEPTH,
        },
        "control": control_identity,
        "policy": {
            "input": "represented_e4m3_q_k_v_dout_x4",
            "probability_storage": "aliased_tmem",
            "direct_tma_dkdv": True,
            "hierarchical_dq_lanes": 1,
            "head_fast_raster": False,
            "exp2_degree": 1,
            "exp2_period": 2,
            "fp8_ds_lift": 16,
            "decoded_gradient_scale": DECODE_SCALE,
            "repeatability_relative_l2_limit": (
                REPEATABILITY_RELATIVE_L2_LIMIT
            ),
            "sequential_reference_runner_count": 1,
            "sequential_reference_inputs_copied_outside_timing": True,
        },
        "correctness": {
            "batch_equivalence_vs_sequential_b1": batch_equivalence,
            "bf16_quality": bf16_quality,
            "batched_repeatability": repeatability,
            "single_sample_dout_sign_flip": {
                "perturbed_sample": perturbed_index,
                "metrics": cross_sample,
                "untouched_sample_relative_l2_limits": leakage_limits,
            },
        },
        "gates": {
            "reference_state_storage_disjoint": (
                reference_state_storage_disjoint
            ),
            "batch_equivalence": equivalence_pass,
            "bf16_quality": bf16_pass,
            "repeatability": repeatability_pass,
            "cross_sample_isolation": isolation_pass,
        },
        "timing": {
            "protocol": (
                "per-sample rotated launch order with destination clears; "
                "one fixed B1 controller launched batch times"
            ),
            "batched_with_clear": timing_batched,
            "sequential_b1_with_clear": timing_sequential,
            "batched_speedup_vs_sequential_b1": (
                timing_sequential["median_us"]
                / timing_batched["median_us"]
            ),
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "memory": memory,
        },
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if not passed:
        raise RuntimeError(
            "batched exact backward validation failed: "
            f"{document['gates']}"
        )


if __name__ == "__main__":
    main()
