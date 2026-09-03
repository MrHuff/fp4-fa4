#!/usr/bin/env python3
"""Validate separately compiled batched v509 extensions against B1 execution.

The gate requires one pairwise-distinct captured attention boundary per batch
lane.  It derives the exact v509 E5M2-dO statistics ABI, executes every lane
through an authenticated B1 v509 binary to form independent references, and
checks both listed and reversed orders.  This detects modulo-smaller-batch
descriptor/raster aliasing.  dK/dV must reproduce B1 bitwise.  dQ is published
by concurrent BF16 TMA store-add operations, so its rounding order changes with
CTA scheduling; it must remain within a tight per-lane relative-L2, peak-error,
and sparsity envelope.  A second gate requires exactly zero dQ/dK/dV for
exactly zero dO/dstat in both orders.

This is a backward raster/address-isolation gate.  It does not replace the
separate fused E5M2 publisher validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import statistics
import sys
from typing import Any

import torch


CONSUMED_FIELDS = (
    "q_fp8",
    "k_fp8",
    "v_fp8",
    "dout_e5m2",
    "lstat",
    "dstat",
    "q_forward_payload_uint8",
    "k_forward_payload_uint8",
    "q_forward_scale_pages_workspace",
    "k_forward_scale_pages_workspace",
    "q_forward_global_scale_workspace",
    "k_forward_global_scale_workspace",
)
CAPTURE_FIELDS = (
    "q_fp8",
    "k_fp8",
    "v_fp8",
    "forward_attention_output",
    "forward_lse",
    "q_forward_payload_uint8",
    "k_forward_payload_uint8",
    "q_forward_scale_pages_workspace",
    "k_forward_scale_pages_workspace",
    "q_forward_global_scale_workspace",
    "k_forward_global_scale_workspace",
)
LOG2_E = 1.4426950408889634
DQ_STORE_ADD_RELATIVE_L2_LIMIT = 1.0e-3
DQ_STORE_ADD_PEAK_RELATIVE_LIMIT = 1.0e-2
DQ_STORE_ADD_CHANGED_FRACTION_LIMIT = 2.5e-4


def load_module(path: pathlib.Path, batch: int) -> Any:
    name = (
        "_C_sm100_gqa_tk_v509_d128_nvfp4_score_e4m3_qkv_e5m2_dout_"
        f"b{batch}_s4096"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    metadata = dict(module.native_tk_d128_backward_metadata())
    expected = {
        "batch": batch,
        "sequence": 4096,
        "query_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "dispatch": f"fail_closed_B{batch}_S4096_only_no_fallback",
        "precleared_dq_out_clears_dk_dv": True,
    }
    mismatches = {
        key: {"actual": metadata.get(key), "expected": value}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"extension metadata mismatch: {mismatches}")
    return module


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(tensor: torch.Tensor, digest: Any) -> None:
    cpu = tensor.detach().cpu().contiguous()
    digest.update(str(cpu.dtype).encode())
    digest.update(str(tuple(cpu.shape)).encode())
    digest.update(memoryview(cpu.view(torch.uint8).numpy()))


def effective_input_digest(capture: dict[str, torch.Tensor]) -> str:
    """Digest the complete tuple consumed by one B1 v509 invocation."""

    digest = hashlib.sha256()
    for field in CONSUMED_FIELDS:
        digest.update(field.encode())
        _tensor_digest(capture[field], digest)
    return digest.hexdigest()


def require_pairwise_distinct_effective_inputs(
    digests: list[str], *, batch: int
) -> None:
    if len(digests) != batch or len(set(digests)) != batch:
        raise ValueError(
            f"exact B{batch} gate requires {batch} pairwise-distinct "
            "effective input tuples"
        )


def require_nontrivial_independent_outputs(
    outputs: list[dict[str, torch.Tensor]],
) -> list[dict[str, int]]:
    """Reject captures whose B1 reference does not exercise every gradient."""

    counts = [
        {
            name: int(torch.count_nonzero(lane[name]))
            for name in ("dq", "dk", "dv")
        }
        for lane in outputs
    ]
    zero_outputs = [
        {name: count for name, count in lane.items() if count == 0}
        for lane in counts
    ]
    if any(zero_outputs):
        raise ValueError(
            "every independent B1 capture must produce nonzero dQ/dK/dV; "
            f"zero outputs by lane: {zero_outputs}"
        )
    return counts


def load_capture(path: pathlib.Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise TypeError(f"capture is not a dictionary: {path}")
    missing = set(CAPTURE_FIELDS) - set(payload)
    if missing:
        raise RuntimeError(f"capture {path} is missing {sorted(missing)}")
    source_dout = payload.get("dout_e5m2", payload.get("dout_fp8"))
    if not isinstance(source_dout, torch.Tensor):
        raise RuntimeError(
            f"capture {path} needs dout_e5m2 or represented-x4 dout_fp8"
        )
    # Both capture variants store the already-x4 represented dO value.  Change
    # only its element format; do not apply another x4 encoding scale.
    dout_e5m2 = source_dout.float().to(torch.float8_e5m2).contiguous()
    attention_output = payload["forward_attention_output"]
    forward_lse = payload["forward_lse"]
    if (
        attention_output.dtype != torch.bfloat16
        or tuple(attention_output.shape) != (1, 4096, 32, 128)
    ):
        raise RuntimeError(f"capture {path} has an invalid attention output")
    if forward_lse.dtype != torch.float32 or tuple(forward_lse.shape) != (
        1,
        32,
        1,
        4096,
    ):
        raise RuntimeError(f"capture {path} has an invalid forward LSE")
    lstat = (8.0 - forward_lse.float() * LOG2_E).contiguous()
    dstat = (
        -4.0
        * (attention_output.float() * dout_e5m2.float()).sum(dim=-1)
    ).permute(0, 2, 1).unsqueeze(2).contiguous()
    result: dict[str, torch.Tensor] = {}
    for field in CAPTURE_FIELDS:
        if field in ("forward_attention_output", "forward_lse"):
            continue
        tensor = payload[field]
        if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != 1:
            raise RuntimeError(f"capture field {field} is not a B1 tensor")
        result[field] = tensor
    result["dout_e5m2"] = dout_e5m2
    result["lstat"] = lstat
    result["dstat"] = dstat
    return result


def concatenate(
    captures: list[dict[str, torch.Tensor]], field: str, device: torch.device
) -> torch.Tensor:
    return torch.cat([capture[field] for capture in captures], dim=0).to(
        device=device, non_blocking=False
    ).contiguous()


def resolve_distinct_capture_paths(
    paths: list[pathlib.Path], *, batch: int
) -> list[pathlib.Path]:
    """Require one distinct resolved capture path for every batch lane."""

    resolved = [path.resolve(strict=True) for path in paths]
    if len(resolved) != batch or len(set(resolved)) != batch:
        raise ValueError(
            f"--capture requires exactly {batch} distinct resolved B1 files"
        )
    return resolved


def batched_capture_orders(capture_count: int, batch: int) -> dict[str, tuple[int, ...]]:
    """Return the one-capture-per-lane order and its reverse."""

    if capture_count != batch:
        raise ValueError("capture_count must equal the exact batch")
    listed = tuple(range(batch))
    reversed_order = tuple(reversed(listed))
    if listed == reversed_order:
        raise ValueError("listed and reversed capture orders must differ")
    return {"listed": listed, "reversed": reversed_order}


def output_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    delta = actual.float() - expected.float()
    if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
        raise TypeError("v509 isolation outputs must be BF16")
    actual_bits = actual.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
    expected_bits = expected.view(torch.int16).to(torch.int32).bitwise_and(
        0xFFFF
    )

    def ordered_bf16(bits: torch.Tensor) -> torch.Tensor:
        magnitude = bits.bitwise_and(0x7FFF)
        return torch.where(
            bits.bitwise_and(0x8000).ne(0),
            0x8000 - magnitude,
            0x8000 + magnitude,
        )

    ulp_distance = (
        ordered_bf16(actual_bits) - ordered_bf16(expected_bits)
    ).abs()
    flat_delta = delta.flatten()
    flat_ulp = ulp_distance.flatten()
    max_abs_flat_index = int(flat_delta.abs().argmax())
    max_ulp_flat_index = int(flat_ulp.argmax())
    flat_actual = actual.flatten()
    flat_expected = expected.flatten()
    delta_l2 = float(torch.linalg.vector_norm(flat_delta))
    expected_l2 = float(torch.linalg.vector_norm(expected.float()))
    expected_peak = float(expected.float().abs().max())
    max_abs = float(delta.abs().max())
    return {
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "finite": bool(torch.isfinite(actual.float()).all()),
        "max_abs": max_abs,
        "mean_abs": float(delta.abs().mean()),
        "delta_l2": delta_l2,
        "expected_l2": expected_l2,
        "relative_l2": delta_l2 / expected_l2 if expected_l2 else None,
        "expected_peak_abs": expected_peak,
        "max_abs_over_expected_peak": (
            max_abs / expected_peak if expected_peak else None
        ),
        "max_bf16_ulp": int(ulp_distance.max()),
        "max_abs_example": {
            "flat_index": max_abs_flat_index,
            "actual": float(flat_actual[max_abs_flat_index]),
            "expected": float(flat_expected[max_abs_flat_index]),
        },
        "max_bf16_ulp_example": {
            "flat_index": max_ulp_flat_index,
            "actual": float(flat_actual[max_ulp_flat_index]),
            "expected": float(flat_expected[max_ulp_flat_index]),
        },
        "over_one_bf16_ulp_elements": int(torch.count_nonzero(ulp_distance > 1)),
        "nonzero_elements": int(torch.count_nonzero(delta)),
        "nonzero_fraction": float(torch.count_nonzero(delta) / delta.numel()),
    }


def dq_store_add_order_ok(error: dict[str, Any]) -> bool:
    """Bound only the expected BF16 concurrent-store accumulation variance."""

    relative_l2 = error["relative_l2"]
    peak_relative = error["max_abs_over_expected_peak"]
    if relative_l2 is None or peak_relative is None:
        return bool(error["bitwise_equal"])
    return bool(
        error["finite"]
        and relative_l2 <= DQ_STORE_ADD_RELATIVE_L2_LIMIT
        and peak_relative <= DQ_STORE_ADD_PEAK_RELATIVE_LIMIT
        and error["nonzero_fraction"]
        <= DQ_STORE_ADD_CHANGED_FRACTION_LIMIT
    )


def precleared_zero_dout_semantics(
    outputs: dict[str, torch.Tensor],
    dq_sentinel: torch.Tensor,
) -> dict[str, Any]:
    """Prove the fused entrypoint preserves dQ and clears dK/dV."""

    result = {
        "dq_sentinel_preserved_bitwise": bool(
            torch.equal(outputs["dq"], dq_sentinel)
        ),
        "dq_nonzero_count": int(torch.count_nonzero(outputs["dq"])),
        "dk_nonzero_count": int(torch.count_nonzero(outputs["dk"])),
        "dv_nonzero_count": int(torch.count_nonzero(outputs["dv"])),
    }
    result["passed"] = bool(
        result["dq_sentinel_preserved_bitwise"]
        and result["dk_nonzero_count"] == 0
        and result["dv_nonzero_count"] == 0
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=pathlib.Path, required=True)
    parser.add_argument("--reference-module", type=pathlib.Path, required=True)
    parser.add_argument("--batch", type=int, choices=(2, 4), required=True)
    parser.add_argument(
        "--capture", type=pathlib.Path, action="append", required=True
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.warmups < 0 or args.samples <= 0:
        raise ValueError("warmups must be nonnegative and samples positive")

    resolved_capture_paths = resolve_distinct_capture_paths(
        args.capture, batch=args.batch
    )
    source_captures = [load_capture(path) for path in resolved_capture_paths]
    effective_digests = [
        effective_input_digest(capture) for capture in source_captures
    ]
    require_pairwise_distinct_effective_inputs(
        effective_digests, batch=args.batch
    )
    capture_orders = batched_capture_orders(len(source_captures), args.batch)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    module_path = args.module.resolve(strict=True)
    reference_module_path = args.reference_module.resolve(strict=True)
    if module_path == reference_module_path:
        raise ValueError("exact-batch and B1 reference modules must differ")
    module = load_module(module_path, args.batch)
    reference_module = load_module(reference_module_path, 1)
    entrypoint = module.backward_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precomputed_out
    precleared_dq_entrypoint = getattr(
        module,
        "backward_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precleared_dq_out",
    )
    reference_entrypoint = getattr(
        reference_module,
        "backward_nvfp4_score_e4m3_qkv_e5m2_dout_bshd_precomputed_out",
    )

    def run_b1_reference(
        capture: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        values = {
            field: capture[field].to(device=device).contiguous()
            for field in CONSUMED_FIELDS
        }
        q_native = values["q_forward_payload_uint8"].view(
            torch.float4_e2m1fn_x2
        )
        k_native = values["k_forward_payload_uint8"].view(
            torch.float4_e2m1fn_x2
        )
        outputs = {
            "dq": torch.empty_like(values["q_fp8"], dtype=torch.bfloat16),
            "dk": torch.empty_like(values["k_fp8"], dtype=torch.bfloat16),
            "dv": torch.empty_like(values["v_fp8"], dtype=torch.bfloat16),
        }
        reference_entrypoint(
            values["q_fp8"],
            values["k_fp8"],
            values["v_fp8"],
            values["dout_e5m2"],
            values["lstat"],
            values["dstat"],
            outputs["dq"],
            outputs["dk"],
            outputs["dv"],
            q_native,
            k_native,
            values["q_forward_scale_pages_workspace"],
            values["k_forward_scale_pages_workspace"],
            values["q_forward_global_scale_workspace"],
            values["k_forward_global_scale_workspace"],
            1.0 / 128.0**0.5,
        )
        torch.cuda.synchronize(device)
        return {name: value.cpu() for name, value in outputs.items()}

    independent_b1_outputs = [
        run_b1_reference(capture) for capture in source_captures
    ]
    independent_b1_nonzero_counts = require_nontrivial_independent_outputs(
        independent_b1_outputs
    )
    repeated_b1_outputs = [
        run_b1_reference(capture) for capture in source_captures
    ]
    b1_repeatability = [
        {
            name: output_error(repeated[name], original[name])
            for name in ("dq", "dk", "dv")
        }
        for original, repeated in zip(
            independent_b1_outputs, repeated_b1_outputs, strict=True
        )
    ]

    torch.cuda.reset_peak_memory_stats(device)

    def validate_order(
        order_name: str, order: tuple[int, ...], *, benchmark: bool
    ) -> dict[str, Any]:
        captures = [source_captures[index] for index in order]
        q = concatenate(captures, "q_fp8", device)
        k = concatenate(captures, "k_fp8", device)
        v = concatenate(captures, "v_fp8", device)
        dout = concatenate(captures, "dout_e5m2", device)
        lstat = concatenate(captures, "lstat", device)
        dstat = concatenate(captures, "dstat", device)
        q_native = concatenate(captures, "q_forward_payload_uint8", device).view(
            torch.float4_e2m1fn_x2
        )
        k_native = concatenate(captures, "k_forward_payload_uint8", device).view(
            torch.float4_e2m1fn_x2
        )
        q_scale = concatenate(captures, "q_forward_scale_pages_workspace", device)
        k_scale = concatenate(captures, "k_forward_scale_pages_workspace", device)
        q_global = concatenate(captures, "q_forward_global_scale_workspace", device)
        k_global = concatenate(captures, "k_forward_global_scale_workspace", device)
        expected = {
            name: torch.cat(
                [independent_b1_outputs[index][name] for index in order],
                dim=0,
            ).to(device=device).contiguous()
            for name in ("dq", "dk", "dv")
        }
        outputs = {
            "dq": torch.empty_like(expected["dq"]),
            "dk": torch.empty_like(expected["dk"]),
            "dv": torch.empty_like(expected["dv"]),
        }

        def run(run_dout: torch.Tensor, run_dstat: torch.Tensor) -> None:
            entrypoint(
                q,
                k,
                v,
                run_dout,
                lstat,
                run_dstat,
                outputs["dq"],
                outputs["dk"],
                outputs["dv"],
                q_native,
                k_native,
                q_scale,
                k_scale,
                q_global,
                k_global,
                1.0 / 128.0**0.5,
            )

        latency_ms: list[float] = []
        if benchmark:
            for _ in range(args.warmups):
                run(dout, dstat)
            torch.cuda.synchronize(device)
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.samples)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.samples)]
            for start, end in zip(starts, ends, strict=True):
                start.record()
                run(dout, dstat)
                end.record()
            torch.cuda.synchronize(device)
            latency_ms = [
                start.elapsed_time(end)
                for start, end in zip(starts, ends, strict=True)
            ]
        else:
            run(dout, dstat)
            torch.cuda.synchronize(device)

        errors = {name: output_error(outputs[name], expected[name]) for name in outputs}
        errors_by_batch = {
            name: [
                output_error(outputs[name][lane], expected[name][lane])
                for lane in range(args.batch)
            ]
            for name in outputs
        }
        first_batched_outputs = {
            name: value.detach().clone() for name, value in outputs.items()
        }
        run(dout, dstat)
        torch.cuda.synchronize(device)
        batched_repeatability = {
            name: output_error(outputs[name], first_batched_outputs[name])
            for name in outputs
        }
        # dK/dV are unique-writer outputs and remain bitwise invariant.  dQ is
        # a BF16 TMA store-add reduction: changing the global CTA population
        # changes legal addition order, especially around cancellation.  Gate
        # every lane with an explicit numerical envelope rather than claiming
        # bitwise or one-ULP determinism that the primitive does not provide.
        dq_ok = all(
            dq_store_add_order_ok(error)
            for error in errors_by_batch["dq"]
        )
        dkdv_ok = bool(
            errors["dk"]["bitwise_equal"] and errors["dv"]["bitwise_equal"]
        )
        if not (dq_ok and dkdv_ok):
            raise RuntimeError(
                f"{order_name} batched output differs from independent B1 receipts: "
                f"aggregate={errors}; per_batch={errors_by_batch}; "
                f"B1_repeatability={b1_repeatability}; "
                f"batched_repeatability={batched_repeatability}"
            )

        # The fused projection publisher has already cleared dQ.  Authenticate
        # the dedicated entrypoint used by that path: it must preserve the
        # publisher-owned dQ allocation while clearing/recomputing dK and dV,
        # and its gradients must be bitwise identical to the standalone-safe
        # entrypoint above.
        clearing_outputs = {
            name: value.detach().clone() for name, value in outputs.items()
        }
        outputs["dq"].zero_()
        outputs["dk"].fill_(1)
        outputs["dv"].fill_(-1)
        precleared_dq_entrypoint(
            q,
            k,
            v,
            dout,
            lstat,
            dstat,
            outputs["dq"],
            outputs["dk"],
            outputs["dv"],
            q_native,
            k_native,
            q_scale,
            k_scale,
            q_global,
            k_global,
            1.0 / 128.0**0.5,
        )
        torch.cuda.synchronize(device)
        precleared_dq_errors = {
            name: output_error(outputs[name], clearing_outputs[name])
            for name in outputs
        }
        precleared_dq_ok = bool(
            dq_store_add_order_ok(precleared_dq_errors["dq"])
            and precleared_dq_errors["dk"]["bitwise_equal"]
            and precleared_dq_errors["dv"]["bitwise_equal"]
        )
        if not precleared_dq_ok:
            raise RuntimeError(
                f"{order_name} precleared-dQ entrypoint differs from the "
                f"standalone-safe entrypoint: {precleared_dq_errors}"
            )

        # An output-equivalence comparison cannot prove that the dedicated
        # wrapper omitted its dQ memset.  With exactly zero dO/dstat the native
        # kernel contributes exactly zero gradients, so a nonzero dQ sentinel
        # must survive bitwise while dK/dV sentinels are cleared by the wrapper.
        zero_dout = torch.zeros_like(dout)
        zero_dstat = torch.zeros_like(dstat)
        dq_sentinel = torch.full_like(outputs["dq"], 0.75)
        outputs["dq"].copy_(dq_sentinel)
        outputs["dk"].fill_(1)
        outputs["dv"].fill_(-1)
        precleared_dq_entrypoint(
            q,
            k,
            v,
            zero_dout,
            lstat,
            zero_dstat,
            outputs["dq"],
            outputs["dk"],
            outputs["dv"],
            q_native,
            k_native,
            q_scale,
            k_scale,
            q_global,
            k_global,
            1.0 / 128.0**0.5,
        )
        torch.cuda.synchronize(device)
        precleared_zero_semantics = precleared_zero_dout_semantics(
            outputs,
            dq_sentinel,
        )
        if not precleared_zero_semantics["passed"]:
            raise RuntimeError(
                f"{order_name} precleared-dQ clear-ownership gate failed: "
                f"{precleared_zero_semantics}"
            )

        outputs["dq"].fill_(0.75)
        outputs["dk"].fill_(1)
        outputs["dv"].fill_(-1)
        run(zero_dout, zero_dstat)
        torch.cuda.synchronize(device)
        zero_counts = {
            name: int(torch.count_nonzero(value)) for name, value in outputs.items()
        }
        if any(zero_counts.values()):
            raise RuntimeError(f"{order_name} zero-dO gate failed: {zero_counts}")

        return {
            "sample_order": list(order),
            "independent_b1_receipt_comparison": errors,
            "independent_b1_receipt_comparison_by_batch": errors_by_batch,
            "batched_repeatability": batched_repeatability,
            "comparison_gate": {
                "dq_bf16_store_add_order_tolerance_passed": dq_ok,
                "dk_dv_bitwise_passed": dkdv_ok,
                "precleared_dq_entrypoint_passed": precleared_dq_ok,
                "precleared_dq_clear_ownership_passed": (
                    precleared_zero_semantics["passed"]
                ),
            },
            "precleared_dq_entrypoint_comparison": precleared_dq_errors,
            "precleared_zero_dout_clear_semantics": (
                precleared_zero_semantics
            ),
            "exact_zero_dout_nonzero_counts": zero_counts,
            "latency_ms": latency_ms,
        }

    order_results = {
        name: validate_order(name, order, benchmark=name == "listed")
        for name, order in capture_orders.items()
    }
    listed_result = order_results["listed"]
    latency_ms = listed_result["latency_ms"]

    receipt = {
        "schema": "tkfa4.v509_batched_isolation_gate.v6",
        "scope": (
            "bounded_backward_exact_batch_raster_and_address_evidence; "
            "not_formal_bitwise_dq_equivalence_proof; "
            "not_fused_publisher_fidelity"
        ),
        "module": {
            "path": str(module_path),
            "bytes": module_path.stat().st_size,
            "sha256": sha256_file(module_path),
        },
        "reference_module": {
            "path": str(reference_module_path),
            "bytes": reference_module_path.stat().st_size,
            "sha256": sha256_file(reference_module_path),
            "metadata": dict(
                reference_module.native_tk_d128_backward_metadata()
            ),
        },
        "metadata": dict(module.native_tk_d128_backward_metadata()),
        "batch": args.batch,
        "captures": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "effective_input_sha256": effective_digest,
            }
            for path, effective_digest in zip(
                resolved_capture_paths, effective_digests, strict=True
            )
        ],
        "derived_abi": {
            "dout": "represented_x4_numeric_cast_to_E5M2_no_rescale",
            "dstat": "-4*sum(forward_attention_output*raw_E5M2_dout)",
            "lstat": "8-forward_lse*log2(e)",
        },
        "dq_bf16_store_add_tolerance": {
            "relative_l2_limit": DQ_STORE_ADD_RELATIVE_L2_LIMIT,
            "max_abs_over_expected_peak_limit": (
                DQ_STORE_ADD_PEAK_RELATIVE_LIMIT
            ),
            "changed_fraction_limit": (
                DQ_STORE_ADD_CHANGED_FRACTION_LIMIT
            ),
            "reason": "concurrent_BF16_TMA_store_add_rounding_order",
        },
        "independent_b1_repeatability": b1_repeatability,
        "independent_b1_nonzero_counts": independent_b1_nonzero_counts,
        "sample_order": listed_result["sample_order"],
        "capture_orders": {
            name: list(order) for name, order in capture_orders.items()
        },
        "independent_b1_receipt_comparison": listed_result[
            "independent_b1_receipt_comparison"
        ],
        "comparison_gate": listed_result["comparison_gate"],
        "exact_zero_dout_nonzero_counts": listed_result[
            "exact_zero_dout_nonzero_counts"
        ],
        "order_comparisons": {
            name: {
                key: value
                for key, value in result.items()
                if key != "latency_ms"
            }
            for name, result in order_results.items()
        },
        "latency_ms": {
            "samples": latency_ms,
            "median": statistics.median(latency_ms),
            "minimum": min(latency_ms),
            "maximum": max(latency_ms),
        },
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
