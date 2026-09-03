"""CPU-testable telemetry helpers for matched-route training probes."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

import torch


@torch.no_grad()
def tensor_statistics(
    tensor: torch.Tensor | None,
    *,
    encoding: str = "numeric",
) -> dict[str, Any]:
    """Summarize a numerically encoded tensor without retaining it."""
    if tensor is None:
        return {"present": False, "encoding": encoding}
    result: dict[str, Any] = {
        "present": bool(tensor.numel()),
        "encoding": encoding,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "elements": tensor.numel(),
    }
    if not tensor.numel():
        return result
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    finite_count = int(finite.sum())
    all_finite = finite_count == value.numel()
    finite_value = value if all_finite else value[finite]
    result.update(
        {
            "all_finite": all_finite,
            "nonfinite": value.numel() - finite_count,
        }
    )
    if not finite_count:
        return {
            **result,
            "minimum": None,
            "maximum": None,
            "max_abs": None,
            "mean": None,
            "rms": None,
            "zero_fraction": None,
            "e4m3_saturation_fraction": None,
        }
    absolute = finite_value.abs()
    result.update(
        {
            "minimum": float(finite_value.min()),
            "maximum": float(finite_value.max()),
            "max_abs": float(absolute.max()),
            "mean": float(finite_value.mean()),
            "rms": float(finite_value.square().mean().sqrt()),
            "zero_fraction": float((finite_value == 0).sum()) / finite_count,
            "e4m3_saturation_fraction": (
                float((absolute == 448.0).sum()) / finite_count
                if tensor.dtype == torch.float8_e4m3fn
                else None
            ),
        }
    )
    return result


@torch.no_grad()
def raw_e8m0_statistics(
    tensor: torch.Tensor | None,
    *,
    encoding: str,
    exponent_bias: int,
    packed_int32: bool = False,
) -> dict[str, Any]:
    """Summarize raw E8M0 bytes without decoding them as their storage dtype.

    MXFP4 block-scale buffers use byte storage even when PyTorch exposes that
    storage as ``float8_e4m3fn``.  Published forward-probability scales pack
    four such bytes into each int32 word and use a replay-specific exponent
    bias.  Numeric conversion of either storage type would be misleading.
    """
    packing = "four_codes_per_int32" if packed_int32 else "one_code_per_byte"
    if tensor is None:
        return {
            "present": False,
            "encoding": encoding,
            "packing": packing,
            "exponent_bias": exponent_bias,
        }
    result: dict[str, Any] = {
        "present": bool(tensor.numel()),
        "encoding": encoding,
        "packing": packing,
        "exponent_bias": exponent_bias,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "elements": tensor.numel(),
    }
    if not tensor.numel():
        return result
    if packed_int32 and tensor.dtype != torch.int32:
        raise ValueError("packed E8M0 probability scales must use int32 storage")
    if not packed_int32 and tensor.element_size() != 1:
        raise ValueError("unpacked E8M0 scales must use one-byte storage")
    if not tensor.is_contiguous():
        raise ValueError("raw E8M0 scale storage must be contiguous")

    # Do not call contiguous() on typed float8 storage: a numerical copy could
    # reinterpret the raw E8M0 bytes before this byte view is formed.
    codes = tensor.detach().view(torch.uint8)
    code_values = codes.to(torch.float32)
    nonzero_codes = codes[codes != 0].to(torch.int32)
    nonzero_count = nonzero_codes.numel()
    result.update(
        {
            "encoded_elements": codes.numel(),
            "zero_code_count": codes.numel() - nonzero_count,
            "zero_code_fraction": (
                float((codes == 0).sum()) / codes.numel()
            ),
            "raw_code_minimum": int(codes.min()),
            "raw_code_maximum": int(codes.max()),
            "raw_code_mean": float(code_values.mean()),
            "nonzero_code_minimum": (
                int(nonzero_codes.min()) if nonzero_count else None
            ),
            "nonzero_code_maximum": (
                int(nonzero_codes.max()) if nonzero_count else None
            ),
            "decoded_exponent_minimum": (
                int(nonzero_codes.min()) - exponent_bias
                if nonzero_count
                else None
            ),
            "decoded_exponent_maximum": (
                int(nonzero_codes.max()) - exponent_bias
                if nonzero_count
                else None
            ),
        }
    )
    return result


def forward_diagnostic_tensor_statistics(
    name: str,
    tensor: torch.Tensor | None,
) -> dict[str, Any]:
    """Apply the storage contract for one captured low-precision tensor."""
    if name == "v_forward_scales":
        return raw_e8m0_statistics(
            tensor,
            encoding="mxfp4_e8m0_block_scale",
            exponent_bias=127,
        )
    if name == "forward_mx_probability_scales":
        # Forward stores max(e - 16 + 127, 0), so replay recovers the
        # represented-probability exponent as code - 111.
        return raw_e8m0_statistics(
            tensor,
            encoding="mxfp4_e8m0_probability_scale",
            exponent_bias=111,
            packed_int32=True,
        )
    encodings = {
        "qk_policy_scales": "qk_policy_fp32",
        "backward_qk_scales": "backward_qk_policy_fp32",
        "q_forward_scales": "nvfp4_e4m3_block_scale",
        "q_forward_global_scale": "nvfp4_global_scale_fp32",
        "k_forward_scales": "nvfp4_e4m3_block_scale",
        "k_forward_global_scale": "nvfp4_global_scale_fp32",
        "q_backward_fp8": "e4m3_payload",
        "k_backward_fp8": "e4m3_payload",
        "v_backward_fp8": "e4m3_payload",
        "attention_output": "bf16_attention_output",
        "lse": "fp32_logsumexp",
    }
    return tensor_statistics(tensor, encoding=encodings.get(name, "numeric"))


def mark_matched_round_timing_eligibility(
    round_records: Mapping[str, MutableMapping[str, Any]],
) -> bool:
    """Mark every route in a complete round eligible only as a clean group."""
    if not round_records:
        raise ValueError("matched timing requires at least one route record")
    rounds = {int(record["round"]) for record in round_records.values()}
    if len(rounds) != 1:
        raise ValueError(f"matched timing records span multiple rounds: {rounds}")
    eligible = not any(
        bool(record.get("diagnostic")) for record in round_records.values()
    )
    for record in round_records.values():
        record["timing_eligible"] = eligible
    return eligible


def select_timing_records(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Select records explicitly marked as matched-round timing eligible."""
    return [
        record
        for record in records
        if bool(
            record.get(
                "timing_eligible",
                not bool(record.get("diagnostic")),
            )
        )
    ]
