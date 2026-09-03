"""Fail-fast checks for matched low-precision backward comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_MISSING = object()


def _same_object(reference: Any, candidate: Any, name: str) -> bool:
    reference_value = getattr(reference, name, _MISSING)
    candidate_value = getattr(candidate, name, _MISSING)
    return (
        reference_value is not _MISSING
        and candidate_value is not _MISSING
        and reference_value is candidate_value
    )


def _same_data_ptr(reference: Any, candidate: Any, name: str) -> bool:
    reference_value = getattr(reference, name, _MISSING)
    candidate_value = getattr(candidate, name, _MISSING)
    if reference_value is _MISSING or candidate_value is _MISSING:
        return False
    reference_data_ptr = getattr(reference_value, "data_ptr", None)
    candidate_data_ptr = getattr(candidate_value, "data_ptr", None)
    if not callable(reference_data_ptr) or not callable(candidate_data_ptr):
        return False
    return int(reference_data_ptr()) == int(candidate_data_ptr())


def _same_sequence_item(
    reference: Any,
    candidate: Any,
    name: str,
    index: int,
) -> bool:
    reference_value = getattr(reference, name, _MISSING)
    candidate_value = getattr(candidate, name, _MISSING)
    try:
        return reference_value[index] is candidate_value[index]
    except (IndexError, KeyError, TypeError):
        return False


def _same_sequence_item_data_ptr(
    reference: Any,
    candidate: Any,
    name: str,
    index: int,
) -> bool:
    reference_value = getattr(reference, name, _MISSING)
    candidate_value = getattr(candidate, name, _MISSING)
    try:
        reference_item = reference_value[index]
        candidate_item = candidate_value[index]
    except (IndexError, KeyError, TypeError):
        return False
    reference_data_ptr = getattr(reference_item, "data_ptr", None)
    candidate_data_ptr = getattr(candidate_item, "data_ptr", None)
    if not callable(reference_data_ptr) or not callable(candidate_data_ptr):
        return False
    return int(reference_data_ptr()) == int(candidate_data_ptr())


def shared_backward_physical_identity(
    reference_runtime: Any,
    candidate_runtime: Any,
) -> dict[str, bool]:
    """Describe whether two runtimes retain one physical backward.

    Runner identity is the primary invariant.  The explicit object and
    allocation checks make the invariant auditable in persisted benchmark
    provenance and fail closed when a partial/mock runner omits a retained
    output or workspace view.
    """
    reference_backward = getattr(reference_runtime, "backward", _MISSING)
    candidate_backward = getattr(candidate_runtime, "backward", _MISSING)
    checks = {
        "shared_backward_execution_runtime_object": _same_object(
            reference_runtime,
            candidate_runtime,
            "backward_execution_runtime",
        ),
        "shared_runner_object": (
            reference_backward is not _MISSING
            and candidate_backward is not _MISSING
            and reference_backward is candidate_backward
        ),
        "shared_control_module_object": _same_object(
            reference_runtime, candidate_runtime, "control"
        ),
        "shared_rope_tuple_object": _same_object(
            reference_runtime, candidate_runtime, "rope"
        ),
        "shared_rope_cos_object": _same_sequence_item(
            reference_runtime, candidate_runtime, "rope", 0
        ),
        "shared_rope_cos_data_ptr": _same_sequence_item_data_ptr(
            reference_runtime, candidate_runtime, "rope", 0
        ),
        "shared_rope_sin_object": _same_sequence_item(
            reference_runtime, candidate_runtime, "rope", 1
        ),
        "shared_rope_sin_data_ptr": _same_sequence_item_data_ptr(
            reference_runtime, candidate_runtime, "rope", 1
        ),
        "shared_packed_rope_object": _same_object(
            reference_runtime, candidate_runtime, "paired_rope"
        ),
        "shared_packed_rope_data_ptr": _same_data_ptr(
            reference_runtime, candidate_runtime, "paired_rope"
        ),
        "shared_gradient_scale_object": _same_object(
            reference_runtime,
            candidate_runtime,
            "gradient_global_scale",
        ),
        "shared_gradient_scale_data_ptr": _same_data_ptr(
            reference_runtime,
            candidate_runtime,
            "gradient_global_scale",
        ),
    }
    for name, label in (
        ("workspace_torch", "workspace"),
        ("dq", "dq"),
        ("dk", "dk"),
        ("dv", "dv"),
        ("dk_partials", "dk_partials"),
        ("dv_partials", "dv_partials"),
    ):
        checks[f"shared_{label}_object"] = _same_object(
            reference_backward, candidate_backward, name
        )
        checks[f"shared_{label}_data_ptr"] = _same_data_ptr(
            reference_backward, candidate_backward, name
        )
    checks["shared_kernel_object"] = _same_object(
        reference_backward, candidate_backward, "kernel"
    )
    checks["shared_compiled_callable_object"] = _same_object(
        reference_backward, candidate_backward, "compiled"
    )
    return checks


def require_shared_backward_physical_identity(
    reference_runtime: Any,
    candidate_runtime: Any,
) -> dict[str, bool]:
    """Reject runtimes that do not retain exactly one physical backward."""
    checks = shared_backward_physical_identity(
        reference_runtime,
        candidate_runtime,
    )
    mismatches = [name for name, matches in checks.items() if not matches]
    if mismatches:
        raise RuntimeError(
            "matched low-precision backward physical identity mismatch: "
            + ", ".join(mismatches)
        )
    return checks


def _different_fields(
    reference: Any,
    candidate: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Return stable dotted paths whose values differ."""
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        fields: list[str] = []
        for key in sorted(set(reference) | set(candidate), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in reference or key not in candidate:
                fields.append(path)
                continue
            fields.extend(
                _different_fields(
                    reference[key], candidate[key], prefix=path
                )
            )
        return fields
    return [] if reference == candidate else [prefix or "<root>"]


def require_matching_backward_contracts(
    contracts: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject a comparison whose effective lowp backward routes differ.

    A shared source file or extension is insufficient: CuTe specializes the
    backward from runtime attributes, and projection epilogues can publish
    different represented operands.  Callers therefore compare the complete
    effective contracts before allocating models or starting a job.
    """
    if len(contracts) < 2:
        return
    reference_route, reference = next(iter(contracts.items()))
    mismatches: list[str] = []
    for route, contract in list(contracts.items())[1:]:
        fields = _different_fields(reference, contract)
        if fields:
            mismatches.append(f"{route}: {', '.join(fields)}")
    if mismatches:
        details = "; ".join(mismatches)
        raise RuntimeError(
            "matched low-precision backward contract mismatch against "
            f"{reference_route}: {details}"
        )
