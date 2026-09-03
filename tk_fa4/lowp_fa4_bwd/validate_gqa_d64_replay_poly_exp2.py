#!/usr/bin/env python3
"""Private B/O/B/O gate for the all-lane degree-2 replay EX2 candidate.

The shared replay validator deliberately exposes only retained contracts.  This
wrapper authenticates the experimental marker, delegates all tensor creation,
forward scale handoff, repeated correctness captures, and rotated timing to
that validator's all-lane path, then records the narrower approximate-math
contract in a distinct result schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from tk_fa4.lowp_fa4_bwd import validate_gqa_d64_replay_preexp as base


CONTRACT = "poly_exp2_degree2"
MARKER = "TK_FORWARD_MX_PROBABILITY_REPLAY_POLY_EXP2_DEGREE2 = True"


def _argument_value(arguments: Sequence[str], name: str) -> str:
    try:
        index = arguments.index(name)
    except ValueError as error:
        raise RuntimeError(f"missing required argument {name}") from error
    if index + 1 >= len(arguments):
        raise RuntimeError(f"missing value for {name}")
    return arguments[index + 1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_all_lane_route(
    path: Path,
    sha256: str,
    size_bytes: int,
    *,
    optimized: bool,
    optimized_contract: str,
):
    """Load two all-lane controls while authenticating their one delta."""
    if optimized_contract != "exact_math_all_lane":
        raise RuntimeError("unexpected delegated gate contract")
    control = base._load_control(
        fp8_p_storage="tmem",
        direct_tma_dkdv=True,
        detached_fp8_p_tmem=False,
        precomposed_control_source=path,
        precomposed_control_sha256=sha256,
        precomposed_control_bytes=size_bytes,
    )
    if not bool(getattr(control, base.ALL_LANE_EXACT_MARKER, False)):
        raise RuntimeError("both routes must retain all-lane replay")
    if bool(
        getattr(
            control,
            "TK_FORWARD_MX_PROBABILITY_REPLAY_PREEXP_NORMALIZER",
            False,
        )
    ):
        raise RuntimeError("pre-exp replay is outside this isolated contract")
    if getattr(control, base.SHARED_METADATA_MARKER, None) is not None:
        raise RuntimeError("shared metadata is outside this isolated contract")
    if getattr(control, base.LOG_CLASSIFIER_MARKER, None) is not None:
        raise RuntimeError("log classifier is outside this isolated contract")
    has_poly = bool(
        getattr(
            control,
            "TK_FORWARD_MX_PROBABILITY_REPLAY_POLY_EXP2_DEGREE2",
            False,
        )
    )
    if has_poly != optimized:
        route = "optimized" if optimized else "baseline"
        raise RuntimeError(
            f"{route} control has wrong degree-2 replay EX2 capability"
        )
    return control


def _run(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    contract = _argument_value(arguments, "--optimized-contract")
    if contract != CONTRACT:
        raise RuntimeError(
            f"--optimized-contract must be {CONTRACT!r}, got {contract!r}"
        )
    output = Path(_argument_value(arguments, "--output"))
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")

    optimized_control = Path(
        _argument_value(arguments, "--optimized-control")
    ).resolve(strict=True)
    control_text = optimized_control.read_text(encoding="utf-8")
    if control_text.count(MARKER) != 1:
        raise RuntimeError(
            "optimized control must declare exactly one degree-2 replay "
            "EX2 contract marker"
        )
    if control_text.count("tk_exp2_alu_degree2_f32x2(") < 2:
        raise RuntimeError(
            "optimized control does not contain the degree-2 helper call"
        )

    translated = list(arguments)
    contract_index = translated.index("--optimized-contract") + 1
    translated[contract_index] = "exact_math_all_lane"
    output_index = translated.index("--output") + 1
    temporary = output.with_name(
        f".{output.name}.poly-exp2-tmp-{os.getpid()}"
    )
    if temporary.exists():
        raise RuntimeError(f"refusing to overwrite temporary output: {temporary}")
    translated[output_index] = str(temporary)

    original_loader = base._load_route_control
    base._load_route_control = _load_all_lane_route
    try:
        try:
            base._run(translated)
            payload = json.loads(temporary.read_text(encoding="utf-8"))
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        base._load_route_control = original_loader

    script_path = Path(__file__).resolve()
    payload["schema"] = "gqa_d64_replay_poly_exp2_ab_v1"
    payload["policy"]["underlying_gate_contract"] = "exact_math_all_lane"
    payload["policy"]["optimized_contract"] = CONTRACT
    payload["policy"]["isolated_baseline"] = (
        "all_lane_exact_native_score_exp2"
    )
    payload["routes"]["optimized"]["poly_exp2_degree2"] = True
    payload["checks"]["optimized_poly_exp2_degree2_contract"] = True
    payload["passed"] = bool(all(payload["checks"].values()))
    payload["provenance"]["wrapper_script"] = {
        "path": str(script_path),
        "sha256": _sha256(script_path),
        "bytes": script_path.stat().st_size,
    }
    payload["provenance"]["command"] = [sys.executable, *sys.argv]

    serialized = json.dumps(payload, indent=2, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")
    print(serialized)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(_run())
