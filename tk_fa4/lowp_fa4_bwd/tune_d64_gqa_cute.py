#!/usr/bin/env python3
"""Tune the SM100 D64/D128 causal GQA backward scaffold.

This driver keeps NVIDIA's CuTe DSL implementation as the control topology
while making its pipeline depths and register split explicit.  It is intended
for the ratio-4 geometries used by Llama-3.2-1B (D64) and Llama-3-8B
(D128), and supports both BF16 and the FP8-input/BF16-gradient hybrid.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import hmac
import importlib.util
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path

try:
    from tk_fa4.lowp_fa4_bwd.backward_policy import (
        resolve_backward_exp2_policy,
    )
except ModuleNotFoundError:  # direct script execution without repo on sys.path
    from backward_policy import resolve_backward_exp2_policy

try:
    from tk_fa4.lowp_fa4_bwd.cutlass_dsl_toolchain import (
        require_d128_mxfp4_v_compile_environment,
        verify_d128_mxfp4_v_toolchain,
    )
except ModuleNotFoundError:  # direct script execution without repo on sys.path
    from cutlass_dsl_toolchain import (
        require_d128_mxfp4_v_compile_environment,
        verify_d128_mxfp4_v_toolchain,
    )


MAX_PRECOMPOSED_CONTROL_BYTES = 8 * 1024 * 1024
MAX_D128_MXFP4_V_DP_PATCH_BYTES = 1024 * 1024


def _read_d128_mxfp4_v_dp_patch(
    source: Path | str,
) -> tuple[bytes, dict[str, int | str]]:
    """Read one stable regular patch image and fingerprint the exact bytes."""
    requested = Path(source)
    try:
        requested_stat = requested.lstat()
    except OSError as error:
        raise RuntimeError(
            f"unable to stat D128 MXFP4 V dP patch: {requested}"
        ) from error
    if not stat.S_ISREG(requested_stat.st_mode):
        raise RuntimeError(
            "D128 MXFP4 V dP patch must be a regular, non-symlink file: "
            f"{requested}"
        )
    if not 0 < requested_stat.st_size <= MAX_D128_MXFP4_V_DP_PATCH_BYTES:
        raise RuntimeError(
            "D128 MXFP4 V dP patch must contain between 1 byte and 1 MiB"
        )
    resolved = requested.resolve(strict=True)
    try:
        open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, open_flags)
        with os.fdopen(descriptor, "rb") as patch_file:
            opened_stat = os.fstat(patch_file.fileno())
            if not stat.S_ISREG(opened_stat.st_mode):
                raise RuntimeError(
                    "D128 MXFP4 V dP patch must remain a regular file"
                )
            payload = patch_file.read(MAX_D128_MXFP4_V_DP_PATCH_BYTES + 1)
            closed_stat = os.fstat(patch_file.fileno())
    except OSError as error:
        raise RuntimeError(
            f"unable to read D128 MXFP4 V dP patch: {resolved}"
        ) from error
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(opened_stat, field) != getattr(closed_stat, field)
        for field in stable_fields
    ):
        raise RuntimeError("D128 MXFP4 V dP patch changed while reading")
    if len(payload) != opened_stat.st_size:
        raise RuntimeError(
            "D128 MXFP4 V dP patch size changed while reading: "
            f"expected {opened_stat.st_size}, found {len(payload)}"
        )
    return payload, {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _require_d128_mxfp4_v_dp_patch_provenance(
    control: object,
    *,
    enabled: bool,
) -> dict[str, int | str] | None:
    """Validate the loader-authenticated candidate patch receipt."""
    marker = bool(getattr(control, "TK_D128_MXFP4_V_DP", False))
    if marker != enabled:
        raise RuntimeError(
            "D128 MXFP4 V dP control marker disagrees with runtime selection"
        )
    provenance = getattr(
        control,
        "TK_D128_MXFP4_V_DP_PATCH_PROVENANCE",
        None,
    )
    if not enabled:
        if provenance is not None:
            raise RuntimeError(
                "retained backward control unexpectedly carries candidate "
                "patch provenance"
            )
        return None
    if not isinstance(provenance, dict) or set(provenance) != {
        "path",
        "sha256",
        "bytes",
    }:
        raise RuntimeError(
            "D128 MXFP4 V dP control is missing exact patch provenance"
        )
    path = provenance.get("path")
    sha256 = provenance.get("sha256")
    byte_count = provenance.get("bytes")
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise RuntimeError("D128 MXFP4 V dP patch path provenance is malformed")
    if not isinstance(sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", sha256
    ) is None:
        raise RuntimeError("D128 MXFP4 V dP patch SHA256 is malformed")
    if (
        type(byte_count) is not int
        or byte_count <= 0
        or byte_count > MAX_D128_MXFP4_V_DP_PATCH_BYTES
    ):
        raise RuntimeError("D128 MXFP4 V dP patch byte count is malformed")
    return {
        "path": path,
        "sha256": sha256,
        "bytes": byte_count,
    }


def _load_precomposed_control(
    source: Path | str | None,
    expected_sha256: str | None,
    expected_bytes: int | None,
    import_directory: Path,
):
    """Load one immutable generated control without composing local patches."""
    if source is None:
        if expected_sha256 is not None or expected_bytes is not None:
            raise ValueError(
                "backward control SHA256/bytes require a control source"
            )
        return None
    if expected_sha256 is None or expected_bytes is None:
        raise ValueError(
            "a precomposed backward control requires source, SHA256, and bytes"
        )
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ValueError("backward control bytes must be a positive integer")
    if expected_bytes > MAX_PRECOMPOSED_CONTROL_BYTES:
        raise ValueError(
            "precomposed backward control exceeds the 8 MiB safety limit"
        )
    if re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None:
        raise ValueError("backward control SHA256 must contain 64 hex digits")

    requested = Path(source)
    try:
        requested_stat = requested.lstat()
    except OSError as error:
        raise RuntimeError(
            f"unable to stat precomposed backward control: {requested}"
        ) from error
    if not stat.S_ISREG(requested_stat.st_mode):
        raise RuntimeError(
            "precomposed backward control must be a regular, non-symlink file: "
            f"{requested}"
        )
    resolved = requested.resolve(strict=True)
    if resolved.suffix != ".py":
        raise ValueError("precomposed backward control must be a Python source")
    if requested_stat.st_size != expected_bytes:
        raise RuntimeError(
            "precomposed backward control size mismatch: "
            f"expected {expected_bytes}, found {requested_stat.st_size}"
        )
    try:
        open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, open_flags)
        with os.fdopen(descriptor, "rb") as control_file:
            opened_stat = os.fstat(control_file.fileno())
            if not stat.S_ISREG(opened_stat.st_mode):
                raise RuntimeError(
                    "precomposed backward control must remain a regular file"
                )
            if opened_stat.st_size != expected_bytes:
                raise RuntimeError(
                    "precomposed backward control size mismatch: "
                    f"expected {expected_bytes}, found {opened_stat.st_size}"
                )
            payload = control_file.read(expected_bytes + 1)
    except OSError as error:
        raise RuntimeError(
            f"unable to read precomposed backward control: {resolved}"
        ) from error
    if len(payload) != expected_bytes:
        raise RuntimeError(
            "precomposed backward control size mismatch: "
            f"expected {expected_bytes}, found {len(payload)}"
        )
    expected_digest = expected_sha256.lower()
    actual_digest = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual_digest, expected_digest):
        raise RuntimeError(
            "precomposed backward control SHA256 mismatch: "
            f"expected {expected_digest}, found {actual_digest}"
        )

    # Import the bytes that were authenticated above rather than reopening the
    # caller-owned path.  Generated CuTe controls derive their ``utils`` import
    # from __file__, so place the verified copy in the canonical Blackwell
    # source directory rather than beside an arbitrary archived artifact.
    if not import_directory.is_dir():
        raise RuntimeError(
            f"CuTe control import directory is missing: {import_directory}"
        )
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="fmha_bwd_d64_gqa_precomposed_",
        suffix=".py",
        dir=import_directory,
        delete=False,
    ) as temporary:
        temporary.write(payload)
        verified_source = Path(temporary.name)
    module_name = f"tk_d64_gqa_precomposed_{actual_digest[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, verified_source)
    if spec is None or spec.loader is None:
        verified_source.unlink(missing_ok=True)
        raise RuntimeError(
            f"unable to load precomposed CuTe control from {resolved}"
        )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        verified_source.unlink(missing_ok=True)
        raise
    atexit.register(verified_source.unlink, missing_ok=True)
    module.TK_DIRECT_TMA_DKDV = True
    module.TK_FP8_P_STORAGE = "tmem"
    module.TK_DETACHED_FP8_P_TMEM = False
    module.TK_D128_MXFP4_V_DP = False
    module.TK_D128_MXFP4_V_TOOLCHAIN_PROVENANCE = None
    module.TK_PRECOMPOSED_CONTROL_PROVENANCE = {
        "mode": "precomposed",
        "source": {
            "path": str(resolved),
            "bytes": len(payload),
            "sha256": actual_digest,
        },
        "required_constants": {
            "TK_DIRECT_TMA_DKDV": True,
            "TK_FP8_P_STORAGE": "tmem",
            "TK_DETACHED_FP8_P_TMEM": False,
        },
        "required_runtime_policy": {
            "owner_fused_dq_scale": False,
        },
    }
    return module


def _load_control(
    *,
    fp8_p_storage: str = "shared",
    owner_fused_dq_scale: bool = False,
    direct_tma_dkdv: bool = False,
    detached_fp8_p_tmem: bool = False,
    use_d128_mxfp4_v_dp: bool = False,
    precomposed_control_source: Path | str | None = None,
    precomposed_control_sha256: str | None = None,
    precomposed_control_bytes: int | None = None,
):
    if detached_fp8_p_tmem and fp8_p_storage != "tmem":
        raise ValueError(
            "detached FP8 P TMEM requires fp8_p_storage='tmem'"
        )
    if use_d128_mxfp4_v_dp and (
        fp8_p_storage != "shared"
        or owner_fused_dq_scale
        or direct_tma_dkdv
        or detached_fp8_p_tmem
        or precomposed_control_source is not None
    ):
        raise ValueError(
            "D128 MXFP4 V dP requires the generated shared-P control without "
            "owner-fused dQ scale, direct dK/dV TMA, detached P, or a "
            "precomposed replay control"
        )
    d128_mxfp4_v_toolchain_provenance = None
    if use_d128_mxfp4_v_dp:
        # This check is intentionally absent from every retained BF16/E4M3-V
        # route.  Version 4.5.2 alone is ambiguous: both the regressing LLVM20
        # payload and the retained LLVM21/cu13 payload publish that version.
        require_d128_mxfp4_v_compile_environment()
        d128_mxfp4_v_toolchain_provenance = (
            verify_d128_mxfp4_v_toolchain()
        )
    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "qutlass"
        / "third_party"
        / "cutlass"
        / "examples"
        / "python"
        / "CuTeDSL"
        / "blackwell"
        / "fmha_bwd.py"
    )
    if precomposed_control_source is not None:
        if (
            fp8_p_storage != "tmem"
            or not direct_tma_dkdv
            or detached_fp8_p_tmem
            or owner_fused_dq_scale
        ):
            raise ValueError(
                "precomposed replay control requires direct TMA, TMEM P, "
                "non-detached P storage, and non-owner-fused dQ scale"
            )
        return _load_precomposed_control(
            precomposed_control_source,
            precomposed_control_sha256,
            precomposed_control_bytes,
            source.parent,
        )
    if (
        precomposed_control_sha256 is not None
        or precomposed_control_bytes is not None
    ):
        raise ValueError(
            "backward control SHA256/bytes require a control source"
        )
    patch = Path(
        os.environ.get(
            "TK_GQA_CONTROL_PATCH",
            Path(__file__).with_name("d64_gqa_cute.patch"),
        )
    )
    patched = subprocess.run(
        ["patch", "--silent", "--output=-", str(source), str(patch)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tile_ready_patch = Path(
        os.environ.get(
            "TK_GQA_TILE_READY_PATCH",
            Path(__file__).with_name("d64_gqa_tile_ready.patch"),
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="fmha_bwd_d64_gqa_base_",
        suffix=".py",
        dir=source.parent,
        delete=False,
    ) as intermediate:
        intermediate.write(patched)
        intermediate_source = Path(intermediate.name)
    try:
        patched = subprocess.run(
            [
                "patch",
                "--silent",
                "--output=-",
                str(intermediate_source),
                str(tile_ready_patch),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        intermediate_source.unlink(missing_ok=True)
    owner_patch = Path(
        os.environ.get(
            "TK_GQA_OWNER_PATCH",
            Path(__file__).with_name("d64_gqa_owner_quantize.patch"),
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="fmha_bwd_d64_gqa_tile_ready_",
        suffix=".py",
        dir=source.parent,
        delete=False,
    ) as intermediate:
        intermediate.write(patched)
        intermediate_source = Path(intermediate.name)
    try:
        patched = subprocess.run(
            [
                "patch",
                "--silent",
                "--output=-",
                str(intermediate_source),
                str(owner_patch),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        intermediate_source.unlink(missing_ok=True)
    fp8_p_lift_patch = Path(
        os.environ.get(
            "TK_GQA_FP8_P_LIFT_PATCH",
            Path(__file__).with_name("d64_gqa_fp8_p_lift.patch"),
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="fmha_bwd_d64_gqa_owner_",
        suffix=".py",
        dir=source.parent,
        delete=False,
    ) as intermediate:
        intermediate.write(patched)
        intermediate_source = Path(intermediate.name)
    try:
        patched = subprocess.run(
            [
                "patch",
                "--silent",
                "--output=-",
                str(intermediate_source),
                str(fp8_p_lift_patch),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        intermediate_source.unlink(missing_ok=True)
    owner_full_operand_patch = Path(
        os.environ.get(
            "TK_GQA_OWNER_FULL_OPERAND_PATCH",
            Path(__file__).with_name("d64_gqa_owner_full_operand.patch"),
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="fmha_bwd_d64_gqa_fp8_p_lift_",
        suffix=".py",
        dir=source.parent,
        delete=False,
    ) as intermediate:
        intermediate.write(patched)
        intermediate_source = Path(intermediate.name)
    try:
        patched = subprocess.run(
            [
                "patch",
                "--silent",
                "--output=-",
                str(intermediate_source),
                str(owner_full_operand_patch),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        intermediate_source.unlink(missing_ok=True)
    owner_kv_patch = Path(
        os.environ.get(
            "TK_GQA_OWNER_KV_PATCH",
            Path(__file__).with_name("d64_gqa_owner_kv_quantize.patch"),
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="fmha_bwd_d64_gqa_owner_full_",
        suffix=".py",
        dir=source.parent,
        delete=False,
    ) as intermediate:
        intermediate.write(patched)
        intermediate_source = Path(intermediate.name)
    try:
        patched = subprocess.run(
            [
                "patch",
                "--silent",
                "--output=-",
                str(intermediate_source),
                str(owner_kv_patch),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        intermediate_source.unlink(missing_ok=True)
    owner_kv_no_materialize_patch = Path(
        os.environ.get(
            "TK_GQA_OWNER_KV_NO_MATERIALIZE_PATCH",
            Path(__file__).with_name(
                "d64_gqa_owner_kv_no_materialize.patch"
            ),
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="fmha_bwd_d64_gqa_owner_kv_",
        suffix=".py",
        dir=source.parent,
        delete=False,
    ) as intermediate:
        intermediate.write(patched)
        intermediate_source = Path(intermediate.name)
    try:
        patched = subprocess.run(
            [
                "patch",
                "--silent",
                "--output=-",
                str(intermediate_source),
                str(owner_kv_no_materialize_patch),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        intermediate_source.unlink(missing_ok=True)
    reverse_query_patch = Path(
        os.environ.get(
            "TK_GQA_REVERSE_QUERY_PATCH",
            Path(__file__).with_name("d64_gqa_reverse_query.patch"),
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="fmha_bwd_d64_gqa_owner_kv_nomat_",
        suffix=".py",
        dir=source.parent,
        delete=False,
    ) as intermediate:
        intermediate.write(patched)
        intermediate_source = Path(intermediate.name)
    try:
        patched = subprocess.run(
            [
                "patch",
                "--silent",
                "--output=-",
                str(intermediate_source),
                str(reverse_query_patch),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        intermediate_source.unlink(missing_ok=True)
    head_fast_raster_patch = Path(
        os.environ.get(
            "TK_GQA_HEAD_FAST_RASTER_PATCH",
            Path(__file__).with_name("d64_gqa_head_fast_raster.patch"),
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="fmha_bwd_d64_gqa_head_fast_raster_",
        suffix=".py",
        dir=source.parent,
        delete=False,
    ) as intermediate:
        intermediate.write(patched)
        intermediate_source = Path(intermediate.name)
    try:
        patched = subprocess.run(
            [
                "patch",
                "--silent",
                "--output=-",
                str(intermediate_source),
                str(head_fast_raster_patch),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        intermediate_source.unlink(missing_ok=True)
    if fp8_p_storage == "shared":
        fp8_p_layout_patch = Path(
            os.environ.get(
                "TK_GQA_FP8_P_LAYOUT_PATCH",
                Path(__file__).with_name("d64_gqa_fp8_p_layout.patch"),
            )
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="fmha_bwd_d64_gqa_reverse_query_",
            suffix=".py",
            dir=source.parent,
            delete=False,
        ) as intermediate:
            intermediate.write(patched)
            intermediate_source = Path(intermediate.name)
        try:
            patched = subprocess.run(
                [
                    "patch",
                    "--silent",
                    "--output=-",
                    str(intermediate_source),
                    str(fp8_p_layout_patch),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        finally:
            intermediate_source.unlink(missing_ok=True)
    elif fp8_p_storage == "tmem":
        fp8_p_tmem_wide_load_patch = Path(
            os.environ.get(
                "TK_GQA_FP8_P_TMEM_WIDE_LOAD_PATCH",
                Path(__file__).with_name("d64_gqa_fp8_p_tmem_wide_load.patch"),
            )
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="fmha_bwd_d64_gqa_reverse_query_",
            suffix=".py",
            dir=source.parent,
            delete=False,
        ) as intermediate:
            intermediate.write(patched)
            intermediate_source = Path(intermediate.name)
        try:
            patched = subprocess.run(
                [
                    "patch",
                    "--silent",
                    "--output=-",
                    str(intermediate_source),
                    str(fp8_p_tmem_wide_load_patch),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        finally:
            intermediate_source.unlink(missing_ok=True)
    elif fp8_p_storage == "tmem-x4":
        fp8_p_tmem_coord_patch = Path(
            os.environ.get(
                "TK_GQA_FP8_P_TMEM_COORD_PATCH",
                Path(__file__).with_name("d64_gqa_fp8_p_tmem_coord.patch"),
            )
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="fmha_bwd_d64_gqa_reverse_query_",
            suffix=".py",
            dir=source.parent,
            delete=False,
        ) as intermediate:
            intermediate.write(patched)
            intermediate_source = Path(intermediate.name)
        try:
            patched = subprocess.run(
                [
                    "patch",
                    "--silent",
                    "--output=-",
                    str(intermediate_source),
                    str(fp8_p_tmem_coord_patch),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        finally:
            intermediate_source.unlink(missing_ok=True)
    fused_probability_lift_patch = Path(
        os.environ.get(
            "TK_GQA_FUSED_PROBABILITY_LIFT_PATCH",
            Path(__file__).with_name(
                "d64_gqa_fused_probability_lift.patch"
            ),
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="fmha_bwd_d64_gqa_probability_lift_",
        suffix=".py",
        dir=source.parent,
        delete=False,
    ) as intermediate:
        intermediate.write(patched)
        intermediate_source = Path(intermediate.name)
    try:
        patched = subprocess.run(
            [
                "patch",
                "--silent",
                "--output=-",
                str(intermediate_source),
                str(fused_probability_lift_patch),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        intermediate_source.unlink(missing_ok=True)
    owner_d64_patch = Path(
        os.environ.get(
            "TK_GQA_OWNER_D64_PATCH",
            Path(__file__).with_name("d64_gqa_owner_d64.patch"),
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="fmha_bwd_d64_gqa_owner_d64_",
        suffix=".py",
        dir=source.parent,
        delete=False,
    ) as intermediate:
        intermediate.write(patched)
        intermediate_source = Path(intermediate.name)
    try:
        patched = subprocess.run(
            [
                "patch",
                "--silent",
                "--output=-",
                str(intermediate_source),
                str(owner_d64_patch),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        intermediate_source.unlink(missing_ok=True)
    if owner_fused_dq_scale:
        owner_fused_dq_scale_patch = Path(
            os.environ.get(
                "TK_GQA_OWNER_FUSED_DQ_SCALE_PATCH",
                Path(__file__).with_name(
                    "d64_gqa_owner_fused_dq_scale.patch"
                ),
            )
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="fmha_bwd_d64_gqa_owner_fused_scale_",
            suffix=".py",
            dir=source.parent,
            delete=False,
        ) as intermediate:
            intermediate.write(patched)
            intermediate_source = Path(intermediate.name)
        try:
            patched = subprocess.run(
                [
                    "patch",
                    "--silent",
                    "--output=-",
                    str(intermediate_source),
                    str(owner_fused_dq_scale_patch),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        finally:
            intermediate_source.unlink(missing_ok=True)
    if direct_tma_dkdv:
        if owner_fused_dq_scale:
            raise ValueError(
                "direct dK/dV TMA reduction is incompatible with the owner "
                "dQ-scale experiment"
            )
        direct_tma_dkdv_patch = Path(
            os.environ.get(
                "TK_GQA_DIRECT_TMA_DKDV_PATCH",
                Path(__file__).with_name("d64_gqa_direct_tma_dkdv.patch"),
            )
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="fmha_bwd_d64_gqa_direct_tma_dkdv_",
            suffix=".py",
            dir=source.parent,
            delete=False,
        ) as intermediate:
            intermediate.write(patched)
            intermediate_source = Path(intermediate.name)
        try:
            patched = subprocess.run(
                [
                    "patch",
                    "--silent",
                    "--output=-",
                    str(intermediate_source),
                    str(direct_tma_dkdv_patch),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        finally:
            intermediate_source.unlink(missing_ok=True)
    if detached_fp8_p_tmem:
        detached_fp8_p_tmem_patch = Path(
            os.environ.get(
                "TK_GQA_DETACHED_FP8_P_TMEM_PATCH",
                Path(__file__).with_name(
                    "d64_gqa_detached_fp8_p_tmem.patch"
                ),
            )
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="fmha_bwd_d64_gqa_detached_fp8_p_tmem_",
            suffix=".py",
            dir=source.parent,
            delete=False,
        ) as intermediate:
            intermediate.write(patched)
            intermediate_source = Path(intermediate.name)
        try:
            patched = subprocess.run(
                [
                    "patch",
                    "--silent",
                    "--output=-",
                    str(intermediate_source),
                    str(detached_fp8_p_tmem_patch),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        finally:
            intermediate_source.unlink(missing_ok=True)
    forward_mx_probability_replay_patch = Path(
        os.environ.get(
            "TK_GQA_FORWARD_MX_PROBABILITY_REPLAY_PATCH",
            Path(__file__).with_name(
                "d64_gqa_forward_mx_probability_replay.patch"
            ),
        )
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="fmha_bwd_d64_gqa_forward_mx_replay_",
        suffix=".py",
        dir=source.parent,
        delete=False,
    ) as intermediate:
        intermediate.write(patched)
        intermediate_source = Path(intermediate.name)
    try:
        patched = subprocess.run(
            [
                "patch",
                "--silent",
                "--output=-",
                str(intermediate_source),
                str(forward_mx_probability_replay_patch),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    finally:
        intermediate_source.unlink(missing_ok=True)
    d128_mxfp4_v_dp_patch_provenance = None
    if use_d128_mxfp4_v_dp:
        requested_d128_mxfp4_v_dp_patch = Path(
            os.environ.get(
                "TK_GQA_D128_MXFP4_V_DP_PATCH",
                Path(__file__).with_name("d128_gqa_mxfp4_v_dp.patch"),
            )
        )
        (
            d128_mxfp4_v_dp_patch_payload,
            d128_mxfp4_v_dp_patch_provenance,
        ) = _read_d128_mxfp4_v_dp_patch(
            requested_d128_mxfp4_v_dp_patch
        )
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="d128_gqa_mxfp4_v_dp_verified_",
            suffix=".patch",
            dir=source.parent,
            delete=False,
        ) as verified_patch:
            verified_patch.write(d128_mxfp4_v_dp_patch_payload)
            verified_patch_path = Path(verified_patch.name)
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="fmha_bwd_d128_gqa_mxfp4_v_dp_",
            suffix=".py",
            dir=source.parent,
            delete=False,
        ) as intermediate:
            intermediate.write(patched)
            intermediate_source = Path(intermediate.name)
        try:
            patched = subprocess.run(
                [
                    "patch",
                    "--silent",
                    "--output=-",
                    str(intermediate_source),
                    str(verified_patch_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        finally:
            intermediate_source.unlink(missing_ok=True)
            verified_patch_path.unlink(missing_ok=True)
    control_class_marker = "class BlackwellFusedMultiHeadAttentionBackward"
    control_class_count = patched.count(control_class_marker)
    if control_class_count != 1:
        raise RuntimeError(
            "the composed backward patch chain must contain exactly one "
            "BlackwellFusedMultiHeadAttentionBackward definition; found "
            f"{control_class_count}"
        )
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="fmha_bwd_d64_gqa_",
        suffix=".py",
        dir=source.parent,
        delete=False,
    ) as temporary:
        temporary.write(patched)
        patched_source = Path(temporary.name)
    atexit.register(os.unlink, patched_source)

    spec = importlib.util.spec_from_file_location(
        "tk_d64_gqa_cute_control", patched_source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load patched CuTe control from {patched_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TK_DIRECT_TMA_DKDV = direct_tma_dkdv
    module.TK_FP8_P_STORAGE = fp8_p_storage
    module.TK_DETACHED_FP8_P_TMEM = detached_fp8_p_tmem
    module.TK_D128_MXFP4_V_DP = bool(use_d128_mxfp4_v_dp)
    module.TK_D128_MXFP4_V_DP_PATCH_PROVENANCE = (
        d128_mxfp4_v_dp_patch_provenance
    )
    module.TK_D128_MXFP4_V_TOOLCHAIN_PROVENANCE = (
        d128_mxfp4_v_toolchain_provenance
    )
    _require_d128_mxfp4_v_dp_patch_provenance(
        module,
        enabled=bool(use_d128_mxfp4_v_dp),
    )
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--q-stages", type=int)
    parser.add_argument("--do-stages", type=int)
    parser.add_argument("--dkdv-stages", type=int)
    parser.add_argument("--reduce-registers", type=int)
    parser.add_argument("--compute-registers", type=int)
    parser.add_argument("--mma-registers", type=int, default=96)
    parser.add_argument("--load-registers", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--reuse-quantized-p", action="store_true")
    parser.add_argument(
        "--fuse-probability-lift",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--prelift-probability-lse",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--fp8-p-storage",
        choices=("shared", "tmem", "tmem-x4", "tmem-legacy"),
        default=None,
        help="handoff used for the quantized P operand consumed by dV",
    )
    parser.add_argument(
        "--fp8-ds-lift",
        type=int,
        choices=(16, 32, 64, 128, 256),
        default=None,
        help=(
            "power-of-two lift applied when publishing FP8 dS; lower values "
            "reduce short-sequence saturation"
        ),
    )
    parser.add_argument(
        "--exp2-alu-period", type=int, choices=tuple(range(17)), default=None
    )
    parser.add_argument("--exp2-alu-degree", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--compact-dq-acc",
        action="store_true",
        help="TMA-reduce dQ through a BF16 accumulator instead of FP32",
    )
    parser.add_argument(
        "--direct-compact-dq",
        action="store_true",
        help=(
            "apply the softmax scale before BF16 conversion and reduce directly "
            "into the caller's zero-initialized dQ tensor"
        ),
    )
    parser.add_argument(
        "--split-gqa-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gqa-reduce-vector", type=int, default=4)
    parser.add_argument("--gqa-reduce-threads", type=int)
    parser.add_argument("--fused-block-seq", type=int)
    parser.add_argument(
        "--fuse-gqa-reduce",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()

    fp8_p_storage = args.fp8_p_storage
    if fp8_p_storage is None:
        fp8_p_storage = (
            "tmem"
            if args.dtype == "fp8" and args.head_dim == 64
            else "shared"
        )
    fuse_probability_lift = args.fuse_probability_lift
    if fuse_probability_lift is None:
        fuse_probability_lift = (
            args.dtype == "fp8" and args.head_dim == 64
        )
    prelift_probability_lse = args.prelift_probability_lse
    if prelift_probability_lse is None:
        prelift_probability_lse = fuse_probability_lift
    exp2_alu_degree = args.exp2_alu_degree
    exp2_alu_period = args.exp2_alu_period
    if args.dtype == "fp8" and args.head_dim == 64:
        exp2_policy = resolve_backward_exp2_policy(
            sequence=args.sequence,
            head_dim=args.head_dim,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            lowp=True,
            exp2_degree=exp2_alu_degree,
            exp2_period=exp2_alu_period,
        )
        exp2_alu_degree = exp2_policy.effective_degree
        exp2_alu_period = exp2_policy.effective_period
    elif exp2_alu_period is None:
        exp2_alu_period = 0

    fp8_ds_lift = args.fp8_ds_lift
    if fp8_ds_lift is None:
        if args.sequence <= 512:
            fp8_ds_lift = 32
        elif args.sequence <= 1024:
            fp8_ds_lift = 64
        elif args.sequence <= 2048:
            fp8_ds_lift = 128
        else:
            fp8_ds_lift = 256

    if args.query_heads % args.kv_heads:
        parser.error("--query-heads must be divisible by --kv-heads")
    if args.head_dim % args.gqa_reduce_vector:
        parser.error("--gqa-reduce-vector must divide --head-dim")
    reduce_registers = args.reduce_registers
    if reduce_registers is None:
        reduce_registers = 136 if args.head_dim == 64 else 152
    compute_registers = args.compute_registers
    if compute_registers is None:
        compute_registers = 136 if args.head_dim == 64 else 128
    gqa_reduce_threads = args.gqa_reduce_threads
    if gqa_reduce_threads is None:
        gqa_reduce_threads = 256
    fuse_gqa_reduce = args.fuse_gqa_reduce
    if fuse_gqa_reduce is None:
        fuse_gqa_reduce = args.head_dim == 128 or args.sequence >= 1024
    fused_block_seq = args.fused_block_seq
    if fused_block_seq is None:
        if args.head_dim == 64:
            fused_block_seq = 32
        elif args.dtype == "fp8" and args.sequence >= 1024:
            fused_block_seq = 64
        else:
            fused_block_seq = 32

    if fuse_gqa_reduce and not args.split_gqa_heads:
        parser.error("--fuse-gqa-reduce requires --split-gqa-heads")
    if fuse_gqa_reduce and gqa_reduce_threads % 16:
        parser.error("fused reduction threads must be divisible by 16")
    if fused_block_seq <= 0:
        parser.error("--fused-block-seq must be positive")
    if reduce_registers % 8 or compute_registers % 8:
        parser.error("warp-group register allocations must be multiples of 8")
    if args.mma_registers != 96 or args.load_registers != 96:
        parser.error(
            "MMA/load/empty warps share one warp group and must retain the "
            "common 96-register deallocation"
        )
    register_budget = reduce_registers + 2 * compute_registers + 96
    if register_budget > 512:
        parser.error(
            "dynamic register allocation exceeds the 512-register warp-group "
            f"budget ({register_budget})"
        )
    if args.head_dim == 64 and compute_registers > 144:
        parser.error(
            "D64 compute allocations above 144 trigger an illegal dynamic "
            "register-allocation instruction on SM100"
        )
    if args.head_dim == 64 and reduce_registers < 128:
        parser.error(
            "D64 reduction allocations below 128 cannot cover the compiled "
            "register footprint and trigger an illegal instruction on SM100"
        )

    q_stages = args.q_stages
    if q_stages is None:
        q_stages = 2
    do_stages = args.do_stages
    if do_stages is None:
        if args.dtype == "fp8":
            do_stages = 1 if args.head_dim == 64 else 2
        else:
            do_stages = 2 if args.head_dim == 64 else 1
    dkdv_stages = args.dkdv_stages
    if dkdv_stages is None:
        dkdv_stages = 1 if args.dtype == "fp8" and args.head_dim == 128 else 2
    if q_stages <= 0 or do_stages <= 0 or dkdv_stages <= 0:
        parser.error("pipeline stage counts must be positive")
    if args.head_dim == 64 and q_stages < 2:
        parser.error(
            "D64 Q pipelines below two stages can deadlock the SM100 mainloop"
        )
    if args.direct_compact_dq and not args.compact_dq_acc:
        parser.error("--direct-compact-dq requires --compact-dq-acc")
    if prelift_probability_lse and not fuse_probability_lift:
        parser.error("--prelift-probability-lse requires --fuse-probability-lift")

    control = _load_control(fp8_p_storage=fp8_p_storage)
    base = control.BlackwellFusedMultiHeadAttentionBackward

    class TunedGqaBackward(base):
        def _setup_attributes(self):
            super()._setup_attributes()
            self.load_mma_Q_stage = q_stages
            self.load_mma_dO_stage = do_stages
            self.mma_compute_dKdV_stage = dkdv_stages

    original_init = TunedGqaBackward.__init__

    def tuned_init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        self.num_regs_reduce = reduce_registers
        self.num_regs_compute = compute_registers
        self.num_regs_mma = args.mma_registers
        self.num_regs_load = args.load_registers
        self.reuse_quantized_p = args.reuse_quantized_p
        self.fuse_probability_lift = fuse_probability_lift
        self.prelift_probability_lse = prelift_probability_lse
        self.fp8_ds_lift = fp8_ds_lift
        self.exp2_alu_period = exp2_alu_period
        self.exp2_alu_degree = exp2_alu_degree
        self.compact_dq_acc = args.compact_dq_acc
        self.direct_compact_dq = args.direct_compact_dq
        self.split_gqa_heads = args.split_gqa_heads
        self.gqa_reduce_vector = args.gqa_reduce_vector
        self.gqa_reduce_threads = gqa_reduce_threads
        self.fuse_gqa_reduce = fuse_gqa_reduce
        self.fused_convert_block_seq = fused_block_seq

    TunedGqaBackward.__init__ = tuned_init
    control.BlackwellFusedMultiHeadAttentionBackward = TunedGqaBackward

    element_dtype = (
        control.BFloat16 if args.dtype == "bf16" else control.Float8E4M3FN
    )
    elapsed_us = control.run(
        args.sequence,
        args.sequence,
        args.query_heads,
        args.kv_heads,
        args.head_dim,
        args.batch,
        True,
        False,
        element_dtype,
        control.Float32,
        (128, 128),
        0.0,
        (-1, -1),
        args.warmup,
        args.iterations,
        not args.check,
        args.cold_l2,
    )
    print(
        json.dumps(
            {
                "sequence": args.sequence,
                "query_heads": args.query_heads,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
                "causal": True,
                "dtype": args.dtype,
                "q_stages": q_stages,
                "do_stages": do_stages,
                "dkdv_stages": dkdv_stages,
                "reduce_registers": reduce_registers,
                "compute_registers": compute_registers,
                "mma_registers": args.mma_registers,
                "load_registers": args.load_registers,
                "register_budget": register_budget,
                "reuse_quantized_p": args.reuse_quantized_p,
                "fuse_probability_lift": fuse_probability_lift,
                "prelift_probability_lse": prelift_probability_lse,
                "fp8_p_storage": fp8_p_storage,
                "fp8_ds_lift": fp8_ds_lift,
                "exp2_alu_period": exp2_alu_period,
                "exp2_alu_degree": exp2_alu_degree,
                "compact_dq_acc": args.compact_dq_acc,
                "direct_compact_dq": args.direct_compact_dq,
                "split_gqa_heads": args.split_gqa_heads,
                "gqa_reduce_vector": args.gqa_reduce_vector,
                "gqa_reduce_threads": gqa_reduce_threads,
                "fuse_gqa_reduce": fuse_gqa_reduce,
                "fused_block_seq": fused_block_seq,
                "elapsed_us": elapsed_us,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
