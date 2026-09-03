"""Fail-closed CUTLASS DSL provenance for experimental D128 MXFP4-V dP.

The retained BF16 and E4M3-V backward routes deliberately do not call this
module.  The experimental MXFP4-V compiler path does: CUTLASS DSL 4.5.2 was
published with materially different LLVM20 and LLVM21 payloads under the same
distribution version, and only the complete LLVM21/cu13 payload retained the
measured D128 performance.

The wheel list is an expected supply-chain recipe, not a claim that pip's
historical overlay order can be reconstructed from an installed environment.
Runtime authentication is instead based on the effective payload and native
library bytes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "d128_mxfp4_v_toolchain.json"
DEFAULT_MANIFEST_SHA256 = (
    "b695346ffe0fb3c80a21005ca55494e60e0decb7b700533004bdf14721a3db0e"
)
DEFAULT_MANIFEST_BYTES = 2599
MAX_MANIFEST_BYTES = 64 * 1024
PAYLOAD_HASH_SCHEMA = "sha256sum_site_relative_path_lines_v1"
EXPECTED_SCHEMA = "fa4_d128_mxfp4_v_cutlass_toolchain_v1"
COMPILE_RECEIPT_SCHEMA = "fa4_d128_mxfp4_v_cute_compile_v1"
TOOLCHAIN_RECEIPT_SCHEMA = "fa4_d128_mxfp4_v_cutlass_toolchain_receipt_v1"


def _stable_regular_file_identity(path: Path | str) -> dict[str, Any]:
    """Hash one stable regular, non-symlink file without trusting its path."""
    requested = Path(path)
    try:
        requested_stat = requested.lstat()
    except OSError as error:
        raise RuntimeError(f"unable to stat toolchain artifact: {requested}") from error
    if not stat.S_ISREG(requested_stat.st_mode):
        raise RuntimeError(
            f"toolchain artifact must be a regular non-symlink file: {requested}"
        )
    resolved = requested.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    try:
        descriptor = os.open(resolved, flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened_stat = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_stat.st_mode):
                raise RuntimeError(
                    f"toolchain artifact stopped being regular: {resolved}"
                )
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            closed_stat = os.fstat(stream.fileno())
    except OSError as error:
        raise RuntimeError(f"unable to read toolchain artifact: {resolved}") from error
    final_stat = resolved.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    observations = (requested_stat, opened_stat, closed_stat, final_stat)
    if any(
        getattr(observation, field) != getattr(requested_stat, field)
        for observation in observations[1:]
        for field in stable_fields
    ):
        raise RuntimeError(f"toolchain artifact changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "bytes": int(final_stat.st_size),
    }


def _read_stable_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identity_before = _stable_regular_file_identity(path)
    if not 0 < identity_before["bytes"] <= MAX_MANIFEST_BYTES:
        raise RuntimeError("CUTLASS toolchain manifest has an invalid byte count")
    try:
        payload = Path(identity_before["path"]).read_bytes()
    except OSError as error:
        raise RuntimeError("unable to read CUTLASS toolchain manifest") from error
    identity_after = _stable_regular_file_identity(path)
    if identity_after != identity_before:
        raise RuntimeError("CUTLASS toolchain manifest changed while reading")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("CUTLASS toolchain manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("CUTLASS toolchain manifest must be a JSON object")
    return manifest, identity_before


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _require_safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise RuntimeError(f"{label} is not a safe POSIX relative path")
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise RuntimeError("unexpected CUTLASS toolchain manifest schema")
    distribution = manifest.get("distribution")
    if distribution != {"name": "nvidia-cutlass-dsl", "version": "4.5.2"}:
        raise RuntimeError("unexpected CUTLASS DSL distribution identity")
    if manifest.get("target_arch") != "sm_100a":
        raise RuntimeError("D128 MXFP4-V toolchain must target sm_100a")

    supply = manifest.get("wheel_supply_chain")
    if not isinstance(supply, dict) or supply.get("provenance") != (
        "expected_inputs_not_inferred_from_overlaid_runtime"
    ):
        raise RuntimeError("CUTLASS wheel provenance semantics are malformed")
    if supply.get("install_order") != ["libs_base", "libs_cu13", "meta"]:
        raise RuntimeError("CUTLASS wheel overlay order is malformed")
    wheels = supply.get("wheels")
    if not isinstance(wheels, list) or [
        wheel.get("id") if isinstance(wheel, dict) else None for wheel in wheels
    ] != supply["install_order"]:
        raise RuntimeError("CUTLASS wheel identities do not match overlay order")
    for index, wheel in enumerate(wheels):
        assert isinstance(wheel, dict)
        _require_safe_relative(wheel.get("filename"), f"wheel {index} filename")
        _require_sha256(wheel.get("sha256"), f"wheel {index} SHA256")
        _require_positive_int(wheel.get("bytes"), f"wheel {index} bytes")

    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("CUTLASS payload identity is missing")
    if payload.get("hash_schema") != PAYLOAD_HASH_SCHEMA:
        raise RuntimeError("CUTLASS payload hash schema is malformed")
    if payload.get("root_entries") != [
        "nvidia_cutlass_dsl",
        "nvidia_cutlass_dsl.pth",
    ]:
        raise RuntimeError("CUTLASS payload roots are malformed")
    if payload.get("excluded") != ["__pycache__", "*.pyc", "*.dist-info"]:
        raise RuntimeError("CUTLASS payload exclusion contract is malformed")
    _require_sha256(payload.get("sha256"), "payload SHA256")
    _require_positive_int(payload.get("files"), "payload files")
    _require_positive_int(payload.get("bytes"), "payload bytes")

    artifacts = manifest.get("native_artifacts")
    expected_artifacts = {
        "cutlass_ir",
        "cute_dsl_runtime",
        "cuda_dialect_runtime_static",
        "pth",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise RuntimeError("CUTLASS native artifact set is malformed")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            raise RuntimeError(f"CUTLASS artifact {name} is malformed")
        _require_safe_relative(
            artifact.get("relative_path"), f"CUTLASS artifact {name} path"
        )
        _require_sha256(artifact.get("sha256"), f"CUTLASS artifact {name} SHA256")
        _require_positive_int(artifact.get("bytes"), f"CUTLASS artifact {name} bytes")
    cutlass_ir = artifacts["cutlass_ir"]
    if cutlass_ir.get("llvm_version") != "21.0.0git" or cutlass_ir.get(
        "llvm_commit"
    ) != "e57c3673ac82461cd3c8a2e5cf6f8a890705c882":
        raise RuntimeError("CUTLASS LLVM21 build identity is malformed")

    if manifest.get("compile_environment") != {
        "CUTE_DSL_ARCH": "sm_100a",
        "CUTE_DSL_KEEP": "ptx,cubin",
        "CUTE_DSL_NO_CACHE": "1",
    }:
        raise RuntimeError("CUTLASS compile environment contract is malformed")


def load_d128_mxfp4_v_toolchain_manifest(
    manifest_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the checked-in toolchain contract."""
    selected = DEFAULT_MANIFEST if manifest_path is None else Path(manifest_path)
    manifest, identity = _read_stable_manifest(selected)
    if manifest_path is None and (
        identity["sha256"] != DEFAULT_MANIFEST_SHA256
        or identity["bytes"] != DEFAULT_MANIFEST_BYTES
    ):
        raise RuntimeError(
            "checked-in D128 MXFP4-V toolchain manifest identity mismatch"
        )
    _validate_manifest(manifest)
    return manifest, identity


def _payload_files(
    site_root: Path,
    root_entries: list[str],
) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    for entry in root_entries:
        relative_entry = _require_safe_relative(entry, "payload root")
        root = site_root / relative_entry
        try:
            root_stat = root.lstat()
        except OSError as error:
            raise RuntimeError(f"CUTLASS payload root is missing: {root}") from error
        if stat.S_ISREG(root_stat.st_mode):
            if root.suffix != ".pyc":
                selected.append((relative_entry, root))
            continue
        if not stat.S_ISDIR(root_stat.st_mode):
            raise RuntimeError(
                f"CUTLASS payload root must be a regular file or directory: {root}"
            )
        for current, directory_names, file_names in os.walk(root, followlinks=False):
            current_path = Path(current)
            kept_directories: list[str] = []
            for name in sorted(directory_names):
                path = current_path / name
                if name == "__pycache__":
                    continue
                path_stat = path.lstat()
                if not stat.S_ISDIR(path_stat.st_mode):
                    raise RuntimeError(
                        f"CUTLASS payload contains an unsafe directory: {path}"
                    )
                kept_directories.append(name)
            directory_names[:] = kept_directories
            for name in sorted(file_names):
                if name.endswith(".pyc"):
                    continue
                path = current_path / name
                relative = path.relative_to(site_root).as_posix()
                path_stat = path.lstat()
                if not stat.S_ISREG(path_stat.st_mode):
                    raise RuntimeError(
                        f"CUTLASS payload contains a non-regular file: {path}"
                    )
                selected.append((relative, path))
    selected.sort(key=lambda item: item[0])
    if len({relative for relative, _ in selected}) != len(selected):
        raise RuntimeError("CUTLASS payload contains duplicate paths")
    return selected


def summarize_cutlass_payload(
    site_root: Path | str,
    root_entries: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Reproduce the audited ``sha256sum`` byte-stream tree identity."""
    root = Path(site_root).resolve(strict=True)
    digest = hashlib.sha256()
    total_bytes = 0
    identities: dict[str, dict[str, Any]] = {}
    for relative, path in _payload_files(root, root_entries):
        identity = _stable_regular_file_identity(path)
        digest.update(f"{identity['sha256']}  {relative}\n".encode("utf-8"))
        total_bytes += int(identity["bytes"])
        identities[relative] = identity
    return (
        {
            "hash_schema": PAYLOAD_HASH_SCHEMA,
            "sha256": digest.hexdigest(),
            "files": len(identities),
            "bytes": total_bytes,
        },
        identities,
    )


def _resolve_cutlass_layout(
    cutlass_origin: Path | str | None,
) -> tuple[Path, Path]:
    if cutlass_origin is None:
        spec = importlib.util.find_spec("cutlass")
        if spec is None or spec.origin is None:
            raise RuntimeError("unable to locate the CUTLASS DSL Python payload")
        origin = Path(spec.origin)
    else:
        origin = Path(cutlass_origin)
    resolved = origin.resolve(strict=True)
    if (
        resolved.name != "__init__.py"
        or resolved.parent.name != "cutlass"
        or resolved.parents[1].name != "python_packages"
        or resolved.parents[2].name != "nvidia_cutlass_dsl"
    ):
        raise RuntimeError(
            "CUTLASS Python origin is outside the pinned wheel payload layout"
        )
    loaded = sys.modules.get("cutlass")
    loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
    if loaded_file is not None and Path(loaded_file).resolve(strict=True) != resolved:
        raise RuntimeError("loaded CUTLASS module disagrees with selected payload")
    return resolved, resolved.parents[3]


def require_d128_mxfp4_v_compile_environment(
    *,
    manifest: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Require artifact-preserving, cache-bypassing candidate compilation."""
    if manifest is None:
        manifest, _ = load_d128_mxfp4_v_toolchain_manifest()
    environment = os.environ if environ is None else environ
    expected = manifest["compile_environment"]
    for name, value in expected.items():
        if environment.get(name) != value:
            raise RuntimeError(
                f"D128 MXFP4-V compilation requires {name}={value!r}"
            )
    dump_value = environment.get("CUTE_DSL_DUMP_DIR")
    if not dump_value or not Path(dump_value).is_absolute():
        raise RuntimeError(
            "D128 MXFP4-V compilation requires an absolute CUTE_DSL_DUMP_DIR"
        )
    dump_path = Path(dump_value)
    try:
        dump_stat = dump_path.lstat()
    except OSError as error:
        raise RuntimeError("CUTE_DSL_DUMP_DIR does not exist") from error
    if not stat.S_ISDIR(dump_stat.st_mode):
        raise RuntimeError("CUTE_DSL_DUMP_DIR must be a non-symlink directory")
    return {**expected, "CUTE_DSL_DUMP_DIR": str(dump_path.resolve(strict=True))}


def configure_d128_mxfp4_v_compile_environment(
    dump_directory: Path | str,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Set the exact candidate-only compiler controls around a prepared dir."""
    manifest, _ = load_d128_mxfp4_v_toolchain_manifest()
    environment = os.environ if environ is None else environ
    dump_path = Path(dump_directory)
    if not dump_path.is_absolute():
        raise RuntimeError("candidate compiler artifact directory must be absolute")
    try:
        dump_stat = dump_path.lstat()
    except OSError as error:
        raise RuntimeError(
            "candidate compiler artifact directory is missing"
        ) from error
    if not stat.S_ISDIR(dump_stat.st_mode):
        raise RuntimeError(
            "candidate compiler artifact directory must be a non-symlink directory"
        )
    required = {
        **manifest["compile_environment"],
        "CUTE_DSL_DUMP_DIR": str(dump_path.resolve(strict=True)),
    }
    for name, value in required.items():
        previous = environment.get(name)
        if previous is not None and previous != value:
            raise RuntimeError(
                f"refusing to replace incompatible candidate compiler setting "
                f"{name}={previous!r}"
            )
        environment[name] = value
    return require_d128_mxfp4_v_compile_environment(
        manifest=manifest,
        environ=environment,
    )


def verify_d128_mxfp4_v_toolchain(
    *,
    manifest_path: Path | str | None = None,
    cutlass_origin: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Authenticate the effective LLVM21/cu13 payload, not pip history."""
    manifest, manifest_identity = load_d128_mxfp4_v_toolchain_manifest(
        manifest_path
    )
    origin, site_root = _resolve_cutlass_layout(cutlass_origin)
    payload_summary, identities = summarize_cutlass_payload(
        site_root,
        manifest["payload"]["root_entries"],
    )
    expected_payload = {
        key: manifest["payload"][key]
        for key in ("hash_schema", "sha256", "files", "bytes")
    }
    if payload_summary != expected_payload:
        raise RuntimeError(
            "CUTLASS DSL effective payload identity mismatch: "
            f"expected {expected_payload}, observed {payload_summary}"
        )

    authenticated_artifacts: dict[str, Any] = {}
    for name, expected in manifest["native_artifacts"].items():
        relative = expected["relative_path"]
        observed = identities.get(relative)
        if observed is None:
            raise RuntimeError(f"CUTLASS payload omitted required artifact {name}")
        if (
            observed["sha256"] != expected["sha256"]
            or observed["bytes"] != expected["bytes"]
        ):
            raise RuntimeError(f"CUTLASS artifact {name} identity mismatch")
        authenticated_artifacts[name] = {
            **expected,
            "path": observed["path"],
        }

    environment = os.environ if environ is None else environ
    runtime_path = Path(authenticated_artifacts["cute_dsl_runtime"]["path"])
    selected_libs = environment.get("CUTE_DSL_LIBS")
    if selected_libs is None:
        runtime_selection = "authenticated_ancestor_auto_discovery"
    else:
        parts = selected_libs.split(os.pathsep)
        if len(parts) != 1 or not parts[0]:
            raise RuntimeError(
                "CUTE_DSL_LIBS must select only the pinned runtime library"
            )
        try:
            selected_runtime = Path(parts[0]).resolve(strict=True)
        except OSError as error:
            raise RuntimeError("CUTE_DSL_LIBS selects a missing library") from error
        if selected_runtime != runtime_path:
            raise RuntimeError("CUTE_DSL_LIBS does not select the pinned runtime")
        runtime_selection = "explicit_authenticated_path"

    return {
        "schema": TOOLCHAIN_RECEIPT_SCHEMA,
        "manifest": manifest_identity,
        "distribution": dict(manifest["distribution"]),
        "target_arch": manifest["target_arch"],
        "wheel_supply_chain": manifest["wheel_supply_chain"],
        "wheel_provenance_is_runtime_inferred": False,
        "payload": {
            **payload_summary,
            "site_root": str(site_root),
            "cutlass_origin": str(origin),
        },
        "native_artifacts": authenticated_artifacts,
        "runtime_selection": runtime_selection,
    }


def _compiled_payload_identity(
    value: Any,
    kind: str,
    target_arch: str,
) -> dict[str, Any]:
    if kind == "ptx":
        if not isinstance(value, str) or not value:
            raise RuntimeError("candidate compilation did not retain PTX text")
        target_pattern = rf"(?m)^\.target {re.escape(target_arch)}(?:,.*)?$"
        if re.search(target_pattern, value) is None:
            raise RuntimeError(
                f"candidate PTX does not declare the pinned {target_arch} target"
            )
        payload = value.encode("utf-8")
        encoding = "utf-8"
    elif kind == "cubin":
        if not isinstance(value, bytes) or not value.startswith(b"\x7fELF"):
            raise RuntimeError("candidate compilation did not retain an ELF CUBIN")
        payload = value
        encoding = "binary"
    else:  # pragma: no cover - internal invariant
        raise AssertionError(f"unsupported compiled artifact kind: {kind}")
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "encoding": encoding,
        "content_source": f"compiled.__{kind}__",
    }


def d128_mxfp4_v_compilation_receipt(
    *,
    control: Any,
    compiled: Any,
    kernel: Any,
) -> dict[str, Any]:
    """Bind generated PTX/CUBIN bytes to the authenticated candidate image."""
    if not bool(getattr(control, "TK_D128_MXFP4_V_DP", False)):
        raise RuntimeError("compilation provenance is candidate-only")
    before = getattr(control, "TK_D128_MXFP4_V_TOOLCHAIN_PROVENANCE", None)
    if not isinstance(before, dict):
        raise RuntimeError("candidate control lacks toolchain provenance")
    after = verify_d128_mxfp4_v_toolchain()
    if after != before:
        raise RuntimeError("CUTLASS toolchain changed during candidate compilation")
    compile_environment = require_d128_mxfp4_v_compile_environment()
    source_path = getattr(control, "__file__", None)
    if not isinstance(source_path, str) or not source_path:
        raise RuntimeError("candidate control lacks generated-source provenance")
    source = _stable_regular_file_identity(source_path)
    patch = getattr(control, "TK_D128_MXFP4_V_DP_PATCH_PROVENANCE", None)
    if not isinstance(patch, dict):
        raise RuntimeError("candidate control lacks patch provenance")

    ptx = _compiled_payload_identity(
        getattr(compiled, "__ptx__", None),
        "ptx",
        str(after["target_arch"]),
    )
    cubin = _compiled_payload_identity(
        getattr(compiled, "__cubin__", None),
        "cubin",
        str(after["target_arch"]),
    )
    registers = {
        "reduce": int(getattr(kernel, "num_regs_reduce")),
        "compute": int(getattr(kernel, "num_regs_compute")),
        "mma": int(getattr(kernel, "num_regs_mma")),
        "load": int(getattr(kernel, "num_regs_load")),
    }
    if registers != {"reduce": 136, "compute": 136, "mma": 96, "load": 96}:
        raise RuntimeError("candidate register split drifted from 136/136/96/96")
    return {
        "schema": COMPILE_RECEIPT_SCHEMA,
        "toolchain": after,
        "compile_environment": compile_environment,
        "generated_control": source,
        "candidate_patch": dict(patch),
        "target_arch": after["target_arch"],
        "registers": registers,
        "ptx": ptx,
        "cubin": cubin,
    }
