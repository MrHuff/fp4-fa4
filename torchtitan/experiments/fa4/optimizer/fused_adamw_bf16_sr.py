# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Authenticated lazy loader for fused stateless BF16-SR AdamW."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
from pathlib import Path
from types import ModuleType

import torch


PROVIDER = "lbt_fused_stateless_adamw_bf16_sr"
PROVIDER_VERSION = 1
CHUNK_SIZE = 262_144
MAX_TENSOR_ELEMENTS = (1 << 32) - 1
CUDA_FLAGS = ("-O3", "-DSR_ADAMW_HASH32=1")
CXX_FLAGS = ("-O3",)

_CSRC_DIRECTORY = Path(__file__).resolve().parent / "csrc" / "adamw_bf16_sr"
_SOURCE_NAMES = (
    "binding.cpp",
    "adamw_bf16_sr.cu",
    "multi_tensor_apply.cuh",
    "compat.h",
    "APEX_LICENSE",
    "NOTICE.md",
)
_COMPILE_SOURCE_NAMES = ("binding.cpp", "adamw_bf16_sr.cu")
_extension: ModuleType | None = None
_thread_lock = threading.Lock()


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for name in _SOURCE_NAMES:
        path = _CSRC_DIRECTORY / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


SOURCE_SHA256 = _source_sha256()


def _assert_source_identity(stage: str) -> None:
    actual = _source_sha256()
    if actual != SOURCE_SHA256:
        raise RuntimeError(
            f"{PROVIDER} source changed {stage}: imported identity "
            f"{SOURCE_SHA256}, current identity {actual}"
        )


def _runtime_fingerprint() -> str:
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    payload = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "capability": capability,
        "provider": PROVIDER,
        "provider_version": PROVIDER_VERSION,
        "source_sha256": SOURCE_SHA256,
        "cxx_flags": CXX_FLAGS,
        "cuda_flags": CUDA_FLAGS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def extension_module_name() -> str:
    """Return the source- and runtime-authenticated extension module name."""
    return (
        f"lbt_adamw_bf16_sr_v{PROVIDER_VERSION}_"
        f"{SOURCE_SHA256[:16]}_{_runtime_fingerprint()[:12]}"
    )


def provider_receipt() -> dict[str, object]:
    """Return the non-secret provider contract recorded by launch receipts."""
    return {
        "provider": PROVIDER,
        "provider_version": PROVIDER_VERSION,
        "source_sha256": SOURCE_SHA256,
        "chunk_size": CHUNK_SIZE,
        "max_tensor_elements": MAX_TENSOR_ELEMENTS,
        "cxx_flags": list(CXX_FLAGS),
        "cuda_flags": list(CUDA_FLAGS),
    }


def _build_root(module_name: str) -> Path:
    configured = os.environ.get("LBT_ADAMW_BF16_SR_BUILD_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        from torch.utils.cpp_extension import get_default_build_root

        root = Path(get_default_build_root()).resolve() / "lbt_adamw_bf16_sr"
    return root / module_name


def get_extension() -> ModuleType:
    """Build once per source/runtime identity and load on the current CUDA rank.

    An outer crash-safe file lock serializes local ranks. PyTorch's internal
    build lock is removed only while holding that outer lock, preventing a
    crashed compiler process from deadlocking every later rank.
    """
    global _extension
    if _extension is not None:
        return _extension
    if not torch.cuda.is_available():
        raise RuntimeError(f"{PROVIDER} requires CUDA")

    with _thread_lock:
        if _extension is not None:
            return _extension

        try:
            from filelock import FileLock
            from torch.utils.cpp_extension import load
        except ImportError as exc:
            raise RuntimeError(
                f"{PROVIDER} requires torch.utils.cpp_extension and filelock"
            ) from exc

        module_name = extension_module_name()
        build_directory = _build_root(module_name)
        build_directory.mkdir(parents=True, exist_ok=True)
        timeout_text = os.environ.get(
            "LBT_ADAMW_BF16_SR_BUILD_LOCK_TIMEOUT_SECONDS", "1800"
        ).strip()
        try:
            lock_timeout = float(timeout_text)
        except ValueError as exc:
            raise ValueError(
                "LBT_ADAMW_BF16_SR_BUILD_LOCK_TIMEOUT_SECONDS must be numeric"
            ) from exc
        if lock_timeout <= 0:
            raise ValueError(
                "LBT_ADAMW_BF16_SR_BUILD_LOCK_TIMEOUT_SECONDS must be positive"
            )

        outer_lock = build_directory.parent / f".{module_name}.build.lock"
        with FileLock(str(outer_lock), timeout=lock_timeout):
            _assert_source_identity("before extension load")
            internal_lock = build_directory / "lock"
            if internal_lock.exists():
                internal_lock.unlink()
            extension = load(
                name=module_name,
                sources=[
                    str(_CSRC_DIRECTORY / name)
                    for name in _COMPILE_SOURCE_NAMES
                ],
                extra_include_paths=[str(_CSRC_DIRECTORY)],
                extra_cflags=list(CXX_FLAGS),
                extra_cuda_cflags=list(CUDA_FLAGS),
                build_directory=str(build_directory),
                verbose=os.environ.get(
                    "LBT_ADAMW_BF16_SR_BUILD_VERBOSE", "0"
                ).strip()
                == "1",
            )
            _assert_source_identity("during extension load")
        if not hasattr(extension, "adamw"):
            raise RuntimeError(
                f"{PROVIDER} extension {module_name} lacks adamw binding"
            )
        _extension = extension
        return extension
