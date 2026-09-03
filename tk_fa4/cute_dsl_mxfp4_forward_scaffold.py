from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib.util
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CUTLASS_PYTHON_ROOT = _REPO_ROOT / "SageAttention" / "sageattention3_blackwell" / "csrc" / "cutlass" / "python"
_CUTLASS_CUTEDSL_ROOT = _CUTLASS_PYTHON_ROOT / "CuTeDSL"
_CUTLASS_EXAMPLES_ROOT = _REPO_ROOT / "SageAttention" / "sageattention3_blackwell" / "csrc" / "cutlass" / "examples" / "python" / "CuTeDSL"
_REFERENCE_FMHA = _REPO_ROOT / "SageAttention" / "sageattention3_blackwell" / "csrc" / "cutlass" / "examples" / "python" / "CuTeDSL" / "blackwell" / "fmha.py"
_REFERENCE_BLOCKSCALED_GEMM = _REPO_ROOT / "SageAttention" / "sageattention3_blackwell" / "csrc" / "cutlass" / "examples" / "python" / "CuTeDSL" / "blackwell" / "dense_blockscaled_gemm_persistent.py"
_REFERENCE_MIXED_INPUT_FMHA_D256 = _REPO_ROOT / "SageAttention" / "sageattention3_blackwell" / "csrc" / "cutlass" / "examples" / "python" / "CuTeDSL" / "blackwell" / "mixed_input_fmha" / "mixed_input_fmha_prefill_d256.py"


@dataclass(frozen=True)
class CuteDslRuntimeStatus:
    cutlass_importable: bool
    torch_importable: bool
    reference_fmha_importable: bool
    reference_mixed_input_fmha_d256_importable: bool
    reference_blockscaled_gemm_importable: bool
    reason: str | None
    torch_reason: str | None
    reference_fmha_reason: str | None
    reference_mixed_input_fmha_d256_reason: str | None
    reference_blockscaled_gemm_reason: str | None
    reference_fmha_path: str
    reference_mixed_input_fmha_d256_path: str
    reference_blockscaled_gemm_path: str
    cutlass_python_root: str
    cutedsl_root: str
    examples_root: str
    setup_script: str


@dataclass(frozen=True)
class Mxfp4ForwardScope:
    qk_backend: str = "v5"
    pv_backend: str = "mxfp4_v3"
    block_m: int = 128
    block_n: int = 128
    head_dim_qk: int = 192
    head_dim_v: int = 128


@contextmanager
def _prepend_sys_path(path: Path):
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def _load_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def probe_cute_dsl_runtime() -> CuteDslRuntimeStatus:
    cutlass_reason: str | None = None
    torch_reason: str | None = None
    reference_fmha_reason: str | None = None
    reference_mixed_input_fmha_d256_reason: str | None = None
    reference_blockscaled_gemm_reason: str | None = None
    cutlass_importable = False
    torch_importable = False
    reference_fmha_importable = False
    reference_mixed_input_fmha_d256_importable = False
    reference_blockscaled_gemm_importable = False

    try:
        import cutlass  # type: ignore
        import cutlass.cute  # type: ignore
        del cutlass
    except Exception as exc:
        cutlass_reason = repr(exc)
    else:
        cutlass_importable = True

    try:
        import torch  # type: ignore
        del torch
    except Exception as exc:
        torch_reason = repr(exc)
    else:
        torch_importable = True

    if cutlass_importable and torch_importable:
        try:
            load_reference_blackwell_fmha_module()
        except Exception as exc:
            reference_fmha_reason = repr(exc)
        else:
            reference_fmha_importable = True

        try:
            load_reference_mixed_input_fmha_d256_module()
        except Exception as exc:
            reference_mixed_input_fmha_d256_reason = repr(exc)
        else:
            reference_mixed_input_fmha_d256_importable = True

        try:
            load_reference_blockscaled_gemm_module()
        except Exception as exc:
            reference_blockscaled_gemm_reason = repr(exc)
        else:
            reference_blockscaled_gemm_importable = True

    return CuteDslRuntimeStatus(
        cutlass_importable=cutlass_importable,
        torch_importable=torch_importable,
        reference_fmha_importable=reference_fmha_importable,
        reference_mixed_input_fmha_d256_importable=reference_mixed_input_fmha_d256_importable,
        reference_blockscaled_gemm_importable=reference_blockscaled_gemm_importable,
        reason=cutlass_reason,
        torch_reason=torch_reason,
        reference_fmha_reason=reference_fmha_reason,
        reference_mixed_input_fmha_d256_reason=reference_mixed_input_fmha_d256_reason,
        reference_blockscaled_gemm_reason=reference_blockscaled_gemm_reason,
        reference_fmha_path=str(_REFERENCE_FMHA),
        reference_mixed_input_fmha_d256_path=str(_REFERENCE_MIXED_INPUT_FMHA_D256),
        reference_blockscaled_gemm_path=str(_REFERENCE_BLOCKSCALED_GEMM),
        cutlass_python_root=str(_CUTLASS_PYTHON_ROOT),
        cutedsl_root=str(_CUTLASS_CUTEDSL_ROOT),
        examples_root=str(_CUTLASS_EXAMPLES_ROOT),
        setup_script=str(_CUTLASS_CUTEDSL_ROOT / "setup.sh"),
    )


def required_mxfp4_extensions() -> tuple[str, ...]:
    return (
        "Add explicit Q/K/V scale tensor arguments; the reference FMHA example only takes dense Q/K/V pointers.",
        "Replace dense QK/PV tiled MMA construction with blockscaled FP4 tiled MMA from the blockscaled GEMM example.",
        "Carry prequantized MXFP4 V payload and scales through the mainloop rather than quantizing in Python.",
        "Quantize P inside the online softmax loop and write P payload plus scale fragments directly to the PV path.",
        "Replace dense V TMA layout assumptions with blockscaled V payload plus scale layouts.",
        "Expose a Python launcher with the same `(Q_fp4, Q_sc, Q_sg, K_fp4, K_sc, K_sg, V_fp4, V_sc)` style contract used in tk_fa4.",
    )


def load_reference_blackwell_fmha_module() -> Any:
    try:
        import cutlass  # type: ignore
        import cutlass.cute  # type: ignore
        import torch  # type: ignore
        del cutlass
        del torch
    except Exception as exc:
        raise RuntimeError(
            "CuTe DSL runtime or torch is not importable in this environment. "
            f"Current failure: {repr(exc)}. "
            f"Expected setup script: {_CUTLASS_CUTEDSL_ROOT / 'setup.sh'}"
        ) from exc
    with _prepend_sys_path(_CUTLASS_EXAMPLES_ROOT):
        return _load_module_from_path("cutlass_blackwell_fmha_reference", _REFERENCE_FMHA)


def load_reference_blockscaled_gemm_module() -> Any:
    try:
        import cutlass  # type: ignore
        import cutlass.cute  # type: ignore
        import torch  # type: ignore
        del cutlass
        del torch
    except Exception as exc:
        raise RuntimeError(
            "CuTe DSL runtime or torch is not importable in this environment. "
            f"Current failure: {repr(exc)}. "
            f"Expected setup script: {_CUTLASS_CUTEDSL_ROOT / 'setup.sh'}"
        ) from exc
    with _prepend_sys_path(_CUTLASS_EXAMPLES_ROOT):
        return _load_module_from_path(
            "cutlass_blackwell_dense_blockscaled_reference",
            _REFERENCE_BLOCKSCALED_GEMM,
        )


def load_reference_mixed_input_fmha_d256_module() -> Any:
    try:
        import cutlass  # type: ignore
        import cutlass.cute  # type: ignore
        import torch  # type: ignore
        del cutlass
        del torch
    except Exception as exc:
        raise RuntimeError(
            "CuTe DSL runtime or torch is not importable in this environment. "
            f"Current failure: {repr(exc)}. "
            f"Expected setup script: {_CUTLASS_CUTEDSL_ROOT / 'setup.sh'}"
        ) from exc
    with _prepend_sys_path(_CUTLASS_EXAMPLES_ROOT):
        return _load_module_from_path(
            "cutlass_blackwell_mixed_input_fmha_d256_reference",
            _REFERENCE_MIXED_INPUT_FMHA_D256,
        )


def reference_entrypoints() -> dict[str, str]:
    return {
        "fmha_class": "blackwell.fmha.BlackwellFusedMultiHeadAttentionForward",
        "fmha_run": "blackwell.fmha.run",
        "mixed_input_fmha_d256_class": "blackwell.mixed_input_fmha.mixed_input_fmha_prefill_d256.MixedInputFusedMultiHeadAttentionPrefillD256",
        "mixed_input_fmha_d256_run": "blackwell.mixed_input_fmha.mixed_input_fmha_prefill_d256.run",
        "blockscaled_gemm_class": "blackwell.dense_blockscaled_gemm_persistent.Sm100BlockScaledPersistentDenseGemmKernel",
        "blockscaled_gemm_run": "blackwell.dense_blockscaled_gemm_persistent.run",
    }


def mxfp4_port_targets() -> tuple[str, ...]:
    return (
        "Reuse the FMHA persistent scheduler, load/correction/epilogue warpgroups, and online softmax structure from `BlackwellFusedMultiHeadAttentionForward`.",
        "Replace dense QK tiled MMA setup with blockscaled FP4 Q/K MMA built like `Sm100BlockScaledPersistentDenseGemmKernel`.",
        "Replace dense PV tiled MMA setup with blockscaled FP4 P/V MMA built like `Sm100BlockScaledPersistentDenseGemmKernel`.",
        "Keep P resident inside the softmax mainloop: quantize the current probability tile to MXFP4 and feed it directly into the PV blockscaled MMA.",
        "Thread prequantized V payload and V scale tensors through the FMHA load pipeline instead of using dense V tensors.",
        "Preserve the FMHA correction/rescale logic, but make its output compatible with blockscaled PV accumulation.",
        "For tk_fa4-style qk_head_dim=192, use `mixed_input_fmha_prefill_d256.py` as the closest large-D in-tree scheduler/mainloop reference rather than assuming `fmha.py` extends directly.",
    )


def summarize_mxfp4_cute_dsl_port(scope: Mxfp4ForwardScope | None = None) -> dict[str, Any]:
    scope = scope or Mxfp4ForwardScope()
    status = probe_cute_dsl_runtime()
    return {
        "scope": {
            "qk_backend": scope.qk_backend,
            "pv_backend": scope.pv_backend,
            "block_m": scope.block_m,
            "block_n": scope.block_n,
            "head_dim_qk": scope.head_dim_qk,
            "head_dim_v": scope.head_dim_v,
        },
        "runtime": {
            "cutlass_importable": status.cutlass_importable,
            "reason": status.reason,
            "torch_importable": status.torch_importable,
            "torch_reason": status.torch_reason,
            "reference_fmha_importable": status.reference_fmha_importable,
            "reference_fmha_reason": status.reference_fmha_reason,
            "reference_mixed_input_fmha_d256_importable": status.reference_mixed_input_fmha_d256_importable,
            "reference_mixed_input_fmha_d256_reason": status.reference_mixed_input_fmha_d256_reason,
            "reference_mixed_input_fmha_d256_path": status.reference_mixed_input_fmha_d256_path,
            "reference_blockscaled_gemm_importable": status.reference_blockscaled_gemm_importable,
            "reference_blockscaled_gemm_reason": status.reference_blockscaled_gemm_reason,
            "reference_fmha_path": status.reference_fmha_path,
            "reference_blockscaled_gemm_path": status.reference_blockscaled_gemm_path,
            "examples_root": status.examples_root,
            "setup_script": status.setup_script,
        },
        "reference_entrypoints": reference_entrypoints(),
        "port_targets": mxfp4_port_targets(),
        "required_extensions": required_mxfp4_extensions(),
    }


def launch_mxfp4_forward_cute_dsl(*_: Any, **__: Any) -> None:
    status = probe_cute_dsl_runtime()
    if not status.cutlass_importable:
        raise RuntimeError(
            "CuTe DSL runtime is not ready here. "
            f"Current failure: {status.reason}. "
            f"Try setting up the in-tree runtime from {status.setup_script} first."
        )
    raise NotImplementedError(
        "MXFP4 CuTe DSL forward is not implemented yet. "
        "Use `summarize_mxfp4_cute_dsl_port()` to inspect the required integration points."
    )
