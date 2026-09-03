from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import stat
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakValueDictionary

import torch


LOWP_BWD_EXTENSION_SOURCE_ENV = "TK_FA4_LOWP_BWD_EXTENSION_SOURCE"
LOWP_BWD_EXTENSION_MODULE = "tk_fa4._C_b300_lowp_bwd"

V509_E5M2_DOUT_PUBLISHER_METADATA = {
    "schema": "tkfa4.v509_e5m2_dout_publisher.v1",
    "source_identity": (
        "v509_fused_nvfp4_output_projection_e5m2_dout_b1_s4096_v1"
    ),
    "experimental": True,
    "production_dispatch_connected": False,
    "dispatch": "fail_closed_B1_S4096_H32_D128_native_score_only",
    "selected_epilogue": "kernel_v509_native_score_e5m2_dout",
    "payload_dtype": "float8_e5m2",
    "payload_layout": "BSHD_contiguous",
    "encode": "(BF16.float()*4).to(float8_e5m2)",
    "encode_scale": 4.0,
    "logical_decode_scale": 0.25,
    "dstat_physical_abi": "-4*sum(O*raw_E5M2_dO)",
    "lstat_abi": "8-LSE*log2(e)",
    "probability_log2_lift": 8.0,
    "batch": 1,
    "sequence": 4096,
    "query_heads": 32,
    "head_dim": 128,
    "store_bf16_dout": False,
    "publish_e4m3_dout": False,
    "publish_stats": True,
    "clear_dq": True,
    "raw_output_slots": 8,
    "e5m2_payload_slot": 7,
}

V510_E5M2_DOUT_PUBLISHER_METADATA = {
    "schema": "tkfa4.v510_e5m2_dout_publisher.v1",
    "source_identity": (
        "v510_fused_nvfp4_output_projection_e5m2_dout_b1_s4096_v1"
    ),
    "experimental": True,
    "production_dispatch_connected": False,
    "dispatch": "fail_closed_B1_S4096_H32_D128_dense_score_only",
    # The payload/statistics implementation is intentionally the exact
    # already-gated v509 epilogue; only the authenticated consumer differs.
    "selected_epilogue": "kernel_v509_native_score_e5m2_dout",
    "payload_dtype": "float8_e5m2",
    "payload_layout": "BSHD_contiguous",
    "encode": "(BF16.float()*4).to(float8_e5m2)",
    "encode_scale": 4.0,
    "logical_decode_scale": 0.25,
    "dstat_physical_abi": "-4*sum(O*raw_E5M2_dO)",
    "lstat_abi": "8-LSE*log2(e)",
    "probability_log2_lift": 8.0,
    "batch": 1,
    "sequence": 4096,
    "query_heads": 32,
    "head_dim": 128,
    "store_bf16_dout": False,
    "publish_e4m3_dout": False,
    "publish_stats": True,
    "clear_dq": True,
    "raw_output_slots": 8,
    "e5m2_payload_slot": 7,
}


def _extension_file_identity(path: Path) -> dict[str, int | str]:
    """Hash one stable regular file image for a subsequent extension load."""
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ImportError(f"extension artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise ImportError(f"extension artifact changed while hashing: {path}")
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": digest.hexdigest(),
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def _load_lowp_bwd_extension_override(source: str):
    requested = Path(source)
    if not requested.is_absolute():
        raise ImportError(
            f"{LOWP_BWD_EXTENSION_SOURCE_ENV} must be an absolute path"
        )
    try:
        requested_stat = requested.lstat()
    except OSError as error:
        raise ImportError(
            f"unable to stat lowp backward extension override: {requested}"
        ) from error
    if not stat.S_ISREG(requested_stat.st_mode):
        raise ImportError(
            "lowp backward extension override must be a regular, non-symlink "
            f"file: {requested}"
        )
    resolved = requested.resolve(strict=True)
    if resolved.suffix != ".so":
        raise ImportError(
            f"lowp backward extension override must be a .so file: {resolved}"
        )
    identity_before = _extension_file_identity(resolved)
    loaded = sys.modules.get(LOWP_BWD_EXTENSION_MODULE)
    if loaded is not None:
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file is None or Path(loaded_file).resolve() != resolved:
            raise ImportError(
                f"{LOWP_BWD_EXTENSION_MODULE} is already loaded from "
                f"{loaded_file!r}, refusing override {resolved}"
            )
        return loaded
    spec = importlib.util.spec_from_file_location(
        LOWP_BWD_EXTENSION_MODULE, resolved
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"unable to create lowp backward extension spec for {resolved}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[LOWP_BWD_EXTENSION_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(LOWP_BWD_EXTENSION_MODULE, None)
        raise
    identity_after = _extension_file_identity(resolved)
    if identity_after != identity_before:
        sys.modules.pop(LOWP_BWD_EXTENSION_MODULE, None)
        raise ImportError(
            "lowp backward extension changed while loading: "
            f"{resolved}"
        )
    module._tk_fa4_loaded_artifact_identity = identity_before
    return module

try:
    from . import _C
except ImportError as exc:  # pragma: no cover - surfaced when the user calls the API.
    _C = None
    _BWD_IMPORT_ERROR = exc
else:
    _BWD_IMPORT_ERROR = None

try:
    from . import _C_b300_causal
except ImportError as exc:  # pragma: no cover - surfaced when the user calls the API.
    _C_b300_causal = None
    _CAUSAL_IMPORT_ERROR = exc
else:
    _CAUSAL_IMPORT_ERROR = None

try:
    from . import _C_b300_causal_bf16_baseline
except ImportError as exc:  # pragma: no cover - surfaced when the user calls the API.
    _C_b300_causal_bf16_baseline = None
    _CAUSAL_BF16_IMPORT_ERROR = exc
else:
    _CAUSAL_BF16_IMPORT_ERROR = None

try:
    from . import _C_b300_noncausal
except ImportError as exc:  # pragma: no cover - surfaced when the user calls the API.
    _C_b300_noncausal = None
    _NONCAUSAL_IMPORT_ERROR = exc
else:
    _NONCAUSAL_IMPORT_ERROR = None

if LOWP_BWD_EXTENSION_SOURCE_ENV in os.environ:
    _C_b300_lowp_bwd = _load_lowp_bwd_extension_override(
        os.environ[LOWP_BWD_EXTENSION_SOURCE_ENV]
    )
    _LOWP_BWD_IMPORT_ERROR = None
else:
    try:
        from . import _C_b300_lowp_bwd
    except ImportError as exc:  # pragma: no cover - surfaced only by lowp API.
        _C_b300_lowp_bwd = None
        _LOWP_BWD_IMPORT_ERROR = exc
    else:
        _LOWP_BWD_IMPORT_ERROR = None


_PAD_MULTIPLE = 128
_EXPERIMENTAL_PAD_MULTIPLE = 256
_QK_HEAD_DIM = 192
_V_HEAD_DIM = 128
_MIN_SEQ_LEN = 2048
_CAUSAL_PERSISTENT_MAX_SEQ = 4096
_B300_ADAPTIVE_PRODUCER_STREAMS: dict[int, torch.cuda.Stream] = {}
_NVFP4_PROJECTION_PUBLICATION_POLICIES = {
    "auto": 0,
    "fused": 1,
    "separate": 2,
}
MXFP4_V_SCALE_POLICY_ROWWISE_D32 = "rowwise_independent_d32_anchors"
MXFP4_V_SCALE_POLICY_SHARED_D32XS32 = (
    "shared_d32xs32_forward_anchors"
)


@dataclass(frozen=True)
class B300AdaptiveLowpOperands:
    """Opaque forward-produced operands for the adaptive lowp backward."""

    q_fp4: torch.Tensor
    score_q_fp4: torch.Tensor
    k_fp4: torch.Tensor
    score_k_fp4: torch.Tensor
    qk_scales: torch.Tensor
    mixed_v: torch.Tensor | None = None


@dataclass(frozen=True)
class B300DualNVFP4ProjectionWeight:
    """One learned weight prepared in forward and physical-transpose layouts.

    The two operands share one global scale and come from one quantization of
    the current BF16 master weight.  They are valid for one forward/backward
    pair and must not be retained across an in-place optimizer update.
    """

    forward: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    transpose: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class B300UnifiedLowpQKV:
    """Projection-native Q/K/V operands shared by forward and backward.

    The compact Q/K payload tensors are stored only once: backward consumes
    their byte views while the FP4 forward consumes the typed E2M1 views
    returned by :meth:`forward_operands`.  ``q``, ``k``, and ``v`` are absent
    when the caller requests the no-BF16-publication projection specialization.
    """

    q: torch.Tensor | None
    k: torch.Tensor | None
    v: torch.Tensor | None
    backward: B300AdaptiveLowpOperands
    # Descriptor-native typed aliases of ``backward.score_{q,k}_fp4``.  Keeping
    # these aliases in the publication bundle prevents a dtype-view TensorImpl
    # allocation in every timed attention launch.
    q_forward_fp4: torch.Tensor
    k_forward_fp4: torch.Tensor
    q_forward_scales: torch.Tensor
    q_forward_global_scale: torch.Tensor
    k_forward_scales: torch.Tensor
    k_forward_global_scale: torch.Tensor
    v_forward_fp4: torch.Tensor
    v_forward_scales: torch.Tensor
    v_backward_fp4: torch.Tensor
    v_backward_scales: torch.Tensor
    v_backward_fp8: torch.Tensor | None
    q_dk_fp4: torch.Tensor | None
    k_dq_fp4: torch.Tensor | None
    q_dk_scales: torch.Tensor | None
    k_dq_scales: torch.Tensor | None
    pure_qk_single_quant: bool = False
    q_backward_fp8: torch.Tensor | None = None
    k_backward_fp8: torch.Tensor | None = None
    q_heads: int | None = None
    kv_heads: int | None = None
    head_dim: int = _QK_HEAD_DIM
    # Optional descriptor-native [B,H,D,S] E4M3 V.  Older projection builds
    # publish only the [B,S,H,D] backward operand, so callers must retain an
    # explicit transpose fallback rather than assuming this field exists.
    v_forward_fp8: torch.Tensor | None = None
    # This is an ABI tag, not descriptive provenance. Consumers that alter
    # the four D32 anchors must require the exact policy they implement.
    v_backward_mxfp4_scale_policy: str | None = None

    def qk_forward_operands(self) -> tuple[torch.Tensor, ...]:
        """Return the six NVFP4 Q/K operands shared by both PV routes."""
        return (
            self.q_forward_fp4,
            self.q_forward_scales,
            self.q_forward_global_scale,
            self.k_forward_fp4,
            self.k_forward_scales,
            self.k_forward_global_scale,
        )

    def forward_operands(self) -> tuple[torch.Tensor, ...]:
        """Return the native FP4-forward Q/K/V operand tuple without copies."""
        if not self.v_forward_fp4.numel() or not self.v_forward_scales.numel():
            raise RuntimeError(
                "this projection bundle has no MXFP4 V publication; use "
                "qk_forward_operands() with its explicit FP8 V operand"
            )
        return (
            *self.qk_forward_operands(),
            self.v_forward_fp4,
            self.v_forward_scales,
        )

    def pure_qk_backward_operands(self) -> tuple[torch.Tensor, ...]:
        """Return compact fixed-scale Q/K operands for pure-FP4 dQ/dK."""
        if self.q_dk_fp4 is None or self.k_dq_fp4 is None:
            raise RuntimeError(
                "compact pure-FP4 Q/K operands were not requested from "
                "the projection epilogue"
            )
        assert self.q_dk_scales is not None
        assert self.k_dq_scales is not None
        return (
            self.q_dk_fp4,
            self.k_dq_fp4,
            self.q_dk_scales,
            self.k_dq_scales,
        )

    def pure_backward_operands(self) -> tuple[torch.Tensor, ...]:
        """Return all six fixed-scale Q/K payloads and optional scale pages."""
        if not self.pure_qk_single_quant:
            raise RuntimeError(
                "the aligned Q/K layouts are adaptive; request "
                "pure_qk_single_quant=True from the projection epilogue"
            )
        return (
            self.backward.q_fp4,
            self.backward.score_q_fp4,
            self.backward.k_fp4,
            self.backward.score_k_fp4,
            *self.pure_qk_backward_operands(),
        )

    def hybrid_fp8_backward_operand(self) -> torch.Tensor:
        """Return projection-native E4M3 V for the retained hybrid backward."""
        if self.v_backward_fp8 is None:
            raise RuntimeError(
                "the QKV projection did not publish the hybrid FP8 V operand"
            )
        return self.v_backward_fp8

    def mxfp4_backward_v_operands(
        self,
        *,
        required_scale_policy: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return packed MXFP4 V and its explicitly tagged scale pages."""
        if not self.v_backward_fp4.numel() or not self.v_backward_scales.numel():
            raise RuntimeError(
                "the QKV projection did not publish an MXFP4 dP V operand"
            )
        if (
            required_scale_policy is not None
            and self.v_backward_mxfp4_scale_policy != required_scale_policy
        ):
            raise RuntimeError(
                "the QKV projection MXFP4 dP V scale policy does not match "
                f"the selected consumer: produced "
                f"{self.v_backward_mxfp4_scale_policy!r}, required "
                f"{required_scale_policy!r}"
            )
        return self.v_backward_fp4, self.v_backward_scales


@dataclass(frozen=True, slots=True, weakref_slot=True)
class B300E4M3QKVForwardWorkspace:
    """Caller-owned publications for one paired-D64 or native-D128 projection.

    Slots 4/6 and 8--13 are common with the legacy 24-publication ABI; slot 23
    is the exact-FP8 feature-major V publication, and slots 20--22 are the
    normal-order E4M3 V/Q/K operands retained by autograd for backward.  Both
    route-specific forward V owners and all three route-neutral backward owners
    are retained so switching the bound runtime never changes allocator
    topology.  The typed Q/K aliases and empty sentinels are constructed once
    with the layer rather than recreated in a timed forward.

    Tensor metadata and storage addresses are part of the private unchecked
    launch contract and must not be mutated after construction.
    """

    q_payload: torch.Tensor
    k_payload: torch.Tensor
    q_scale_pages: torch.Tensor
    q_global_scale: torch.Tensor
    k_scale_pages: torch.Tensor
    k_global_scale: torch.Tensor
    v_mxfp4_payload: torch.Tensor
    v_mxfp4_scale_pages: torch.Tensor
    v_fp8_payload: torch.Tensor
    v_backward_fp8: torch.Tensor
    q_backward_fp8: torch.Tensor
    k_backward_fp8: torch.Tensor
    q_payload_fp4: torch.Tensor
    k_payload_fp4: torch.Tensor
    empty_bf16: torch.Tensor
    empty_byte: torch.Tensor
    empty_fp8: torch.Tensor
    empty_fp4: torch.Tensor
    v_backward_mxfp4: torch.Tensor | None = None
    v_backward_mxfp4_scale_pages: torch.Tensor | None = None

    def __post_init__(self) -> None:
        typed_tensors = (
            ("q_payload", self.q_payload, torch.uint8),
            ("k_payload", self.k_payload, torch.uint8),
            ("q_scale_pages", self.q_scale_pages, torch.float8_e4m3fn),
            ("q_global_scale", self.q_global_scale, torch.float32),
            ("k_scale_pages", self.k_scale_pages, torch.float8_e4m3fn),
            ("k_global_scale", self.k_global_scale, torch.float32),
            (
                "v_mxfp4_payload",
                self.v_mxfp4_payload,
                torch.float4_e2m1fn_x2,
            ),
            (
                "v_mxfp4_scale_pages",
                self.v_mxfp4_scale_pages,
                torch.float8_e4m3fn,
            ),
            ("v_fp8_payload", self.v_fp8_payload, torch.float8_e4m3fn),
            ("v_backward_fp8", self.v_backward_fp8, torch.float8_e4m3fn),
            ("q_backward_fp8", self.q_backward_fp8, torch.float8_e4m3fn),
            ("k_backward_fp8", self.k_backward_fp8, torch.float8_e4m3fn),
            ("q_payload_fp4", self.q_payload_fp4, torch.float4_e2m1fn_x2),
            ("k_payload_fp4", self.k_payload_fp4, torch.float4_e2m1fn_x2),
            ("empty_bf16", self.empty_bf16, torch.bfloat16),
            ("empty_byte", self.empty_byte, torch.uint8),
            ("empty_fp8", self.empty_fp8, torch.float8_e4m3fn),
            ("empty_fp4", self.empty_fp4, torch.float4_e2m1fn_x2),
        )
        device = self.q_payload.device
        for name, tensor, dtype in typed_tensors:
            if (
                tensor.dtype != dtype
                or not tensor.is_cuda
                or not tensor.is_contiguous()
                or tensor.device != device
            ):
                raise ValueError(
                    f"{name} must be contiguous CUDA {dtype} on {device}"
                )
        if (self.v_backward_mxfp4 is None) != (
            self.v_backward_mxfp4_scale_pages is None
        ):
            raise ValueError(
                "v_backward_mxfp4 and its scale pages must be supplied together"
            )
        for name, tensor in (
            ("v_backward_mxfp4", self.v_backward_mxfp4),
            (
                "v_backward_mxfp4_scale_pages",
                self.v_backward_mxfp4_scale_pages,
            ),
        ):
            if tensor is not None and (
                tensor.dtype != torch.uint8
                or not tensor.is_cuda
                or not tensor.is_contiguous()
                or tensor.device != device
            ):
                raise ValueError(
                    f"{name} must be contiguous CUDA uint8 on {device}"
                )
        for name, sentinel in (
            ("empty_bf16", self.empty_bf16),
            ("empty_byte", self.empty_byte),
            ("empty_fp8", self.empty_fp8),
            ("empty_fp4", self.empty_fp4),
        ):
            if sentinel.numel() != 0:
                raise ValueError(f"{name} must be a reusable zero-element tensor")
        for name, owner, alias in (
            ("Q", self.q_payload, self.q_payload_fp4),
            ("K", self.k_payload, self.k_payload_fp4),
        ):
            if (
                tuple(alias.shape) != tuple(owner.shape)
                or alias.data_ptr() != owner.data_ptr()
            ):
                raise ValueError(
                    f"cached {name} FP4 alias must share the payload storage "
                    "and shape"
                )

    def compact_outputs(self) -> tuple[torch.Tensor, ...]:
        """Return compact ABI outputs in their append-compatible call order."""
        return (
            self.q_payload,
            self.k_payload,
            self.q_scale_pages,
            self.q_global_scale,
            self.k_scale_pages,
            self.k_global_scale,
            self.v_mxfp4_payload,
            self.v_mxfp4_scale_pages,
            self.v_fp8_payload,
            self.v_backward_fp8,
            self.q_backward_fp8,
            self.k_backward_fp8,
        )

    def compact_mx_backward_v_outputs(self) -> tuple[torch.Tensor, ...]:
        """Append caller-owned MX backward V storage to the stable 12 slots."""
        if (
            self.v_backward_mxfp4 is None
            or self.v_backward_mxfp4_scale_pages is None
        ):
            raise RuntimeError(
                "this forward workspace has no MXFP4 backward V storage"
            )
        return (
            *self.compact_outputs(),
            self.v_backward_mxfp4,
            self.v_backward_mxfp4_scale_pages,
        )


def _b300_typed_fp4_alias(payload: torch.Tensor) -> torch.Tensor:
    """Create one descriptor-native FP4 alias while assembling a bundle."""
    if payload.dtype == torch.float4_e2m1fn_x2:
        return payload
    return payload.view(torch.float4_e2m1fn_x2)


@dataclass(frozen=True)
class B300UnifiedLowpDout:
    """Output-projection result plus producer-native MXFP4 backward inputs."""

    dout: torch.Tensor | None
    # Shape-matched storage retained for native backward descriptor plumbing.
    # It is intentionally uninitialized when ``dout`` is None; producer-native
    # consumers read ``dout_backward_fp8`` instead.
    dout_storage: torch.Tensor
    dout_dp_fp4: torch.Tensor
    dout_dp_scales: torch.Tensor
    dout_dv_fp4: torch.Tensor
    dout_dv_scales: torch.Tensor
    dpsum: torch.Tensor
    lse_log2: torch.Tensor
    dout_backward_fp8: torch.Tensor | None

    def backward_operands(self) -> tuple[torch.Tensor, ...]:
        """Return the dO payload/scale/statistics suffix consumed by backward."""
        return (
            self.dout_dp_fp4,
            self.dout_dp_scales,
            self.dout_dv_fp4,
            self.dout_dv_scales,
            self.dpsum,
            self.lse_log2,
        )

    def hybrid_fp8_backward_operands(self) -> tuple[torch.Tensor, ...]:
        """Return projection-native E4M3 dO and softmax statistics."""
        if self.dout_backward_fp8 is None:
            raise RuntimeError(
                "the dO projection did not publish the hybrid FP8 operand"
            )
        return self.dout_backward_fp8, self.dpsum, self.lse_log2


@dataclass(frozen=True)
class B300V509E5M2DoutPublication:
    """Exact fused dO publication consumed only by v509 native score."""

    # Descriptor plumbing reuses attention_output and never stores BF16 dO.
    dout_storage: torch.Tensor
    dpsum: torch.Tensor
    lse_log2: torch.Tensor
    dout_backward_e5m2: torch.Tensor

    def backward_operands(self) -> tuple[torch.Tensor, ...]:
        return self.dout_backward_e5m2, self.dpsum, self.lse_log2


@dataclass(frozen=True)
class B300V510E5M2DoutPublication:
    """Exact fused dO publication consumed only by v510 dense score."""

    # Descriptor plumbing reuses attention_output and never stores BF16 dO.
    dout_storage: torch.Tensor
    dpsum: torch.Tensor
    lse_log2: torch.Tensor
    dout_backward_e5m2: torch.Tensor

    def backward_operands(self) -> tuple[torch.Tensor, ...]:
        return self.dout_backward_e5m2, self.dpsum, self.lse_log2


def _b300_adaptive_producer_stream(device: torch.device) -> torch.cuda.Stream:
    index = torch.cuda.current_device() if device.index is None else device.index
    stream = _B300_ADAPTIVE_PRODUCER_STREAMS.get(index)
    if stream is None:
        with torch.cuda.device(index):
            stream = torch.cuda.Stream(device=torch.device("cuda", index))
        _B300_ADAPTIVE_PRODUCER_STREAMS[index] = stream
    return stream


def _ensure_backward_extension() -> None:
    if _C is not None:
        return
    build_hint = Path(__file__).resolve().parent
    raise ImportError(
        f"tk_fa4 backward extension is not built. Run `make` in {build_hint}."
    ) from _BWD_IMPORT_ERROR


def _ensure_forward_extensions() -> None:
    if _C_b300_causal_bf16_baseline is None or _C_b300_noncausal is None:
        build_hint = Path(__file__).resolve().parent
        missing = []
        if _C_b300_causal_bf16_baseline is None:
            missing.append("causal_bf16_baseline")
        if _C_b300_noncausal is None:
            missing.append("noncausal")
        raise ImportError(
            f"tk_fa4 forward extension(s) missing for {', '.join(missing)}. Run `make` in {build_hint}."
        ) from (_CAUSAL_BF16_IMPORT_ERROR or _NONCAUSAL_IMPORT_ERROR)


def _ensure_lowp_bwd_extension() -> None:
    if _C_b300_lowp_bwd is not None:
        return
    build_hint = Path(__file__).resolve().parent / "lowp_fa4_bwd"
    raise ImportError(
        f"tk_fa4 low-precision backward extension is not built. Run `make` in {build_hint}."
    ) from _LOWP_BWD_IMPORT_ERROR


def _check_cuda_bf16_bshd(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16")
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape (batch, seqlen, heads, head_dim)")


def _check_cuda_bf16_bhsd(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16")
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape (batch, heads, seqlen, head_dim)")


def _check_sm100(device: torch.device) -> None:
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        raise RuntimeError(
            f"tk_fa4 exact B300 path requires GB200 / SM100, got compute capability {(major, minor)}"
        )


def _check_exact_qkv_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    _check_cuda_bf16_bshd(q, "q")
    _check_cuda_bf16_bshd(k, "k")
    _check_cuda_bf16_bshd(v, "v")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("batch dimensions must match")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("sequence lengths must match")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise ValueError("exact B300 path requires equal q, k, and v head counts")
    if q.shape[3] != _QK_HEAD_DIM or k.shape[3] != _QK_HEAD_DIM:
        raise ValueError(f"q and k head_dim must be {_QK_HEAD_DIM}")
    if v.shape[3] != _V_HEAD_DIM:
        raise ValueError(f"v head_dim must be {_V_HEAD_DIM}")
    if q.shape[1] < _MIN_SEQ_LEN:
        raise ValueError(f"exact B300 path requires seqlen >= {_MIN_SEQ_LEN}")
    _check_sm100(q.device)


def _check_adaptive_lowp_operands(
    q: torch.Tensor,
    k: torch.Tensor,
    operands: B300AdaptiveLowpOperands,
) -> None:
    _check_cuda_bf16_bshd(q, "q")
    _check_cuda_bf16_bshd(k, "k")
    if q.shape != k.shape or q.device != k.device:
        raise ValueError("adaptive lowp Q and K must have identical shapes and devices")
    if q.shape[-1] != _QK_HEAD_DIM:
        raise ValueError(f"adaptive lowp Q/K head_dim must be {_QK_HEAD_DIM}")

    batch, seqlen, heads, depth = q.shape
    byte_layouts = (
        ("q_fp4", operands.q_fp4, (batch, heads, depth, seqlen)),
        ("score_q_fp4", operands.score_q_fp4, (batch, heads, seqlen, depth // 2)),
        ("k_fp4", operands.k_fp4, (batch, seqlen, heads, depth)),
        ("score_k_fp4", operands.score_k_fp4, (batch, heads, seqlen, depth // 2)),
    )
    for name, tensor, expected_shape in byte_layouts:
        if tensor.dtype != torch.uint8 or tensor.device != q.device:
            raise ValueError(f"{name} must be a CUDA uint8 tensor on the Q/K device")
        if tuple(tensor.shape) != expected_shape or not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous with shape {expected_shape}, got {tuple(tensor.shape)}"
            )
    scales = operands.qk_scales
    if (
        scales.dtype != torch.float32
        or scales.device != q.device
        or not scales.is_contiguous()
        or tuple(scales.shape) != (batch, heads, 7)
    ):
        raise ValueError(
            "qk_scales must be contiguous CUDA float32 with shape "
            f"{(batch, heads, 7)}"
        )
    if operands.mixed_v is not None:
        expected_mixed_v = (batch, seqlen, heads, _V_HEAD_DIM)
        if (
            operands.mixed_v.dtype != torch.float8_e4m3fn
            or operands.mixed_v.device != q.device
            or not operands.mixed_v.is_contiguous()
            or tuple(operands.mixed_v.shape) != expected_mixed_v
        ):
            raise ValueError(
                "mixed_v must be contiguous E4M3 with shape "
                f"{expected_mixed_v} on the Q/K device"
            )


def _check_exact_out(x: torch.Tensor, reference: torch.Tensor, name: str) -> None:
    _check_cuda_bf16_bshd(x, name)
    if x.shape != reference.shape:
        raise ValueError(f"{name} must match v/out shape")
    if x.device != reference.device:
        raise ValueError(f"{name} must be on the same CUDA device as q")


def _normalize_lse_bsh(lse: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if not lse.is_cuda:
        raise ValueError("lse must be a CUDA tensor")
    if lse.dtype != torch.float32:
        raise ValueError("lse must be float32")
    if lse.device != q.device:
        raise ValueError("lse must be on the same CUDA device as q")
    if lse.shape == q.shape[:3]:
        return lse.contiguous()
    if lse.ndim == 3 and lse.shape == (q.shape[0], q.shape[2], q.shape[1]):
        return lse.permute(0, 2, 1).contiguous()
    raise ValueError("lse must have shape (batch, seqlen, heads)")


def _pad_bshd(x: torch.Tensor, multiple: int = _PAD_MULTIPLE) -> torch.Tensor:
    pad = (-x.shape[1]) % multiple
    if pad == 0:
        return x.contiguous()
    pad_tensor = torch.zeros(
        (x.shape[0], pad, x.shape[2], x.shape[3]),
        dtype=x.dtype,
        device=x.device,
    )
    return torch.cat((x, pad_tensor), dim=1).contiguous()


def _pad_bsh(x: torch.Tensor, value: float = 0.0, multiple: int = _PAD_MULTIPLE) -> torch.Tensor:
    pad = (-x.shape[1]) % multiple
    if pad == 0:
        return x.contiguous()
    pad_tensor = torch.full(
        (x.shape[0], pad, x.shape[2]),
        fill_value=value,
        dtype=x.dtype,
        device=x.device,
    )
    return torch.cat((x, pad_tensor), dim=1).contiguous()


def _to_bhsd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def _from_bhsd(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1, 3).contiguous()


def _maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


_V382_ADVANCED_LONG_HEADS = {
    8192: frozenset((8, 16)),
    16384: frozenset((4, 8, 16, 32, 64, 128)),
    32768: frozenset((16, 32, 64, 128)),
    65536: frozenset((16, 32, 64, 128)),
}


def _select_hot_backward_kernel(batch: int, seqlen: int, heads: int):
    supported_heads = _V382_ADVANCED_LONG_HEADS.get(seqlen)
    if batch == 1 and supported_heads is not None and heads in supported_heads:
        advanced_kernel = getattr(
            _C,
            f"b300_mha_bwd_hot_cute16_candidate_s{seqlen}_v382_advanced_long_internal",
            None,
        )
        if advanced_kernel is not None:
            return advanced_kernel
    return _C.b300_mha_bwd_hot_cute16_candidate_internal


def _try_hot_fastpath(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float | None,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not causal or deterministic:
        return None
    if not (q.is_cuda and k.is_cuda and v.is_cuda and out.is_cuda and lse.is_cuda and dout.is_cuda):
        return None
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
        or dout.dtype != torch.bfloat16
        or lse.dtype != torch.float32
    ):
        return None
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or out.ndim != 4 or dout.ndim != 4 or lse.ndim != 3:
        return None
    if q.device != k.device or q.device != v.device or q.device != out.device or q.device != lse.device or q.device != dout.device:
        return None
    q_shape = q.shape
    v_shape = v.shape
    if q_shape != k.shape or out.shape != v_shape or dout.shape != v_shape:
        return None
    batch, seqlen, heads, q_dim = q_shape
    if (v_shape[0], v_shape[1], v_shape[2]) != (batch, seqlen, heads):
        return None
    if q_dim != _QK_HEAD_DIM or v_shape[3] != _V_HEAD_DIM:
        return None
    if seqlen < _MIN_SEQ_LEN or seqlen % _EXPERIMENTAL_PAD_MULTIPLE != 0:
        return None
    if lse.shape != q_shape[:3]:
        return None
    default_scale = q_dim ** -0.5
    if softmax_scale is not None and not math.isclose(float(softmax_scale), float(default_scale), rel_tol=0.0, abs_tol=1e-7):
        return None
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and out.is_contiguous() and lse.is_contiguous() and dout.is_contiguous()):
        return None
    hot_backward_kernel = _select_hot_backward_kernel(batch, seqlen, heads)
    return hot_backward_kernel(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        float(default_scale if softmax_scale is None else softmax_scale),
        seqlen,
        deterministic,
    )


def _resolve_scale(q: torch.Tensor, softmax_scale: float | None) -> float:
    default_scale = q.shape[-1] ** -0.5
    if softmax_scale is None:
        return float(default_scale)
    if not math.isclose(float(softmax_scale), float(default_scale), rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(
            f"exact B300 path only supports softmax_scale={default_scale} for head_dim={q.shape[-1]}"
        )
    return float(softmax_scale)


def _select_forward_kernel(causal: bool, seqlen: int):
    if not causal:
        return _C_b300_noncausal.forward
    if seqlen <= _CAUSAL_PERSISTENT_MAX_SEQ:
        return _C_b300_causal_bf16_baseline.forward_persistent
    return _C_b300_causal_bf16_baseline.forward


def _experimental_hot_supported(seqlen: int, causal: bool, deterministic: bool) -> bool:
    return causal and (not deterministic) and seqlen % _EXPERIMENTAL_PAD_MULTIPLE == 0


def _resolve_experimental_impl(
    implementation: str,
    seqlen: int,
    causal: bool,
    deterministic: bool,
) -> str:
    if implementation not in {"auto", "ref", "hot"}:
        raise ValueError("implementation must be one of 'auto', 'ref', or 'hot'")
    if implementation == "ref":
        return "ref"
    if implementation == "hot":
        if not _experimental_hot_supported(seqlen, causal, deterministic):
            raise ValueError(
                "CuTe16 hot mode not implemented yet; current stage only supports causal=True, deterministic=False with seqlen divisible by 256"
            )
        return "hot"
    return "ref"


def _lse_to_l_aux(lse: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    l_aux = (-lse) / softmax_scale
    l_aux_pad = _pad_bsh(l_aux)
    return l_aux_pad.permute(0, 2, 1).contiguous().unsqueeze(2)


def _lse_to_bh1s(lse: torch.Tensor, multiple: int) -> torch.Tensor:
    return _pad_bsh(lse, multiple=multiple).permute(0, 2, 1).contiguous().unsqueeze(2)


def b300_adaptive_lowp_operands_from_projection(
    q: torch.Tensor,
    k: torch.Tensor,
    q_fp4: torch.Tensor,
    score_q_fp4: torch.Tensor,
    k_fp4: torch.Tensor,
    score_k_fp4: torch.Tensor,
    qk_scales: torch.Tensor,
    *,
    mixed_v: torch.Tensor | None = None,
) -> B300AdaptiveLowpOperands:
    """Wrap layouts emitted directly by an upstream Q/K projection.

    This performs validation only: the supplied tensors are retained without
    copies or quantization, so a projection epilogue can publish the backward
    operands at no additional launch cost.
    """
    operands = B300AdaptiveLowpOperands(
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        qk_scales,
        mixed_v,
    )
    _check_adaptive_lowp_operands(q, k, operands)
    return operands


def b300_adaptive_lowp_operands_from_scales(
    q: torch.Tensor,
    k: torch.Tensor,
    qk_scales: torch.Tensor,
    *,
    mixed_v: torch.Tensor | None = None,
) -> B300AdaptiveLowpOperands:
    """Pack Q/K using adaptive metadata already produced upstream."""
    _ensure_lowp_bwd_extension()
    _check_cuda_bf16_bshd(q, "q")
    _check_cuda_bf16_bshd(k, "k")
    if q.shape != k.shape or q.device != k.device:
        raise ValueError("adaptive lowp Q and K must have identical shapes and devices")
    packed = _C_b300_lowp_bwd.quantize_fp4_dual_qk_precomputed_scales(
        q,
        k,
        qk_scales,
    )
    operands = B300AdaptiveLowpOperands(*packed, mixed_v=mixed_v)
    _check_adaptive_lowp_operands(q, k, operands)
    return operands


def b300_prepare_nvfp4_projection_operand(
    tensor: torch.Tensor,
    *,
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a BF16 activation with row-by-K16 NVFP4 scaling.

    This helper is for activations, not learned weights.  Learned projection
    weights must use :func:`b300_prepare_nvfp4_projection_weight` so their
    forward and physical-transpose paths share true 16x16 scale blocks.
    Preparing an activation here is a functional fallback; an upstream
    producer should emit this packed representation directly when possible.
    """
    _ensure_lowp_bwd_extension()
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        raise ValueError("tensor must be a two-dimensional CUDA BF16 matrix")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if tensor.shape[0] % 128 or tensor.shape[1] % 128:
        raise ValueError("both matrix dimensions must be divisible by 128")
    if global_scale is None:
        packed = _C_b300_lowp_bwd.quantize_nvfp4_projection_operand(tensor)
    else:
        if (
            global_scale.dtype != torch.float32
            or not global_scale.is_cuda
            or global_scale.device != tensor.device
            or not global_scale.is_contiguous()
            or global_scale.numel() != 1
        ):
            raise ValueError(
                "global_scale must be one contiguous CUDA float32 value on "
                "the tensor device"
            )
        packed = (
            _C_b300_lowp_bwd.
            quantize_nvfp4_projection_operand_precomputed_scale(
                tensor,
                global_scale,
            )
        )
    return tuple(packed)


def b300_prepare_nvfp4_projection_operand_rmsnorm(
    tensor: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Fuse RMSNorm with exact-dynamic native NVFP4 activation packing.

    The first three results have the identical payload, row-by-K16 scale-page,
    and global-decode ABI returned by
    :func:`b300_prepare_nvfp4_projection_operand`. ``inv_rms`` is retained for
    backward, and the final BF16 publication preserves the existing QKV
    weight-gradient input while this experimental path is validated.
    """
    _ensure_lowp_bwd_extension()
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        raise ValueError("tensor must be a two-dimensional CUDA BF16 matrix")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if tensor.shape[0] <= 0 or tensor.shape[0] % 128 or tensor.shape[1] != 2048:
        raise ValueError(
            "tensor must have positive M divisible by 128 and K=2048"
        )
    if (
        gamma.ndim != 1
        or gamma.dtype != torch.bfloat16
        or not gamma.is_cuda
        or not gamma.is_contiguous()
        or gamma.device != tensor.device
        or gamma.numel() != tensor.shape[1]
    ):
        raise ValueError(
            "gamma must be contiguous CUDA BF16 [K] on the tensor device"
        )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    prepared = tuple(
        _C_b300_lowp_bwd.quantize_nvfp4_projection_operand_rmsnorm(
            tensor,
            gamma,
            float(epsilon),
        )
    )
    if len(prepared) != 5:
        raise RuntimeError(
            "fused RMSNorm NVFP4 preparation must return five tensors"
        )
    rows, columns = tensor.shape
    expected = (
        ((rows, columns // 2), torch.float4_e2m1fn_x2),
        ((rows // 128, columns // 64, 512), torch.float8_e4m3fn),
        ((1,), torch.float32),
        ((rows,), torch.float32),
        ((rows, columns), torch.bfloat16),
    )
    for index, (output, (shape, dtype)) in enumerate(
        zip(prepared, expected, strict=True)
    ):
        if (
            tuple(output.shape) != shape
            or output.dtype != dtype
            or output.device != tensor.device
            or not output.is_contiguous()
        ):
            raise RuntimeError(
                "fused RMSNorm NVFP4 preparation returned invalid tensor "
                f"{index}: {tuple(output.shape)}, {output.dtype}, "
                f"contiguous={output.is_contiguous()} on {output.device}; "
                f"expected {shape}, {dtype}, contiguous on {tensor.device}"
            )
    return prepared


def b300_rmsnorm_backward(
    tensor: torch.Tensor,
    gamma: torch.Tensor,
    inv_rms: torch.Tensor,
    gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact RMSNorm ``dx`` and ``dgamma`` without FP32 tensors.

    This exact-shape experimental path consumes the forward ``inv_rms`` and
    returns contiguous BF16 gradients matching ``tensor`` and ``gamma``.
    """
    _ensure_lowp_bwd_extension()
    if (
        tensor.ndim != 2
        or tensor.dtype != torch.bfloat16
        or not tensor.is_cuda
        or not tensor.is_contiguous()
        or tensor.shape[1] != 2048
        or tensor.shape[0] % 16
    ):
        raise ValueError(
            "tensor must be contiguous CUDA BF16 [M, 2048] with M divisible "
            "by 16"
        )
    if (
        gradient.shape != tensor.shape
        or gradient.dtype != torch.bfloat16
        or gradient.device != tensor.device
        or not gradient.is_contiguous()
    ):
        raise ValueError(
            "gradient must match the contiguous CUDA BF16 tensor"
        )
    if (
        gamma.shape != (tensor.shape[1],)
        or gamma.dtype != torch.bfloat16
        or gamma.device != tensor.device
        or not gamma.is_contiguous()
    ):
        raise ValueError(
            "gamma must be contiguous CUDA BF16 [2048] on the tensor device"
        )
    if (
        inv_rms.shape != (tensor.shape[0],)
        or inv_rms.dtype != torch.float32
        or inv_rms.device != tensor.device
        or not inv_rms.is_contiguous()
    ):
        raise ValueError(
            "inv_rms must be contiguous CUDA FP32 [M] on the tensor device"
        )
    gradients = tuple(
        _C_b300_lowp_bwd.rmsnorm_backward_bf16(
            tensor,
            gamma,
            inv_rms,
            gradient,
        )
    )
    expected = (
        (tensor.shape, torch.bfloat16),
        (gamma.shape, torch.bfloat16),
    )
    if len(gradients) != len(expected):
        raise RuntimeError("fused RMSNorm backward must return two tensors")
    for index, (output, (shape, dtype)) in enumerate(
        zip(gradients, expected, strict=True)
    ):
        if (
            output.shape != shape
            or output.dtype != dtype
            or output.device != tensor.device
            or not output.is_contiguous()
        ):
            raise RuntimeError(
                "fused RMSNorm backward returned invalid tensor "
                f"{index}: {tuple(output.shape)}, {output.dtype}, "
                f"contiguous={output.is_contiguous()} on {output.device}; "
                f"expected {tuple(shape)}, {dtype}, contiguous on "
                f"{tensor.device}"
            )
    return gradients


def b300_prepare_e4m3_projection_operand(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize projection activations with one E4M3 scale per row.

    This is the functional preparation fallback for the projection-native
    dense-E4M3 kernel.  A production integration should emit the returned
    payload and decode vector from the upstream normalization epilogue.
    """
    _ensure_lowp_bwd_extension()
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        raise ValueError("tensor must be a two-dimensional CUDA BF16 matrix")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if tensor.shape[0] % 128 or tensor.shape[1] % 128:
        raise ValueError("both matrix dimensions must be divisible by 128")
    return tuple(
        _C_b300_lowp_bwd.quantize_e4m3_projection_operand(tensor)
    )


def b300_prepare_e4m3_projection_weight(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a learned projection weight per output channel.

    ``tensor`` retains the standard PyTorch ``[N,K]`` linear-weight layout.
    The channelwise decode vector is applied to the corresponding accumulator
    columns before RoPE and low-precision attention-operand publication.
    Immutable weights may be prepared once.  Training integrations must
    prepare the current weight after every optimizer update; caching by tensor
    identity is unsafe because optimizers mutate parameters in place.
    """
    _ensure_lowp_bwd_extension()
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        raise ValueError("tensor must be a two-dimensional CUDA BF16 matrix")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if tensor.shape[0] % 128 or tensor.shape[1] % 128:
        raise ValueError("both matrix dimensions must be divisible by 128")
    return tuple(
        _C_b300_lowp_bwd.quantize_e4m3_projection_weight(tensor)
    )


def b300_prepare_nvfp4_projection_weight(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare a learned projection weight with 16x16 NVFP4 scaling.

    The hardware consumes one scale per 1x16 group, but every group in a
    16-row block receives the same E4M3 scale.  This makes quantization
    transpose-consistent between forward and input-gradient projections.
    Activations should continue to use
    :func:`b300_prepare_nvfp4_projection_operand` and its 1x16 scaling.
    A cached result is valid only for that exact weight version and must be
    refreshed after an in-place optimizer update.
    """
    _ensure_lowp_bwd_extension()
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        raise ValueError("tensor must be a two-dimensional CUDA BF16 matrix")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if tensor.shape[0] % 128 or tensor.shape[1] % 128:
        raise ValueError("both matrix dimensions must be divisible by 128")
    return tuple(
        _C_b300_lowp_bwd.quantize_nvfp4_projection_weight(tensor)
    )


def b300_prepare_nvfp4_projection_weight_dual(
    tensor: torch.Tensor,
) -> B300DualNVFP4ProjectionWeight:
    """Prepare a learned weight and its exact physical transpose once.

    This is restricted to true 16x16 learned-weight scaling.  The extension
    quantizes the current BF16 ``[N,K]`` value once, then transposes its FP4
    codes and 16x16 scale-tile grid into the ``[K,N]`` operand consumed by
    input-gradient projection.  The global decode scale is shared bitwise.
    """
    _ensure_lowp_bwd_extension()
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        raise ValueError("tensor must be a two-dimensional CUDA BF16 matrix")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if tensor.shape[0] % 128 or tensor.shape[1] % 128:
        raise ValueError("both matrix dimensions must be divisible by 128")
    prepared = tuple(
        _C_b300_lowp_bwd.quantize_nvfp4_projection_weight_dual(tensor)
    )
    if len(prepared) != 6:
        raise RuntimeError(
            "dual NVFP4 weight preparation must return six tensors"
        )
    forward = prepared[:3]
    transpose = prepared[3:]
    rows, columns = tensor.shape
    expected_shapes = (
        (rows, columns // 2),
        (rows // 128, columns // 64, 512),
        (1,),
        (columns, rows // 2),
        (columns // 128, rows // 64, 512),
        (1,),
    )
    expected_dtypes = (
        torch.float4_e2m1fn_x2,
        torch.float8_e4m3fn,
        torch.float32,
        torch.float4_e2m1fn_x2,
        torch.float8_e4m3fn,
        torch.float32,
    )
    for index, (output, expected_shape, expected_dtype) in enumerate(
        zip(prepared, expected_shapes, expected_dtypes, strict=True)
    ):
        if (
            tuple(output.shape) != expected_shape
            or output.device != tensor.device
            or output.dtype != expected_dtype
            or not output.is_contiguous()
        ):
            raise RuntimeError(
                "dual NVFP4 weight preparation returned invalid tensor "
                f"{index}: {tuple(output.shape)}, {output.dtype}, "
                f"contiguous={output.is_contiguous()} on {output.device}; "
                f"expected {expected_shape}, {expected_dtype}, contiguous "
                f"on {tensor.device}"
            )
    if forward[2].data_ptr() != transpose[2].data_ptr():
        raise RuntimeError(
            "dual NVFP4 weight layouts must share one global decode scale"
        )
    return B300DualNVFP4ProjectionWeight(
        forward=forward,
        transpose=transpose,
    )


def _b300_require_disjoint_weight_publications(
    inputs: tuple[tuple[str, torch.Tensor], ...],
    outputs: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    """Reject overlapping byte ranges before a caller-owned native launch."""

    def byte_range(tensor: torch.Tensor) -> tuple[int, int]:
        return tensor.data_ptr(), tensor.numel() * tensor.element_size()

    def ranges_overlap(
        left: tuple[int, int],
        right: tuple[int, int],
    ) -> bool:
        left_begin, left_bytes = left
        right_begin, right_bytes = right
        if left_begin <= right_begin:
            return right_begin - left_begin < left_bytes
        return left_begin - right_begin < right_bytes

    output_ranges = tuple(
        (name, byte_range(tensor)) for name, tensor in outputs
    )
    for index, (left_name, left_range) in enumerate(output_ranges):
        for right_name, right_range in output_ranges[index + 1 :]:
            if ranges_overlap(left_range, right_range):
                raise ValueError(
                    f"{left_name} and {right_name} must use disjoint storage"
                )
        for input_name, input_tensor in inputs:
            if ranges_overlap(left_range, byte_range(input_tensor)):
                raise ValueError(
                    f"{input_name} and {left_name} must use disjoint storage"
                )


def b300_prepare_nvfp4_projection_weight_dual_out(
    tensor: torch.Tensor,
    forward_packed: torch.Tensor,
    forward_scales: torch.Tensor,
    backward_packed: torch.Tensor,
    backward_scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    checked: bool = True,
    authenticate: bool = False,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Publish true-2D NVFP4 ``W`` and ``W.T`` into owned storage.

    After a shared global-amax reduction, one tiled quantization/publication
    pass writes both physical GEMM orientations directly into the supplied
    tensors.  The two operands share ``global_scale``.  ``authenticate=True``
    is a first-use preflight against independent exact-2D preparations; a
    shape-bound hot path may select ``checked=False`` only after that preflight
    succeeds.  Neither native route allocates output tensors.
    """
    if authenticate and not checked:
        raise ValueError("bitwise authentication requires the checked path")
    inputs = (("input", tensor),)
    outputs = (
        ("forward_packed", forward_packed),
        ("forward_scales", forward_scales),
        ("backward_packed", backward_packed),
        ("backward_scales", backward_scales),
        ("global_scale", global_scale),
    )
    if checked:
        _b300_require_disjoint_weight_publications(inputs, outputs)
    _ensure_lowp_bwd_extension()
    suffix = "" if checked else "_unchecked"
    symbol = "quantize_nvfp4_projection_weight_dual_out" + suffix
    function = getattr(_C_b300_lowp_bwd, symbol, None)
    if function is None:
        extension_path = getattr(_C_b300_lowp_bwd, "__file__", "<unknown>")
        raise RuntimeError(
            "the selected low-precision extension "
            f"{extension_path!r} does not provide required caller-owned "
            f"dual-weight preparation symbol {symbol!r}"
        )
    function(
        tensor,
        forward_packed,
        forward_scales,
        backward_packed,
        backward_scales,
        global_scale,
    )
    forward = (forward_packed, forward_scales, global_scale)
    backward = (backward_packed, backward_scales, global_scale)
    if authenticate:
        references = (
            (
                "forward",
                forward,
                tuple(b300_prepare_nvfp4_projection_weight(tensor)),
            ),
            (
                "backward",
                backward,
                tuple(
                    b300_prepare_nvfp4_projection_weight(
                        tensor.T.contiguous()
                    )
                ),
            ),
        )
        for orientation, actual, reference in references:
            for field, actual_tensor, reference_tensor in zip(
                ("payload", "scales", "global scale"),
                actual,
                reference,
                strict=True,
            ):
                _b300_require_bitwise_equal(
                    f"{orientation} projection weight {field}",
                    reference_tensor,
                    actual_tensor,
                )
    return forward, backward


def b300_prepare_gqa_d128_qkv_projection_weight_dual_out(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    forward_packed: torch.Tensor,
    forward_scales: torch.Tensor,
    backward_packed: torch.Tensor,
    backward_scales: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    checked: bool = True,
    authenticate: bool = False,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Publish both D128 GQA QKV weight orientations without BF16 packing.

    The source tensors retain canonical PyTorch linear-weight order.  The
    native producer pair-interleaves each D128 Q/K head while quantizing,
    appends canonical V, and writes true-2D NVFP4 forward and physical-
    transpose operands directly into the supplied storage.  Both operands
    share ``global_scale``.

    ``authenticate=True`` is a construction-time preflight: it requires the
    checked symbol and compares every published byte with the established
    pair-interleave, concatenate, and independent true-2D quantization path.
    A shape-bound training path may use ``checked=False`` only after that
    exact storage contract has passed preflight.
    """
    if authenticate and not checked:
        raise ValueError("bitwise authentication requires the checked path")
    inputs = (
        ("Q weight", q_weight),
        ("K weight", k_weight),
        ("V weight", v_weight),
    )
    outputs = (
        ("forward_packed", forward_packed),
        ("forward_scales", forward_scales),
        ("backward_packed", backward_packed),
        ("backward_scales", backward_scales),
        ("global_scale", global_scale),
    )
    if checked:
        _b300_require_disjoint_weight_publications(inputs, outputs)
    _ensure_lowp_bwd_extension()
    suffix = "" if checked else "_unchecked"
    symbol = "quantize_gqa_d128_qkv_projection_weight_dual_out" + suffix
    function = getattr(_C_b300_lowp_bwd, symbol, None)
    if function is None:
        extension_path = getattr(_C_b300_lowp_bwd, "__file__", "<unknown>")
        raise RuntimeError(
            "the selected low-precision extension "
            f"{extension_path!r} does not provide required direct D128 "
            f"dual-weight preparation symbol {symbol!r}"
        )
    function(
        q_weight,
        k_weight,
        v_weight,
        forward_packed,
        forward_scales,
        backward_packed,
        backward_scales,
        global_scale,
    )
    forward = (forward_packed, forward_scales, global_scale)
    backward = (backward_packed, backward_scales, global_scale)
    if authenticate:
        interleaved_q, interleaved_k = (
            b300_pair_interleave_gqa_d128_qk_projection_weights(
                q_weight,
                k_weight,
            )
        )
        physical = b300_stack_gqa_d128_qkv_projection_weights(
            interleaved_q,
            interleaved_k,
            v_weight,
        )
        reference_forward = tuple(
            b300_prepare_nvfp4_projection_weight(physical)
        )
        reference_backward = tuple(
            b300_prepare_nvfp4_projection_weight(physical.T.contiguous())
        )
        for orientation, actual, reference in (
            ("forward", forward, reference_forward),
            ("backward", backward, reference_backward),
        ):
            for field, actual_tensor, reference_tensor in zip(
                ("payload", "scales", "global scale"),
                actual,
                reference,
                strict=True,
            ):
                _b300_require_bitwise_equal(
                    f"D128 QKV {orientation} weight {field}",
                    reference_tensor,
                    actual_tensor,
                )
    return forward, backward


def b300_prepare_nvfp4_projection_operand_scaled(
    tensor: torch.Tensor,
    value_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare NVFP4 while folding a uniform value scale into metadata.

    Dynamic NVFP4 normalization makes a positive uniform multiplier cancel
    from the FP4 payload and block scales.  Only the single global decode
    scalar changes.  Folding the multiplier there avoids materializing a
    scaled BF16 activation while remaining equivalent for power-of-two loss
    scales.
    """
    _ensure_lowp_bwd_extension()
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        raise ValueError("tensor must be a two-dimensional CUDA BF16 matrix")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if tensor.shape[0] % 128 or tensor.shape[1] % 128:
        raise ValueError("both matrix dimensions must be divisible by 128")
    if not math.isfinite(value_scale) or value_scale <= 0.0:
        raise ValueError("value_scale must be finite and positive")
    packed = _C_b300_lowp_bwd.quantize_nvfp4_projection_operand_scaled(
        tensor,
        float(value_scale),
    )
    return tuple(packed)


def b300_prepare_nvfp4_projection_operand_inverse_rope(
    tensor: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    global_scale: torch.Tensor,
    publish_inverse_bf16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse pair-native inverse RoPE into delayed-scale NVFP4 packing.

    ``tensor`` is the interleaved projection-gradient matrix
    ``[B*S, H*512]`` with per-head fields ``[dQ192, dK192, dV128]``.  The
    packed operand always represents inverse-rotated dQ/dK.  When
    ``publish_inverse_bf16`` is true, the same transformed register values are
    also written back to ``tensor`` for projection weight-gradient consumers;
    otherwise the BF16 tensor remains in the rotated attention basis.
    """
    _ensure_lowp_bwd_extension()
    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not tensor.is_cuda:
        raise ValueError("tensor must be a two-dimensional CUDA BF16 matrix")
    if not tensor.is_contiguous():
        raise ValueError("tensor must be contiguous")
    if tensor.shape[0] % 128 or tensor.shape[1] % 512:
        raise ValueError("tensor must have shape [B*S, H*512] with B*S % 128 == 0")
    if (
        global_scale.dtype != torch.float32
        or not global_scale.is_cuda
        or global_scale.device != tensor.device
        or not global_scale.is_contiguous()
        or global_scale.numel() != 1
    ):
        raise ValueError(
            "global_scale must be one contiguous CUDA float32 value on the "
            "tensor device"
        )
    packed = (
        _C_b300_lowp_bwd.
        quantize_nvfp4_projection_operand_precomputed_scale_inverse_rope(
            tensor,
            global_scale,
            rope_cos,
            rope_sin,
            bool(publish_inverse_bf16),
        )
    )
    return tuple(packed)


def b300_project_nvfp4(
    input_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Multiply native NVFP4 activation and linear-weight operands.

    The activation represents ``[M, K]`` and the weight ``[N, K]``; the
    returned BF16 matrix has shape ``[M, N]``. Projection weights should be
    prepared once. Supplying delayed scale state while preparing an activation
    avoids the matrix-wide amax pass and models a projection-native epilogue.
    """
    _ensure_lowp_bwd_extension()
    if len(input_operand) != 3 or len(weight_operand) != 3:
        raise ValueError(
            "NVFP4 operands must contain packed data, block scales, and one "
            "global scale"
        )
    return _C_b300_lowp_bwd.project_nvfp4_generic(
        *input_operand,
        *weight_operand,
    )


def b300_project_e4m3(
    input_operand: tuple[torch.Tensor, torch.Tensor],
    weight_operand: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Multiply rowwise/channelwise E4M3 learned-projection operands.

    The activation represents ``[M, K]`` with one FP32 decode scale per row,
    and the standard linear weight represents ``[N, K]`` with one decode scale
    per output channel. The returned matrix is contiguous BF16 ``[M, N]``.
    Prepare the operands with :func:`b300_prepare_e4m3_projection_operand` and
    :func:`b300_prepare_e4m3_projection_weight`, respectively.
    """
    _ensure_lowp_bwd_extension()
    if len(input_operand) != 2 or len(weight_operand) != 2:
        raise ValueError(
            "E4M3 operands must contain an E4M3 payload and one FP32 decode "
            "vector"
        )
    return _C_b300_lowp_bwd.project_e4m3_generic(
        *input_operand,
        *weight_operand,
    )


def b300_project_gqa_d128_hierarchical_qkv_gradient_nvfp4(
    dq_or_lanes: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    projection_weight_operand: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    gradient_global_scale: torch.Tensor,
    rope_packed: torch.Tensor,
    *,
    return_operand: bool = False,
    dq_decode_scale: float = 1.0,
    dk_decode_scale: float = 1.0,
    dv_decode_scale: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Consume D128 GQA dQ reduction lanes in projection backward.

    ``dq_or_lanes`` is either a materialized ``[B,S,Hq,128]`` control, which
    is authenticated for B=1 or B=2, or the private head-major
    ``[2,1,Hq,S,128]`` workspace produced by hierarchical attention. The
    hierarchical specialization remains B=1 only. Its lanes are folded in
    registers, inverse-RoPE is applied to dQ/dK, and one delayed-scale NVFP4
    ``[dQ|dK|dV]`` operand is projected without publishing standalone dQ.
    """
    _ensure_lowp_bwd_extension()
    if len(projection_weight_operand) != 3:
        raise ValueError(
            "NVFP4 projection weight must contain payload, block scales, "
            "and one global scale"
        )
    if (
        dk.ndim != 4
        or dv.ndim != 4
        or tuple(dk.shape) != tuple(dv.shape)
        or dk.shape[0] <= 0
        or dk.shape[1] <= 0
        or dk.shape[2] <= 0
        or dk.shape[3] != 128
    ):
        raise ValueError(
            "dK/dV must have matching positive [B,S,Hkv,128] shapes"
        )
    batch, sequence, kv_heads, _ = (int(value) for value in dk.shape)
    if dq_or_lanes.ndim == 5:
        if batch != 1:
            raise ValueError(
                "hierarchical D128 GQA projection is authenticated only for "
                "batch 1"
            )
        expected_prefix = (2, batch)
        if (
            tuple(dq_or_lanes.shape[:2]) != expected_prefix
            or dq_or_lanes.shape[3] != sequence
            or dq_or_lanes.shape[4] != 128
        ):
            raise ValueError(
                "hierarchical dQ must have shape [2,1,Hq,S,128] matching "
                "dK/dV"
            )
        q_heads = int(dq_or_lanes.shape[2])
    elif dq_or_lanes.ndim == 4:
        if batch not in (1, 2):
            raise ValueError(
                "materialized D128 GQA projection is authenticated only for "
                "batch 1 or 2"
            )
        if (
            int(dq_or_lanes.shape[0]) != batch
            or int(dq_or_lanes.shape[1]) != sequence
            or int(dq_or_lanes.shape[3]) != 128
        ):
            raise ValueError(
                "materialized dQ must have shape [B,S,Hq,128] matching dK/dV"
            )
        q_heads = int(dq_or_lanes.shape[2])
    else:
        raise ValueError(
            "dQ must be materialized [B,S,Hq,128] or hierarchical "
            "[2,1,Hq,S,128]"
        )
    if q_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError("D128 GQA projection requires Hq divisible by Hkv")
    if batch == 2 and (sequence, q_heads, kv_heads) != (4096, 32, 8):
        raise ValueError(
            "materialized D128 B2 projection is authenticated only for "
            "S4096/Hq32/Hkv8"
        )
    result = (
        _C_b300_lowp_bwd.
        project_gqa_d128_hierarchical_qkv_gradient_nvfp4(
            dq_or_lanes,
            dk,
            dv,
            *projection_weight_operand,
            gradient_global_scale,
            rope_packed,
            float(dq_decode_scale),
            float(dk_decode_scale),
            float(dv_decode_scale),
        )
    )
    if len(result) != 3:
        raise RuntimeError(
            "D128 QKV gradient projection returned an unexpected ABI: "
            f"{len(result)} tensors, expected 3"
        )
    if return_operand:
        return tuple(result)
    return result[0]


def b300_pack_gqa_d64_paired_rope(
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    """Pack D64 RoPE for the adjacent-head D128 physical consumer.

    Each physical D128 tile holds two independent D64 heads, so the 32-pair
    D64 table is repeated across both 64-value halves.  This is static model
    metadata and should be prepared once rather than in the training step.
    """
    if (
        rope_cos.dtype != torch.bfloat16
        or rope_sin.dtype != torch.bfloat16
        or rope_cos.ndim != 3
        or tuple(rope_cos.shape) != tuple(rope_sin.shape)
        or rope_cos.shape[2] != 32
    ):
        raise ValueError("D64 RoPE tables must be matching BF16 [B,S,32]")
    if rope_cos.device != rope_sin.device or not rope_cos.is_cuda:
        raise ValueError("D64 RoPE tables must share one CUDA device")
    packed = (
        torch.stack((rope_cos, rope_sin), dim=-1)
        .contiguous()
        .view(torch.int32)
        .reshape(rope_cos.shape)
    )
    return torch.cat((packed, packed), dim=-1).contiguous()


def b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4(
    input_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    qkv_weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    paired_qk_scales: torch.Tensor,
    paired_rope_packed: torch.Tensor,
    *,
    batch: int,
    seqlen: int,
    q_heads: int,
    kv_heads: int,
    store_bf16: bool = False,
    publish_fp8_backward: bool = True,
    interleave_causal_kv: bool = False,
    v_mxfp4_scale_2d: bool = False,
    represented_backward: bool = False,
    per_block_qk_scales: bool = False,
    experimental_split_v_backward: bool = False,
) -> B300UnifiedLowpQKV:
    """Project D64 GQA and publish native FA4 operands in one epilogue.

    Adjacent logical D64 heads share one physical D128 projection tile.  RoPE
    is nevertheless applied independently to each D64 half, and Q/K compact
    payloads, tensor scales, V payload/scales, and FP8 backward operands are
    written directly in logical-head order.  The opt-in causal layout spreads
    consecutive K/V rows across the four physical N128 quarters so each early
    MXFP4 P block sees the whole key tile.  No post-projection transpose or
    metadata repack is required.  ``paired_qk_scales`` has shape
    ``[B, Hq/2, 7]`` because its fixed tensor-wide scale is shared by the two
    D64 halves; local block scales remain independent. MXFP4 V defaults to
    one E8M0 scale per depth row and 32 sequence values, which avoids coupling
    unrelated depth rows. Pass ``v_mxfp4_scale_2d=True`` only to reproduce the
    coarser 32x32 policy.
    """
    _ensure_lowp_bwd_extension()
    if per_block_qk_scales and not represented_backward:
        raise ValueError(
            "per_block_qk_scales requires represented_backward=True"
        )
    if represented_backward and not per_block_qk_scales:
        raise ValueError(
            "represented native NVFP4 backward currently requires "
            "per_block_qk_scales=True"
        )
    if experimental_split_v_backward and not (
        interleave_causal_kv
        and represented_backward
        and per_block_qk_scales
    ):
        raise ValueError(
            "experimental_split_v_backward requires interleaved MXFP4 V, "
            "represented backward operands, and per-block Q/K scales"
        )
    if represented_backward and (
        bool(interleave_causal_kv) != bool(experimental_split_v_backward)
    ):
        raise ValueError(
            "represented native NVFP4 uses unsplit FP8 V or split "
            "interleaved MXFP4 V backward publication"
        )
    if len(input_operand) != 3 or len(qkv_weight_operand) != 3:
        raise ValueError(
            "NVFP4 projection operands must contain packed data, block "
            "scales, and one global scale"
        )
    if (
        q_heads <= 0
        or kv_heads <= 0
        or q_heads % 2
        or kv_heads % 2
        or q_heads % kv_heads
    ):
        raise ValueError(
            "paired D64 projection requires positive even Hq/Hkv and Hq "
            "divisible by Hkv"
        )
    if (
        paired_qk_scales.dtype != torch.float32
        or not paired_qk_scales.is_cuda
        or not paired_qk_scales.is_contiguous()
        or tuple(paired_qk_scales.shape) != (batch, q_heads // 2, 7)
    ):
        raise ValueError(
            "paired_qk_scales must be contiguous CUDA float32 [B,Hq/2,7]"
        )
    if (
        paired_rope_packed.dtype != torch.int32
        or not paired_rope_packed.is_cuda
        or not paired_rope_packed.is_contiguous()
        or tuple(paired_rope_packed.shape) != (batch, seqlen, 64)
    ):
        raise ValueError(
            "paired_rope_packed must be contiguous CUDA int32 [B,S,64]"
        )
    project_name = (
        "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
        "interleaved_causal"
        if interleave_causal_kv
        else "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed"
    )
    if represented_backward:
        project_name += "_represented_backward"
    if per_block_qk_scales:
        project_name += "_perblock_qk"
    if experimental_split_v_backward:
        project_name += "_split_v_backward"
    project = getattr(_C_b300_lowp_bwd, project_name, None)
    if project is None:
        extension_path = getattr(_C_b300_lowp_bwd, "__file__", "<unknown>")
        raise RuntimeError(
            "the selected low-precision extension "
            f"{extension_path!r} does not provide required native NVFP4 "
            f"projection specialization {project_name!r}"
        )
    projected = (
        project(
            *input_operand,
            *qkv_weight_operand,
            paired_qk_scales,
            paired_rope_packed,
            int(batch),
            int(seqlen),
            int(q_heads),
            int(kv_heads),
            bool(store_bf16),
            bool(publish_fp8_backward),
            bool(v_mxfp4_scale_2d),
        )
    )
    backward = B300AdaptiveLowpOperands(
        q_fp4=projected[3],
        score_q_fp4=projected[4],
        k_fp4=projected[5],
        score_k_fp4=projected[6],
        qk_scales=projected[7],
    )
    expected_payloads = (
        (backward.score_q_fp4, (batch, q_heads, seqlen, 32)),
        (backward.score_k_fp4, (batch, kv_heads, seqlen, 32)),
    )
    for tensor, shape in expected_payloads:
        if tensor.dtype != torch.uint8 or tuple(tensor.shape) != shape:
            raise RuntimeError(
                "paired D64 projection returned an invalid Q/K payload: "
                f"expected uint8 {shape}, got {tensor.dtype} "
                f"{tuple(tensor.shape)}"
            )
    if not store_bf16 and (backward.q_fp4.numel() or backward.k_fp4.numel()):
        raise RuntimeError(
            "paired D64 hybrid projection unexpectedly published aligned Q/K"
        )
    q_forward_scales = projected[8]
    q_forward_global_scale = projected[9]
    k_forward_scales = projected[10]
    k_forward_global_scale = projected[11]
    v_forward_fp4 = projected[12]
    v_forward_scales = projected[13]
    expected_forward = (
        (
            q_forward_scales,
            torch.float8_e4m3fn,
            (batch, seqlen // 128, q_heads, 512),
        ),
        (q_forward_global_scale, torch.float32, (batch, q_heads)),
        (
            k_forward_scales,
            torch.float8_e4m3fn,
            (batch, seqlen // 64, kv_heads, 512),
        ),
        (k_forward_global_scale, torch.float32, (batch, kv_heads)),
        (
            v_forward_fp4,
            torch.float4_e2m1fn_x2,
            (batch, kv_heads, 64, seqlen // 2),
        ),
        (
            v_forward_scales,
            torch.float8_e4m3fn,
            (batch, seqlen // 128, kv_heads, 512),
        ),
    )
    for tensor, dtype, shape in expected_forward:
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise RuntimeError(
                "paired D64 projection returned an invalid forward operand: "
                f"expected {dtype} {shape}, got {tensor.dtype} "
                f"{tuple(tensor.shape)}"
            )
    if not publish_fp8_backward:
        raise ValueError(
            "the paired D64 path currently requires FP8 backward publication"
        )
    q_backward_fp8 = projected[21]
    k_backward_fp8 = projected[22]
    v_backward_fp8 = projected[20]
    v_forward_fp8 = (
        projected[23]
        if len(projected) > 23 and projected[23].numel()
        else None
    )
    expected_fp8 = (
        (q_backward_fp8, (batch, seqlen, q_heads, 64)),
        (k_backward_fp8, (batch, seqlen, kv_heads, 64)),
        (v_backward_fp8, (batch, seqlen, kv_heads, 64)),
    )
    for tensor, shape in expected_fp8:
        if (
            tensor.dtype != torch.float8_e4m3fn
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(
                "paired D64 projection returned an invalid FP8 backward "
                f"operand: expected contiguous {shape}, got {tensor.dtype} "
                f"{tuple(tensor.shape)}"
            )
    if v_forward_fp8 is not None:
        expected_forward_fp8 = (batch, kv_heads, 64, seqlen)
        if (
            v_forward_fp8.dtype != torch.float8_e4m3fn
            or tuple(v_forward_fp8.shape) != expected_forward_fp8
            or not v_forward_fp8.is_contiguous()
        ):
            raise RuntimeError(
                "paired D64 projection returned an invalid feature-major "
                "FP8 V operand: expected contiguous float8_e4m3fn "
                f"{expected_forward_fp8}, got {v_forward_fp8.dtype} "
                f"{tuple(v_forward_fp8.shape)}"
            )
    q_raw, k_raw, v_raw = projected[:3]
    if store_bf16:
        expected_bf16 = (
            (q_raw, (batch, seqlen, q_heads, 64)),
            (k_raw, (batch, seqlen, kv_heads, 64)),
            (v_raw, (batch, seqlen, kv_heads, 64)),
        )
        for tensor, shape in expected_bf16:
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
                raise RuntimeError(
                    "paired D64 projection returned an invalid BF16 tensor: "
                    f"expected {shape}, got {tensor.dtype} "
                    f"{tuple(tensor.shape)}"
                )
    return B300UnifiedLowpQKV(
        q=q_raw if store_bf16 else None,
        k=k_raw if store_bf16 else None,
        v=v_raw if store_bf16 else None,
        backward=backward,
        q_forward_fp4=_b300_typed_fp4_alias(backward.score_q_fp4),
        k_forward_fp4=_b300_typed_fp4_alias(backward.score_k_fp4),
        q_forward_scales=q_forward_scales,
        q_forward_global_scale=q_forward_global_scale,
        k_forward_scales=k_forward_scales,
        k_forward_global_scale=k_forward_global_scale,
        v_forward_fp4=v_forward_fp4,
        v_forward_scales=v_forward_scales,
        v_backward_fp4=projected[14],
        v_backward_scales=projected[15],
        v_backward_fp8=v_backward_fp8,
        q_backward_fp8=q_backward_fp8,
        k_backward_fp8=k_backward_fp8,
        q_dk_fp4=None,
        k_dq_fp4=None,
        q_dk_scales=None,
        k_dq_scales=None,
        q_heads=int(q_heads),
        kv_heads=int(kv_heads),
        head_dim=64,
        v_forward_fp8=v_forward_fp8,
    )


def b300_require_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection(
    *,
    publish_mxfp4_v: bool,
) -> str:
    """Require the experimental paired-D64 native NVFP4 projection ABI."""
    _ensure_lowp_bwd_extension()
    project_name = "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed"
    if publish_mxfp4_v:
        project_name += (
            "_interleaved_causal_represented_backward_perblock_qk_"
            "split_v_backward"
        )
    else:
        project_name += "_represented_backward_perblock_qk"
    if getattr(_C_b300_lowp_bwd, project_name, None) is None:
        extension_path = getattr(_C_b300_lowp_bwd, "__file__", "<unknown>")
        raise RuntimeError(
            "the selected low-precision extension "
            f"{extension_path!r} does not provide required experimental "
            f"native NVFP4 projection specialization {project_name!r}"
        )
    return project_name


class B300BoundNVFP4QKVProjection:
    """Experimental shape-bound native NVFP4 projection dispatcher.

    First use authenticates a caller-owned publication workspace against the
    matching allocating ABI and executes the checked out-parameter symbol.
    Later calls to that exact workspace execute the unchecked route-specific
    symbol. Q/K backward publications lift the represented NVFP4 codes with
    per-row-K16 scales. The MX route publishes forward MXFP4 V and direct
    projection-accumulator E4M3 V backward; exact FP8 uses that same direct V
    backward source without the inactive MX publication. The experimental
    E4M3-derived MX route is an explicit construction-time opt-in: it keeps
    the same direct E4M3 backward V bytes, then derives forward MXFP4 from
    those bytes inside the projection epilogue. Its first-use authentication
    compares that MX publication with the standalone E4M3(x4)-to-MXFP4
    converter. The output-shared split-V route instead preserves both direct
    publication semantics and removes only the redundant V reload/restaging.
    Explicit ``None`` selects it automatically only for the authenticated
    direct rowwise MX shape when ``hidden=2048`` is supplied. Old callers
    omitting both ``hidden`` and the selector retain the historical publisher.
    """

    def __init__(
        self,
        *,
        batch: int,
        seqlen: int,
        hidden: int | None = None,
        q_heads: int,
        kv_heads: int,
        publish_mxfp4_v: bool,
        v_mxfp4_scale_2d: bool,
        experimental_e4m3_derived_mxfp4_v: bool = False,
        experimental_output_shared_split_v: bool | None = False,
    ) -> None:
        if (
            experimental_output_shared_split_v is not None
            and type(experimental_output_shared_split_v) is not bool
        ):
            raise TypeError(
                "experimental_output_shared_split_v must be exactly bool "
                "or None"
            )
        self.batch = int(batch)
        self.seqlen = int(seqlen)
        self.hidden = None if hidden is None else int(hidden)
        self.q_heads = int(q_heads)
        self.kv_heads = int(kv_heads)
        self.publish_mxfp4_v = bool(publish_mxfp4_v)
        self.v_mxfp4_scale_2d = bool(v_mxfp4_scale_2d)
        self.experimental_e4m3_derived_mxfp4_v = bool(
            experimental_e4m3_derived_mxfp4_v
        )
        self.experimental_output_shared_split_v_requested = (
            experimental_output_shared_split_v
        )
        output_shared_authenticated_shape = bool(
            self.batch == 16
            and self.seqlen == 4096
            and self.hidden == 2048
            and self.q_heads == 32
            and self.kv_heads == 8
        )
        output_shared_eligible = bool(
            output_shared_authenticated_shape
            and self.publish_mxfp4_v
            and not self.experimental_e4m3_derived_mxfp4_v
            and not self.v_mxfp4_scale_2d
        )
        if (
            experimental_output_shared_split_v is True
            and self.experimental_e4m3_derived_mxfp4_v
        ):
            raise ValueError(
                "experimental output-shared split V and E4M3-derived MXFP4 "
                "V are mutually exclusive"
            )
        if (
            experimental_output_shared_split_v is True
            and not output_shared_eligible
        ):
            if not self.publish_mxfp4_v:
                reason = "publish_mxfp4_v=True"
            elif self.experimental_e4m3_derived_mxfp4_v:
                reason = "the direct-MX rather than E4M3-derived route"
            elif self.v_mxfp4_scale_2d:
                reason = "the rowwise 1x32 MXFP4 V scale policy"
            else:
                reason = (
                    "the authenticated B16/S4096/H2048/Hq32/Hkv8/D64 shape"
                )
            raise ValueError(
                "experimental output-shared split V requires " + reason
            )
        self.experimental_output_shared_split_v = bool(
            output_shared_eligible
            if experimental_output_shared_split_v is None
            else experimental_output_shared_split_v
        )
        self.experimental_output_shared_split_v_resolved = (
            self.experimental_output_shared_split_v
        )
        if (
            self.experimental_e4m3_derived_mxfp4_v
            and not self.publish_mxfp4_v
        ):
            raise ValueError(
                "experimental E4M3-derived MXFP4 V requires "
                "publish_mxfp4_v=True"
            )
        if (
            self.experimental_output_shared_split_v
            and self.experimental_e4m3_derived_mxfp4_v
        ):
            raise ValueError(
                "experimental output-shared split V and E4M3-derived MXFP4 V "
                "are mutually exclusive"
            )
        if (
            self.experimental_e4m3_derived_mxfp4_v
            and self.v_mxfp4_scale_2d
        ):
            raise ValueError(
                "experimental E4M3-derived MXFP4 V supports only the "
                "rowwise 1x32 scale policy (v_mxfp4_scale_2d=False)"
            )
        self.represented_backward = True
        self.per_block_qk_scales = True
        self.experimental_split_v_backward = bool(
            self.publish_mxfp4_v
            and not self.experimental_e4m3_derived_mxfp4_v
        )
        self.abi_validation_symbol = (
            b300_require_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection(
                publish_mxfp4_v=self.publish_mxfp4_v,
            )
        )
        if self.experimental_e4m3_derived_mxfp4_v:
            self.checked_symbol = (
                "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
                "interleaved_causal_represented_backward_perblock_qk_"
                "e4m3_derived_mx_forward_out"
            )
        elif self.experimental_output_shared_split_v:
            self.checked_symbol = (
                "project_qkv_gqa_d64_paired_unified_fp4_nvfp4_rope_packed_"
                "interleaved_causal_represented_backward_perblock_qk_"
                "output_shared_split_v_mx_forward_out"
            )
        else:
            compact_suffix = (
                "_mx_forward_out"
                if self.publish_mxfp4_v
                else "_fp8_forward_out"
            )
            self.checked_symbol = self.abi_validation_symbol + compact_suffix
        self.unchecked_symbol = self.checked_symbol + "_unchecked"
        self.symbol = self.unchecked_symbol
        self.output_shared_split_v_path = (
            "output_shared_split_v"
            if self.experimental_output_shared_split_v
            else (
                "e4m3_derived_mx"
                if self.experimental_e4m3_derived_mxfp4_v
                else "retained_split_v"
                if self.publish_mxfp4_v
                else "fp8"
            )
        )
        self.requires_v_mxfp4_scales_out = False
        self.requires_forward_workspace = True
        self.experimental = True
        self.backward_publication_semantics = (
            "represented_nvfp4_qk_per_row_k16_with_"
            "projection_accumulator_e4m3_v"
        )
        required_symbols = [self.checked_symbol, self.unchecked_symbol]
        if self.experimental_e4m3_derived_mxfp4_v:
            required_symbols.append(
                "convert_e4m3_x4_v_bhds_to_causal_mxfp4"
            )
        missing_symbols = [
            symbol
            for symbol in required_symbols
            if getattr(_C_b300_lowp_bwd, symbol, None) is None
        ]
        if missing_symbols:
            extension_path = getattr(
                _C_b300_lowp_bwd, "__file__", "<unknown>"
            )
            raise RuntimeError(
                "the selected low-precision extension "
                f"{extension_path!r} does not provide required compact "
                f"experimental native NVFP4 projection specializations "
                f"{missing_symbols!r}"
            )
        self._project_checked = getattr(
            _C_b300_lowp_bwd, self.checked_symbol
        )
        self._project_unchecked = getattr(
            _C_b300_lowp_bwd, self.unchecked_symbol
        )
        # Weak values preserve id-collision protection while allowing a
        # layer's replaced private workspace to release its HBM allocation.
        self._validated_forward_workspaces: WeakValueDictionary[
            int, B300E4M3QKVForwardWorkspace
        ] = WeakValueDictionary()
        self._successful_full_abi_validation_count = 0

    @property
    def abi_validated(self) -> bool:
        return self._successful_full_abi_validation_count > 0

    @property
    def forward_workspace_abi_validated(self) -> bool:
        return self._successful_full_abi_validation_count > 0

    @property
    def validated_forward_workspace_count(self) -> int:
        return len(self._validated_forward_workspaces)

    @property
    def successful_full_abi_validation_count(self) -> int:
        return self._successful_full_abi_validation_count

    @property
    def vscale_out_abi_validated(self) -> bool:
        return bool(
            self.publish_mxfp4_v
            and self._successful_full_abi_validation_count > 0
        )

    def __call__(
        self,
        input_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        qkv_weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        paired_qk_scales: torch.Tensor,
        paired_rope_packed: torch.Tensor,
        *,
        forward_workspace: B300E4M3QKVForwardWorkspace,
    ) -> B300UnifiedLowpQKV:
        if self.experimental_output_shared_split_v:
            assert self.hidden is not None
            expected_packed_k = self.hidden // 2
            for name, packed in (
                ("input", input_operand[0]),
                ("QKV weight", qkv_weight_operand[0]),
            ):
                if packed.dim() != 2 or packed.size(1) != expected_packed_k:
                    raise ValueError(
                        "output-shared split-V requires rank-2 packed "
                        f"{name} with K width {expected_packed_k} for "
                        f"H{self.hidden}"
                    )
        if not isinstance(
            forward_workspace, B300E4M3QKVForwardWorkspace
        ):
            raise TypeError(
                "the bound native NVFP4 projection requires a "
                "B300E4M3QKVForwardWorkspace"
            )
        workspace_key = id(forward_workspace)
        validated_workspace = self._validated_forward_workspaces.get(
            workspace_key
        )
        first_use = validated_workspace is None
        if not first_use and validated_workspace is not forward_workspace:
            raise RuntimeError("forward workspace identity collision")
        legacy_bundle = None
        if first_use:
            legacy_bundle = (
                b300_project_qkv_gqa_d64_paired_unified_lowp_nvfp4(
                    input_operand,
                    qkv_weight_operand,
                    paired_qk_scales,
                    paired_rope_packed,
                    batch=self.batch,
                    seqlen=self.seqlen,
                    q_heads=self.q_heads,
                    kv_heads=self.kv_heads,
                    store_bf16=False,
                    publish_fp8_backward=True,
                    interleave_causal_kv=self.publish_mxfp4_v,
                    v_mxfp4_scale_2d=self.v_mxfp4_scale_2d,
                    represented_backward=self.represented_backward,
                    per_block_qk_scales=self.per_block_qk_scales,
                    experimental_split_v_backward=(
                        self.publish_mxfp4_v
                    ),
                )
            )
        project = (
            self._project_checked if first_use else self._project_unchecked
        )
        backward_publications = project(
            *input_operand,
            *qkv_weight_operand,
            paired_qk_scales,
            paired_rope_packed,
            self.batch,
            self.seqlen,
            self.q_heads,
            self.kv_heads,
            self.v_mxfp4_scale_2d,
            *forward_workspace.compact_outputs(),
        )
        try:
            backward_tuple = tuple(backward_publications)
        except TypeError as error:
            raise RuntimeError(
                "compact native NVFP4 projection must return backward "
                "{V,Q,K} tensors"
            ) from error
        if len(backward_tuple) != 3:
            raise RuntimeError(
                "compact native NVFP4 projection must return exactly three "
                "backward {V,Q,K} tensors"
            )
        expected_backward_owners = (
            forward_workspace.v_backward_fp8,
            forward_workspace.q_backward_fp8,
            forward_workspace.k_backward_fp8,
        )
        for name, returned, owner in zip(
            ("V", "Q", "K"),
            backward_tuple,
            expected_backward_owners,
            strict=True,
        ):
            if returned.data_ptr() != owner.data_ptr():
                raise RuntimeError(
                    "compact native NVFP4 projection returned backward "
                    f"{name} outside its caller-owned workspace"
                )
        if first_use:
            assert legacy_bundle is not None
            common_pairs = (
                (
                    "Q payload",
                    legacy_bundle.backward.score_q_fp4,
                    forward_workspace.q_payload,
                ),
                (
                    "K payload",
                    legacy_bundle.backward.score_k_fp4,
                    forward_workspace.k_payload,
                ),
                (
                    "Q scale pages",
                    legacy_bundle.q_forward_scales,
                    forward_workspace.q_scale_pages,
                ),
                (
                    "Q global scale",
                    legacy_bundle.q_forward_global_scale,
                    forward_workspace.q_global_scale,
                ),
                (
                    "K scale pages",
                    legacy_bundle.k_forward_scales,
                    forward_workspace.k_scale_pages,
                ),
                (
                    "K global scale",
                    legacy_bundle.k_forward_global_scale,
                    forward_workspace.k_global_scale,
                ),
            )
            for name, legacy, compact in common_pairs:
                _b300_require_bitwise_equal(name, legacy, compact)
            if self.publish_mxfp4_v:
                if self.experimental_e4m3_derived_mxfp4_v:
                    if legacy_bundle.v_backward_fp8 is None:
                        raise RuntimeError(
                            "legacy native NVFP4 projection omitted the "
                            "E4M3 V source required to authenticate derived "
                            "MXFP4"
                        )
                    feature_major_v = (
                        legacy_bundle.v_backward_fp8.permute(0, 2, 3, 1)
                        .contiguous()
                    )
                    reference_payload, reference_scales = getattr(
                        _C_b300_lowp_bwd,
                        "convert_e4m3_x4_v_bhds_to_causal_mxfp4",
                    )(feature_major_v)
                    _b300_require_bitwise_equal(
                        "E4M3-derived MXFP4 V payload",
                        reference_payload,
                        forward_workspace.v_mxfp4_payload,
                    )
                    _b300_require_bitwise_equal(
                        "E4M3-derived MXFP4 V scale pages",
                        reference_scales,
                        forward_workspace.v_mxfp4_scale_pages,
                        valid_last_dim_indices=tuple(
                            depth_lane * 16
                            + depth_group * 4
                            + sequence_quarter
                            for depth_lane in range(32)
                            for depth_group in range(2)
                            for sequence_quarter in range(4)
                        ),
                    )
                else:
                    _b300_require_bitwise_equal(
                        "MXFP4 V payload",
                        legacy_bundle.v_forward_fp4,
                        forward_workspace.v_mxfp4_payload,
                    )
                    _b300_require_bitwise_equal(
                        "MXFP4 V scale pages",
                        legacy_bundle.v_forward_scales,
                        forward_workspace.v_mxfp4_scale_pages,
                        valid_last_dim_indices=tuple(
                            depth_lane * 16
                            + depth_group * 4
                            + sequence_quarter
                            for depth_lane in range(32)
                            for depth_group in range(2)
                            for sequence_quarter in range(4)
                        ),
                    )
            else:
                if legacy_bundle.v_forward_fp8 is None:
                    raise RuntimeError(
                        "legacy native NVFP4 projection omitted forward FP8 V"
                    )
                _b300_require_bitwise_equal(
                    "FP8 V payload",
                    legacy_bundle.v_forward_fp8,
                    forward_workspace.v_fp8_payload,
                )
            legacy_backward = (
                legacy_bundle.v_backward_fp8,
                legacy_bundle.q_backward_fp8,
                legacy_bundle.k_backward_fp8,
            )
            if any(tensor is None for tensor in legacy_backward):
                raise RuntimeError(
                    "legacy native NVFP4 projection omitted a backward "
                    "{V,Q,K} publication"
                )
            for name, legacy, compact in zip(
                ("backward V", "backward Q", "backward K"),
                legacy_backward,
                backward_tuple,
                strict=True,
            ):
                assert legacy is not None
                _b300_require_bitwise_equal(name, legacy, compact)
            self._validated_forward_workspaces[
                workspace_key
            ] = forward_workspace
            self._successful_full_abi_validation_count += 1
        return _b300_compact_e4m3_qkv_bundle(
            forward_workspace,
            backward_tuple,
            paired_qk_scales,
            q_heads=self.q_heads,
            kv_heads=self.kv_heads,
            publish_mxfp4_v=self.publish_mxfp4_v,
        )


def b300_bind_qkv_gqa_d64_paired_unified_lowp_nvfp4_projection(
    *,
    batch: int,
    seqlen: int,
    hidden: int | None = None,
    q_heads: int,
    kv_heads: int,
    publish_mxfp4_v: bool,
    v_mxfp4_scale_2d: bool = False,
    experimental_e4m3_derived_mxfp4_v: bool = False,
    experimental_output_shared_split_v: bool | None = False,
) -> B300BoundNVFP4QKVProjection:
    """Bind the experimental native NVFP4 projection at one fixed shape.

    ``experimental_e4m3_derived_mxfp4_v`` is fail-closed and defaults to
    false. It is valid only for the MX forward route and never changes the
    existing direct-MX or exact-FP8 dispatchers.

    An omitted argument or explicit false retains the historical publisher.
    Explicit ``None`` automatically selects the accepted output-shared
    publisher only for direct rowwise MXFP4 split-V at its authenticated shape
    when ``hidden=2048`` is supplied. An unknown or different hidden size falls
    back for ``None`` and fails closed for true. Every candidate call validates
    both packed K widths, and the checked C++ call independently enforces
    H2048 from the packed input. Explicit true also fails closed for FP8,
    E4M3-derived MX, and incompatible scale policies.
    """
    return B300BoundNVFP4QKVProjection(
        batch=batch,
        seqlen=seqlen,
        hidden=hidden,
        q_heads=q_heads,
        kv_heads=kv_heads,
        publish_mxfp4_v=publish_mxfp4_v,
        v_mxfp4_scale_2d=v_mxfp4_scale_2d,
        experimental_e4m3_derived_mxfp4_v=(
            experimental_e4m3_derived_mxfp4_v
        ),
        experimental_output_shared_split_v=(
            experimental_output_shared_split_v
        ),
    )


def _b300_d128_nvfp4_projection_policy(
    seqlen: int,
    *,
    packed_rope: bool,
) -> tuple[int, bool, bool]:
    """Return the measured D128 cluster and packed-RoPE cache policy."""
    sequence = int(seqlen)
    has_packed_rope = bool(packed_rope)
    cluster_cap = {4096: 68, 8192: 72}.get(sequence, 0)
    return (
        cluster_cap,
        has_packed_rope,
        has_packed_rope and sequence >= 2048,
    )


def b300_require_qkv_gqa_d128_unified_lowp_nvfp4_projection() -> str:
    """Require the allocating D128 ABI used for first-call authentication."""
    _ensure_lowp_bwd_extension()
    project_name = (
        "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
    )
    if getattr(_C_b300_lowp_bwd, project_name, None) is None:
        extension_path = getattr(_C_b300_lowp_bwd, "__file__", "<unknown>")
        raise RuntimeError(
            "the selected low-precision extension "
            f"{extension_path!r} does not provide required D128 native "
            f"NVFP4 projection specialization {project_name!r}"
        )
    return project_name


class B300BoundD128NVFP4QKVProjection:
    """Shape-bound, route-selective D128 native-NVFP4 QKV projection.

    First use authenticates caller-owned storage bitwise against the allocating
    projection. Steady-state calls execute only one unchecked, route-specific
    symbol and never publish the inactive forward V representation. The normal
    FP8 and retained-dual-V routes normally share projection-accumulator E4M3
    Q/K/V for backward. The opt-in represented FP8-PV route instead derives its
    E4M3 Q/K backward bytes from the exact per-row-K16 NVFP4 codes and scales
    used by forward while retaining projection-accumulator E4M3 V. The rowwise
    and shared-tile MX-backward experiments keep direct E4M3 Q/K but make
    MXFP4 V the sole backward V publication. In particular,
    the shared-tile route quantizes each resident BF16 D32xS32 tile once and
    publishes the exact same codes in the forward and backward physical
    layouts. The output-shared dual-V candidate remains a separate route that
    publishes ordinary-order MXFP4 forward V and exact E4M3 backward V.
    Explicit ``None`` selects that dual-V route automatically only at the
    authenticated 8B B1/B2 shapes; the default false preserves the existing
    D128 route.
    """

    def __init__(
        self,
        *,
        batch: int,
        seqlen: int,
        hidden: int,
        q_heads: int,
        kv_heads: int,
        publish_mxfp4_v: bool,
        v_mxfp4_scale_2d: bool,
        per_block_qk_scales: bool,
        represented_backward: bool = False,
        experimental_output_shared_dual_v: bool | None = False,
        experimental_mx_backward_v: bool = False,
        experimental_shared_tile_mx_backward_v: bool = False,
    ) -> None:
        if type(represented_backward) is not bool:
            raise TypeError("represented_backward must be exactly bool")
        if (
            experimental_output_shared_dual_v is not None
            and type(experimental_output_shared_dual_v) is not bool
        ):
            raise TypeError(
                "experimental_output_shared_dual_v must be exactly bool "
                "or None"
            )
        if type(experimental_mx_backward_v) is not bool:
            raise TypeError("experimental_mx_backward_v must be exactly bool")
        if type(experimental_shared_tile_mx_backward_v) is not bool:
            raise TypeError(
                "experimental_shared_tile_mx_backward_v must be exactly bool"
            )
        if (
            experimental_mx_backward_v
            and experimental_shared_tile_mx_backward_v
        ):
            raise ValueError(
                "rowwise and shared-tile MX backward V are mutually exclusive"
            )
        self.batch = int(batch)
        self.seqlen = int(seqlen)
        self.hidden = int(hidden)
        self.q_heads = int(q_heads)
        self.kv_heads = int(kv_heads)
        self.publish_mxfp4_v = bool(publish_mxfp4_v)
        self.v_mxfp4_scale_2d = bool(v_mxfp4_scale_2d)
        self.per_block_qk_scales = bool(per_block_qk_scales)
        self.represented_backward = represented_backward
        self.experimental_rowwise_mx_backward_v = bool(
            experimental_mx_backward_v
        )
        self.experimental_shared_tile_mx_backward_v = bool(
            experimental_shared_tile_mx_backward_v
        )
        self.experimental_mx_backward_v = bool(
            self.experimental_rowwise_mx_backward_v
            or self.experimental_shared_tile_mx_backward_v
        )
        if self.represented_backward:
            if (
                self.batch not in (1, 2)
                or self.seqlen != 4096
                or self.hidden != 4096
                or self.q_heads != 32
                or self.kv_heads != 8
            ):
                raise ValueError(
                    "represented D128 Q/K backward is authenticated only for "
                    "B1/B2/S4096/H4096/Hq32/Hkv8/D128"
                )
            if self.publish_mxfp4_v:
                raise ValueError(
                    "represented D128 Q/K backward requires FP8-PV "
                    "(publish_mxfp4_v=False)"
                )
            if not self.per_block_qk_scales:
                raise ValueError(
                    "represented D128 Q/K backward requires per-row-K16 "
                    "Q/K scales"
                )
            if self.v_mxfp4_scale_2d:
                raise ValueError(
                    "represented D128 Q/K backward does not accept an MXFP4 "
                    "V scale policy"
                )
            if experimental_output_shared_dual_v is not False:
                raise ValueError(
                    "represented D128 Q/K backward requires "
                    "experimental_output_shared_dual_v=False"
                )
            if self.experimental_mx_backward_v:
                raise ValueError(
                    "represented D128 Q/K backward is incompatible with MX "
                    "backward-V candidates"
                )
        self.v_backward_mxfp4_scale_policy = (
            MXFP4_V_SCALE_POLICY_SHARED_D32XS32
            if self.experimental_shared_tile_mx_backward_v
            else MXFP4_V_SCALE_POLICY_ROWWISE_D32
            if self.experimental_rowwise_mx_backward_v
            else None
        )
        if self.batch <= 0 or self.seqlen <= 0:
            raise ValueError("bound D128 projection requires a positive shape")
        if self.hidden <= 0 or self.hidden % 256:
            raise ValueError(
                "bound D128 projection requires hidden divisible by 256"
            )
        if (
            self.q_heads <= 0
            or self.kv_heads <= 0
            or self.q_heads % self.kv_heads
        ):
            raise ValueError(
                "bound D128 projection requires positive divisible Q/KV heads"
            )
        self.experimental_output_shared_dual_v_requested = (
            experimental_output_shared_dual_v
        )
        output_shared_authenticated_shape = bool(
            self.batch in (1, 2)
            and self.seqlen == 4096
            and self.hidden == 4096
            and self.q_heads == 32
            and self.kv_heads == 8
        )
        output_shared_eligible = bool(
            output_shared_authenticated_shape
            and self.publish_mxfp4_v
            and not self.v_mxfp4_scale_2d
            and self.per_block_qk_scales
        )
        if (
            experimental_output_shared_dual_v is True
            and not output_shared_eligible
        ):
            if not self.publish_mxfp4_v:
                reason = "publish_mxfp4_v=True"
            elif self.v_mxfp4_scale_2d:
                reason = "the rowwise 1x32 MXFP4 V scale policy"
            elif not self.per_block_qk_scales:
                reason = "per-row-K16 Q/K scales"
            else:
                reason = (
                    "the authenticated B1/B2/S4096/H4096/Hq32/Hkv8/D128 "
                    "shape"
                )
            raise ValueError(
                "experimental D128 output-shared dual V requires " + reason
            )
        self.experimental_output_shared_dual_v = bool(
            output_shared_eligible
            if experimental_output_shared_dual_v is None
            else experimental_output_shared_dual_v
        )
        self.experimental_output_shared_dual_v_resolved = (
            self.experimental_output_shared_dual_v
        )
        if self.experimental_mx_backward_v:
            if not self.publish_mxfp4_v:
                raise ValueError(
                    "experimental MX backward V requires publish_mxfp4_v=True"
                )
            if (
                self.experimental_shared_tile_mx_backward_v
                and not self.v_mxfp4_scale_2d
            ):
                raise ValueError(
                    "shared-tile MX backward V requires D32xS32 scales"
                )
            if (
                self.experimental_rowwise_mx_backward_v
                and self.v_mxfp4_scale_2d
            ):
                raise ValueError(
                    "experimental MX backward V requires rowwise 1x32 scales"
                )
            if (
                self.experimental_shared_tile_mx_backward_v
                and not output_shared_authenticated_shape
            ):
                raise ValueError(
                    "shared-tile MX backward V is authenticated only for "
                    "B1/B2/S4096/H4096/Hq32/Hkv8/D128"
                )
            if (
                self.experimental_shared_tile_mx_backward_v
                and not self.per_block_qk_scales
            ):
                raise ValueError(
                    "shared-tile MX backward V requires per-row-K16 Q/K "
                    "scales"
                )
            if self.experimental_output_shared_dual_v:
                raise ValueError(
                    "MX-only backward V and output-shared E4M3 backward V "
                    "are mutually exclusive"
                )
        # LowpAttentionRuntime already reports the D64 implementation through
        # these split-V fields. Keep aliases so the D128 candidate can flow
        # through the same provenance plumbing without obscuring its clearer
        # dual-V public name.
        self.experimental_output_shared_split_v_requested = (
            self.experimental_output_shared_dual_v_requested
        )
        self.experimental_output_shared_split_v = (
            self.experimental_output_shared_dual_v
        )
        self.experimental_output_shared_split_v_resolved = (
            self.experimental_output_shared_dual_v_resolved
        )
        (
            self.cluster_cap,
            self.cache_packed_rope,
            self.cache_adaptive_qk_scale,
        ) = _b300_d128_nvfp4_projection_policy(
            self.seqlen,
            packed_rope=True,
        )
        self.abi_validation_symbol = (
            b300_require_qkv_gqa_d128_unified_lowp_nvfp4_projection()
        )
        if self.represented_backward:
            route_suffix = "_represented_backward_perblock_qk_fp8_forward_out"
        elif self.experimental_shared_tile_mx_backward_v:
            route_suffix = "_shared_tile_mx_backward_v_mx_forward_out"
        elif self.experimental_mx_backward_v:
            route_suffix = "_mx_backward_v_mx_forward_out"
        elif self.experimental_output_shared_dual_v:
            route_suffix = "_output_shared_dual_v_mx_forward_out"
        else:
            route_suffix = (
                "_mx_forward_out"
                if self.publish_mxfp4_v
                else "_fp8_forward_out"
            )
        self.checked_symbol = self.abi_validation_symbol + route_suffix
        self.unchecked_symbol = self.checked_symbol + "_unchecked"
        self.symbol = self.unchecked_symbol
        self.requires_v_mxfp4_scales_out = False
        self.requires_forward_workspace = True
        self.experimental = True
        self.output_shared_dual_v_path = (
            "shared_tile_mx_backward_v"
            if self.experimental_shared_tile_mx_backward_v
            else "mx_backward_v"
            if self.experimental_rowwise_mx_backward_v
            else "output_shared_dual_v"
            if self.experimental_output_shared_dual_v
            else "retained_dual_v"
            if self.publish_mxfp4_v
            else "fp8"
        )
        self.output_shared_split_v_path = self.output_shared_dual_v_path
        self.projection_forward_publication_path = (
            "caller_owned_represented_qk_fp8_pv_d128"
            if self.represented_backward
            else "caller_owned_shared_tile_mx_backward_v_d128"
            if self.experimental_shared_tile_mx_backward_v
            else "caller_owned_mx_backward_v_d128"
            if self.experimental_rowwise_mx_backward_v
            else "caller_owned_output_shared_dual_v_d128"
            if self.experimental_output_shared_dual_v
            else "caller_owned_route_selective_d128"
        )
        self.qk_backward_source = (
            "represented_nvfp4_codes_per_row_k16"
            if self.represented_backward
            else "projection_accumulator_e4m3"
        )
        self.v_backward_source = (
            "shared_d32xs32_forward_anchor_mxfp4_v"
            if self.experimental_shared_tile_mx_backward_v
            else "rowwise_width6_mxfp4_v"
            if self.experimental_rowwise_mx_backward_v
            else "projection_accumulator_e4m3"
        )
        self.backward_publication_semantics = (
            "represented_nvfp4_qk_per_row_k16_with_projection_accumulator_e4m3_v"
            if self.represented_backward
            else "single_quantized_d32xs32_mxfp4_v_with_projection_accumulator_e4m3_qk"
            if self.experimental_shared_tile_mx_backward_v
            else "rowwise_width6_mxfp4_v_with_projection_accumulator_e4m3_qk"
            if self.experimental_rowwise_mx_backward_v
            else "projection_accumulator_e4m3_qkv_shared_across_pv_routes"
        )
        missing_symbols = [
            symbol
            for symbol in (self.checked_symbol, self.unchecked_symbol)
            if getattr(_C_b300_lowp_bwd, symbol, None) is None
        ]
        if missing_symbols:
            extension_path = getattr(
                _C_b300_lowp_bwd, "__file__", "<unknown>"
            )
            raise RuntimeError(
                "the selected low-precision extension "
                f"{extension_path!r} does not provide required compact D128 "
                f"native NVFP4 projection specializations {missing_symbols!r}"
            )
        self._project_checked = getattr(
            _C_b300_lowp_bwd, self.checked_symbol
        )
        self._project_unchecked = getattr(
            _C_b300_lowp_bwd, self.unchecked_symbol
        )
        self._validated_forward_workspaces: WeakValueDictionary[
            int, B300E4M3QKVForwardWorkspace
        ] = WeakValueDictionary()
        self._successful_full_abi_validation_count = 0

    @property
    def abi_validated(self) -> bool:
        return self._successful_full_abi_validation_count > 0

    @property
    def forward_workspace_abi_validated(self) -> bool:
        return self._successful_full_abi_validation_count > 0

    @property
    def validated_forward_workspace_count(self) -> int:
        return len(self._validated_forward_workspaces)

    @property
    def successful_full_abi_validation_count(self) -> int:
        return self._successful_full_abi_validation_count

    @property
    def vscale_out_abi_validated(self) -> bool:
        return bool(
            self.publish_mxfp4_v
            and self._successful_full_abi_validation_count > 0
        )

    def __call__(
        self,
        input_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        qkv_weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        qk_scales: torch.Tensor,
        rope_packed: torch.Tensor,
        *,
        forward_workspace: B300E4M3QKVForwardWorkspace,
    ) -> B300UnifiedLowpQKV:
        if not isinstance(
            forward_workspace, B300E4M3QKVForwardWorkspace
        ):
            raise TypeError(
                "the bound D128 native NVFP4 projection requires a "
                "B300E4M3QKVForwardWorkspace"
            )
        if len(input_operand) != 3 or len(qkv_weight_operand) != 3:
            raise ValueError(
                "the bound D128 native NVFP4 projection requires three-part "
                "input and weight operands"
            )
        expected_packed_k = self.hidden // 2
        expected_weight_rows = self.q_heads * 128 + self.kv_heads * 256
        packed_input = input_operand[0]
        packed_weight = qkv_weight_operand[0]
        if (
            packed_input.ndim != 2
            or tuple(packed_input.shape)
            != (self.batch * self.seqlen, expected_packed_k)
        ):
            raise ValueError(
                "bound D128 projection requires packed input shape "
                f"[{self.batch * self.seqlen},{expected_packed_k}]"
            )
        if (
            packed_weight.ndim != 2
            or tuple(packed_weight.shape)
            != (expected_weight_rows, expected_packed_k)
        ):
            raise ValueError(
                "bound D128 projection requires packed QKV weight shape "
                f"[{expected_weight_rows},{expected_packed_k}]"
            )
        workspace_key = id(forward_workspace)
        validated_workspace = self._validated_forward_workspaces.get(
            workspace_key
        )
        first_use = validated_workspace is None
        if not first_use and validated_workspace is not forward_workspace:
            raise RuntimeError("forward workspace identity collision")

        legacy_bundle = None
        mx_backward_legacy_bundle = None
        inactive_owners = (
            (
                ("inactive FP8 V payload", forward_workspace.v_fp8_payload),
                *(
                    (
                        (
                            "inactive backward E4M3 V",
                            forward_workspace.v_backward_fp8,
                        ),
                    )
                    if self.experimental_mx_backward_v
                    else ()
                ),
            )
            if self.publish_mxfp4_v
            else (
                (
                    "inactive MXFP4 V payload",
                    forward_workspace.v_mxfp4_payload,
                ),
                (
                    "inactive MXFP4 V scale pages",
                    forward_workspace.v_mxfp4_scale_pages,
                ),
            )
        )
        inactive_snapshots = (
            tuple(
                (name, owner.clone(), owner)
                for name, owner in inactive_owners
            )
            if first_use
            else ()
        )
        if first_use:
            legacy_bundle = b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
                input_operand,
                qkv_weight_operand,
                qk_scales,
                batch=self.batch,
                seqlen=self.seqlen,
                q_heads=self.q_heads,
                kv_heads=self.kv_heads,
                store_bf16=False,
                publish_fp8_backward=True,
                v_mxfp4_scale_2d=self.v_mxfp4_scale_2d,
                per_block_qk_scales=self.per_block_qk_scales,
                rope_packed=rope_packed,
                cluster_cap=self.cluster_cap,
                cache_packed_rope=self.cache_packed_rope,
                cache_adaptive_qk_scale=self.cache_adaptive_qk_scale,
            )
            if self.experimental_rowwise_mx_backward_v:
                mx_backward_legacy_bundle = (
                    b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
                        input_operand,
                        qkv_weight_operand,
                        qk_scales,
                        batch=self.batch,
                        seqlen=self.seqlen,
                        q_heads=self.q_heads,
                        kv_heads=self.kv_heads,
                        store_bf16=False,
                        publish_fp8_backward=False,
                        v_mxfp4_scale_2d=self.v_mxfp4_scale_2d,
                        per_block_qk_scales=False,
                        rope_packed=rope_packed,
                        cluster_cap=self.cluster_cap,
                        cache_packed_rope=self.cache_packed_rope,
                        cache_adaptive_qk_scale=(
                            self.cache_adaptive_qk_scale
                        ),
                    )
                )
        project = (
            self._project_checked if first_use else self._project_unchecked
        )
        compact_outputs = (
            forward_workspace.compact_mx_backward_v_outputs()
            if self.experimental_mx_backward_v
            else forward_workspace.compact_outputs()
        )
        backward_publications = project(
            *input_operand,
            *qkv_weight_operand,
            qk_scales,
            rope_packed,
            self.batch,
            self.seqlen,
            self.q_heads,
            self.kv_heads,
            self.v_mxfp4_scale_2d,
            self.per_block_qk_scales,
            self.cluster_cap,
            self.cache_packed_rope,
            self.cache_adaptive_qk_scale,
            *compact_outputs,
        )
        try:
            backward_tuple = tuple(backward_publications)
        except TypeError as error:
            raise RuntimeError(
                "compact D128 native NVFP4 projection must return backward "
                "V/Q/K tensors"
            ) from error
        expected_backward_count = 4 if self.experimental_mx_backward_v else 3
        if len(backward_tuple) != expected_backward_count:
            raise RuntimeError(
                "compact D128 native NVFP4 projection returned an invalid "
                f"backward tuple length {len(backward_tuple)}; expected "
                f"{expected_backward_count}"
            )
        if self.experimental_mx_backward_v:
            assert forward_workspace.v_backward_mxfp4 is not None
            assert forward_workspace.v_backward_mxfp4_scale_pages is not None
            expected_backward_owners = (
                forward_workspace.v_backward_mxfp4,
                forward_workspace.v_backward_mxfp4_scale_pages,
                forward_workspace.q_backward_fp8,
                forward_workspace.k_backward_fp8,
            )
            backward_names = ("MX V", "MX V scales", "Q", "K")
        else:
            expected_backward_owners = (
                forward_workspace.v_backward_fp8,
                forward_workspace.q_backward_fp8,
                forward_workspace.k_backward_fp8,
            )
            backward_names = ("V", "Q", "K")
        for name, returned, owner in zip(
            backward_names,
            backward_tuple,
            expected_backward_owners,
            strict=True,
        ):
            if returned.data_ptr() != owner.data_ptr():
                raise RuntimeError(
                    "compact D128 native NVFP4 projection returned backward "
                    f"{name} outside its caller-owned workspace"
                )

        if first_use:
            assert legacy_bundle is not None
            for name, snapshot, owner in inactive_snapshots:
                _b300_require_bitwise_equal(name, snapshot, owner)
            common_pairs = (
                (
                    "Q payload",
                    legacy_bundle.backward.score_q_fp4,
                    forward_workspace.q_payload,
                ),
                (
                    "K payload",
                    legacy_bundle.backward.score_k_fp4,
                    forward_workspace.k_payload,
                ),
                (
                    "Q scale pages",
                    legacy_bundle.q_forward_scales,
                    forward_workspace.q_scale_pages,
                ),
                (
                    "Q global scale",
                    legacy_bundle.q_forward_global_scale,
                    forward_workspace.q_global_scale,
                ),
                (
                    "K scale pages",
                    legacy_bundle.k_forward_scales,
                    forward_workspace.k_scale_pages,
                ),
                (
                    "K global scale",
                    legacy_bundle.k_forward_global_scale,
                    forward_workspace.k_global_scale,
                ),
            )
            for name, legacy, compact in common_pairs:
                _b300_require_bitwise_equal(name, legacy, compact)
            if self.publish_mxfp4_v:
                _b300_require_bitwise_equal(
                    "MXFP4 V payload",
                    legacy_bundle.v_forward_fp4,
                    forward_workspace.v_mxfp4_payload,
                )
                _b300_require_bitwise_equal(
                    "MXFP4 V scale pages",
                    legacy_bundle.v_forward_scales,
                    forward_workspace.v_mxfp4_scale_pages,
                )
            else:
                if legacy_bundle.v_forward_fp8 is None:
                    raise RuntimeError(
                        "legacy D128 native NVFP4 projection omitted forward "
                        "FP8 V"
                    )
                _b300_require_bitwise_equal(
                    "FP8 V payload",
                    legacy_bundle.v_forward_fp8,
                    forward_workspace.v_fp8_payload,
                )
            if self.experimental_shared_tile_mx_backward_v:
                _b300_require_shared_tile_mxfp4_v(
                    forward_workspace.v_mxfp4_payload,
                    forward_workspace.v_mxfp4_scale_pages,
                    forward_workspace.v_backward_mxfp4,
                    forward_workspace.v_backward_mxfp4_scale_pages,
                )
                legacy_backward = (
                    legacy_bundle.q_backward_fp8,
                    legacy_bundle.k_backward_fp8,
                )
                compact_backward = backward_tuple[2:]
                legacy_backward_names = ("backward Q", "backward K")
            elif self.experimental_mx_backward_v:
                assert mx_backward_legacy_bundle is not None
                legacy_backward = (
                    mx_backward_legacy_bundle.v_backward_fp4,
                    mx_backward_legacy_bundle.v_backward_scales,
                    legacy_bundle.q_backward_fp8,
                    legacy_bundle.k_backward_fp8,
                )
                legacy_backward_names = (
                    "backward MX V",
                    "backward MX V scales",
                    "backward Q",
                    "backward K",
                )
                compact_backward = backward_tuple
            elif self.represented_backward:
                legacy_backward = (legacy_bundle.v_backward_fp8,)
                legacy_backward_names = ("backward V",)
                compact_backward = backward_tuple[:1]
            else:
                legacy_backward = (
                    legacy_bundle.v_backward_fp8,
                    legacy_bundle.q_backward_fp8,
                    legacy_bundle.k_backward_fp8,
                )
                legacy_backward_names = (
                    "backward V",
                    "backward Q",
                    "backward K",
                )
                compact_backward = backward_tuple
            if any(tensor is None for tensor in legacy_backward):
                raise RuntimeError(
                    "legacy D128 native NVFP4 projection omitted a backward "
                    "{V,Q,K} publication"
                )
            for name, legacy, compact in zip(
                legacy_backward_names,
                legacy_backward,
                compact_backward,
                strict=True,
            ):
                assert legacy is not None
                _b300_require_bitwise_equal(name, legacy, compact)
            if self.represented_backward:
                _b300_require_represented_d128_nvfp4_qk_backward(
                    forward_workspace,
                    backward_tuple[1],
                    backward_tuple[2],
                )
            self._validated_forward_workspaces[
                workspace_key
            ] = forward_workspace
            self._successful_full_abi_validation_count += 1

        return _b300_compact_e4m3_qkv_bundle(
            forward_workspace,
            backward_tuple,
            qk_scales,
            q_heads=self.q_heads,
            kv_heads=self.kv_heads,
            publish_mxfp4_v=self.publish_mxfp4_v,
            head_dim=128,
            mx_backward_v=self.experimental_mx_backward_v,
            mx_backward_v_scale_policy=(
                self.v_backward_mxfp4_scale_policy
            ),
        )


def b300_bind_qkv_gqa_d128_unified_lowp_nvfp4_projection(
    *,
    batch: int,
    seqlen: int,
    hidden: int,
    q_heads: int,
    kv_heads: int,
    publish_mxfp4_v: bool,
    v_mxfp4_scale_2d: bool = False,
    per_block_qk_scales: bool = False,
    represented_backward: bool = False,
    experimental_output_shared_dual_v: bool | None = False,
    experimental_mx_backward_v: bool = False,
    experimental_shared_tile_mx_backward_v: bool = False,
) -> B300BoundD128NVFP4QKVProjection:
    """Bind the production clustered D128 projection at one fixed shape.

    The represented-Q/K candidate is exact-bool, FP8-PV-only, and requires
    per-row-K16 scales with every MX/output-shared candidate disabled. Its
    first use authenticates Q/K against the exact forward NVFP4 representation
    instead of the numerically different direct-E4M3 projection publication.
    The output-shared dual-V candidate is fail-closed. False preserves the
    retained publisher; explicit ``None`` automatically selects it only for
    B1/B2/S4096/H4096/Hq32/Hkv8 with direct rowwise MXFP4 V and per-row-K16
    Q/K.
    First use still authenticates every active forward and backward byte
    against the allocating route before steady-state unchecked dispatch.
    """
    if type(represented_backward) is not bool:
        raise TypeError("represented_backward must be exactly bool")
    return B300BoundD128NVFP4QKVProjection(
        batch=batch,
        seqlen=seqlen,
        hidden=hidden,
        q_heads=q_heads,
        kv_heads=kv_heads,
        publish_mxfp4_v=publish_mxfp4_v,
        v_mxfp4_scale_2d=v_mxfp4_scale_2d,
        per_block_qk_scales=per_block_qk_scales,
        represented_backward=represented_backward,
        experimental_output_shared_dual_v=(
            experimental_output_shared_dual_v
        ),
        experimental_mx_backward_v=experimental_mx_backward_v,
        experimental_shared_tile_mx_backward_v=(
            experimental_shared_tile_mx_backward_v
        ),
    )


def b300_require_qkv_gqa_d128_unified_lowp_e4m3_projection() -> str:
    """Require the native-D128 dense-E4M3 QKV projection ABI."""
    _ensure_lowp_bwd_extension()
    project_name = (
        "project_qkv_gqa_d128_unified_fp8_e4m3_rope_packed_clustered"
    )
    required = (
        project_name,
        project_name + "_fp8_forward_out",
        project_name + "_fp8_forward_out_unchecked",
        project_name + "_mx_forward_out",
        project_name + "_mx_forward_out_unchecked",
    )
    missing = [
        symbol
        for symbol in required
        if getattr(_C_b300_lowp_bwd, symbol, None) is None
    ]
    if missing:
        extension_path = getattr(_C_b300_lowp_bwd, "__file__", "<unknown>")
        raise RuntimeError(
            "the selected low-precision extension "
            f"{extension_path!r} does not provide required native-D128 "
            f"dense-E4M3 QKV symbols {missing!r}"
        )
    return project_name


class B300BoundD128E4M3QKVProjection:
    """Shape-bound native-D128 dense-E4M3 QKV projection.

    Q/K are published as row-K16 NVFP4 for causal attention. Both FP8-PV and
    MXFP4-PV retain ordinary K/V order and publish backward E4M3 Q/K/V
    directly from the projection accumulator. Represented-Q/K publication is
    intentionally absent from this ABI.
    """

    def __init__(
        self,
        *,
        batch: int,
        seqlen: int,
        hidden: int,
        q_heads: int,
        kv_heads: int,
        publish_mxfp4_v: bool,
        v_mxfp4_scale_2d: bool = False,
        cluster_cap: int = 0,
    ) -> None:
        self.batch = int(batch)
        self.seqlen = int(seqlen)
        self.hidden = int(hidden)
        self.q_heads = int(q_heads)
        self.kv_heads = int(kv_heads)
        self.publish_mxfp4_v = bool(publish_mxfp4_v)
        self.v_mxfp4_scale_2d = bool(v_mxfp4_scale_2d)
        self.cluster_cap = int(cluster_cap)
        if self.batch <= 0 or self.seqlen <= 0:
            raise ValueError("bound D128 E4M3 projection requires a positive shape")
        if self.seqlen % 256:
            raise ValueError(
                "bound D128 E4M3 projection requires seqlen divisible by 256"
            )
        if self.hidden <= 0 or self.hidden % 128:
            raise ValueError(
                "bound D128 E4M3 projection requires hidden divisible by 128"
            )
        if (
            self.q_heads <= 0
            or self.kv_heads <= 0
            or self.q_heads % self.kv_heads
        ):
            raise ValueError(
                "bound D128 E4M3 projection requires positive divisible "
                "Q/KV heads"
            )
        total_width = (self.q_heads + 2 * self.kv_heads) * 128
        if total_width % 256:
            raise ValueError(
                "bound D128 E4M3 projection requires QKV width divisible "
                "by 256"
            )
        if self.cluster_cap < 0:
            raise ValueError("cluster_cap must be non-negative")

        self.per_block_qk_scales = True
        self.represented_backward = False
        self.backward_match_forward_operands = False
        self.experimental_split_v_backward = False
        self.interleave_causal_kv = False
        self.cache_packed_rope = False
        self.cache_adaptive_qk_scale = False
        self.requires_v_mxfp4_scales_out = False
        self.requires_forward_workspace = True
        self.experimental = True
        self.output_shared_split_v_path = (
            "ordinary_mx" if self.publish_mxfp4_v else "fp8"
        )
        self.output_shared_dual_v_path = self.output_shared_split_v_path
        self.projection_forward_publication_path = (
            "caller_owned_dense_e4m3_d128"
        )
        self.backward_publication_semantics = (
            "projection_accumulator_e4m3_qkv_shared_across_pv_routes"
        )

        self.abi_validation_symbol = (
            b300_require_qkv_gqa_d128_unified_lowp_e4m3_projection()
        )
        suffix = (
            "_mx_forward_out"
            if self.publish_mxfp4_v
            else "_fp8_forward_out"
        )
        self.checked_symbol = self.abi_validation_symbol + suffix
        self.unchecked_symbol = self.checked_symbol + "_unchecked"
        self.symbol = self.unchecked_symbol
        self._project_allocating = getattr(
            _C_b300_lowp_bwd, self.abi_validation_symbol
        )
        self._project_checked = getattr(
            _C_b300_lowp_bwd, self.checked_symbol
        )
        self._project_unchecked = getattr(
            _C_b300_lowp_bwd, self.unchecked_symbol
        )
        self._validated_forward_workspaces: WeakValueDictionary[
            int, B300E4M3QKVForwardWorkspace
        ] = WeakValueDictionary()
        self._successful_full_abi_validation_count = 0

    @property
    def abi_validated(self) -> bool:
        return self._successful_full_abi_validation_count > 0

    @property
    def forward_workspace_abi_validated(self) -> bool:
        return self._successful_full_abi_validation_count > 0

    @property
    def validated_forward_workspace_count(self) -> int:
        return len(self._validated_forward_workspaces)

    @property
    def successful_full_abi_validation_count(self) -> int:
        return self._successful_full_abi_validation_count

    @property
    def vscale_out_abi_validated(self) -> bool:
        return bool(
            self.publish_mxfp4_v
            and self._successful_full_abi_validation_count > 0
        )

    def __call__(
        self,
        input_operand: tuple[torch.Tensor, torch.Tensor],
        qkv_weight_operand: tuple[torch.Tensor, torch.Tensor],
        qk_scales: torch.Tensor,
        rope_packed: torch.Tensor,
        *,
        forward_workspace: B300E4M3QKVForwardWorkspace,
    ) -> B300UnifiedLowpQKV:
        if not isinstance(forward_workspace, B300E4M3QKVForwardWorkspace):
            raise TypeError(
                "the bound D128 dense-E4M3 projection requires a "
                "B300E4M3QKVForwardWorkspace"
            )
        if len(input_operand) != 2 or len(qkv_weight_operand) != 2:
            raise ValueError(
                "the bound D128 dense-E4M3 projection requires two-part "
                "payload/decode input and weight operands"
            )
        input_fp8, input_decode = input_operand
        weight_fp8, weight_decode = qkv_weight_operand
        expected_rows = self.batch * self.seqlen
        expected_weight_rows = (self.q_heads + 2 * self.kv_heads) * 128
        if tuple(input_fp8.shape) != (expected_rows, self.hidden):
            raise ValueError(
                "bound D128 E4M3 projection requires input shape "
                f"[{expected_rows},{self.hidden}]"
            )
        if tuple(weight_fp8.shape) != (expected_weight_rows, self.hidden):
            raise ValueError(
                "bound D128 E4M3 projection requires QKV weight shape "
                f"[{expected_weight_rows},{self.hidden}]"
            )
        if tuple(input_decode.shape) != (expected_rows,):
            raise ValueError(
                "bound D128 E4M3 input decode requires one value per row"
            )
        if tuple(weight_decode.shape) != (expected_weight_rows,):
            raise ValueError(
                "bound D128 E4M3 weight decode requires one value per output"
            )
        if tuple(qk_scales.shape) != (self.batch, self.q_heads, 7):
            raise ValueError(
                "bound D128 E4M3 row-K16 scale policy requires "
                f"[{self.batch},{self.q_heads},7] metadata"
            )
        if tuple(rope_packed.shape) != (self.batch, self.seqlen, 64):
            raise ValueError(
                "bound D128 E4M3 RoPE requires packed [B,S,64] metadata"
            )

        workspace_key = id(forward_workspace)
        validated_workspace = self._validated_forward_workspaces.get(
            workspace_key
        )
        first_use = validated_workspace is None
        if not first_use and validated_workspace is not forward_workspace:
            raise RuntimeError("forward workspace identity collision")

        legacy = None
        inactive_owners = (
            (("inactive FP8 V payload", forward_workspace.v_fp8_payload),)
            if self.publish_mxfp4_v
            else (
                (
                    "inactive MXFP4 V payload",
                    forward_workspace.v_mxfp4_payload,
                ),
                (
                    "inactive MXFP4 V scale pages",
                    forward_workspace.v_mxfp4_scale_pages,
                ),
            )
        )
        inactive_snapshots = (
            tuple((name, owner.clone(), owner) for name, owner in inactive_owners)
            if first_use
            else ()
        )
        if first_use:
            legacy = tuple(
                self._project_allocating(
                    input_fp8,
                    input_decode,
                    weight_fp8,
                    weight_decode,
                    qk_scales,
                    rope_packed,
                    self.batch,
                    self.seqlen,
                    self.q_heads,
                    self.kv_heads,
                    self.publish_mxfp4_v,
                    self.v_mxfp4_scale_2d,
                    self.cluster_cap,
                )
            )
            if len(legacy) != 24:
                raise RuntimeError(
                    "allocating D128 dense-E4M3 projection returned an "
                    f"invalid publication tuple length {len(legacy)}"
                )

        project = self._project_checked if first_use else self._project_unchecked
        backward_publications = project(
            input_fp8,
            input_decode,
            weight_fp8,
            weight_decode,
            qk_scales,
            rope_packed,
            self.batch,
            self.seqlen,
            self.q_heads,
            self.kv_heads,
            self.v_mxfp4_scale_2d,
            self.cluster_cap,
            *forward_workspace.compact_outputs(),
        )
        try:
            backward_tuple = tuple(backward_publications)
        except TypeError as error:
            raise RuntimeError(
                "compact D128 dense-E4M3 projection must return backward "
                "V/Q/K tensors"
            ) from error
        if len(backward_tuple) != 3:
            raise RuntimeError(
                "compact D128 dense-E4M3 projection must return exactly "
                "three backward V/Q/K tensors"
            )
        for name, returned, owner in zip(
            ("V", "Q", "K"),
            backward_tuple,
            (
                forward_workspace.v_backward_fp8,
                forward_workspace.q_backward_fp8,
                forward_workspace.k_backward_fp8,
            ),
            strict=True,
        ):
            if returned.data_ptr() != owner.data_ptr():
                raise RuntimeError(
                    "compact D128 dense-E4M3 projection returned backward "
                    f"{name} outside its caller-owned workspace"
                )

        if first_use:
            assert legacy is not None
            for name, snapshot, owner in inactive_snapshots:
                _b300_require_bitwise_equal(name, snapshot, owner)
            common_pairs = (
                ("Q payload", legacy[4], forward_workspace.q_payload),
                ("K payload", legacy[6], forward_workspace.k_payload),
                ("Q scale pages", legacy[8], forward_workspace.q_scale_pages),
                ("Q global scale", legacy[9], forward_workspace.q_global_scale),
                ("K scale pages", legacy[10], forward_workspace.k_scale_pages),
                ("K global scale", legacy[11], forward_workspace.k_global_scale),
            )
            for name, allocated, compact in common_pairs:
                _b300_require_bitwise_equal(name, allocated, compact)
            if self.publish_mxfp4_v:
                _b300_require_bitwise_equal(
                    "MXFP4 V payload", legacy[12], forward_workspace.v_mxfp4_payload
                )
                _b300_require_bitwise_equal(
                    "MXFP4 V scale pages",
                    legacy[13],
                    forward_workspace.v_mxfp4_scale_pages,
                )
            else:
                _b300_require_bitwise_equal(
                    "FP8 V payload", legacy[23], forward_workspace.v_fp8_payload
                )
            for name, allocated, compact in zip(
                ("backward V", "backward Q", "backward K"),
                legacy[20:23],
                backward_tuple,
                strict=True,
            ):
                _b300_require_bitwise_equal(name, allocated, compact)
            self._validated_forward_workspaces[workspace_key] = forward_workspace
            self._successful_full_abi_validation_count += 1

        return _b300_compact_e4m3_qkv_bundle(
            forward_workspace,
            backward_tuple,
            qk_scales,
            q_heads=self.q_heads,
            kv_heads=self.kv_heads,
            publish_mxfp4_v=self.publish_mxfp4_v,
            head_dim=128,
        )


def b300_bind_qkv_gqa_d128_unified_lowp_e4m3_projection(
    *,
    batch: int,
    seqlen: int,
    hidden: int,
    q_heads: int,
    kv_heads: int,
    publish_mxfp4_v: bool,
    v_mxfp4_scale_2d: bool = False,
    cluster_cap: int = 0,
) -> B300BoundD128E4M3QKVProjection:
    """Bind the native-D128 dense-E4M3 projection at one fixed shape."""
    return B300BoundD128E4M3QKVProjection(
        batch=batch,
        seqlen=seqlen,
        hidden=hidden,
        q_heads=q_heads,
        kv_heads=kv_heads,
        publish_mxfp4_v=publish_mxfp4_v,
        v_mxfp4_scale_2d=v_mxfp4_scale_2d,
        cluster_cap=cluster_cap,
    )


def b300_require_qkv_gqa_d64_paired_unified_lowp_e4m3_projection(
    *,
    publish_mxfp4_v: bool = False,
    represented_backward: bool = False,
    per_block_qk_scales: bool = False,
    experimental_split_v_backward: bool = False,
) -> str:
    """Require and return the selected paired-D64 E4M3 projection symbol.

    This is intentionally tensor-free so launchers can reject an extension
    that lacks the requested publication specialization before allocating a
    model or compiling the attention backward kernel.
    """
    _ensure_lowp_bwd_extension()
    if per_block_qk_scales and not represented_backward:
        raise ValueError(
            "per_block_qk_scales requires represented_backward=True"
        )
    if experimental_split_v_backward and not (
        publish_mxfp4_v and represented_backward and per_block_qk_scales
    ):
        raise ValueError(
            "experimental_split_v_backward requires MXFP4 V, represented "
            "backward operands, and per-block Q/K scales"
        )
    project_name = (
        "project_qkv_gqa_d64_paired_unified_fp8_e4m3_rope_packed_"
        "interleaved_causal"
    )
    if represented_backward:
        project_name += "_represented_backward"
    if per_block_qk_scales:
        project_name += "_perblock_qk"
    if experimental_split_v_backward:
        project_name += "_split_v_backward"
    if getattr(_C_b300_lowp_bwd, project_name, None) is None:
        extension_path = getattr(_C_b300_lowp_bwd, "__file__", "<unknown>")
        raise RuntimeError(
            "the selected low-precision extension "
            f"{extension_path!r} does not provide required projection "
            f"specialization {project_name!r}"
        )
    return project_name


def _b300_compact_e4m3_qkv_bundle(
    forward_workspace: B300E4M3QKVForwardWorkspace,
    backward_publications: tuple[torch.Tensor, ...],
    paired_qk_scales: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
    publish_mxfp4_v: bool,
    head_dim: int = 64,
    mx_backward_v: bool = False,
    mx_backward_v_scale_policy: str | None = None,
) -> B300UnifiedLowpQKV:
    """Assemble the legacy bundle view over compact forward publications."""
    if mx_backward_v:
        (
            v_backward_fp4,
            v_backward_scales,
            q_backward_fp8,
            k_backward_fp8,
        ) = backward_publications
        v_backward_fp8 = None
    else:
        v_backward_fp8, q_backward_fp8, k_backward_fp8 = (
            backward_publications
        )
        v_backward_fp4 = forward_workspace.empty_byte
        v_backward_scales = forward_workspace.empty_byte
    backward = B300AdaptiveLowpOperands(
        q_fp4=forward_workspace.empty_byte,
        score_q_fp4=forward_workspace.q_payload,
        k_fp4=forward_workspace.empty_byte,
        score_k_fp4=forward_workspace.k_payload,
        qk_scales=paired_qk_scales,
    )
    return B300UnifiedLowpQKV(
        q=None,
        k=None,
        v=None,
        backward=backward,
        q_forward_fp4=forward_workspace.q_payload_fp4,
        k_forward_fp4=forward_workspace.k_payload_fp4,
        q_forward_scales=forward_workspace.q_scale_pages,
        q_forward_global_scale=forward_workspace.q_global_scale,
        k_forward_scales=forward_workspace.k_scale_pages,
        k_forward_global_scale=forward_workspace.k_global_scale,
        v_forward_fp4=(
            forward_workspace.v_mxfp4_payload
            if publish_mxfp4_v
            else forward_workspace.empty_fp4
        ),
        v_forward_scales=(
            forward_workspace.v_mxfp4_scale_pages
            if publish_mxfp4_v
            else forward_workspace.empty_fp8
        ),
        v_backward_fp4=v_backward_fp4,
        v_backward_scales=v_backward_scales,
        v_backward_fp8=v_backward_fp8,
        q_backward_fp8=q_backward_fp8,
        k_backward_fp8=k_backward_fp8,
        q_dk_fp4=None,
        k_dq_fp4=None,
        q_dk_scales=None,
        k_dq_scales=None,
        q_heads=int(q_heads),
        kv_heads=int(kv_heads),
        head_dim=int(head_dim),
        v_forward_fp8=(
            None
            if publish_mxfp4_v
            else forward_workspace.v_fp8_payload
        ),
        v_backward_mxfp4_scale_policy=(
            mx_backward_v_scale_policy if mx_backward_v else None
        ),
    )


def _b300_require_bitwise_equal(
    name: str,
    legacy: torch.Tensor,
    compact: torch.Tensor,
    *,
    valid_last_dim_indices: tuple[int, ...] | None = None,
) -> None:
    """Authenticate one compact publication without float-value semantics."""
    if (
        legacy.dtype != compact.dtype
        or tuple(legacy.shape) != tuple(compact.shape)
        or legacy.device != compact.device
        or not compact.is_contiguous()
    ):
        raise RuntimeError(
            f"compact {name} metadata differs from the legacy publication"
        )
    legacy_bytes = legacy.view(torch.uint8)
    compact_bytes = compact.view(torch.uint8)
    if valid_last_dim_indices is not None:
        # D64 MX V-scale pages reserve the upper half of each 16-byte lane
        # slot for the general D128 layout.  Those bytes are deliberately
        # unwritten, so authenticating them would compare allocator history.
        # This path runs only once per caller workspace, outside timing.
        indices = torch.tensor(
            valid_last_dim_indices,
            dtype=torch.long,
            device=legacy.device,
        )
        legacy_bytes = legacy_bytes.index_select(-1, indices)
        compact_bytes = compact_bytes.index_select(-1, indices)
    legacy_bytes = legacy_bytes.reshape(-1)
    compact_bytes = compact_bytes.reshape(-1)
    if not torch.equal(legacy_bytes, compact_bytes):
        raise RuntimeError(
            f"compact {name} is not bitwise identical to the legacy publication"
        )


def _b300_require_represented_d128_nvfp4_qk_backward(
    forward_workspace: B300E4M3QKVForwardWorkspace,
    q_backward_fp8: torch.Tensor,
    k_backward_fp8: torch.Tensor,
) -> None:
    """Authenticate represented Q/K from the definitive forward NVFP4 bytes.

    The allocating D128 canary publishes direct projection-accumulator E4M3
    Q/K, which is intentionally not the represented candidate's ABI. Decode
    the caller-owned forward codes and their exact per-row-K16 scale pages,
    lift that represented value by four, and require byte-identical E4M3 Q/K.
    This runs once per caller workspace, outside steady-state dispatch.
    """
    from .lowp_fa4_bwd.projection_quantization_reference import (
        decode_native_nvfp4_qk,
    )

    def require_one(
        name: str,
        payload: torch.Tensor,
        scale_pages: torch.Tensor,
        global_scale: torch.Tensor,
        publication: torch.Tensor,
    ) -> None:
        with torch.inference_mode():
            represented = decode_native_nvfp4_qk(
                payload,
                scale_pages,
                global_scale,
                scale_tile_rows=128,
            )
            reference = (
                represented.mul_(4.0)
                .clamp_(-448.0, 448.0)
                .to(torch.float8_e4m3fn)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
        _b300_require_bitwise_equal(
            f"represented backward {name}", reference, publication
        )

    require_one(
        "Q",
        forward_workspace.q_payload,
        forward_workspace.q_scale_pages,
        forward_workspace.q_global_scale,
        q_backward_fp8,
    )
    # D128 K duplicates each S128 scale page into the two physical S64 pages.
    # Select one page per logical S128 tile so the readable decoder consumes
    # the same four row quarters as the fused represented publisher.
    require_one(
        "K",
        forward_workspace.k_payload,
        forward_workspace.k_scale_pages[:, ::2].contiguous(),
        forward_workspace.k_global_scale,
        k_backward_fp8,
    )


def _b300_require_shared_tile_mxfp4_v(
    forward_payload: torch.Tensor,
    forward_scales: torch.Tensor,
    backward_payload: torch.Tensor | None,
    backward_scales: torch.Tensor | None,
) -> None:
    """Authenticate one D32xS32 code matrix under both physical V ABIs."""
    if backward_payload is None or backward_scales is None:
        raise RuntimeError("shared-tile MXFP4 V workspace is incomplete")
    if forward_payload.ndim != 4 or backward_payload.ndim != 4:
        raise RuntimeError("shared-tile MXFP4 V payload ranks are invalid")
    batch, heads, depth, packed_sequence = forward_payload.shape
    sequence = packed_sequence * 2
    if (
        depth != 128
        or tuple(backward_payload.shape)
        != (batch, sequence, heads, depth // 2)
        or tuple(forward_scales.shape)
        != (batch, sequence // 128, heads, 512)
        or tuple(backward_scales.shape) != tuple(forward_scales.shape)
    ):
        raise RuntimeError("shared-tile MXFP4 V workspace shapes are invalid")

    forward_bytes = forward_payload.contiguous().view(torch.uint8)
    backward_bytes = backward_payload.contiguous().view(torch.uint8)
    forward_codes = torch.stack(
        (forward_bytes & 0x0F, forward_bytes >> 4), dim=-1
    ).reshape(batch, heads, depth, sequence).permute(0, 3, 1, 2).contiguous()
    backward_codes = torch.stack(
        (backward_bytes & 0x0F, backward_bytes >> 4), dim=-1
    ).reshape(batch, sequence, heads, depth)
    _b300_require_bitwise_equal(
        "shared-tile forward/backward MXFP4 V code matrix",
        forward_codes,
        backward_codes,
    )

    sequence_tiles = sequence // 128
    forward_pages = (
        forward_scales.contiguous()
        .view(torch.uint8)
        .reshape(batch, sequence_tiles, heads, 32, 4, 4)
        .permute(0, 1, 2, 5, 4, 3)
        .contiguous()
    )
    backward_pages = (
        backward_scales.contiguous()
        .view(torch.uint8)
        .reshape(batch, sequence_tiles, heads, 32, 4, 4)
        .permute(0, 1, 2, 4, 5, 3)
        .contiguous()
    )
    forward_repeated = (
        forward_pages[..., :1].expand_as(forward_pages).contiguous()
    )
    backward_repeated = (
        backward_pages[..., :1].expand_as(backward_pages).contiguous()
    )
    _b300_require_bitwise_equal(
        "shared-tile forward MXFP4 V scale replication",
        forward_repeated,
        forward_pages,
    )
    _b300_require_bitwise_equal(
        "shared-tile backward MXFP4 V scale replication",
        backward_repeated,
        backward_pages,
    )
    _b300_require_bitwise_equal(
        "shared-tile forward/backward MXFP4 V anchors",
        forward_pages[..., 0].contiguous(),
        backward_pages[..., 0].contiguous(),
    )


class B300BoundE4M3QKVProjection:
    """Construction-bound projection dispatcher for a fixed training shape.

    Each caller-owned workspace is authenticated once by comparing the full
    legacy ABI with the checked compact out-parameter ABI.  Every later use of
    that exact workspace calls only the fixed-shape unchecked symbol.  There is
    deliberately no fallback from an unchecked launch to an allocating path.
    """

    def __init__(
        self,
        *,
        batch: int,
        seqlen: int,
        q_heads: int,
        kv_heads: int,
        publish_mxfp4_v: bool,
        v_mxfp4_scale_2d: bool,
        represented_backward: bool,
        per_block_qk_scales: bool,
        experimental_split_v_backward: bool,
    ) -> None:
        self.batch = int(batch)
        self.seqlen = int(seqlen)
        self.q_heads = int(q_heads)
        self.kv_heads = int(kv_heads)
        self.publish_mxfp4_v = bool(publish_mxfp4_v)
        self.v_mxfp4_scale_2d = bool(v_mxfp4_scale_2d)
        self.represented_backward = bool(represented_backward)
        self.per_block_qk_scales = bool(per_block_qk_scales)
        self.experimental_split_v_backward = bool(
            experimental_split_v_backward
        )
        self.abi_validation_symbol = (
            b300_require_qkv_gqa_d64_paired_unified_lowp_e4m3_projection(
                publish_mxfp4_v=self.publish_mxfp4_v,
                represented_backward=self.represented_backward,
                per_block_qk_scales=self.per_block_qk_scales,
                experimental_split_v_backward=(
                    self.experimental_split_v_backward
                ),
            )
        )
        self.requires_v_mxfp4_scales_out = False
        self.requires_forward_workspace = True
        compact_suffix = (
            "_mx_forward_out"
            if self.publish_mxfp4_v
            else "_fp8_forward_out"
        )
        self.checked_symbol = self.abi_validation_symbol + compact_suffix
        self.unchecked_symbol = self.checked_symbol + "_unchecked"
        # ``symbol`` describes the steady-state timed target.  The checked
        # symbol remains explicit provenance for first-use authentication.
        self.symbol = self.unchecked_symbol
        missing_symbols = [
            symbol
            for symbol in (self.checked_symbol, self.unchecked_symbol)
            if getattr(_C_b300_lowp_bwd, symbol, None) is None
        ]
        if missing_symbols:
            extension_path = getattr(
                _C_b300_lowp_bwd, "__file__", "<unknown>"
            )
            raise RuntimeError(
                "the selected low-precision extension "
                f"{extension_path!r} does not provide required compact "
                f"projection specializations {missing_symbols!r}"
            )
        self._project_checked = getattr(
            _C_b300_lowp_bwd, self.checked_symbol
        )
        self._project_unchecked = getattr(
            _C_b300_lowp_bwd, self.unchecked_symbol
        )
        # Weak values preserve id-collision protection while allowing a
        # layer's replaced private workspace to release its HBM allocation.
        self._validated_forward_workspaces: WeakValueDictionary[
            int, B300E4M3QKVForwardWorkspace
        ] = WeakValueDictionary()
        self._successful_full_abi_validation_count = 0

    @property
    def abi_validated(self) -> bool:
        return self._successful_full_abi_validation_count > 0

    @property
    def forward_workspace_abi_validated(self) -> bool:
        return self._successful_full_abi_validation_count > 0

    @property
    def validated_forward_workspace_count(self) -> int:
        return len(self._validated_forward_workspaces)

    @property
    def successful_full_abi_validation_count(self) -> int:
        return self._successful_full_abi_validation_count

    @property
    def vscale_out_abi_validated(self) -> bool:
        return bool(
            self.publish_mxfp4_v
            and self._successful_full_abi_validation_count > 0
        )

    def __call__(
        self,
        input_operand: tuple[torch.Tensor, torch.Tensor],
        qkv_weight_operand: tuple[torch.Tensor, torch.Tensor],
        paired_qk_scales: torch.Tensor,
        paired_rope_packed: torch.Tensor,
        *,
        forward_workspace: B300E4M3QKVForwardWorkspace,
    ) -> B300UnifiedLowpQKV:
        if not isinstance(
            forward_workspace, B300E4M3QKVForwardWorkspace
        ):
            raise TypeError(
                "the bound production projection requires a "
                "B300E4M3QKVForwardWorkspace"
            )
        input_fp8, input_row_decode = input_operand
        weight_fp8, weight_channel_decode = qkv_weight_operand
        workspace_key = id(forward_workspace)
        validated_workspace = self._validated_forward_workspaces.get(
            workspace_key
        )
        first_use = validated_workspace is None
        if not first_use and validated_workspace is not forward_workspace:
            raise RuntimeError("forward workspace identity collision")
        legacy_bundle = None
        if first_use:
            legacy_bundle = (
                b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(
                    input_operand,
                    qkv_weight_operand,
                    paired_qk_scales,
                    paired_rope_packed,
                    batch=self.batch,
                    seqlen=self.seqlen,
                    q_heads=self.q_heads,
                    kv_heads=self.kv_heads,
                    publish_mxfp4_v=self.publish_mxfp4_v,
                    v_mxfp4_scale_2d=self.v_mxfp4_scale_2d,
                    interleave_causal_kv=self.publish_mxfp4_v,
                    represented_backward=self.represented_backward,
                    per_block_qk_scales=self.per_block_qk_scales,
                    experimental_split_v_backward=(
                        self.experimental_split_v_backward
                    ),
                )
            )
        project = (
            self._project_checked if first_use else self._project_unchecked
        )
        backward_publications = project(
            input_fp8,
            input_row_decode,
            weight_fp8,
            weight_channel_decode,
            paired_qk_scales,
            paired_rope_packed,
            self.batch,
            self.seqlen,
            self.q_heads,
            self.kv_heads,
            self.v_mxfp4_scale_2d,
            forward_workspace.q_payload,
            forward_workspace.k_payload,
            forward_workspace.q_scale_pages,
            forward_workspace.q_global_scale,
            forward_workspace.k_scale_pages,
            forward_workspace.k_global_scale,
            forward_workspace.v_mxfp4_payload,
            forward_workspace.v_mxfp4_scale_pages,
            forward_workspace.v_fp8_payload,
            forward_workspace.v_backward_fp8,
            forward_workspace.q_backward_fp8,
            forward_workspace.k_backward_fp8,
        )
        try:
            backward_tuple = tuple(backward_publications)
        except TypeError as error:
            raise RuntimeError(
                "compact projection must return backward {V,Q,K} tensors"
            ) from error
        if len(backward_tuple) != 3:
            raise RuntimeError(
                "compact projection must return exactly three backward "
                "{V,Q,K} tensors"
            )
        if first_use:
            assert legacy_bundle is not None
            common_pairs = (
                (
                    "Q payload",
                    legacy_bundle.backward.score_q_fp4,
                    forward_workspace.q_payload,
                ),
                (
                    "K payload",
                    legacy_bundle.backward.score_k_fp4,
                    forward_workspace.k_payload,
                ),
                (
                    "Q scale pages",
                    legacy_bundle.q_forward_scales,
                    forward_workspace.q_scale_pages,
                ),
                (
                    "Q global scale",
                    legacy_bundle.q_forward_global_scale,
                    forward_workspace.q_global_scale,
                ),
                (
                    "K scale pages",
                    legacy_bundle.k_forward_scales,
                    forward_workspace.k_scale_pages,
                ),
                (
                    "K global scale",
                    legacy_bundle.k_forward_global_scale,
                    forward_workspace.k_global_scale,
                ),
            )
            for name, legacy, compact in common_pairs:
                _b300_require_bitwise_equal(name, legacy, compact)
            if self.publish_mxfp4_v:
                _b300_require_bitwise_equal(
                    "MXFP4 V payload",
                    legacy_bundle.v_forward_fp4,
                    forward_workspace.v_mxfp4_payload,
                )
                _b300_require_bitwise_equal(
                    "MXFP4 V scale pages",
                    legacy_bundle.v_forward_scales,
                    forward_workspace.v_mxfp4_scale_pages,
                    valid_last_dim_indices=tuple(
                        depth_lane * 16
                        + depth_group * 4
                        + sequence_quarter
                        for depth_lane in range(32)
                        for depth_group in range(2)
                        for sequence_quarter in range(4)
                    ),
                )
            else:
                if legacy_bundle.v_forward_fp8 is None:
                    raise RuntimeError(
                        "legacy exact-FP8 projection omitted forward V"
                    )
                _b300_require_bitwise_equal(
                    "FP8 V payload",
                    legacy_bundle.v_forward_fp8,
                    forward_workspace.v_fp8_payload,
                )
            legacy_backward = (
                legacy_bundle.v_backward_fp8,
                legacy_bundle.q_backward_fp8,
                legacy_bundle.k_backward_fp8,
            )
            if any(tensor is None for tensor in legacy_backward):
                raise RuntimeError(
                    "legacy projection omitted a backward {V,Q,K} publication"
                )
            for name, legacy, compact in zip(
                ("backward V", "backward Q", "backward K"),
                legacy_backward,
                backward_tuple,
                strict=True,
            ):
                assert legacy is not None
                _b300_require_bitwise_equal(name, legacy, compact)
            self._validated_forward_workspaces[
                workspace_key
            ] = forward_workspace
            self._successful_full_abi_validation_count += 1
        return _b300_compact_e4m3_qkv_bundle(
            forward_workspace,
            backward_tuple,
            paired_qk_scales,
            q_heads=self.q_heads,
            kv_heads=self.kv_heads,
            publish_mxfp4_v=self.publish_mxfp4_v,
        )


def b300_bind_qkv_gqa_d64_paired_unified_lowp_e4m3_projection(
    *,
    batch: int,
    seqlen: int,
    q_heads: int,
    kv_heads: int,
    publish_mxfp4_v: bool = False,
    v_mxfp4_scale_2d: bool = False,
    represented_backward: bool = False,
    per_block_qk_scales: bool = False,
    experimental_split_v_backward: bool = False,
) -> B300BoundE4M3QKVProjection:
    """Bind one exact E4M3 projection specialization and fixed shape."""
    return B300BoundE4M3QKVProjection(
        batch=batch,
        seqlen=seqlen,
        q_heads=q_heads,
        kv_heads=kv_heads,
        publish_mxfp4_v=publish_mxfp4_v,
        v_mxfp4_scale_2d=v_mxfp4_scale_2d,
        represented_backward=represented_backward,
        per_block_qk_scales=per_block_qk_scales,
        experimental_split_v_backward=experimental_split_v_backward,
    )


def b300_project_qkv_gqa_d64_paired_unified_lowp_e4m3(
    input_operand: tuple[torch.Tensor, torch.Tensor],
    qkv_weight_operand: tuple[torch.Tensor, torch.Tensor],
    paired_qk_scales: torch.Tensor,
    paired_rope_packed: torch.Tensor,
    *,
    batch: int,
    seqlen: int,
    q_heads: int,
    kv_heads: int,
    publish_mxfp4_v: bool = False,
    v_mxfp4_scale_2d: bool = False,
    interleave_causal_kv: bool | None = None,
    represented_backward: bool = False,
    per_block_qk_scales: bool = False,
    experimental_split_v_backward: bool = False,
) -> B300UnifiedLowpQKV:
    """Project paired-D64 QKV with dense E4M3 and publish FA4 operands.

    Activations use one decode scale per input row and weights one decode
    scale per output channel.  Decode is applied to the FP32 tensor-core
    accumulator in registers; pair-native RoPE and the existing NVFP4-QK /
    E4M3-V publishers consume that fragment directly.  By default the
    specialization instantiates no MXFP4-V work.  Set ``publish_mxfp4_v`` for
    the interleaved causal MX forward; slots 12/13 then carry the existing
    feature-major payload and E8M0 scale pages.  The forward-only publication
    defaults to one scale per depth row and 32 sequence values; this avoids
    coupling unrelated depth rows and removes the warp-wide tile-max
    reduction.  Set ``v_mxfp4_scale_2d=True`` only for compatibility with the
    coarser 32x32 policy used when forward and backward MX publications must
    represent one transposable operand. ``interleave_causal_kv`` defaults to
    the selected route: exact FP8 uses normal K/V order, while MXFP4 uses the
    causal-interleaved K/V order expected by its forward kernel. Both modes
    retain normal-order FP8 backward operands and never materialize a BF16
    QKV matrix. Exact FP8 additionally publishes its feature-major V operand;
    MX omits that redundant mirror. ``represented_backward=True`` publishes
    backward E4M3 Q/K from the retained forward NVFP4 codes; for MXFP4 it
    also publishes backward V from the retained forward MXFP4 codes/scales.
    ``per_block_qk_scales=True`` selects one E4M3 scale per logical Q/K row
    and K16 block; it requires represented backward publication so forward
    and backward consume the same retained E2M1 codes. The experimental
    ``experimental_split_v_backward=True`` specialization is an MX-only A/B:
    Q/K still come from those represented per-block NVFP4 codes, but backward
    V is published directly from the projection accumulator rather than from
    an MXFP4 encode/decode/lift round trip. It intentionally changes backward
    V numerics and is not the default.
    The default preserves the independent backward quantization contract for
    compatibility with older extensions and checkpoints.
    """
    _ensure_lowp_bwd_extension()
    if len(input_operand) != 2 or len(qkv_weight_operand) != 2:
        raise ValueError(
            "E4M3 projection operands must contain a payload and decode "
            "vector"
        )
    input_fp8, input_row_decode = input_operand
    weight_fp8, weight_channel_decode = qkv_weight_operand
    for name, payload, decode in (
        ("input", input_fp8, input_row_decode),
        ("qkv_weight", weight_fp8, weight_channel_decode),
    ):
        if (
            payload.dtype != torch.float8_e4m3fn
            or not payload.is_cuda
            or payload.ndim != 2
            or not payload.is_contiguous()
        ):
            raise ValueError(
                f"{name} payload must be a contiguous two-dimensional "
                "CUDA float8_e4m3fn matrix"
            )
        if (
            decode.dtype != torch.float32
            or not decode.is_cuda
            or decode.ndim != 1
            or not decode.is_contiguous()
            or decode.device != payload.device
        ):
            raise ValueError(
                f"{name} decode must be a contiguous one-dimensional CUDA "
                "float32 vector on the payload device"
            )
        if decode.numel() != payload.shape[0]:
            raise ValueError(
                f"{name} decode must contain one value per payload row"
            )
    if input_fp8.device != weight_fp8.device:
        raise ValueError("input and QKV weight operands must share one device")
    if interleave_causal_kv is None:
        interleave_causal_kv = bool(publish_mxfp4_v)
    if bool(interleave_causal_kv) != bool(publish_mxfp4_v):
        raise ValueError(
            "exact-FP8 requires normal K/V order while MXFP4 requires "
            "causal-interleaved K/V order; interleave_causal_kv must match "
            "publish_mxfp4_v"
        )
    if batch <= 0 or seqlen <= 0:
        raise ValueError("batch and sequence length must be positive")
    if (
        q_heads <= 0
        or kv_heads <= 0
        or q_heads % 2
        or kv_heads % 2
        or q_heads % kv_heads
    ):
        raise ValueError(
            "paired D64 projection requires positive even Hq/Hkv and Hq "
            "divisible by Hkv"
        )
    rows = batch * seqlen
    total_width = (q_heads + 2 * kv_heads) * 64
    if (
        tuple(input_fp8.shape[:1]) != (rows,)
        or weight_fp8.shape[0] != total_width
        or input_fp8.shape[1] != weight_fp8.shape[1]
    ):
        raise ValueError(
            "paired E4M3 projection requires input [B*S,K] and weight "
            "[(Hq+2*Hkv)*64,K]"
        )
    if seqlen % 256 or input_fp8.shape[1] % 128 or total_width % 256:
        raise ValueError(
            "paired E4M3 projection requires S divisible by 256, K "
            "divisible by 128, and total QKV width divisible by 256"
        )
    if (
        paired_qk_scales.dtype != torch.float32
        or not paired_qk_scales.is_cuda
        or not paired_qk_scales.is_contiguous()
        or paired_qk_scales.device != input_fp8.device
        or tuple(paired_qk_scales.shape) != (batch, q_heads // 2, 7)
    ):
        raise ValueError(
            "paired_qk_scales must be contiguous CUDA float32 [B,Hq/2,7] "
            "on the operand device"
        )
    if (
        paired_rope_packed.dtype != torch.int32
        or not paired_rope_packed.is_cuda
        or not paired_rope_packed.is_contiguous()
        or paired_rope_packed.device != input_fp8.device
        or tuple(paired_rope_packed.shape) != (batch, seqlen, 64)
    ):
        raise ValueError(
            "paired_rope_packed must be contiguous CUDA int32 [B,S,64] on "
            "the operand device"
        )
    project_name = (
        b300_require_qkv_gqa_d64_paired_unified_lowp_e4m3_projection(
            publish_mxfp4_v=publish_mxfp4_v,
            represented_backward=represented_backward,
            per_block_qk_scales=per_block_qk_scales,
            experimental_split_v_backward=experimental_split_v_backward,
        )
    )
    project = getattr(_C_b300_lowp_bwd, project_name)
    projected = project(
        input_fp8,
        input_row_decode,
        weight_fp8,
        weight_channel_decode,
        paired_qk_scales,
        paired_rope_packed,
        int(batch),
        int(seqlen),
        int(q_heads),
        int(kv_heads),
        bool(publish_mxfp4_v),
        bool(v_mxfp4_scale_2d),
        bool(interleave_causal_kv),
    )
    if len(projected) not in (23, 24):
        raise RuntimeError(
            "paired E4M3 projection returned an unsupported publication "
            f"tuple of length {len(projected)} (expected 23 or 24)"
        )
    empty_publications = (0, 1, 2, 3, 5, *range(14, 20))
    if any(projected[index].numel() for index in empty_publications):
        raise RuntimeError(
            "paired E4M3 no-materialization route unexpectedly returned "
            "BF16, aligned-Q/K, backward-MXFP4, or pure-Q/K publications"
        )
    if publish_mxfp4_v:
        for tensor, dtype, shape in (
            (
                projected[12],
                torch.float4_e2m1fn_x2,
                (batch, kv_heads, 64, seqlen // 2),
            ),
            (
                projected[13],
                torch.float8_e4m3fn,
                (batch, seqlen // 128, kv_heads, 512),
            ),
        ):
            if (
                tensor.dtype != dtype
                or tuple(tensor.shape) != shape
                or not tensor.is_contiguous()
                or tensor.device != input_fp8.device
            ):
                raise RuntimeError(
                    "paired E4M3 projection returned an invalid MXFP4 V "
                    f"publication: expected contiguous {dtype} {shape}, got "
                    f"{tensor.dtype} {tuple(tensor.shape)}"
                )
    elif projected[12].numel() or projected[13].numel():
        raise RuntimeError(
            "FP8-only E4M3 projection unexpectedly published MXFP4 V"
        )
    backward = B300AdaptiveLowpOperands(
        q_fp4=projected[3],
        score_q_fp4=projected[4],
        k_fp4=projected[5],
        score_k_fp4=projected[6],
        qk_scales=projected[7],
    )
    if (
        backward.qk_scales.dtype != torch.float32
        or tuple(backward.qk_scales.shape) != (batch, q_heads // 2, 7)
        or not backward.qk_scales.is_contiguous()
        or backward.qk_scales.device != input_fp8.device
    ):
        raise RuntimeError(
            "paired E4M3 projection returned invalid adaptive Q/K scales"
        )
    for tensor, shape in (
        (backward.score_q_fp4, (batch, q_heads, seqlen, 32)),
        (backward.score_k_fp4, (batch, kv_heads, seqlen, 32)),
    ):
        if (
            tensor.dtype != torch.uint8
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.device != input_fp8.device
        ):
            raise RuntimeError(
                "paired E4M3 projection returned an invalid Q/K payload: "
                f"expected contiguous uint8 {shape}, got {tensor.dtype} "
                f"{tuple(tensor.shape)}"
            )
    v_backward_fp8 = projected[20]
    q_backward_fp8 = projected[21]
    k_backward_fp8 = projected[22]
    for tensor, dtype, shape in (
        (
            projected[8],
            torch.float8_e4m3fn,
            (batch, seqlen // 128, q_heads, 512),
        ),
        (projected[9], torch.float32, (batch, q_heads)),
        (
            projected[10],
            torch.float8_e4m3fn,
            (batch, seqlen // 64, kv_heads, 512),
        ),
        (projected[11], torch.float32, (batch, kv_heads)),
        (
            v_backward_fp8,
            torch.float8_e4m3fn,
            (batch, seqlen, kv_heads, 64),
        ),
        (
            q_backward_fp8,
            torch.float8_e4m3fn,
            (batch, seqlen, q_heads, 64),
        ),
        (
            k_backward_fp8,
            torch.float8_e4m3fn,
            (batch, seqlen, kv_heads, 64),
        ),
    ):
        if (
            tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.device != input_fp8.device
        ):
            raise RuntimeError(
                "paired E4M3 projection returned invalid publication: "
                f"expected contiguous {dtype} {shape}, got {tensor.dtype} "
                f"{tuple(tensor.shape)}"
            )
    v_forward_fp8_publication = (
        projected[23] if len(projected) == 24 else None
    )
    if publish_mxfp4_v:
        if (
            v_forward_fp8_publication is not None
            and v_forward_fp8_publication.numel()
        ):
            raise RuntimeError(
                "paired E4M3 MXFP4-V projection unexpectedly published "
                "redundant feature-major FP8 V"
            )
        v_forward_fp8 = None
    else:
        v_forward_fp8 = v_forward_fp8_publication
        if v_forward_fp8 is not None:
            expected_forward_fp8 = (batch, kv_heads, 64, seqlen)
            if (
                v_forward_fp8.dtype != torch.float8_e4m3fn
                or tuple(v_forward_fp8.shape) != expected_forward_fp8
                or not v_forward_fp8.is_contiguous()
                or v_forward_fp8.device != input_fp8.device
            ):
                raise RuntimeError(
                    "paired E4M3 projection returned an invalid "
                    "feature-major FP8 V operand: expected contiguous "
                    f"float8_e4m3fn {expected_forward_fp8}, got "
                    f"{v_forward_fp8.dtype} "
                    f"{tuple(v_forward_fp8.shape)}"
                )
    return B300UnifiedLowpQKV(
        q=None,
        k=None,
        v=None,
        backward=backward,
        q_forward_fp4=_b300_typed_fp4_alias(backward.score_q_fp4),
        k_forward_fp4=_b300_typed_fp4_alias(backward.score_k_fp4),
        q_forward_scales=projected[8],
        q_forward_global_scale=projected[9],
        k_forward_scales=projected[10],
        k_forward_global_scale=projected[11],
        v_forward_fp4=projected[12],
        v_forward_scales=projected[13],
        v_backward_fp4=projected[14],
        v_backward_scales=projected[15],
        v_backward_fp8=v_backward_fp8,
        q_backward_fp8=q_backward_fp8,
        k_backward_fp8=k_backward_fp8,
        q_dk_fp4=None,
        k_dq_fp4=None,
        q_dk_scales=None,
        k_dq_scales=None,
        q_heads=int(q_heads),
        kv_heads=int(kv_heads),
        head_dim=64,
        v_forward_fp8=v_forward_fp8,
    )


def b300_project_gqa_d64_paired_qkv_gradient_nvfp4(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    projection_weight_operand: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    gradient_global_scale: torch.Tensor,
    paired_rope_packed: torch.Tensor,
    *,
    return_operand: bool = False,
    dq_decode_scale: float = 1.0,
    dk_decode_scale: float = 1.0,
    dv_decode_scale: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project D64 GQA gradients through paired physical D128 tiles.

    Adjacent D64 heads are contiguous, so viewing each pair as one D128 head
    lets the established inverse-RoPE/NVFP4 consumer issue full-width TMA
    loads.  The resulting logical matrix remains [all dQ | all dK | all dV]
    and is byte-identical to packing the materialized D64 matrix directly.
    ``paired_rope_packed`` should be prepared once with
    :func:`b300_pack_gqa_d64_paired_rope`.
    """
    for name, tensor in (("dQ", dq), ("dK", dk), ("dV", dv)):
        if (
            tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.ndim != 4
            or tensor.shape[-1] != 64
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA BF16 [B,S,H,64]"
            )
    if dq.shape[:2] != dk.shape[:2] or dk.shape != dv.shape:
        raise ValueError("D64 dQ/dK/dV must share batch and sequence")
    q_heads = dq.shape[2]
    kv_heads = dk.shape[2]
    if q_heads % 2 or kv_heads % 2 or q_heads % kv_heads:
        raise ValueError(
            "paired D64 projection requires even Hq/Hkv and Hq divisible by Hkv"
        )
    if (
        paired_rope_packed.dtype != torch.int32
        or not paired_rope_packed.is_cuda
        or not paired_rope_packed.is_contiguous()
        or tuple(paired_rope_packed.shape)
        != (dq.shape[0], dq.shape[1], 64)
    ):
        raise ValueError(
            "paired D64 RoPE must be contiguous CUDA int32 [B,S,64]"
        )
    return b300_project_gqa_d128_hierarchical_qkv_gradient_nvfp4(
        dq.view(dq.shape[0], dq.shape[1], q_heads // 2, 128),
        dk.view(dk.shape[0], dk.shape[1], kv_heads // 2, 128),
        dv.view(dv.shape[0], dv.shape[1], kv_heads // 2, 128),
        projection_weight_operand,
        gradient_global_scale,
        paired_rope_packed,
        return_operand=return_operand,
        dq_decode_scale=dq_decode_scale,
        dk_decode_scale=dk_decode_scale,
        dv_decode_scale=dv_decode_scale,
    )


def b300_stitch_gqa_d64_inverse_rope_gradient(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    gradient_scale: float | None = None,
    dv_decode: float = 0.25,
    q_gradient_scale: float | None = None,
    k_gradient_scale: float | None = None,
    v_gradient_scale: float | None = None,
) -> torch.Tensor:
    """Publish the projection-native D64 GQA gradient matrix in one pass.

    The consumer layout is ``[all dQ | all dK | all dV]``.  dQ/dK receive
    pair-native inverse RoPE.  The explicit per-field scales make Q/K/V decode
    ownership independent.  ``gradient_scale`` plus ``dv_decode`` remains a
    compatibility shorthand when no per-field scale is supplied.  This
    replaces the materialized float conversions, two inverse-RoPE graphs, and
    three-way concatenation used by the functional fallback.
    """
    _ensure_lowp_bwd_extension()
    explicit = (q_gradient_scale, k_gradient_scale, v_gradient_scale)
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError("Q/K/V gradient scales must be supplied together")
        if gradient_scale is not None:
            raise ValueError(
                "gradient_scale cannot be combined with per-field scales"
            )
        q_scale = float(q_gradient_scale)
        k_scale = float(k_gradient_scale)
        v_scale = float(v_gradient_scale)
    else:
        if gradient_scale is None:
            raise ValueError("gradient_scale or explicit Q/K/V scales required")
        q_scale = float(gradient_scale)
        k_scale = float(gradient_scale)
        v_scale = float(gradient_scale) * float(dv_decode)
    return _C_b300_lowp_bwd.stitch_gqa_d64_inverse_rope_grad(
        dq,
        dk,
        dv,
        rope_cos,
        rope_sin,
        q_scale,
        k_scale,
        v_scale,
    )


def b300_stitch_gqa_d128_inverse_rope_gradient(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    q_gradient_scale: float,
    k_gradient_scale: float,
    v_gradient_scale: float,
) -> torch.Tensor:
    """Publish the projection-native D128 GQA gradient matrix in one pass.

    The consumer layout is ``[all dQ | all dK | all dV]``. dQ/dK receive
    pair-native inverse RoPE while all fields are decoded directly into the
    BF16 matrix consumed by the QKV weight-gradient GEMM. This avoids the two
    functional inverse-RoPE graphs and three-way concatenation.
    """
    _ensure_lowp_bwd_extension()
    return _C_b300_lowp_bwd.stitch_gqa_d128_inverse_rope_grad(
        dq,
        dk,
        dv,
        rope_cos,
        rope_sin,
        float(q_gradient_scale),
        float(k_gradient_scale),
        float(v_gradient_scale),
    )


def b300_pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(
    dq_or_lanes: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    gradient_global_scale: torch.Tensor,
    rope_packed: torch.Tensor,
    output_operand: tuple[torch.Tensor, torch.Tensor],
    dq_tile_arrivals: torch.Tensor,
    *,
    row_tile_begin: int,
    row_tile_end: int,
    col_tile_begin: int,
    col_tile_end: int,
    arrival_epoch: int = 0,
) -> None:
    """Publish a ready tile range of the hierarchical D128 QKV operand.

    This frontier-driven API is authenticated only for B1. dQ may be
    row-major [1,S,Hq,128] or hierarchical
    [2,1,Hq,S,128]. A positive arrival_epoch makes the current CUDA stream
    wait until every selected dQ head has completed the causal reduction
    frontier ending at row_tile_end. K/V ranges use epoch zero and are
    normally issued after the attention completion event.
    """
    _ensure_lowp_bwd_extension()
    if (
        dk.ndim != 4
        or dv.ndim != 4
        or int(dk.shape[0]) != 1
        or int(dv.shape[0]) != 1
    ):
        raise ValueError(
            "tile-ready D128 QKV packing is authenticated only for batch 1"
        )
    if len(output_operand) != 2:
        raise ValueError("tile-ready output must contain payload and scales")
    _C_b300_lowp_bwd.pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(
        dq_or_lanes,
        dk,
        dv,
        gradient_global_scale,
        rope_packed,
        *output_operand,
        dq_tile_arrivals,
        int(row_tile_begin),
        int(row_tile_end),
        int(col_tile_begin),
        int(col_tile_end),
        int(arrival_epoch),
    )


def b300_project_qk_adaptive_lowp_nvfp4(
    input_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    qk_weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    qk_scales: torch.Tensor,
    *,
    batch: int,
    seqlen: int,
    heads: int,
    publication: str = "auto",
    mixed_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, B300AdaptiveLowpOperands]:
    """Project Q/K with NVFP4 and publish every adaptive backward layout.

    ``input_operand`` and ``qk_weight_operand`` are the packed data, E4M3
    block scales, and global scale produced by the native NVFP4 operand
    :func:`b300_prepare_nvfp4_projection_operand` (or an equivalent upstream
    producer).  Projection weights should be prepared once rather than
    quantized in the training hot path.

    ``publication='auto'`` keeps packing inside the projection epilogue for
    reduction widths of at least 3072.  Shallower projections use the lighter
    projection specialization followed by the wider standalone packer.  Both
    routes return the same :class:`B300AdaptiveLowpOperands`, so every consumer
    of the adaptive Q/K contract sees identical tensors and metadata.
    """
    _ensure_lowp_bwd_extension()
    if publication not in _NVFP4_PROJECTION_PUBLICATION_POLICIES:
        raise ValueError(
            "publication must be one of 'auto', 'fused', or 'separate'"
        )
    if len(input_operand) != 3 or len(qk_weight_operand) != 3:
        raise ValueError(
            "NVFP4 projection operands must contain packed data, block "
            "scales, and one global scale"
        )
    projected = _C_b300_lowp_bwd.project_qk_adaptive_fp4_nvfp4_dispatch(
        *input_operand,
        *qk_weight_operand,
        qk_scales,
        int(batch),
        int(seqlen),
        int(heads),
        _NVFP4_PROJECTION_PUBLICATION_POLICIES[publication],
    )
    q, k = projected[:2]
    operands = B300AdaptiveLowpOperands(*projected[2:], mixed_v=mixed_v)
    _check_adaptive_lowp_operands(q, k, operands)
    return q, k, operands


def b300_project_qkv_unified_lowp_nvfp4(
    input_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    qkv_weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    qk_scales: torch.Tensor,
    *,
    batch: int,
    seqlen: int,
    heads: int,
    store_bf16: bool = True,
    publish_pure_qk: bool = False,
    pure_qk_single_quant: bool = False,
    publish_fp8_backward: bool = False,
    rope_cos: torch.Tensor | None = None,
    rope_sin: torch.Tensor | None = None,
) -> B300UnifiedLowpQKV:
    """Project Q/K/V once and publish the native forward/backward layouts.

    The packed weight rows are ordered as all Q heads, then all K heads, then
    all V heads.  Q and K have depth 192 and V has depth 128.  The projection
    epilogue publishes the four adaptive backward layouts, the forward NVFP4
    scale pages, and transposed MXFP4 V directly from its BF16-rounded register
    fragment.  Set ``publish_pure_qk=True`` to additionally emit the compact
    fixed-scale Q/K operands consumed by pure-FP4 dK/dQ.  This is opt-in so
    the hybrid route does not pay for a second Q/K quantization.
    ``pure_qk_single_quant=True`` is the aggressive pure-only epilogue: it
    quantizes Q/K once at the fixed x16 scale and fans those codes into all six
    backward layouts.  ``qk_scales`` may be an empty CUDA float32 tensor in
    that mode because no adaptive metadata is read.  ``bundle.backward`` is
    intentionally not an adaptive-hybrid operand; use
    :meth:`B300UnifiedLowpQKV.pure_backward_operands`.
    Set ``store_bf16=False`` when every downstream consumer accepts the
    low-precision bundle.  Supplying ``rope_cos`` and ``rope_sin`` enables
    fused full-head RoPE before every Q/K publication.  The tables use the
    compact BF16 shape ``[B, S, 96]``.  Q/K weight rows must first be converted
    with :func:`b300_pair_interleave_qk_projection_weights`; this one-time
    layout change lets each epilogue register rotate an adjacent coordinate
    pair without cross-TMEM traffic.  V uses one E8M0 scale per 32x32
    sequence--depth tile, shared by the forward- and backward-oriented
    publications.
    """
    _ensure_lowp_bwd_extension()
    if len(input_operand) != 3 or len(qkv_weight_operand) != 3:
        raise ValueError(
            "NVFP4 projection operands must contain packed data, block "
            "scales, and one global scale"
        )
    if (rope_cos is None) != (rope_sin is None):
        raise ValueError("rope_cos and rope_sin must be supplied together")
    if rope_cos is None:
        projected = _C_b300_lowp_bwd.project_qkv_unified_fp4_nvfp4(
            *input_operand,
            *qkv_weight_operand,
            qk_scales,
            int(batch),
            int(seqlen),
            int(heads),
            bool(store_bf16),
            bool(publish_pure_qk),
            bool(pure_qk_single_quant),
            bool(publish_fp8_backward),
        )
    else:
        projected = _C_b300_lowp_bwd.project_qkv_unified_fp4_nvfp4_rope(
            *input_operand,
            *qkv_weight_operand,
            qk_scales,
            rope_cos,
            rope_sin,
            int(batch),
            int(seqlen),
            int(heads),
            bool(store_bf16),
            bool(publish_pure_qk),
            bool(pure_qk_single_quant),
            bool(publish_fp8_backward),
        )
    q_raw, k_raw, v_raw = projected[:3]
    backward = B300AdaptiveLowpOperands(
        q_fp4=projected[3],
        score_q_fp4=projected[4],
        k_fp4=projected[5],
        score_k_fp4=projected[6],
        qk_scales=projected[7],
    )
    expected_byte_layouts = (
        (backward.q_fp4, (batch, heads, _QK_HEAD_DIM, seqlen)),
        (backward.score_q_fp4, (batch, heads, seqlen, _QK_HEAD_DIM // 2)),
        (backward.k_fp4, (batch, seqlen, heads, _QK_HEAD_DIM)),
        (backward.score_k_fp4, (batch, heads, seqlen, _QK_HEAD_DIM // 2)),
    )
    for tensor, shape in expected_byte_layouts:
        if tensor.dtype != torch.uint8 or tuple(tensor.shape) != shape:
            raise RuntimeError(
                f"unified projection returned an invalid Q/K layout: "
                f"expected uint8 {shape}, got {tensor.dtype} {tuple(tensor.shape)}"
            )
    q_forward_scales = projected[8]
    q_forward_global_scale = projected[9]
    k_forward_scales = projected[10]
    k_forward_global_scale = projected[11]
    v_forward_fp4 = projected[12]
    v_forward_scales = projected[13]
    v_backward_fp4 = projected[14]
    v_backward_scales = projected[15]
    q_dk_raw = projected[16]
    k_dq_raw = projected[17]
    q_dk_scales_raw = projected[18]
    k_dq_scales_raw = projected[19]
    v_backward_fp8_raw = projected[20]
    expected_forward = (
        (
            q_forward_scales,
            torch.float8_e4m3fn,
            (batch, seqlen // 128, heads * 3, 512),
        ),
        (q_forward_global_scale, torch.float32, (batch, heads)),
        (
            k_forward_scales,
            torch.float8_e4m3fn,
            (batch, seqlen // 64, heads * 3, 512),
        ),
        (k_forward_global_scale, torch.float32, (batch, heads)),
        (
            v_forward_fp4,
            torch.float4_e2m1fn_x2,
            (batch, heads, _V_HEAD_DIM, seqlen // 2),
        ),
        (
            v_forward_scales,
            torch.uint8,
            (batch, heads, 2, seqlen // 128, 32, 16),
        ),
    )
    for tensor, dtype, shape in expected_forward:
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise RuntimeError(
                "unified projection returned an invalid forward operand: "
                f"expected {dtype} {shape}, got {tensor.dtype} {tuple(tensor.shape)}"
            )
    if publish_fp8_backward:
        if v_backward_fp4.numel() or v_backward_scales.numel():
            raise RuntimeError(
                "hybrid QKV projection unexpectedly published MXFP4 V "
                "backward operands"
            )
        expected_fp8_shape = (batch, seqlen, heads, _V_HEAD_DIM)
        if (
            v_backward_fp8_raw.dtype != torch.float8_e4m3fn
            or tuple(v_backward_fp8_raw.shape) != expected_fp8_shape
        ):
            raise RuntimeError(
                "unified projection returned an invalid hybrid FP8 V "
                f"operand: expected float8_e4m3fn {expected_fp8_shape}, "
                f"got {v_backward_fp8_raw.dtype} "
                f"{tuple(v_backward_fp8_raw.shape)}"
            )
        v_backward_fp8 = v_backward_fp8_raw
    else:
        expected_backward_mx = (
            (
                v_backward_fp4,
                torch.uint8,
                (batch, seqlen, heads, _V_HEAD_DIM // 2),
            ),
            (
                v_backward_scales,
                torch.uint8,
                (batch, seqlen // 128, heads, 512),
            ),
        )
        for tensor, dtype, shape in expected_backward_mx:
            if tensor.dtype != dtype or tuple(tensor.shape) != shape:
                raise RuntimeError(
                    "unified projection returned an invalid backward V "
                    f"operand: expected {dtype} {shape}, got {tensor.dtype} "
                    f"{tuple(tensor.shape)}"
                )
        if v_backward_fp8_raw.numel():
            raise RuntimeError("hybrid FP8 V was unexpectedly published")
        v_backward_fp8 = None
    if publish_pure_qk:
        expected_pure_qk = (
            (q_dk_raw, (batch, heads, _QK_HEAD_DIM, seqlen // 2)),
            (k_dq_raw, (batch, heads, _QK_HEAD_DIM, seqlen // 2)),
        )
        for tensor, shape in expected_pure_qk:
            if tensor.dtype != torch.uint8 or tuple(tensor.shape) != shape:
                raise RuntimeError(
                    "unified projection returned an invalid compact pure-Q/K "
                    f"layout: expected uint8 {shape}, got {tensor.dtype} "
                    f"{tuple(tensor.shape)}"
                )
        if q_dk_scales_raw.numel() or k_dq_scales_raw.numel():
            raise RuntimeError(
                "fixed-scale pure-Q/K projection must return empty scale tensors"
            )
        q_dk_fp4 = q_dk_raw
        k_dq_fp4 = k_dq_raw
        q_dk_scales = q_dk_scales_raw
        k_dq_scales = k_dq_scales_raw
    else:
        if q_dk_raw.numel() or k_dq_raw.numel():
            raise RuntimeError(
                "compact pure-Q/K operands were unexpectedly published"
            )
        q_dk_fp4 = None
        k_dq_fp4 = None
        q_dk_scales = None
        k_dq_scales = None
    q = q_raw if store_bf16 else None
    k = k_raw if store_bf16 else None
    v = v_raw if store_bf16 else None
    if store_bf16 and not pure_qk_single_quant:
        _check_adaptive_lowp_operands(q_raw, k_raw, backward)
    elif store_bf16:
        if tuple(q_raw.shape) != (
            batch,
            seqlen,
            heads,
            _QK_HEAD_DIM,
        ) or tuple(k_raw.shape) != tuple(q_raw.shape):
            raise RuntimeError(
                "pure unified projection returned invalid BF16 Q/K tensors"
            )
        if tuple(v_raw.shape) != (batch, seqlen, heads, _V_HEAD_DIM):
            raise RuntimeError("unified projection returned an invalid BF16 V")
    return B300UnifiedLowpQKV(
        q=q,
        k=k,
        v=v,
        backward=backward,
        q_forward_fp4=_b300_typed_fp4_alias(backward.score_q_fp4),
        k_forward_fp4=_b300_typed_fp4_alias(backward.score_k_fp4),
        q_forward_scales=q_forward_scales,
        q_forward_global_scale=q_forward_global_scale,
        k_forward_scales=k_forward_scales,
        k_forward_global_scale=k_forward_global_scale,
        v_forward_fp4=v_forward_fp4,
        v_forward_scales=v_forward_scales,
        v_backward_fp4=v_backward_fp4,
        v_backward_scales=v_backward_scales,
        v_backward_fp8=v_backward_fp8,
        q_dk_fp4=q_dk_fp4,
        k_dq_fp4=k_dq_fp4,
        q_dk_scales=q_dk_scales,
        k_dq_scales=k_dq_scales,
        pure_qk_single_quant=bool(pure_qk_single_quant),
        q_heads=int(heads),
        kv_heads=int(heads),
        head_dim=_QK_HEAD_DIM,
    )


def b300_pack_gqa_d128_rope(
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    """Pack static D128 BF16 cosine/sine tables into one word per pair."""
    if (
        rope_cos.dtype != torch.bfloat16
        or rope_sin.dtype != torch.bfloat16
        or rope_cos.ndim != 3
        or tuple(rope_cos.shape) != tuple(rope_sin.shape)
        or rope_cos.shape[2] != 64
    ):
        raise ValueError("D128 RoPE tables must be matching BF16 [B, S, 64]")
    if rope_cos.device != rope_sin.device or not rope_cos.is_cuda:
        raise ValueError("D128 RoPE tables must share one CUDA device")
    paired = torch.stack((rope_cos, rope_sin), dim=-1).contiguous()
    return paired.view(torch.int32).reshape(rope_cos.shape).contiguous()


def b300_project_qkv_gqa_d128_unified_lowp_nvfp4(
    input_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    qkv_weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    qk_scales: torch.Tensor,
    *,
    batch: int,
    seqlen: int,
    q_heads: int,
    kv_heads: int,
    store_bf16: bool = True,
    publish_fp8_backward: bool = True,
    v_mxfp4_scale_2d: bool = False,
    per_block_qk_scales: bool = False,
    rope_cos: torch.Tensor | None = None,
    rope_sin: torch.Tensor | None = None,
    rope_packed: torch.Tensor | None = None,
    cluster_cap: int | None = None,
    cache_packed_rope: bool | None = None,
    cache_adaptive_qk_scale: bool | None = None,
) -> B300UnifiedLowpQKV:
    """Project real-Llama D128 GQA directly into FA4 operand layouts.

    Weight rows use ``[all Q, all K, all V]`` order with widths
    ``Hq*128``, ``Hkv*128``, and ``Hkv*128``.  The projection epilogue can
    apply pair-native RoPE, publishes NVFP4 Q/K and MXFP4 V for the causal GQA
    forward, and publishes scaled E4M3 Q/K/V for backward.  By default each
    MXFP4 V scale covers one 1x32 sequence group.  ``v_mxfp4_scale_2d=True``
    is an explicit 32x32 ablation and must never be inferred from a C++
    default.  ``per_block_qk_scales=True`` publishes true row-by-K16 Q/K
    scales for forward while leaving the independent E4M3 backward operands
    unchanged.  A table prepared once by :func:`b300_pack_gqa_d128_rope`
    halves RoPE load instructions.
    BF16 Q/K/V publication is independently optional, so the production path
    does not materialize redundant full-precision operands.  No standalone
    Q/K/V packing launch is required.
    """
    _ensure_lowp_bwd_extension()
    if len(input_operand) != 3 or len(qkv_weight_operand) != 3:
        raise ValueError(
            "NVFP4 projection operands must contain packed data, block "
            "scales, and one global scale"
        )
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("q_heads must be positive and divisible by kv_heads")
    if (rope_cos is None) != (rope_sin is None):
        raise ValueError("rope_cos and rope_sin must be supplied together")
    if rope_packed is not None and rope_cos is not None:
        raise ValueError("split and packed RoPE tables are mutually exclusive")
    if cluster_cap is None:
        # The D128 Hq32/Hkv8 projection has 384 and 768 output tiles at the
        # two long-context Llama-8B shapes.  Controlled, interleaved launch
        # sweeps show that slightly reducing resident clusters trims the
        # uneven final wave without giving up too much parallelism.
        cluster_cap = {4096: 68, 8192: 72}.get(seqlen, 0)
    if cache_packed_rope is None:
        cache_packed_rope = rope_packed is not None
    if cache_adaptive_qk_scale is None:
        # One lookup per D128 head wins once enough row tiles amortize the
        # slightly longer scalar lifetime.  Same-binary sweeps retain the
        # original per-slice loads at the two shortest production shapes.
        cache_adaptive_qk_scale = (
            rope_packed is not None and seqlen >= 2048
        )
    if cluster_cap < 0:
        raise ValueError("cluster_cap must be non-negative")
    if (
        cluster_cap or cache_packed_rope or cache_adaptive_qk_scale
    ) and rope_packed is None:
        raise ValueError(
            "cluster_cap, cache_packed_rope, and cache_adaptive_qk_scale "
            "require packed RoPE"
        )
    arguments = (
        *input_operand,
        *qkv_weight_operand,
        qk_scales,
    )
    suffix = (
        int(batch),
        int(seqlen),
        int(q_heads),
        int(kv_heads),
        bool(store_bf16),
        bool(publish_fp8_backward),
        bool(v_mxfp4_scale_2d),
        bool(per_block_qk_scales),
    )
    if rope_packed is not None:
        packed_project_name = (
            "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed_clustered"
            if cluster_cap or cache_packed_rope or cache_adaptive_qk_scale
            else "project_qkv_gqa_d128_unified_fp4_nvfp4_rope_packed"
        )
        packed_project = getattr(_C_b300_lowp_bwd, packed_project_name)
        projected = (
            packed_project(
                *arguments,
                rope_packed,
                *suffix,
                *(
                    (
                        int(cluster_cap),
                        bool(cache_packed_rope),
                        bool(cache_adaptive_qk_scale),
                    )
                    if (
                        cluster_cap or cache_packed_rope or
                        cache_adaptive_qk_scale
                    )
                    else ()
                ),
            )
        )
    elif rope_cos is None:
        projected = (
            _C_b300_lowp_bwd.project_qkv_gqa_d128_unified_fp4_nvfp4(
                *arguments,
                *suffix,
            )
        )
    else:
        projected = (
            _C_b300_lowp_bwd.project_qkv_gqa_d128_unified_fp4_nvfp4_rope(
                *arguments,
                rope_cos,
                rope_sin,
                *suffix,
            )
        )

    q_raw, k_raw, v_raw = projected[:3]
    backward = B300AdaptiveLowpOperands(
        q_fp4=projected[3],
        score_q_fp4=projected[4],
        k_fp4=projected[5],
        score_k_fp4=projected[6],
        qk_scales=projected[7],
    )
    compact_only_qk = publish_fp8_backward and not store_bf16
    expected_qk = (
        (projected[4], (batch, q_heads, seqlen, 64)),
        (projected[6], (batch, kv_heads, seqlen, 64)),
    )
    for tensor, shape in expected_qk:
        if tensor.dtype != torch.uint8 or tuple(tensor.shape) != shape:
            raise RuntimeError(
                "D128 GQA projection returned an invalid Q/K operand: "
                f"expected uint8 {shape}, got {tensor.dtype} "
                f"{tuple(tensor.shape)}"
            )
    if compact_only_qk:
        if projected[3].numel() or projected[5].numel():
            raise RuntimeError(
                "D128 GQA compact-only projection unexpectedly published "
                "aligned Q/K backward operands"
            )
    else:
        expected_aligned_qk = (
            (projected[3], (batch, q_heads, 128, seqlen)),
            (projected[5], (batch, seqlen, kv_heads, 128)),
        )
        for tensor, shape in expected_aligned_qk:
            if tensor.dtype != torch.uint8 or tuple(tensor.shape) != shape:
                raise RuntimeError(
                    "D128 GQA projection returned an invalid aligned Q/K "
                    f"operand: expected uint8 {shape}, got {tensor.dtype} "
                    f"{tuple(tensor.shape)}"
                )
    q_forward_scales = projected[8]
    q_forward_global_scale = projected[9]
    k_forward_scales = projected[10]
    k_forward_global_scale = projected[11]
    v_forward_fp4 = projected[12]
    v_forward_scales = projected[13]
    expected_forward = (
        (
            q_forward_scales,
            torch.float8_e4m3fn,
            (batch, seqlen // 128, q_heads * 2, 512),
        ),
        (q_forward_global_scale, torch.float32, (batch, q_heads)),
        (
            k_forward_scales,
            torch.float8_e4m3fn,
            (batch, seqlen // 64, kv_heads * 2, 512),
        ),
        (k_forward_global_scale, torch.float32, (batch, kv_heads)),
        (
            v_forward_fp4,
            torch.float4_e2m1fn_x2,
            (batch, kv_heads, 128, seqlen // 2),
        ),
        (
            v_forward_scales,
            torch.float8_e4m3fn,
            (batch, seqlen // 128, kv_heads, 512),
        ),
    )
    for tensor, dtype, shape in expected_forward:
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise RuntimeError(
                "D128 GQA projection returned an invalid forward operand: "
                f"expected {dtype} {shape}, got {tensor.dtype} "
                f"{tuple(tensor.shape)}"
            )
    v_backward_fp4 = projected[14]
    v_backward_scales = projected[15]
    v_backward_fp8_raw = projected[20]
    q_backward_fp8_raw = projected[21]
    k_backward_fp8_raw = projected[22]
    v_forward_fp8_raw = (
        projected[23] if len(projected) > 23 and projected[23].numel() else None
    )
    if publish_fp8_backward:
        expected_v_fp8 = (batch, seqlen, kv_heads, 128)
        if (
            v_backward_fp8_raw.dtype != torch.float8_e4m3fn
            or tuple(v_backward_fp8_raw.shape) != expected_v_fp8
        ):
            raise RuntimeError(
                "D128 GQA projection returned an invalid FP8 V operand: "
                f"expected {expected_v_fp8}, got "
                f"{tuple(v_backward_fp8_raw.shape)}"
            )
        v_backward_fp8 = v_backward_fp8_raw
        expected_q_fp8 = (batch, seqlen, q_heads, 128)
        expected_k_fp8 = (batch, seqlen, kv_heads, 128)
        for name, tensor, shape in (
            ("Q", q_backward_fp8_raw, expected_q_fp8),
            ("K", k_backward_fp8_raw, expected_k_fp8),
        ):
            if (
                tensor.dtype != torch.float8_e4m3fn
                or tuple(tensor.shape) != shape
            ):
                raise RuntimeError(
                    f"D128 GQA projection returned an invalid FP8 {name} "
                    f"operand: expected {shape}, got {tuple(tensor.shape)}"
                )
        q_backward_fp8 = q_backward_fp8_raw
        k_backward_fp8 = k_backward_fp8_raw
        expected_forward_v_fp8 = (batch, kv_heads, 128, seqlen)
        if (
            v_forward_fp8_raw is None
            or v_forward_fp8_raw.dtype != torch.float8_e4m3fn
            or tuple(v_forward_fp8_raw.shape) != expected_forward_v_fp8
            or not v_forward_fp8_raw.is_contiguous()
            or v_forward_fp8_raw.device != input_operand[0].device
        ):
            actual = (
                None
                if v_forward_fp8_raw is None
                else (
                    v_forward_fp8_raw.dtype,
                    tuple(v_forward_fp8_raw.shape),
                )
            )
            raise RuntimeError(
                "D128 GQA projection returned an invalid feature-major FP8 "
                f"V operand: expected contiguous float8_e4m3fn "
                f"{expected_forward_v_fp8}, got {actual}"
            )
        v_forward_fp8 = v_forward_fp8_raw
    else:
        v_backward_fp8 = None
        q_backward_fp8 = None
        k_backward_fp8 = None
        if v_forward_fp8_raw is not None:
            raise RuntimeError(
                "D128 GQA projection published feature-major FP8 V while "
                "FP8 backward publication was disabled"
            )
        v_forward_fp8 = None
    if store_bf16:
        expected_bf16 = (
            (q_raw, (batch, seqlen, q_heads, 128)),
            (k_raw, (batch, seqlen, kv_heads, 128)),
            (v_raw, (batch, seqlen, kv_heads, 128)),
        )
        for tensor, shape in expected_bf16:
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
                raise RuntimeError(
                    "D128 GQA projection returned an invalid BF16 tensor: "
                    f"expected {shape}, got {tensor.dtype} "
                    f"{tuple(tensor.shape)}"
                )
    return B300UnifiedLowpQKV(
        q=q_raw if store_bf16 else None,
        k=k_raw if store_bf16 else None,
        v=v_raw if store_bf16 else None,
        backward=backward,
        q_forward_fp4=_b300_typed_fp4_alias(backward.score_q_fp4),
        k_forward_fp4=_b300_typed_fp4_alias(backward.score_k_fp4),
        q_forward_scales=q_forward_scales,
        q_forward_global_scale=q_forward_global_scale,
        k_forward_scales=k_forward_scales,
        k_forward_global_scale=k_forward_global_scale,
        v_forward_fp4=v_forward_fp4,
        v_forward_scales=v_forward_scales,
        v_backward_fp4=v_backward_fp4,
        v_backward_scales=v_backward_scales,
        v_backward_fp8=v_backward_fp8,
        q_backward_fp8=q_backward_fp8,
        k_backward_fp8=k_backward_fp8,
        q_dk_fp4=None,
        k_dq_fp4=None,
        q_dk_scales=None,
        k_dq_scales=None,
        q_heads=int(q_heads),
        kv_heads=int(kv_heads),
        head_dim=128,
        v_forward_fp8=v_forward_fp8,
    )


def b300_require_v509_e5m2_dout_route(
    backward_metadata: dict[str, object],
) -> dict[str, object]:
    """Authenticate the fused publisher/backward pair before v509 selection."""
    _ensure_lowp_bwd_extension()
    metadata_fn = getattr(
        _C_b300_lowp_bwd,
        "project_dout_unified_fp4_nvfp4_v509_e5m2_metadata",
        None,
    )
    if not callable(metadata_fn):
        raise RuntimeError(
            "v509 E5M2 dO route lacks exact fused publisher metadata"
        )
    publisher_raw = dict(metadata_fn())
    missing = {*V509_E5M2_DOUT_PUBLISHER_METADATA, "source_file"} - set(
        publisher_raw
    )
    publisher_mismatches = {
        field: {
            "actual": publisher_raw.get(field),
            "expected": expected,
        }
        for field, expected in V509_E5M2_DOUT_PUBLISHER_METADATA.items()
        if publisher_raw.get(field) != expected
        or type(publisher_raw.get(field)) is not type(expected)
    }
    source_file = publisher_raw.get("source_file")
    normalized_source = (
        source_file.replace("\\", "/")
        if isinstance(source_file, str)
        else ""
    )
    expected_suffix = "/tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu"
    if not (
        normalized_source == "lowp_fa4_bwd.cu"
        or normalized_source == expected_suffix.removeprefix("/")
        or normalized_source.endswith(expected_suffix)
    ):
        publisher_mismatches["source_file"] = {
            "actual": source_file,
            "expected_suffix": expected_suffix,
        }
    if missing:
        publisher_mismatches["missing"] = sorted(missing)

    from tk_fa4.lowp_fa4_bwd.native_tk_d128_nvfp4_score_e5m2_dout_backward import (
        EXPECTED_EXTENSION_METADATA as expected_backward,
    )

    if not isinstance(backward_metadata, dict):
        raise RuntimeError(
            "v509 E5M2 dO route requires exact backward metadata"
        )
    backward_mismatches = {
        field: {
            "actual": backward_metadata.get(field),
            "expected": expected,
        }
        for field, expected in expected_backward.items()
        if backward_metadata.get(field) != expected
        or type(backward_metadata.get(field)) is not type(expected)
    }
    missing_backward = set(expected_backward) - set(backward_metadata)
    if missing_backward:
        backward_mismatches["missing"] = sorted(missing_backward)
    if publisher_mismatches or backward_mismatches:
        raise RuntimeError(
            "v509 E5M2 dO route metadata mismatch: "
            f"publisher={publisher_mismatches}, "
            f"backward={backward_mismatches}"
        )
    return {
        "route": "v509_only_fail_closed",
        "publisher": dict(V509_E5M2_DOUT_PUBLISHER_METADATA),
        "backward": dict(expected_backward),
    }


def b300_require_v510_e5m2_dout_route(
    backward_metadata: dict[str, object],
) -> dict[str, object]:
    """Authenticate the fused publisher/dense-score v510 pair."""
    _ensure_lowp_bwd_extension()
    metadata_fn = getattr(
        _C_b300_lowp_bwd,
        "project_dout_unified_fp4_nvfp4_v510_e5m2_metadata",
        None,
    )
    if not callable(metadata_fn):
        raise RuntimeError(
            "v510 E5M2 dO route lacks exact fused publisher metadata"
        )
    publisher_raw = dict(metadata_fn())
    missing = {*V510_E5M2_DOUT_PUBLISHER_METADATA, "source_file"} - set(
        publisher_raw
    )
    publisher_mismatches = {
        field: {
            "actual": publisher_raw.get(field),
            "expected": expected,
        }
        for field, expected in V510_E5M2_DOUT_PUBLISHER_METADATA.items()
        if publisher_raw.get(field) != expected
        or type(publisher_raw.get(field)) is not type(expected)
    }
    source_file = publisher_raw.get("source_file")
    normalized_source = (
        source_file.replace("\\", "/")
        if isinstance(source_file, str)
        else ""
    )
    expected_suffix = "/tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu"
    if not (
        normalized_source == "lowp_fa4_bwd.cu"
        or normalized_source == expected_suffix.removeprefix("/")
        or normalized_source.endswith(expected_suffix)
    ):
        publisher_mismatches["source_file"] = {
            "actual": source_file,
            "expected_suffix": expected_suffix,
        }
    if missing:
        publisher_mismatches["missing"] = sorted(missing)

    from tk_fa4.lowp_fa4_bwd.native_tk_d128_dense_score_e5m2_dout_backward import (
        EXPECTED_EXTENSION_METADATA as expected_backward,
    )

    if not isinstance(backward_metadata, dict):
        raise RuntimeError(
            "v510 E5M2 dO route requires exact backward metadata"
        )
    backward_mismatches = {
        field: {
            "actual": backward_metadata.get(field),
            "expected": expected,
        }
        for field, expected in expected_backward.items()
        if backward_metadata.get(field) != expected
        or type(backward_metadata.get(field)) is not type(expected)
    }
    missing_backward = set(expected_backward) - set(backward_metadata)
    if missing_backward:
        backward_mismatches["missing"] = sorted(missing_backward)
    if publisher_mismatches or backward_mismatches:
        raise RuntimeError(
            "v510 E5M2 dO route metadata mismatch: "
            f"publisher={publisher_mismatches}, "
            f"backward={backward_mismatches}"
        )
    return {
        "route": "v510_only_fail_closed",
        "publisher": dict(V510_E5M2_DOUT_PUBLISHER_METADATA),
        "backward": dict(expected_backward),
    }


def b300_project_dout_unified_lowp_nvfp4(
    input_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    attention_output: torch.Tensor,
    lse: torch.Tensor,
    *,
    batch: int,
    seqlen: int,
    heads: int,
    store_bf16: bool = True,
    publish_fp8_backward: bool = False,
    publish_stats: bool = True,
    stats_workspace: torch.Tensor | None = None,
    dq_clear: torch.Tensor | None = None,
    probability_log2_lift: float = 0.0,
) -> B300UnifiedLowpDout:
    """Project dO and fuse every pure-MXFP4 backward publication.

    The epilogue emits row-major dO for dP, feature-major dO for dV, their
    exact E8M0 pages, ``dPsum``, and log2 LSE.  Fixed-E4M3 ``dPsum`` is
    accumulated from the exact rounded/saturated bytes consumed by dP;
    block-scaled publications use the BF16-rounded projection fragment.
    ``store_bf16=False`` avoids materializing dO when
    the downstream attention backward consumes only the low-precision bundle.
    For the hybrid route, ``publish_stats=False`` publishes only fixed-scale
    E4M3 dO; backward can then derive its row statistic from that same operand
    without extending the projection critical path or recreating dO.
    ``stats_workspace`` lets the hybrid epilogue publish CuTe-native negative
    ``dPsum`` and negative log2-LSE directly into the first two float32 pages
    of a backward uint8 workspace, eliminating a separate copy/sign launch.
    ``dq_clear`` assigns an otherwise idle producer warp to zero the direct
    BF16 dQ reduction target while projection is in flight.
    ``probability_log2_lift`` is a D128-only fused lstat offset. Native TK
    requests 8 so its probability operand is lifted by 256; the CuTe route
    keeps the default zero because it applies that lift internally.
    """
    _ensure_lowp_bwd_extension()
    probability_log2_lift = float(probability_log2_lift)
    if probability_log2_lift not in (0.0, 8.0):
        raise ValueError("probability_log2_lift must be exactly zero or eight")
    if probability_log2_lift and (
        int(attention_output.shape[-1]) != 128
        or not publish_fp8_backward
        or not publish_stats
        or stats_workspace is None
    ):
        raise ValueError(
            "a nonzero probability_log2_lift requires D128 direct FP8 "
            "backward statistics workspace publication"
        )
    if len(input_operand) != 3 or len(weight_operand) != 3:
        raise ValueError(
            "NVFP4 projection operands must contain packed data, block "
            "scales, and one global scale"
        )
    if stats_workspace is not None:
        if not publish_stats:
            raise ValueError("stats_workspace requires publish_stats=True")
        if not publish_fp8_backward:
            raise ValueError(
                "stats_workspace currently requires publish_fp8_backward=True"
            )
        minimum_bytes = 2 * batch * heads * seqlen * 4
        if (
            stats_workspace.dtype != torch.uint8
            or not stats_workspace.is_cuda
            or not stats_workspace.is_contiguous()
            or stats_workspace.ndim != 1
            or stats_workspace.device != input_operand[0].device
            or stats_workspace.numel() < minimum_bytes
        ):
            raise ValueError(
                "stats_workspace must be a contiguous 1D CUDA uint8 tensor "
                f"on the projection device with at least {minimum_bytes} bytes"
            )
    if dq_clear is not None:
        if not publish_fp8_backward:
            raise ValueError("dq_clear requires publish_fp8_backward=True")
        depth = int(attention_output.shape[-1])
        direct_shape = (batch, seqlen, heads, depth)
        hierarchical_shape = (2, batch, heads, seqlen, depth)
        if (
            dq_clear.dtype != torch.bfloat16
            or not dq_clear.is_cuda
            or not dq_clear.is_contiguous()
            or tuple(dq_clear.shape)
                not in (direct_shape, hierarchical_shape)
            or dq_clear.device != input_operand[0].device
        ):
            raise ValueError(
                "dq_clear must be contiguous BF16 [B,S,H,D] or "
                "[2,B,H,S,D] on the projection device"
            )
    projected = _C_b300_lowp_bwd.project_dout_unified_fp4_nvfp4(
        *input_operand,
        *weight_operand,
        attention_output,
        lse,
        int(batch),
        int(seqlen),
        int(heads),
        bool(store_bf16),
        bool(publish_fp8_backward),
        bool(publish_stats),
        stats_workspace,
        dq_clear,
        probability_log2_lift,
    )
    dout_raw = projected[0]
    result = B300UnifiedLowpDout(
        dout=dout_raw if store_bf16 else None,
        dout_storage=dout_raw,
        dout_dp_fp4=projected[1],
        dout_dp_scales=projected[2],
        dout_dv_fp4=projected[3],
        dout_dv_scales=projected[4],
        dpsum=projected[5],
        lse_log2=projected[6],
        dout_backward_fp8=(projected[7] if publish_fp8_backward else None),
    )
    expected_common = (
        (result.dpsum, torch.float32, (batch, heads, 1, seqlen)),
        (result.lse_log2, torch.float32, (batch, heads, 1, seqlen)),
    )
    expected_mx = (
        (result.dout_dp_fp4, torch.uint8, (batch, seqlen, heads, 64)),
        (
            result.dout_dp_scales,
            torch.uint8,
            (batch, seqlen // 128, heads, 512),
        ),
        (result.dout_dv_fp4, torch.uint8, (batch, heads, 128, seqlen // 2)),
        (
            result.dout_dv_scales,
            torch.uint8,
            (batch, seqlen // 128, heads, 512),
        ),
    )
    expected = (
        expected_common if publish_fp8_backward else expected_common + expected_mx
    ) if publish_stats else ()
    for tensor, dtype, shape in expected:
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise RuntimeError(
                "unified dO projection returned an invalid operand: "
                f"expected {dtype} {shape}, got {tensor.dtype} "
                f"{tuple(tensor.shape)}"
            )
    if publish_fp8_backward:
        for tensor in (
            result.dout_dp_fp4,
            result.dout_dp_scales,
            result.dout_dv_fp4,
            result.dout_dv_scales,
        ):
            if tensor.numel():
                raise RuntimeError(
                    "hybrid dO projection unexpectedly published MXFP4 operands"
                )
        assert result.dout_backward_fp8 is not None
        expected_fp8_shape = (
            batch,
            seqlen,
            heads,
            int(attention_output.shape[-1]),
        )
        if (
            result.dout_backward_fp8.dtype != torch.float8_e4m3fn
            or tuple(result.dout_backward_fp8.shape) != expected_fp8_shape
        ):
            raise RuntimeError(
                "unified dO projection returned an invalid hybrid FP8 "
                f"operand: expected float8_e4m3fn {expected_fp8_shape}, "
                f"got {result.dout_backward_fp8.dtype} "
                f"{tuple(result.dout_backward_fp8.shape)}"
            )
    elif projected[7].numel():
        raise RuntimeError("hybrid FP8 dO was unexpectedly published")
    if not publish_stats and (result.dpsum.numel() or result.lse_log2.numel()):
        raise RuntimeError(
            "statistics-free dO projection unexpectedly returned statistics"
        )
    return result


def b300_project_dout_unified_lowp_nvfp4_v509_e5m2(
    input_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    attention_output: torch.Tensor,
    lse: torch.Tensor,
    *,
    stats_workspace: torch.Tensor,
    dq_clear: torch.Tensor,
) -> B300V509E5M2DoutPublication:
    """Fuse the exact B1/S4096/H32/D128 E5M2 dO publication for v509.

    Route metadata pairing is deliberately authenticated by
    :func:`b300_require_v509_e5m2_dout_route` when the runtime selects v509,
    outside this low-level tensor API. This exact symbol has no format,
    scaling, statistics, or BF16-storage switches to broaden its ABI.
    """
    _ensure_lowp_bwd_extension()
    if len(input_operand) != 3 or len(weight_operand) != 3:
        raise ValueError(
            "v509 NVFP4 projection operands must contain packed data, block "
            "scales, and one global scale"
        )
    expected_shape = (1, 4096, 32, 128)
    if (
        attention_output.dtype != torch.bfloat16
        or not attention_output.is_cuda
        or not attention_output.is_contiguous()
        or tuple(attention_output.shape) != expected_shape
    ):
        raise ValueError(
            "v509 E5M2 dO publication requires contiguous CUDA BF16 "
            f"attention output {expected_shape}"
        )
    expected_stats_bytes = 2 * 1 * 32 * 4096 * torch.float32.itemsize
    if (
        stats_workspace.dtype != torch.uint8
        or not stats_workspace.is_cuda
        or not stats_workspace.is_contiguous()
        or stats_workspace.ndim != 1
        or stats_workspace.device != attention_output.device
        or stats_workspace.numel() != expected_stats_bytes
    ):
        raise ValueError(
            "v509 E5M2 dO publication requires the exact contiguous CUDA "
            f"uint8 statistics workspace with exactly {expected_stats_bytes} "
            "bytes"
        )
    if (
        dq_clear.dtype != torch.bfloat16
        or not dq_clear.is_cuda
        or not dq_clear.is_contiguous()
        or tuple(dq_clear.shape) != expected_shape
        or dq_clear.device != attention_output.device
    ):
        raise ValueError(
            "v509 E5M2 dO publication requires contiguous CUDA BF16 dQ "
            f"clear storage {expected_shape}"
        )
    projected = (
        _C_b300_lowp_bwd.project_dout_unified_fp4_nvfp4_v509_e5m2(
            *input_operand,
            *weight_operand,
            attention_output,
            lse,
            stats_workspace,
            dq_clear,
        )
    )
    if len(projected) != 8:
        raise RuntimeError(
            "v509 E5M2 dO projection returned a noncanonical raw ABI"
        )
    dout_storage = projected[0]
    for sentinel in projected[1:5]:
        if sentinel.numel():
            raise RuntimeError(
                "v509 E5M2 dO projection unexpectedly published MXFP4 or "
                "legacy E4M3 intermediates"
            )
    dpsum = projected[5]
    lse_log2 = projected[6]
    dout_backward_e5m2 = projected[7]
    expected_stats_shape = (1, 32, 1, 4096)
    for name, tensor in (("dpsum", dpsum), ("lse_log2", lse_log2)):
        if (
            tensor.dtype != torch.float32
            or tuple(tensor.shape) != expected_stats_shape
        ):
            raise RuntimeError(
                f"v509 E5M2 dO projection returned invalid {name}: "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
    if (
        dout_backward_e5m2.dtype != torch.float8_e5m2
        or tuple(dout_backward_e5m2.shape) != expected_shape
        or not dout_backward_e5m2.is_contiguous()
    ):
        raise RuntimeError(
            "v509 E5M2 dO projection slot 7 must be genuine contiguous "
            f"float8_e5m2 {expected_shape}"
        )
    if (
        dout_storage.dtype != torch.bfloat16
        or tuple(dout_storage.shape) != expected_shape
        or dout_storage.data_ptr() != attention_output.data_ptr()
    ):
        raise RuntimeError(
            "v509 E5M2 dO projection must reuse attention output only as "
            "descriptor storage and must not materialize BF16 dO"
        )
    return B300V509E5M2DoutPublication(
        dout_storage=dout_storage,
        dpsum=dpsum,
        lse_log2=lse_log2,
        dout_backward_e5m2=projected[7],
    )


def b300_project_dout_unified_lowp_nvfp4_v510_e5m2(
    input_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weight_operand: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    attention_output: torch.Tensor,
    lse: torch.Tensor,
    *,
    stats_workspace: torch.Tensor,
    dq_clear: torch.Tensor,
) -> B300V510E5M2DoutPublication:
    """Fuse exact B1/S4096/H32/D128 E5M2 dO for dense-score v510.

    Route metadata pairing is deliberately authenticated by
    :func:`b300_require_v510_e5m2_dout_route` when the runtime selects v510,
    outside this low-level tensor API. This exact symbol has no format,
    scaling, statistics, or BF16-storage switches to broaden its ABI.
    """
    _ensure_lowp_bwd_extension()
    if len(input_operand) != 3 or len(weight_operand) != 3:
        raise ValueError(
            "v510 NVFP4 projection operands must contain packed data, block "
            "scales, and one global scale"
        )
    expected_shape = (1, 4096, 32, 128)
    if (
        attention_output.dtype != torch.bfloat16
        or not attention_output.is_cuda
        or not attention_output.is_contiguous()
        or tuple(attention_output.shape) != expected_shape
    ):
        raise ValueError(
            "v510 E5M2 dO publication requires contiguous CUDA BF16 "
            f"attention output {expected_shape}"
        )
    expected_stats_bytes = 2 * 1 * 32 * 4096 * torch.float32.itemsize
    if (
        stats_workspace.dtype != torch.uint8
        or not stats_workspace.is_cuda
        or not stats_workspace.is_contiguous()
        or stats_workspace.ndim != 1
        or stats_workspace.device != attention_output.device
        or stats_workspace.numel() != expected_stats_bytes
    ):
        raise ValueError(
            "v510 E5M2 dO publication requires the exact contiguous CUDA "
            f"uint8 statistics workspace with exactly {expected_stats_bytes} "
            "bytes"
        )
    if (
        dq_clear.dtype != torch.bfloat16
        or not dq_clear.is_cuda
        or not dq_clear.is_contiguous()
        or tuple(dq_clear.shape) != expected_shape
        or dq_clear.device != attention_output.device
    ):
        raise ValueError(
            "v510 E5M2 dO publication requires contiguous CUDA BF16 dQ "
            f"clear storage {expected_shape}"
        )
    projected = (
        _C_b300_lowp_bwd.project_dout_unified_fp4_nvfp4_v510_e5m2(
            *input_operand,
            *weight_operand,
            attention_output,
            lse,
            stats_workspace,
            dq_clear,
        )
    )
    if len(projected) != 8:
        raise RuntimeError(
            "v510 E5M2 dO projection returned a noncanonical raw ABI"
        )
    dout_storage = projected[0]
    for sentinel in projected[1:5]:
        if sentinel.numel():
            raise RuntimeError(
                "v510 E5M2 dO projection unexpectedly published MXFP4 or "
                "legacy E4M3 intermediates"
            )
    dpsum = projected[5]
    lse_log2 = projected[6]
    dout_backward_e5m2 = projected[7]
    expected_stats_shape = (1, 32, 1, 4096)
    for name, tensor in (("dpsum", dpsum), ("lse_log2", lse_log2)):
        if (
            tensor.dtype != torch.float32
            or tuple(tensor.shape) != expected_stats_shape
        ):
            raise RuntimeError(
                f"v510 E5M2 dO projection returned invalid {name}: "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
    if (
        dout_backward_e5m2.dtype != torch.float8_e5m2
        or tuple(dout_backward_e5m2.shape) != expected_shape
        or not dout_backward_e5m2.is_contiguous()
    ):
        raise RuntimeError(
            "v510 E5M2 dO projection slot 7 must be genuine contiguous "
            f"float8_e5m2 {expected_shape}"
        )
    if (
        dout_storage.dtype != torch.bfloat16
        or tuple(dout_storage.shape) != expected_shape
        or dout_storage.data_ptr() != attention_output.data_ptr()
    ):
        raise RuntimeError(
            "v510 E5M2 dO projection must reuse attention output only as "
            "descriptor storage and must not materialize BF16 dO"
        )
    return B300V510E5M2DoutPublication(
        dout_storage=dout_storage,
        dpsum=dpsum,
        lse_log2=lse_log2,
        dout_backward_e5m2=projected[7],
    )


def b300_mha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    return_lse: bool = False,
):
    _ensure_forward_extensions()
    _check_exact_qkv_inputs(q, k, v)
    _resolve_scale(q, softmax_scale)

    seqlen = q.shape[1]
    q_pad = _pad_bshd(q)
    k_pad = _pad_bshd(k)
    v_pad = _pad_bshd(v)

    out_pad = torch.empty_like(v_pad)
    lse_pad = torch.empty(
        (q_pad.shape[0], q_pad.shape[2], 1, q_pad.shape[1]),
        dtype=torch.float32,
        device=q.device,
    )
    _select_forward_kernel(causal, seqlen)(q_pad, k_pad, v_pad, out_pad, lse_pad)

    out = out_pad[:, :seqlen].contiguous()
    lse = lse_pad[:, :, 0, :seqlen].permute(0, 2, 1).contiguous()
    return (out, lse) if return_lse else out


def b300_mha_fwd_with_mixed_v(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    return_lse: bool = False,
):
    """Run forward while preparing the reusable mixed FP8/MXFP4 V operand.

    The V conversion is a once-per-row sidecar enqueued directly after forward.
    Keeping it on the caller's stream avoids competing with the already
    occupancy-saturating forward kernel.  The packed tensor is intentionally
    opaque; pass it to the low-precision backward prepacked-V entry point.
    """
    _ensure_forward_extensions()
    _ensure_lowp_bwd_extension()
    _check_exact_qkv_inputs(q, k, v)
    _resolve_scale(q, softmax_scale)

    seqlen = q.shape[1]
    q_pad = _pad_bshd(q)
    k_pad = _pad_bshd(k)
    v_pad = _pad_bshd(v)

    out_pad = torch.empty_like(v_pad)
    lse_pad = torch.empty(
        (q_pad.shape[0], q_pad.shape[2], 1, q_pad.shape[1]),
        dtype=torch.float32,
        device=q.device,
    )
    _select_forward_kernel(causal, seqlen)(q_pad, k_pad, v_pad, out_pad, lse_pad)
    mixed_v = _C_b300_lowp_bwd.prepack_mixed_v(v_pad)

    out = out_pad[:, :seqlen].contiguous()
    lse = lse_pad[:, :, 0, :seqlen].permute(0, 2, 1).contiguous()
    return (out, lse, mixed_v) if return_lse else (out, mixed_v)


def b300_mha_fwd_with_adaptive_lowp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    return_lse: bool = False,
    prepare_mixed_v: bool = False,
    max_quant_scale: float = 16.0,
    min_quant_scale: float = 2.0**-12,
    scale_headroom: float = 0.325,
    rms_clip_multiple: float = 2.75,
    prepared_operands: B300AdaptiveLowpOperands | None = None,
    overlap_producer: bool = True,
):
    """Run forward while preparing adaptive Q/K operands for lowp backward.

    By default the sidecar producer overlaps forward on a dedicated CUDA
    stream and rejoins before return.  ``prepared_operands`` is a zero-copy
    path for a Q/K projection that emits the four E2M1 layouts directly.  A
    projection that emits only the seven-word record for each head can use
    :func:`b300_adaptive_lowp_operands_from_scales` to skip the reduction.

    Each head has seven scale words.  Words 0--4 are the common Q/K multiplier
    and backward correction factors, word 5 is producer scratch, and word 6
    contains the repeated E4M3 tensor-scale byte pattern used by score MMA.
    The producer clips at the larger of ``scale_headroom * amax`` and
    ``rms_clip_multiple * RMS`` so sparse outliers do not erase the bulk.
    """
    _ensure_forward_extensions()
    _ensure_lowp_bwd_extension()
    _check_exact_qkv_inputs(q, k, v)
    scale = _resolve_scale(q, softmax_scale)

    seqlen = q.shape[1]
    q_pad = _pad_bshd(q)
    k_pad = _pad_bshd(k)
    v_pad = _pad_bshd(v)
    out_pad = torch.empty_like(v_pad)
    lse_pad = torch.empty(
        (q_pad.shape[0], q_pad.shape[2], 1, q_pad.shape[1]),
        dtype=torch.float32,
        device=q.device,
    )
    if prepared_operands is not None:
        if q_pad.shape != q.shape:
            raise ValueError(
                "projection-produced adaptive operands require an already padded sequence"
            )
        _check_adaptive_lowp_operands(q_pad, k_pad, prepared_operands)

    needs_qk_producer = prepared_operands is None
    needs_mixed_v_producer = prepare_mixed_v and (
        prepared_operands is None or prepared_operands.mixed_v is None
    )
    can_overlap = (
        overlap_producer
        and (needs_qk_producer or needs_mixed_v_producer)
        and not torch.cuda.is_current_stream_capturing()
    )

    packed = None
    mixed_v = None if prepared_operands is None else prepared_operands.mixed_v
    if can_overlap:
        current_stream = torch.cuda.current_stream(q.device)
        producer_stream = _b300_adaptive_producer_stream(q.device)
        producer_stream.wait_stream(current_stream)
        with torch.cuda.stream(producer_stream):
            if needs_qk_producer:
                packed = _C_b300_lowp_bwd.quantize_fp4_dual_qk_adaptive(
                    q_pad,
                    k_pad,
                    float(max_quant_scale),
                    float(min_quant_scale),
                    float(scale_headroom),
                    float(rms_clip_multiple),
                    float(scale),
                    4096.0,
                )
            if needs_mixed_v_producer:
                mixed_v = _C_b300_lowp_bwd.prepack_mixed_v(v_pad)

        _select_forward_kernel(causal, seqlen)(
            q_pad,
            k_pad,
            v_pad,
            out_pad,
            lse_pad,
        )
        out = out_pad[:, :seqlen].contiguous()
        lse = lse_pad[:, :, 0, :seqlen].permute(0, 2, 1).contiguous()
        current_stream.wait_stream(producer_stream)
    else:
        _select_forward_kernel(causal, seqlen)(
            q_pad,
            k_pad,
            v_pad,
            out_pad,
            lse_pad,
        )
        if needs_qk_producer:
            packed = _C_b300_lowp_bwd.quantize_fp4_dual_qk_adaptive(
                q_pad,
                k_pad,
                float(max_quant_scale),
                float(min_quant_scale),
                float(scale_headroom),
                float(rms_clip_multiple),
                float(scale),
                4096.0,
            )
        if needs_mixed_v_producer:
            mixed_v = _C_b300_lowp_bwd.prepack_mixed_v(v_pad)
        out = out_pad[:, :seqlen].contiguous()
        lse = lse_pad[:, :, 0, :seqlen].permute(0, 2, 1).contiguous()

    if prepared_operands is None:
        assert packed is not None
        operands = B300AdaptiveLowpOperands(*packed, mixed_v=mixed_v)
    elif mixed_v is prepared_operands.mixed_v:
        operands = prepared_operands
    else:
        operands = B300AdaptiveLowpOperands(
            prepared_operands.q_fp4,
            prepared_operands.score_q_fp4,
            prepared_operands.k_fp4,
            prepared_operands.score_k_fp4,
            prepared_operands.qk_scales,
            mixed_v,
        )
    if can_overlap:
        for tensor in (
            operands.q_fp4,
            operands.score_q_fp4,
            operands.k_fp4,
            operands.score_k_fp4,
            operands.qk_scales,
        ):
            tensor.record_stream(current_stream)
        if operands.mixed_v is not None:
            operands.mixed_v.record_stream(current_stream)
    return (out, lse, operands) if return_lse else (out, operands)


def b300_mha_bwd_adaptive_lowp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    operands: B300AdaptiveLowpOperands,
    *,
    route: str = "winner",
    causal: bool = True,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    return_bf16_dq: bool = False,
):
    """Consume adaptive operands; optionally keep dQ in projection-ready BF16."""
    _ensure_lowp_bwd_extension()
    _check_exact_qkv_inputs(q, k, v)
    _check_exact_out(out, v, "out")
    _check_exact_out(dout, v, "dout")
    _check_adaptive_lowp_operands(q, k, operands)
    lse = _normalize_lse_bsh(lse, q)
    scale = _resolve_scale(q, softmax_scale)
    if route not in ("winner", "mixed"):
        raise ValueError(
            "adaptive Q/K route must be 'winner' or 'mixed'"
        )
    if route == "mixed" and return_bf16_dq:
        raise ValueError("the adaptive mixed route does not publish BF16 dQ")
    if route == "mixed" and operands.mixed_v is None:
        raise ValueError(
            "route='mixed' requires operands prepared with prepare_mixed_v=True"
        )
    if route == "mixed":
        return _C_b300_lowp_bwd.backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_adaptive_prepacked_v_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            operands.q_fp4,
            operands.score_q_fp4,
            operands.k_fp4,
            operands.score_k_fp4,
            operands.qk_scales,
            4096.0,
            causal,
            scale,
            deterministic,
            operands.mixed_v,
        )
    backward_name = (
        "backward_fp4_fp8dpdv_x32_split_dk_adaptive_bf16dq_native"
        if return_bf16_dq
        else "backward_fp4_fp8dpdv_x32_split_dk_adaptive_native"
    )
    backward = getattr(_C_b300_lowp_bwd, backward_name)
    return backward(
        q,
        k,
        v,
        out,
        lse,
        dout,
        operands.q_fp4,
        operands.score_q_fp4,
        operands.k_fp4,
        operands.score_k_fp4,
        operands.qk_scales,
        4096.0,
        causal,
        scale,
        deterministic,
    )


def b300_mha_bwd_adaptive_lowp_nvfp4_projection_dgrad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    operands: B300AdaptiveLowpOperands,
    projection_weight_operand: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    dq_global_scale: torch.Tensor,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Consume private BF16 dQ through a compact NVFP4 projection handoff.

    ``projection_weight_operand`` is prepared once from a linear weight with
    shape ``[hidden, heads * 192]``. ``dq_global_scale`` is delayed scaling
    state from a prior step (or calibration); keeping it device-resident lets
    the compact producer skip a matrix-wide dQ amax reduction. The returned
    tuple is projected dX, dK, and dV—standalone dQ is not published.
    """
    _ensure_lowp_bwd_extension()
    _check_exact_qkv_inputs(q, k, v)
    _check_exact_out(out, v, "out")
    _check_exact_out(dout, v, "dout")
    _check_adaptive_lowp_operands(q, k, operands)
    if len(projection_weight_operand) != 3:
        raise ValueError(
            "NVFP4 projection weight must contain packed data, block scales, "
            "and one global scale"
        )
    if (
        dq_global_scale.dtype != torch.float32
        or not dq_global_scale.is_cuda
        or dq_global_scale.device != q.device
        or not dq_global_scale.is_contiguous()
        or dq_global_scale.numel() != 1
    ):
        raise ValueError(
            "dq_global_scale must be one contiguous CUDA float32 value on "
            "the Q/K/V device"
        )
    lse = _normalize_lse_bsh(lse, q)
    scale = _resolve_scale(q, softmax_scale)
    projected, dk, dv = (
        _C_b300_lowp_bwd.
        backward_fp4_fp8dpdv_x32_split_dk_adaptive_nvfp4_dq_projection_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            operands.q_fp4,
            operands.score_q_fp4,
            operands.k_fp4,
            operands.score_k_fp4,
            operands.qk_scales,
            *projection_weight_operand,
            dq_global_scale,
            4096.0,
            causal,
            scale,
            deterministic,
        )
    )
    return projected, dk, dv


def b300_mha_bwd_adaptive_lowp_hierarchical_nvfp4_qkv_projection_dgrad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    operands: B300AdaptiveLowpOperands,
    dout_fp8: torch.Tensor,
    v_fp8: torch.Tensor,
    projection_weight_operand: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    gradient_global_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    stats_from_packed_dout: bool = True,
    causal: bool = True,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    _hierarchical_dq_reduction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold private dQ lanes directly into a stacked NVFP4 QKV projection.

    Attention keeps the even/odd dQ partials private.  The projection-native
    producer performs their final BF16 sum in registers, applies inverse RoPE
    to dQ/dK, and emits one delayed-scale NVFP4 ``[dQ | dK | dV]`` operand.
    Standalone completed BF16 dQ is therefore never materialized.  The cached
    projection weight operand must represent the stacked matrix
    ``[hidden, heads * 512]``.

    The returned tuple is projected dX plus the ordinary BF16 dK and dV
    outputs.  This interface covers the activation-gradient path; a training
    integration that also consumes the private tile for the QKV weight
    gradient remains a separate contract.
    """
    _ensure_lowp_bwd_extension()
    _check_exact_qkv_inputs(q, k, v)
    _check_exact_out(out, v, "out")
    _check_exact_out(dout, v, "dout")
    _check_adaptive_lowp_operands(q, k, operands)
    if len(projection_weight_operand) != 3:
        raise ValueError(
            "NVFP4 projection weight must contain packed data, block scales, "
            "and one global scale"
        )
    for name, tensor, reference in (
        ("dout_fp8", dout_fp8, dout),
        ("v_fp8", v_fp8, v),
    ):
        if (
            tensor.dtype != torch.float8_e4m3fn
            or not tensor.is_cuda
            or tensor.device != q.device
            or not tensor.is_contiguous()
            or tensor.shape != reference.shape
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA E4M3 with shape "
                f"{tuple(reference.shape)} on the Q/K/V device"
            )
    if (
        gradient_global_scale.dtype != torch.float32
        or not gradient_global_scale.is_cuda
        or gradient_global_scale.device != q.device
        or not gradient_global_scale.is_contiguous()
        or gradient_global_scale.numel() != 1
    ):
        raise ValueError(
            "gradient_global_scale must be one contiguous CUDA float32 value "
            "on the Q/K/V device"
        )
    expected_rope = q.shape[0] * q.shape[1] * (_QK_HEAD_DIM // 2)
    for name, tensor in (("rope_cos", rope_cos), ("rope_sin", rope_sin)):
        if (
            tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or tensor.device != q.device
            or not tensor.is_contiguous()
            or tensor.numel() != expected_rope
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA BF16 with "
                f"{expected_rope} elements on the Q/K/V device"
            )
    lse = _normalize_lse_bsh(lse, q)
    scale = _resolve_scale(q, softmax_scale)
    native = getattr(
        _C_b300_lowp_bwd,
        (
            "backward_fp4_fp8dpdv_x32_split_dk_adaptive_"
            "hierarchical_qkv_projection_native"
            if _hierarchical_dq_reduction
            else
            "backward_fp4_fp8dpdv_x32_split_dk_adaptive_"
            "stacked_qkv_projection_native"
        ),
    )
    projected, dk, dv = (
        native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            operands.q_fp4,
            operands.score_q_fp4,
            operands.k_fp4,
            operands.score_k_fp4,
            operands.qk_scales,
            dout_fp8,
            v_fp8,
            bool(stats_from_packed_dout),
            *projection_weight_operand,
            gradient_global_scale,
            rope_cos,
            rope_sin,
            4096.0,
            causal,
            scale,
            deterministic,
        )
    )
    return projected, dk, dv


def b300_mha_bwd_adaptive_lowp_stacked_nvfp4_qkv_projection_dgrad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    operands: B300AdaptiveLowpOperands,
    dout_fp8: torch.Tensor,
    v_fp8: torch.Tensor,
    projection_weight_operand: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    gradient_global_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    stats_from_packed_dout: bool = True,
    causal: bool = True,
    softmax_scale: float | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one completed BF16 dQ/dK/dV set through stacked NVFP4.

    This control keeps the normal fast attention schedule, then applies
    inverse RoPE while packing the three gradients into one stacked delayed-
    scale NVFP4 operand.  It avoids the tile-ready signaling overhead of the
    hierarchical experiment while preserving its projection geometry.
    """
    return b300_mha_bwd_adaptive_lowp_hierarchical_nvfp4_qkv_projection_dgrad(
        q,
        k,
        v,
        out,
        lse,
        dout,
        operands,
        dout_fp8,
        v_fp8,
        projection_weight_operand,
        gradient_global_scale,
        rope_cos,
        rope_sin,
        stats_from_packed_dout=stats_from_packed_dout,
        causal=causal,
        softmax_scale=softmax_scale,
        deterministic=deterministic,
        _hierarchical_dq_reduction=False,
    )


def b300_pair_interleave_qk_projection_weights(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert standard split-half Q/K rows to adjacent rotary pairs.

    This is a one-time model-weight layout conversion.  Within every
    192-dimensional head, rows ``[x_0..x_95, y_0..y_95]`` become
    ``[x_0, y_0, x_1, y_1, ...]``.  Applying the same permutation to Q and K
    preserves their dot product and makes full-head Llama RoPE local to each
    projection-epilogue ``bf16_2`` register.
    """
    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be a two-dimensional BF16 weight")
        if not weight.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if weight.shape[0] % _QK_HEAD_DIM:
            raise ValueError(f"{name} output width must be divisible by 192")
    if q_weight.device != k_weight.device:
        raise ValueError("Q/K projection weights must share one device")
    if q_weight.shape != k_weight.shape:
        raise ValueError("Q/K projection weights must have matching shapes")
    heads = q_weight.shape[0] // _QK_HEAD_DIM
    hidden = q_weight.shape[1]

    def pair_interleave(weight: torch.Tensor) -> torch.Tensor:
        per_head = weight.reshape(heads, _QK_HEAD_DIM, hidden)
        return torch.stack(
            (
                per_head[:, : _QK_HEAD_DIM // 2],
                per_head[:, _QK_HEAD_DIM // 2 :],
            ),
            dim=2,
        ).reshape_as(weight).contiguous()

    return pair_interleave(q_weight), pair_interleave(k_weight)


def b300_pair_interleave_gqa_d128_qk_projection_weights(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert asymmetric D128 GQA Q/K rows to adjacent rotary pairs."""
    for name, weight in (("q_weight", q_weight), ("k_weight", k_weight)):
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be a two-dimensional BF16 weight")
        if not weight.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if weight.shape[0] % 128:
            raise ValueError(f"{name} output width must be divisible by 128")
    if q_weight.device != k_weight.device:
        raise ValueError("Q/K projection weights must share one device")
    if q_weight.shape[1] != k_weight.shape[1]:
        raise ValueError("Q/K projection input widths must match")

    def pair_interleave(weight: torch.Tensor) -> torch.Tensor:
        heads = weight.shape[0] // 128
        hidden = weight.shape[1]
        per_head = weight.reshape(heads, 128, hidden)
        return torch.stack(
            (per_head[:, :64], per_head[:, 64:]),
            dim=2,
        ).reshape_as(weight).contiguous()

    return pair_interleave(q_weight), pair_interleave(k_weight)


def b300_inverse_rope_interleaved_qkv_grad_(
    qkv_grad: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply inverse pair-native RoPE to Q/K gradient fields in-place.

    ``qkv_grad`` uses the per-head ``[dQ192, dK192, dV128]`` layout returned
    by the fused low-precision backward.  The result is ready for projection
    dgrad or weight-gradient consumers whose Q/K weights use the adjacent-pair
    row order.
    """
    _ensure_lowp_bwd_extension()
    return _C_b300_lowp_bwd.inverse_rope_interleaved_qkv_grad_inplace(
        qkv_grad,
        rope_cos,
        rope_sin,
    )


def b300_rope_pair_qk_(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply full-head pair-native RoPE to contiguous BF16 Q/K in-place."""
    _ensure_lowp_bwd_extension()
    return tuple(
        _C_b300_lowp_bwd.rope_pair_qk_inplace(
            q,
            k,
            rope_cos,
            rope_sin,
        )
    )


def b300_stack_qkv_projection_weights(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> torch.Tensor:
    """Stack learned Q/K/V weights in the unified forward-projection order."""
    weights = (q_weight, k_weight, v_weight)
    names = ("q_weight", "k_weight", "v_weight")
    for name, weight in zip(names, weights):
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be a two-dimensional BF16 weight")
        if not weight.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not (q_weight.device == k_weight.device == v_weight.device):
        raise ValueError("Q/K/V projection weights must share one device")
    if not (q_weight.shape[1] == k_weight.shape[1] == v_weight.shape[1]):
        raise ValueError("Q/K/V projection input widths must match")
    if q_weight.shape[0] % _QK_HEAD_DIM != 0:
        raise ValueError("Q projection output width must be divisible by 192")
    heads = q_weight.shape[0] // _QK_HEAD_DIM
    if k_weight.shape[0] != heads * _QK_HEAD_DIM:
        raise ValueError("K projection weight does not match the Q head count")
    if v_weight.shape[0] != heads * _V_HEAD_DIM:
        raise ValueError("V projection weight does not match the Q head count")
    return torch.cat(weights, dim=0).contiguous()


def b300_stack_gqa_d128_qkv_projection_weights(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> torch.Tensor:
    """Stack asymmetric D128 GQA weights as all Q, then K, then V rows."""
    weights = (q_weight, k_weight, v_weight)
    names = ("q_weight", "k_weight", "v_weight")
    for name, weight in zip(names, weights):
        if weight.ndim != 2 or weight.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be a two-dimensional BF16 weight")
        if not weight.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if weight.shape[0] % 128:
            raise ValueError(f"{name} output width must be divisible by 128")
    if not (q_weight.device == k_weight.device == v_weight.device):
        raise ValueError("Q/K/V projection weights must share one device")
    if not (q_weight.shape[1] == k_weight.shape[1] == v_weight.shape[1]):
        raise ValueError("Q/K/V projection input widths must match")
    kv_heads = k_weight.shape[0] // 128
    if v_weight.shape[0] != kv_heads * 128:
        raise ValueError("V projection weight must match the KV head count")
    q_heads = q_weight.shape[0] // 128
    if q_heads % kv_heads:
        raise ValueError("Q head count must be divisible by KV head count")
    return torch.cat(weights, dim=0).contiguous()


def b300_interleave_qkv_projection_weights(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> torch.Tensor:
    """Pack Q/K/V projection weights for the fused backward consumer.

    Each input uses the PyTorch linear-weight layout ``[out_features,
    in_features]``. The returned ``[heads * 512, in_features]`` tensor is
    ordered per head as ``[Q192, K192, V128]`` and should be prepared once,
    rather than repacked in the training hot path.
    """
    weights = (q_weight, k_weight, v_weight)
    names = ("q_weight", "k_weight", "v_weight")
    for name, weight in zip(names, weights):
        if weight.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional linear weight")
        if weight.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16")
    if not (q_weight.device == k_weight.device == v_weight.device):
        raise ValueError("Q/K/V projection weights must share one device")
    if not (q_weight.shape[1] == k_weight.shape[1] == v_weight.shape[1]):
        raise ValueError("Q/K/V projection input widths must match")
    if q_weight.shape[0] % _QK_HEAD_DIM != 0:
        raise ValueError("Q projection output width must be divisible by 192")
    heads = q_weight.shape[0] // _QK_HEAD_DIM
    if k_weight.shape[0] != heads * _QK_HEAD_DIM:
        raise ValueError("K projection weight does not match the Q head count")
    if v_weight.shape[0] != heads * _V_HEAD_DIM:
        raise ValueError("V projection weight does not match the Q head count")
    hidden = q_weight.shape[1]
    return torch.cat(
        (
            q_weight.reshape(heads, _QK_HEAD_DIM, hidden),
            k_weight.reshape(heads, _QK_HEAD_DIM, hidden),
            v_weight.reshape(heads, _V_HEAD_DIM, hidden),
        ),
        dim=1,
    ).reshape(heads * (_QK_HEAD_DIM * 2 + _V_HEAD_DIM), hidden).contiguous()


def b300_mha_bwd_adaptive_lowp_projection_dgrad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    operands: B300AdaptiveLowpOperands,
    projection_weight: torch.Tensor,
    *,
    causal: bool = True,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    projection_splits: int | None = None,
    return_qkv_grad: bool = False,
):
    """Run adaptive attention backward into a projection-ready BF16 buffer.

    ``projection_weight`` must be packed once with
    :func:`b300_interleave_qkv_projection_weights`. The attention kernel emits
    only fully reduced BF16 gradients in per-head ``[Q, K, V]`` order, so no
    standalone dQ/dK/dV tensors or concatenation are needed. By default the
    consumer chooses one to four head-contiguous GEMMs according to the shape;
    this avoids inefficient monolithic projection shapes at H16/H24/H64.
    """
    _ensure_lowp_bwd_extension()
    _check_exact_qkv_inputs(q, k, v)
    _check_exact_out(out, v, "out")
    _check_exact_out(dout, v, "dout")
    _check_adaptive_lowp_operands(q, k, operands)
    lse = _normalize_lse_bsh(lse, q)
    if projection_weight.ndim != 2:
        raise ValueError("projection_weight must be two-dimensional")
    if projection_weight.dtype != torch.bfloat16:
        raise ValueError("projection_weight must be bfloat16")
    if projection_weight.device != q.device:
        raise ValueError("projection_weight must be on the Q/K/V device")
    if not projection_weight.is_contiguous():
        raise ValueError("projection_weight must be contiguous")
    projection_width = q.shape[2] * (_QK_HEAD_DIM * 2 + _V_HEAD_DIM)
    if projection_weight.shape[0] != projection_width:
        raise ValueError(
            "projection_weight leading dimension must equal heads * 512"
        )
    if projection_splits is None:
        heads = q.shape[2]
        if heads == 64:
            projection_splits = 4
        elif heads == 16:
            projection_splits = 2
        elif heads == 24 and q.shape[1] >= 8192:
            projection_splits = 2
        else:
            projection_splits = 1
    if projection_splits <= 0 or q.shape[2] % projection_splits != 0:
        raise ValueError(
            "projection_splits must be positive and divide the head count"
        )
    scale = _resolve_scale(q, softmax_scale)
    backward = getattr(
        _C_b300_lowp_bwd,
        "backward_fp4_fp8dpdv_x32_split_dk_adaptive_qkv_bf16_native",
    )
    (qkv_grad,) = backward(
        q,
        k,
        v,
        out,
        lse,
        dout,
        operands.q_fp4,
        operands.score_q_fp4,
        operands.k_fp4,
        operands.score_k_fp4,
        operands.qk_scales,
        4096.0,
        causal,
        scale,
        deterministic,
    )
    batch, seqlen = q.shape[:2]
    projection_input = qkv_grad.reshape(batch * seqlen, projection_width)
    split_width = projection_width // projection_splits
    dx = torch.mm(
        projection_input[:, :split_width],
        projection_weight[:split_width],
    )
    for split in range(1, projection_splits):
        start = split * split_width
        end = start + split_width
        dx.addmm_(
            projection_input[:, start:end],
            projection_weight[start:end],
        )
    dx = dx.reshape(batch, seqlen, projection_weight.shape[1])
    return (dx, qkv_grad) if return_qkv_grad else dx


def b300_mha_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    deterministic: bool = False,
):
    _ensure_backward_extension()
    _check_exact_qkv_inputs(q, k, v)
    _check_exact_out(out, v, "out")
    _check_exact_out(dout, v, "dout")
    lse = _normalize_lse_bsh(lse, q)

    softmax_scale = _resolve_scale(q, softmax_scale)
    seqlen = q.shape[1]

    q_pad = _to_bhsd(_pad_bshd(q))
    k_pad = _to_bhsd(_pad_bshd(k))
    v_pad = _to_bhsd(_pad_bshd(v))
    out_pad = _to_bhsd(_pad_bshd(out))
    dout_pad = _to_bhsd(_pad_bshd(dout))
    l_aux_pad = _lse_to_l_aux(lse, softmax_scale)

    dq, dk, dv = _C.b300_mha_bwd(
        q_pad,
        k_pad,
        v_pad,
        out_pad,
        l_aux_pad,
        dout_pad,
        causal,
        softmax_scale,
        seqlen,
        deterministic,
    )
    dq = _from_bhsd(dq[:, :, :seqlen, :])
    dk = _from_bhsd(dk[:, :, :seqlen, :])
    dv = _from_bhsd(dv[:, :, :seqlen, :])
    return dq, dk, dv


def b300_mha_bwd_dv_only(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    deterministic: bool = False,
):
    _ensure_backward_extension()
    _check_cuda_bf16_bshd(q, "q")
    _check_cuda_bf16_bshd(k, "k")
    _check_cuda_bf16_bshd(dout, "dout")
    if q.device != k.device or q.device != dout.device:
        raise ValueError("q, k, and dout must be on the same CUDA device")
    if q.shape[0] != k.shape[0] or q.shape[0] != dout.shape[0]:
        raise ValueError("batch dimensions must match")
    if q.shape[1] != k.shape[1] or q.shape[1] != dout.shape[1]:
        raise ValueError("sequence lengths must match")
    if q.shape[2] != k.shape[2] or q.shape[2] != dout.shape[2]:
        raise ValueError("head counts must match")
    if q.shape[3] != _QK_HEAD_DIM or k.shape[3] != _QK_HEAD_DIM:
        raise ValueError(f"q and k head_dim must be {_QK_HEAD_DIM}")
    if dout.shape[3] != _V_HEAD_DIM:
        raise ValueError(f"dout head_dim must be {_V_HEAD_DIM}")
    if q.shape[1] < _MIN_SEQ_LEN:
        raise ValueError(f"exact B300 path requires seqlen >= {_MIN_SEQ_LEN}")
    _check_sm100(q.device)
    lse = _normalize_lse_bsh(lse, q)

    softmax_scale = _resolve_scale(q, softmax_scale)
    seqlen = q.shape[1]

    q_pad = _to_bhsd(_pad_bshd(q))
    k_pad = _to_bhsd(_pad_bshd(k))
    dout_pad = _to_bhsd(_pad_bshd(dout))
    l_aux_pad = _lse_to_l_aux(lse, softmax_scale)

    dv = _C.b300_mha_bwd_dv_only_internal(
        q_pad,
        k_pad,
        dout_pad,
        l_aux_pad,
        causal,
        softmax_scale,
        seqlen,
        deterministic,
    )
    return _from_bhsd(dv[:, :, :seqlen, :])


def b300_mha_bwd_experimental(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    implementation: str = "auto",
):
    _ensure_backward_extension()
    if implementation == "hot":
        hot_fastpath = _try_hot_fastpath(
            q,
            k,
            v,
            out,
            lse,
            dout,
            causal=causal,
            softmax_scale=softmax_scale,
            deterministic=deterministic,
        )
        if hot_fastpath is not None:
            return hot_fastpath
    _check_exact_qkv_inputs(q, k, v)
    _check_exact_out(out, v, "out")
    _check_exact_out(dout, v, "dout")
    lse = _normalize_lse_bsh(lse, q)

    softmax_scale = _resolve_scale(q, softmax_scale)
    seqlen = q.shape[1]
    implementation = _resolve_experimental_impl(implementation, seqlen, causal, deterministic)

    if implementation == "hot":
        batch, seqlen, heads, _ = q.shape
        hot_backward_kernel = _select_hot_backward_kernel(batch, seqlen, heads)
        dq, dk, dv = hot_backward_kernel(
            _maybe_contiguous(q),
            _maybe_contiguous(k),
            _maybe_contiguous(v),
            _maybe_contiguous(out),
            _maybe_contiguous(lse),
            _maybe_contiguous(dout),
            causal,
            softmax_scale,
            seqlen,
            deterministic,
        )
        return dq[:, :seqlen, :, :], dk[:, :seqlen, :, :], dv[:, :seqlen, :, :]

    q_pad = _to_bhsd(_pad_bshd(q, multiple=_EXPERIMENTAL_PAD_MULTIPLE))
    k_pad = _to_bhsd(_pad_bshd(k, multiple=_EXPERIMENTAL_PAD_MULTIPLE))
    v_pad = _to_bhsd(_pad_bshd(v, multiple=_EXPERIMENTAL_PAD_MULTIPLE))
    out_pad = _to_bhsd(_pad_bshd(out, multiple=_EXPERIMENTAL_PAD_MULTIPLE))
    dout_pad = _to_bhsd(_pad_bshd(dout, multiple=_EXPERIMENTAL_PAD_MULTIPLE))
    lse_pad = _lse_to_bh1s(lse, _EXPERIMENTAL_PAD_MULTIPLE)

    if implementation == "hot":
        raise AssertionError("unreachable")
    else:
        dq, dk, dv = _C.b300_mha_bwd_fa4_style_ref(
            q_pad,
            k_pad,
            v_pad,
            out_pad,
            lse_pad,
            dout_pad,
            causal,
            softmax_scale,
            seqlen,
            deterministic,
        )
    dq = _from_bhsd(dq[:, :, :seqlen, :])
    dk = _from_bhsd(dk[:, :, :seqlen, :])
    dv = _from_bhsd(dv[:, :, :seqlen, :])
    return dq, dk, dv


class _B300FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
        deterministic: bool,
    ):
        out, lse = b300_mha_fwd(
            q,
            k,
            v,
            causal=causal,
            softmax_scale=softmax_scale,
            return_lse=True,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = _resolve_scale(q, softmax_scale)
        ctx.causal = causal
        ctx.deterministic = deterministic
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, dlse: torch.Tensor | None):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = b300_mha_bwd(
            q,
            k,
            v,
            out,
            lse,
            dout.contiguous(),
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None


class _B300FlashAttnFuncExperimental(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
        deterministic: bool,
        implementation: str,
    ):
        out, lse = b300_mha_fwd(
            q,
            k,
            v,
            causal=causal,
            softmax_scale=softmax_scale,
            return_lse=True,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = _resolve_scale(q, softmax_scale)
        ctx.causal = causal
        ctx.deterministic = deterministic
        ctx.implementation = implementation
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, dlse: torch.Tensor | None):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = b300_mha_bwd_experimental(
            q,
            k,
            v,
            out,
            lse,
            dout.contiguous(),
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
            deterministic=ctx.deterministic,
            implementation=ctx.implementation,
        )
        return dq, dk, dv, None, None, None, None


def b300_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
):
    out, lse = _B300FlashAttnFunc.apply(q, k, v, softmax_scale, causal, deterministic)
    return (out, lse) if return_lse else out


def b300_flash_attn_func_experimental(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    return_lse: bool = False,
    implementation: str = "auto",
):
    out, lse = _B300FlashAttnFuncExperimental.apply(
        q, k, v, softmax_scale, causal, deterministic, implementation
    )
    return (out, lse) if return_lse else out


def _warn_deprecated(name: str) -> None:
    warnings.warn(
        f"`tk_fa4.{name}` is deprecated and still routes through the legacy broad-shape implementation in `tk_fa4.deprecated`. "
        "Use `b300_mha_fwd`, `b300_mha_bwd`, or `b300_flash_attn_func` for the exact B300 fast path.",
        DeprecationWarning,
        stacklevel=2,
    )


def mha_fwd(*args, **kwargs):
    _warn_deprecated("mha_fwd")
    from .deprecated.interface import mha_fwd as _legacy_mha_fwd

    return _legacy_mha_fwd(*args, **kwargs)


def mha_bwd(*args, **kwargs):
    _warn_deprecated("mha_bwd")
    from .deprecated.interface import mha_bwd as _legacy_mha_bwd

    return _legacy_mha_bwd(*args, **kwargs)


def flash_attn_func(*args, **kwargs):
    _warn_deprecated("flash_attn_func")
    from .deprecated.interface import flash_attn_func as _legacy_flash_attn_func

    return _legacy_flash_attn_func(*args, **kwargs)
