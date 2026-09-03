from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import os
from pathlib import Path
from typing import Callable, List, Optional, Union
import sys

import torch
from torch import nn

from torchtitan.protocols.model_converter import ModelConverter, register_model_converter
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

from .job_config import JobConfig


_BF16_TOPOLOGY_CONVERTER = "fa4_exact_bf16_topology"
_SHA256_LENGTH = 64
_AUTHENTICATED_FLASH_OVERLAY = (
    (
        "flash_attn/cute/fp4_flash_bwd_sm100.py",
        "0e3c152ebcd0c2bf1ef0edc76fa108c0bb04c497d76c056749dbb57b1ed293f2",
        289_268,
    ),
    (
        "flash_attn/cute/interface.py",
        "13a1edbd711ae29141fceb69c54a8a93bc18384511792cebf3ee433ff220cd75",
        159_112,
    ),
    (
        "flash_attn/cute/mma_sm100_desc.py",
        "86efe9315696b7bdb7bfa915c7946e301d3e88d089964cb4f27a74c43d604d09",
        16_101,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _module_is_below(module: object, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        return Path(module_file).resolve().is_relative_to(root)
    except OSError:
        return False


@dataclass(frozen=True)
class _AuthenticatedFlashRuntime:
    root: Path
    interface_sha256: str

    @classmethod
    def from_job_config(cls, job_config: JobConfig) -> "_AuthenticatedFlashRuntime":
        cfg = job_config.fa4
        configured_root = str(getattr(cfg, "exact_flash_attn_root", ""))
        root = Path(configured_root)
        if not root.is_absolute():
            raise ValueError(
                "BF16 FA4 requires fa4.exact_flash_attn_root to be an "
                "absolute path"
            )
        root = root.resolve()
        configured_sha256 = str(
            getattr(cfg, "exact_flash_attn_source_sha256", "")
        ).lower()
        if not _is_sha256(configured_sha256):
            raise ValueError(
                "BF16 FA4 requires an exact "
                "fa4.exact_flash_attn_source_sha256 identity"
            )
        settings = cls(root=root, interface_sha256=configured_sha256)
        settings.authenticate_files()
        return settings

    def authenticate_files(self) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"authenticated FlashAttention root does not exist: {self.root}"
            )
        for relative, expected_sha256, expected_bytes in (
            _AUTHENTICATED_FLASH_OVERLAY
        ):
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise FileNotFoundError(
                    "authenticated FlashAttention overlay is incomplete: "
                    f"{path}"
                )
            actual_bytes = path.stat().st_size
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    "authenticated FlashAttention overlay byte mismatch for "
                    f"{relative}: {actual_bytes} != {expected_bytes}"
                )
            actual_sha256 = _sha256(path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "authenticated FlashAttention overlay SHA256 mismatch for "
                    f"{relative}: {actual_sha256} != {expected_sha256}"
                )
            if relative == "flash_attn/cute/interface.py" and (
                actual_sha256 != self.interface_sha256
            ):
                raise RuntimeError(
                    "authenticated FlashAttention interface SHA256 does not "
                    "match fa4.exact_flash_attn_source_sha256"
                )


def _prepend_python_path(path: Path) -> None:
    path_str = str(path)
    if path.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _purge_stale_modules(prefix: str, expected_root: Path) -> None:
    for name, module in list(sys.modules.items()):
        if name != prefix and not name.startswith(prefix + "."):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            module_path = Path(module_file).resolve()
        except OSError:
            continue
        if not module_path.is_relative_to(expected_root):
            del sys.modules[name]


def _prepend_exact_python_path(path: Path) -> None:
    path_str = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != path_str]
    sys.path.insert(0, path_str)


def _extend_fa4_python_paths() -> None:
    # The publication repository vendors the authenticated runtime beside
    # TorchTitan.  An explicit environment override is useful for developers,
    # but implicit workspace-wide discovery would make results depend on an
    # unrelated checkout and is therefore intentionally unsupported here.
    repository_root = Path(__file__).resolve().parents[3]
    configured_root = os.environ.get("FA4_FLASH_ATTN_ROOT", "").strip()
    flash_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else (repository_root / "flash-attention").resolve()
    )
    if not (flash_root / "flash_attn" / "cute").is_dir():
        raise FileNotFoundError(
            "FA4 FlashAttention source is missing. Initialize the pinned "
            f"flash-attention submodule or set FA4_FLASH_ATTN_ROOT: {flash_root}"
        )
    candidate_paths = [
        flash_root.parent,
        flash_root,
        flash_root / "csrc" / "cutlass" / "python" / "CuTeDSL",
    ]
    for path in candidate_paths:
        _prepend_python_path(path)
    _purge_stale_modules("flash_attn", flash_root)


@dataclass(frozen=True)
class _ResolvedFA4Config:
    mode: str
    softcap: float = 0.0
    score_mod: Optional[Callable] = None
    score_mod_bwd: Optional[Callable] = None
    sigmoid_attention: bool = False
    sigmoid_sfu_freq: int = 16
    sigmoid_sfu_res: int = 0
    sigmoid_sfu_freq_bwd: int | None = None
    sigmoid_sfu_res_bwd: int | None = None
    sigmoid_use_direct_bwd_poly: bool = False
    sigmoid_bias: float | None = None
    sigmoid_poly_backend: str = "cute"
    sigmoid_qk_norm: bool = True
    authenticated_runtime: _AuthenticatedFlashRuntime | None = None


def _assert_authenticated_runtime_origins(
    runtime: dict[str, Callable],
    root: Path,
) -> None:
    # The authenticated comparator is deliberately softmax-only.  Its capsule
    # needs the CuTe interface, but not the optional polynomial-audit helpers
    # used by the generic softcap/sigmoid routes.
    for module_name in ("flash_attn.cute.interface",):
        module = sys.modules.get(module_name)
        if module is None or not _module_is_below(module, root):
            raise RuntimeError(
                f"authenticated BF16 FA4 module {module_name} resolved "
                f"outside {root}"
            )
    for name, function in runtime.items():
        module_name = getattr(function, "__module__", "")
        module = sys.modules.get(module_name)
        if module is None or not _module_is_below(module, root):
            raise RuntimeError(
                f"authenticated BF16 FA4 callable {name} resolved outside "
                f"{root}: module={module_name!r}"
            )
    for module_name, module in list(sys.modules.items()):
        if module_name != "flash_attn" and not module_name.startswith(
            "flash_attn."
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file and not _module_is_below(module, root):
            raise RuntimeError(
                "authenticated BF16 FA4 imported a mixed-origin module: "
                f"{module_name} from {module_file}"
            )


@lru_cache(maxsize=4)
def _load_fa4_runtime(
    authenticated_runtime: _AuthenticatedFlashRuntime | None = None,
):
    if authenticated_runtime is None:
        _extend_fa4_python_paths()
    else:
        authenticated_runtime.authenticate_files()
        _purge_stale_modules("flash_attn", authenticated_runtime.root)
        _prepend_exact_python_path(authenticated_runtime.root)
        importlib.invalidate_caches()

    interface = importlib.import_module("flash_attn.cute.interface")
    runtime = {
        "_flash_attn_fwd": interface._flash_attn_fwd,
        "_flash_attn_bwd": interface._flash_attn_bwd,
    }
    if authenticated_runtime is None:
        polynomial_manifest = importlib.import_module(
            "flash_attn.cute.polynomial_manifest"
        )
        utils = importlib.import_module("flash_attn.cute.utils")
        runtime.update(
            {
                "run_polynomial_coefficient_audit": (
                    polynomial_manifest.run_polynomial_coefficient_audit
                ),
                "create_softcap_scoremod_backend": (
                    utils.create_softcap_scoremod_backend
                ),
                "create_softcap_scoremod_bwd_backend": (
                    utils.create_softcap_scoremod_bwd_backend
                ),
            }
        )
    if authenticated_runtime is not None:
        authenticated_runtime.authenticate_files()
        _assert_authenticated_runtime_origins(
            runtime,
            authenticated_runtime.root,
        )
    return runtime


def _resolve_fa4_config(job_config: JobConfig) -> _ResolvedFA4Config:
    cfg = job_config.fa4
    converters = tuple(job_config.model.converters or ())
    authenticated_runtime = None
    if _BF16_TOPOLOGY_CONVERTER in converters:
        authenticated_runtime = _AuthenticatedFlashRuntime.from_job_config(
            job_config
        )
    runtime = None
    if cfg.audit_coefficients or cfg.mode != "softmax":
        runtime = _load_fa4_runtime(authenticated_runtime)
    if cfg.audit_coefficients:
        assert runtime is not None
        audited = runtime["run_polynomial_coefficient_audit"]()
        logger.info("FA4 polynomial audit passed: %s", ", ".join(audited))

    if cfg.mode == "softmax":
        return _ResolvedFA4Config(
            mode="softmax",
            sigmoid_qk_norm=False,
            authenticated_runtime=authenticated_runtime,
        )

    if cfg.mode == "softcap":
        if cfg.softcap_backend == "native":
            return _ResolvedFA4Config(
                mode="softcap",
                softcap=float(cfg.softcap),
                sigmoid_qk_norm=False,
                authenticated_runtime=authenticated_runtime,
            )
        assert runtime is not None
        score_mod = runtime["create_softcap_scoremod_backend"](
            cfg.softcap,
            degree=cfg.softcap_degree,
            backend=cfg.softcap_backend,
        )
        score_mod_bwd = runtime["create_softcap_scoremod_bwd_backend"](
            cfg.softcap,
            degree=cfg.softcap_degree,
            backend=cfg.softcap_backend,
            backward_mode=cfg.softcap_backward_mode,
        )
        return _ResolvedFA4Config(
            mode="softcap",
            softcap=0.0,
            score_mod=score_mod,
            score_mod_bwd=score_mod_bwd,
            sigmoid_qk_norm=False,
            authenticated_runtime=authenticated_runtime,
        )

    if cfg.mode == "sigmoid_attention":
        if cfg.sigmoid_variant == "sfu":
            sfu_freq = 1
            sfu_res = 1
        else:
            sfu_freq = cfg.sigmoid_sfu_freq
            sfu_res = cfg.sigmoid_sfu_res
        return _ResolvedFA4Config(
            mode="sigmoid_attention",
            softcap=0.0,
            sigmoid_attention=True,
            sigmoid_sfu_freq=sfu_freq,
            sigmoid_sfu_res=sfu_res,
            sigmoid_sfu_freq_bwd=cfg.sigmoid_sfu_freq_bwd,
            sigmoid_sfu_res_bwd=cfg.sigmoid_sfu_res_bwd,
            sigmoid_use_direct_bwd_poly=(cfg.sigmoid_backward_mode == "direct"),
            sigmoid_bias=cfg.sigmoid_bias,
            sigmoid_poly_backend=cfg.sigmoid_poly_backend,
            sigmoid_qk_norm=cfg.sigmoid_qk_norm,
            authenticated_runtime=authenticated_runtime,
        )

    raise ValueError(f"Unsupported fa4.mode={cfg.mode!r}")


class _FA4Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        softcap,
        sigmoid_attention,
        sigmoid_sfu_freq,
        sigmoid_sfu_res,
        sigmoid_sfu_freq_bwd,
        sigmoid_sfu_res_bwd,
        sigmoid_use_direct_bwd_poly,
        sigmoid_bias,
        sigmoid_poly_backend,
        score_mod,
        score_mod_bwd,
        authenticated_runtime,
    ):
        runtime = _load_fa4_runtime(authenticated_runtime)
        if sigmoid_sfu_freq_bwd is None:
            sigmoid_sfu_freq_bwd = sigmoid_sfu_freq
        if sigmoid_sfu_res_bwd is None:
            sigmoid_sfu_res_bwd = sigmoid_sfu_res
        effective_softcap = softcap if score_mod is None else 0.0
        if authenticated_runtime is not None:
            if (
                score_mod is not None
                or score_mod_bwd is not None
                or sigmoid_attention
                or effective_softcap != 0.0
            ):
                raise RuntimeError(
                    "authenticated BF16 FA4 runtime is softmax-only"
                )
            # The pinned production interface predates the experimental
            # sigmoid keyword surface used by the generic FA4 development
            # tree.  Call only its authenticated softmax ABI.
            out, lse = runtime["_flash_attn_fwd"](
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                softcap=0.0,
                return_lse=True,
            )
        else:
            out, lse = runtime["_flash_attn_fwd"](
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                softcap=effective_softcap,
                score_mod=score_mod,
                sigmoid_attention=sigmoid_attention,
                sigmoid_sfu_freq=sigmoid_sfu_freq,
                sigmoid_sfu_res=sigmoid_sfu_res,
                sigmoid_bias=sigmoid_bias,
                sigmoid_poly_backend=sigmoid_poly_backend,
            )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.softcap = effective_softcap
        ctx.score_mod = score_mod
        ctx.score_mod_bwd = score_mod_bwd
        ctx.sigmoid_attention = sigmoid_attention
        ctx.sigmoid_sfu_freq_bwd = sigmoid_sfu_freq_bwd
        ctx.sigmoid_sfu_res_bwd = sigmoid_sfu_res_bwd
        ctx.sigmoid_use_direct_bwd_poly = sigmoid_use_direct_bwd_poly
        ctx.sigmoid_bias = sigmoid_bias
        ctx.sigmoid_poly_backend = sigmoid_poly_backend
        ctx.authenticated_runtime = authenticated_runtime
        return out

    @staticmethod
    def backward(ctx, dout):
        runtime = _load_fa4_runtime(ctx.authenticated_runtime)
        q, k, v, out, lse = ctx.saved_tensors
        if ctx.authenticated_runtime is not None:
            dq, dk, dv = runtime["_flash_attn_bwd"](
                q,
                k,
                v,
                out,
                dout,
                lse,
                ctx.softmax_scale,
                ctx.causal,
                ctx.softcap,
            )
        else:
            dq, dk, dv = runtime["_flash_attn_bwd"](
                q,
                k,
                v,
                out,
                dout,
                lse,
                ctx.softmax_scale,
                ctx.causal,
                ctx.softcap,
                score_mod=ctx.score_mod,
                score_mod_bwd=ctx.score_mod_bwd,
                sigmoid_attention=ctx.sigmoid_attention,
                sigmoid_bias=ctx.sigmoid_bias,
                sigmoid_sfu_freq=ctx.sigmoid_sfu_freq_bwd,
                sigmoid_sfu_res=ctx.sigmoid_sfu_res_bwd,
                sigmoid_use_direct_bwd_poly=ctx.sigmoid_use_direct_bwd_poly,
                sigmoid_poly_backend=ctx.sigmoid_poly_backend,
            )
        return dq, dk, dv, *((None,) * 14)


def _fa4_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None,
    causal: bool,
    config: _ResolvedFA4Config,
) -> torch.Tensor:
    return _FA4Func.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        config.softcap,
        config.sigmoid_attention,
        config.sigmoid_sfu_freq,
        config.sigmoid_sfu_res,
        config.sigmoid_sfu_freq_bwd,
        config.sigmoid_sfu_res_bwd,
        config.sigmoid_use_direct_bwd_poly,
        config.sigmoid_bias,
        config.sigmoid_poly_backend,
        config.score_mod,
        config.score_mod_bwd,
        config.authenticated_runtime,
    )


class FA4AttentionWrapper(nn.Module):
    def __init__(self, config: _ResolvedFA4Config, head_dim: int):
        super().__init__()
        self.config = config
        self.head_dim = head_dim
        if config.sigmoid_attention and config.sigmoid_qk_norm:
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        score_mod: Callable | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        if score_mod is not None:
            raise NotImplementedError(
                "FA4AttentionWrapper does not accept an external score_mod; configure low_bits_training.fa4 instead."
            )
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        target_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled()
            else q.dtype
        )
        out = _fa4_func(
            q.transpose(1, 2).contiguous().to(target_dtype),
            k.transpose(1, 2).contiguous().to(target_dtype),
            v.transpose(1, 2).contiguous().to(target_dtype),
            softmax_scale=scale,
            causal=True,
            config=self.config,
        )
        return out.transpose(1, 2)


def _patch_attention_modules(model: nn.Module, config: _ResolvedFA4Config) -> int:
    patched = 0
    for module in model.modules():
        if not hasattr(module, "inner_attention") or not hasattr(module, "head_dim"):
            continue
        if isinstance(module.inner_attention, FA4AttentionWrapper):
            continue
        wrapper = FA4AttentionWrapper(config=config, head_dim=module.head_dim)
        param = next(module.parameters(), None)
        if param is not None and param.device.type != "meta":
            wrapper = wrapper.to(device=param.device, dtype=param.dtype)
        elif param is not None:
            wrapper = wrapper.to(dtype=param.dtype)
        module.inner_attention = wrapper
        if hasattr(module, "use_flex_attn"):
            module.use_flex_attn = False
        if hasattr(module, "attn_score_modifier"):
            module.attn_score_modifier = None
        patched += 1
    return patched


class FA4AttentionConverter(ModelConverter):
    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.config = _resolve_fa4_config(job_config)

    def convert(self, model: nn.Module):
        native_gqa_modules = [
            module
            for module in model.modules()
            if getattr(module, "_bf16_fa4_native_gqa", False)
        ]
        if self.config.authenticated_runtime is not None and not native_gqa_modules:
            raise RuntimeError(
                "authenticated BF16 FA4 requires the topology converter to "
                "install native GQA before fa4_attention"
            )
        patched = _patch_attention_modules(model, self.config)
        if self.config.authenticated_runtime is not None:
            if patched != len(native_gqa_modules):
                raise RuntimeError(
                    "authenticated BF16 FA4 did not patch exactly the native "
                    f"GQA modules: patched={patched} native={len(native_gqa_modules)}"
                )
            if any(
                not isinstance(module.inner_attention, FA4AttentionWrapper)
                for module in native_gqa_modules
            ):
                raise RuntimeError(
                    "authenticated BF16 FA4 left a native GQA module unpatched"
                )
        logger.info("Patched %d attention modules to FA4 mode=%s", patched, self.config.mode)

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        pass


register_model_converter(FA4AttentionConverter, "fa4_attention")
