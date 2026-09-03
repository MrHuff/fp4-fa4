# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Checkpointed fused AdamW with stateless BF16 stochastic writeback.

The CUDA kernel computes AdamW in FP32 registers, stores both moments in BF16
with deterministic round-to-nearest, and writes BF16 parameters with a
counter-based hash keyed by base seed, optimizer step, stable tensor index,
and element index. It never reads or advances an ambient PyTorch generator.

Provider identity, base seed, and the next stochastic phase live in the single
parameter group, independently of the ordinary Adam bias-correction step, so
TorchTitan's flattened optimizer state and DCP preserve the exact stochastic
continuation. Older or differently built providers fail closed unless the
operator explicitly requests a non-bitwise new phase outside an authenticated
launch.
"""

from __future__ import annotations

import os
import types
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch.optim import Optimizer

from .fused_adamw_bf16_sr import (
    CHUNK_SIZE,
    MAX_TENSOR_ELEMENTS,
    PROVIDER,
    PROVIDER_VERSION,
    SOURCE_SHA256,
    get_extension,
)


STATE_VERSION = 2
UINT64_MAX = (1 << 64) - 1
SEED_ENV = "LBT_ADAMW_BF16_SR_SEED"
PROVIDER_ENV = "LBT_ADAMW_BF16_SR_PROVIDER"
PROVIDER_VERSION_ENV = "LBT_ADAMW_BF16_SR_PROVIDER_VERSION"
SOURCE_SHA256_ENV = "LBT_ADAMW_BF16_SR_SOURCE_SHA256"
CHECKPOINT_SCHEMA_ENV = "LBT_ADAMW_BF16_SR_CHECKPOINT_SCHEMA"
MISSING_POLICY_ENV = "LBT_ADAMW_BF16_SR_MISSING_POLICY"
START_NEW_PHASE_POLICY = "start_new_phase"

VERSION_KEY = "lbt_bf16_sr_version"
BASE_SEED_KEY = "lbt_bf16_sr_base_seed"
STEP_KEY = "lbt_bf16_sr_step"
PROVIDER_KEY = "lbt_bf16_sr_provider"
PROVIDER_VERSION_KEY = "lbt_bf16_sr_provider_version"
SOURCE_SHA256_KEY = "lbt_bf16_sr_source_sha256"
_PHASE_KEYS = (
    VERSION_KEY,
    BASE_SEED_KEY,
    STEP_KEY,
    PROVIDER_KEY,
    PROVIDER_VERSION_KEY,
    SOURCE_SHA256_KEY,
)
CHECKPOINT_SCHEMA = "v2-fused-stateless"
_RUNTIME_IDENTITY_ENV = {
    PROVIDER_ENV: PROVIDER,
    PROVIDER_VERSION_ENV: str(PROVIDER_VERSION),
    SOURCE_SHA256_ENV: SOURCE_SHA256,
    CHECKPOINT_SCHEMA_ENV: CHECKPOINT_SCHEMA,
}


def configured_sr_seed() -> int:
    raw_seed = os.environ.get(SEED_ENV, "0").strip()
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise ValueError(f"{SEED_ENV} must be an integer, got {raw_seed!r}") from exc
    if not 0 <= seed <= UINT64_MAX:
        raise ValueError(f"{SEED_ENV} must be in [0, 2**64 - 1], got {seed}")
    return seed


def validate_runtime_provider_environment(configured_seed: int) -> None:
    """Bind an authenticated launch environment to the running provider.

    Local runs may omit every identity variable. If a renderer supplies any
    identity field, all fields (including the seed) become mandatory and must
    match the provider compiled from this source tree. Authenticated launches
    also reject the non-bitwise legacy/new-phase escape hatch.
    """
    names = (*_RUNTIME_IDENTITY_ENV, SEED_ENV)
    present_identity = {
        name for name in _RUNTIME_IDENTITY_ENV if name in os.environ
    }
    if not present_identity:
        return
    present = {name for name in names if name in os.environ}
    missing = sorted(set(names) - present)
    if missing:
        raise RuntimeError(
            "partial AdamWBF16SR runtime provider identity; missing "
            f"{missing}"
        )
    for name, expected in _RUNTIME_IDENTITY_ENV.items():
        actual = os.environ[name]
        if actual != expected:
            raise RuntimeError(
                f"AdamWBF16SR runtime identity mismatch for {name}: "
                f"expected {expected!r}, got {actual!r}"
            )
    environment_seed = configured_sr_seed()
    if environment_seed != configured_seed:
        raise RuntimeError(
            "AdamWBF16SR runtime seed differs from the constructed optimizer: "
            f"environment={environment_seed}, optimizer={configured_seed}"
        )
    policy = os.environ.get(MISSING_POLICY_ENV, "").strip()
    if policy:
        raise RuntimeError(
            "authenticated AdamWBF16SR launches forbid "
            f"{MISSING_POLICY_ENV}; production resumes must preserve the exact "
            "provider and stochastic phase"
        )


def _require_plain_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} must be an integer, got {type(value).__name__}")
    return value


def _require_plain_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{name} must be a string, got {type(value).__name__}")
    return value


class AdamWBF16SR(Optimizer):
    """Single-group fused CUDA AdamW with checkpointed stateless BF16 SR."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        bf16_stochastic_round: bool = True,
        sr_seed: int = 0,
    ) -> None:
        parameters = list(params)
        if not parameters:
            raise ValueError("AdamWBF16SR requires at least one parameter")
        if any(not isinstance(parameter, torch.nn.Parameter) for parameter in parameters):
            raise TypeError("AdamWBF16SR requires a flat iterable of Parameters")
        if not bf16_stochastic_round:
            raise ValueError("AdamWBF16SR requires bf16_stochastic_round=True")
        if amsgrad:
            raise ValueError("AdamWBF16SR does not support amsgrad")
        if isinstance(sr_seed, bool) or not isinstance(sr_seed, int):
            raise ValueError("AdamWBF16SR sr_seed must be an integer")
        if not 0 <= sr_seed <= UINT64_MAX:
            raise ValueError(
                f"AdamWBF16SR sr_seed must be in [0, 2**64 - 1], got {sr_seed}"
            )
        validate_runtime_provider_environment(sr_seed)
        self._lbt_configured_sr_seed = sr_seed
        self._lbt_sr_device = self._validate_initial_parameters(parameters)
        self._lbt_parameter_ids = tuple(id(parameter) for parameter in parameters)
        self._lbt_noop_flag: torch.Tensor | None = None
        self._lbt_shared_adam_step: torch.Tensor | None = None
        self._lbt_allow_initial_group = True
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": False,
        }
        super().__init__(parameters, defaults)
        self._lbt_allow_initial_group = False
        self._validated_hyperparameters()
        self._reset_phase_metadata()

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if not getattr(self, "_lbt_allow_initial_group", False):
            raise RuntimeError("AdamWBF16SR requires exactly one parameter group")
        if getattr(self, "param_groups", []):
            raise RuntimeError("AdamWBF16SR requires exactly one parameter group")
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        if not isinstance(group["lr"], torch.Tensor):
            group["lr"] = torch.tensor(group["lr"], dtype=torch.float32)

    @staticmethod
    def _validate_initial_parameters(
        parameters: list[torch.nn.Parameter],
    ) -> torch.device:
        if any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("AdamWBF16SR accepts trainable parameters only")
        invalid_dtypes = sorted(
            {
                str(parameter.dtype)
                for parameter in parameters
                if parameter.dtype != torch.bfloat16
            }
        )
        if invalid_dtypes:
            raise ValueError(
                "AdamWBF16SR requires BF16 parameters; found "
                + ", ".join(invalid_dtypes)
            )
        devices = {torch.device(parameter.device) for parameter in parameters}
        if len(devices) != 1:
            raise ValueError(
                "AdamWBF16SR requires single-device parameter ownership; found "
                f"{sorted(map(str, devices))}"
            )
        device = devices.pop()
        if device.type != "cuda":
            raise ValueError(
                "AdamWBF16SR requires one BF16 CUDA device; found " f"{device}"
            )
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.current_device() != device.index:
            raise RuntimeError(
                "AdamWBF16SR parameters must belong to the current CUDA device; "
                f"parameters={device}, current=cuda:{torch.cuda.current_device()}"
            )
        for index, parameter in enumerate(parameters):
            if not parameter.is_contiguous():
                raise ValueError(
                    f"AdamWBF16SR parameter {index} must be contiguous"
                )
            if parameter.numel() > MAX_TENSOR_ELEMENTS:
                raise ValueError(
                    f"AdamWBF16SR parameter {index} exceeds 2**32 - 1 elements"
                )
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise ValueError("AdamWBF16SR parameter list contains duplicates")
        return device

    def _validated_parameters(
        self, *, require_gradients: bool
    ) -> tuple[list[torch.nn.Parameter], list[torch.Tensor]]:
        if len(self.param_groups) != 1:
            raise RuntimeError("AdamWBF16SR requires exactly one parameter group")
        parameters = self.param_groups[0]["params"]
        if tuple(id(parameter) for parameter in parameters) != self._lbt_parameter_ids:
            raise RuntimeError(
                "AdamWBF16SR canonical parameter order changed after construction"
            )
        current = torch.cuda.current_device()
        if current != self._lbt_sr_device.index:
            raise RuntimeError(
                "AdamWBF16SR current CUDA device changed after construction: "
                f"owned={self._lbt_sr_device}, current=cuda:{current}"
            )
        gradients: list[torch.Tensor] = []
        for index, parameter in enumerate(parameters):
            if (
                parameter.dtype != torch.bfloat16
                or parameter.device != self._lbt_sr_device
                or not parameter.is_contiguous()
                or parameter.numel() > MAX_TENSOR_ELEMENTS
            ):
                raise RuntimeError(
                    f"AdamWBF16SR parameter {index} violated its BF16 CUDA ABI"
                )
            gradient = parameter.grad
            if not require_gradients and gradient is None:
                continue
            if gradient is None:
                raise RuntimeError(
                    "AdamWBF16SR requires every trainable parameter to have a "
                    f"gradient; missing parameter index {index}"
                )
            if gradient.is_sparse:
                raise RuntimeError("AdamWBF16SR does not support sparse gradients")
            if (
                gradient.dtype != torch.bfloat16
                or gradient.device != self._lbt_sr_device
                or not gradient.is_contiguous()
                or gradient.numel() != parameter.numel()
            ):
                raise RuntimeError(
                    f"AdamWBF16SR gradient {index} violated its BF16 CUDA ABI"
                )
            gradients.append(gradient)
        return list(parameters), gradients

    def _validated_hyperparameters(
        self,
    ) -> tuple[float, float, float, float, float]:
        if len(self.param_groups) != 1:
            raise RuntimeError("AdamWBF16SR requires exactly one parameter group")
        group = self.param_groups[0]
        learning_rate = group["lr"]
        if isinstance(learning_rate, torch.Tensor):
            if learning_rate.numel() != 1:
                raise RuntimeError("AdamWBF16SR learning rate must be scalar")
            learning_rate_value = float(learning_rate.item())
        else:
            learning_rate_value = float(learning_rate)
        beta1, beta2 = map(float, group["betas"])
        epsilon = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        if learning_rate_value < 0:
            raise ValueError("AdamWBF16SR learning rate must be non-negative")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("AdamWBF16SR betas must be in [0, 1)")
        if epsilon < 0 or weight_decay < 0:
            raise ValueError(
                "AdamWBF16SR epsilon and weight decay must be non-negative"
            )
        return learning_rate_value, beta1, beta2, epsilon, weight_decay

    def _reset_phase_metadata(self) -> None:
        group = self.param_groups[0]
        group[VERSION_KEY] = STATE_VERSION
        group[BASE_SEED_KEY] = self._lbt_configured_sr_seed
        group[STEP_KEY] = 0
        group[PROVIDER_KEY] = PROVIDER
        group[PROVIDER_VERSION_KEY] = PROVIDER_VERSION
        group[SOURCE_SHA256_KEY] = SOURCE_SHA256

    def _validated_phase_metadata(self) -> tuple[int, int]:
        if len(self.param_groups) != 1:
            raise RuntimeError("AdamWBF16SR requires exactly one parameter group")
        group = self.param_groups[0]
        missing = [key for key in _PHASE_KEYS if key not in group]
        if missing:
            raise RuntimeError(
                "AdamWBF16SR checkpoint is missing stochastic-phase fields: "
                f"{missing}"
            )
        version = _require_plain_int(VERSION_KEY, group[VERSION_KEY])
        base_seed = _require_plain_int(BASE_SEED_KEY, group[BASE_SEED_KEY])
        step = _require_plain_int(STEP_KEY, group[STEP_KEY])
        provider = _require_plain_string(PROVIDER_KEY, group[PROVIDER_KEY])
        provider_version = _require_plain_int(
            PROVIDER_VERSION_KEY, group[PROVIDER_VERSION_KEY]
        )
        source_sha256 = _require_plain_string(
            SOURCE_SHA256_KEY, group[SOURCE_SHA256_KEY]
        )
        if version != STATE_VERSION:
            raise RuntimeError(
                "unsupported AdamWBF16SR stochastic-phase version "
                f"{version}; expected {STATE_VERSION}"
            )
        if provider != PROVIDER or provider_version != PROVIDER_VERSION:
            raise RuntimeError(
                "AdamWBF16SR checkpoint provider differs from the running provider"
            )
        if source_sha256 != SOURCE_SHA256:
            raise RuntimeError(
                "AdamWBF16SR checkpoint kernel source hash differs from the "
                "running provider"
            )
        if not 0 <= base_seed <= UINT64_MAX:
            raise RuntimeError(
                f"{BASE_SEED_KEY} must be in [0, 2**64 - 1], got {base_seed}"
            )
        if step < 0:
            raise RuntimeError(f"{STEP_KEY} must be non-negative, got {step}")
        if base_seed != self._lbt_configured_sr_seed:
            raise RuntimeError(
                "AdamWBF16SR checkpoint seed differs from the configured seed: "
                f"checkpoint={base_seed}, configured={self._lbt_configured_sr_seed}"
            )
        return base_seed, step

    def _new_phase_policy_enabled(self) -> bool:
        return (
            os.environ.get(MISSING_POLICY_ENV, "").strip().lower()
            == START_NEW_PHASE_POLICY
        )

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore AdamW and its exact next SR phase, or fail closed."""
        saved_groups = state_dict.get("param_groups")
        metadata_error: Exception | None = None
        if not isinstance(saved_groups, list) or len(saved_groups) != len(
            self.param_groups
        ):
            metadata_error = RuntimeError(
                "AdamWBF16SR checkpoint has invalid parameter-group metadata"
            )
        else:
            try:
                for group_index, group in enumerate(saved_groups):
                    if not isinstance(group, Mapping):
                        raise RuntimeError(
                            f"AdamWBF16SR param_group {group_index} is not a mapping"
                        )
                    missing = [key for key in _PHASE_KEYS if key not in group]
                    if missing:
                        raise RuntimeError(
                            "AdamWBF16SR checkpoint predates isolated stochastic "
                            f"phase state; param_group {group_index} is missing {missing}"
                        )
                    version = _require_plain_int(VERSION_KEY, group[VERSION_KEY])
                    base_seed = _require_plain_int(BASE_SEED_KEY, group[BASE_SEED_KEY])
                    step = _require_plain_int(STEP_KEY, group[STEP_KEY])
                    provider = _require_plain_string(
                        PROVIDER_KEY, group[PROVIDER_KEY]
                    )
                    provider_version = _require_plain_int(
                        PROVIDER_VERSION_KEY, group[PROVIDER_VERSION_KEY]
                    )
                    source_sha256 = _require_plain_string(
                        SOURCE_SHA256_KEY, group[SOURCE_SHA256_KEY]
                    )
                    if version != STATE_VERSION:
                        raise RuntimeError(
                            "unsupported AdamWBF16SR stochastic-phase version "
                            f"{version}; expected {STATE_VERSION}"
                        )
                    if not 0 <= base_seed <= UINT64_MAX or step < 0:
                        raise RuntimeError(
                            "AdamWBF16SR checkpoint contains invalid "
                            "stochastic-phase values"
                        )
                    if base_seed != self._lbt_configured_sr_seed:
                        raise RuntimeError(
                            "AdamWBF16SR checkpoint seed differs from the "
                            "configured seed: "
                            f"checkpoint={base_seed}, "
                            f"configured={self._lbt_configured_sr_seed}"
                        )
                    if (
                        provider != PROVIDER
                        or provider_version != PROVIDER_VERSION
                        or source_sha256 != SOURCE_SHA256
                    ):
                        raise RuntimeError(
                            "AdamWBF16SR checkpoint provider identity differs from "
                            "the running fused stateless provider"
                        )
            except Exception as exc:
                metadata_error = exc

        if metadata_error is not None and not self._new_phase_policy_enabled():
            raise RuntimeError(
                f"{metadata_error}. Resume is rejected because the next optimizer "
                "stochastic phase is ambiguous. For an explicitly non-bitwise "
                f"continuation only, set {MISSING_POLICY_ENV}="
                f"{START_NEW_PHASE_POLICY} to reset the isolated phase to step zero."
            ) from metadata_error

        super().load_state_dict(dict(state_dict))
        if metadata_error is not None:
            self._reset_phase_metadata()
            warnings.warn(
                "Loaded AdamW state without a compatible AdamWBF16SR phase and "
                "explicitly started a new stochastic phase at configured step zero; "
                "this continuation is not bitwise-equivalent to the source run.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._validated_phase_metadata()
        self._validated_parameters(require_gradients=False)
        self._canonicalize_loaded_state()

    def _canonicalize_loaded_state(self) -> None:
        parameters = self.param_groups[0]["params"]
        populated = [bool(self.state[parameter]) for parameter in parameters]
        if not any(populated):
            self._lbt_shared_adam_step = None
            return
        if not all(populated):
            raise RuntimeError("AdamWBF16SR checkpoint has partial optimizer state")
        completed_steps: set[int] = set()
        for index, parameter in enumerate(parameters):
            state = self.state[parameter]
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise RuntimeError(
                    f"AdamWBF16SR state {index} has unsupported fields {set(state)}"
                )
            step_tensor = state["step"]
            if not isinstance(step_tensor, torch.Tensor) or step_tensor.numel() != 1:
                raise RuntimeError("AdamWBF16SR Adam step must be one tensor value")
            completed_step_value = float(step_tensor.item())
            completed_step = int(completed_step_value)
            if completed_step < 0 or completed_step_value != completed_step:
                raise RuntimeError("AdamWBF16SR Adam step must be a non-negative integer")
            completed_steps.add(completed_step)
            for name in ("exp_avg", "exp_avg_sq"):
                value = state[name]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype != torch.bfloat16
                    or value.device != self._lbt_sr_device
                    or not value.is_contiguous()
                    or value.numel() != parameter.numel()
                ):
                    raise RuntimeError(
                        f"AdamWBF16SR {name} state {index} violated its BF16 CUDA ABI"
                    )
        if len(completed_steps) != 1:
            raise RuntimeError("AdamWBF16SR per-parameter Adam steps differ")
        completed_step = completed_steps.pop()
        shared_step = torch.tensor(float(completed_step), dtype=torch.float32)
        for parameter in parameters:
            self.state[parameter]["step"] = shared_step
        self._lbt_shared_adam_step = shared_step

    def _state_for_update(
        self, parameters: list[torch.nn.Parameter]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
        populated = [bool(self.state[parameter]) for parameter in parameters]
        if not any(populated):
            shared_step = torch.tensor(0.0, dtype=torch.float32)
            first_moments = []
            second_moments = []
            for parameter in parameters:
                first_moment = torch.zeros_like(parameter)
                second_moment = torch.zeros_like(parameter)
                self.state[parameter]["step"] = shared_step
                self.state[parameter]["exp_avg"] = first_moment
                self.state[parameter]["exp_avg_sq"] = second_moment
                first_moments.append(first_moment)
                second_moments.append(second_moment)
            self._lbt_shared_adam_step = shared_step
            return first_moments, second_moments, 0
        if not all(populated):
            raise RuntimeError("AdamWBF16SR has partial optimizer state")
        if self._lbt_shared_adam_step is None:
            self._canonicalize_loaded_state()
        assert self._lbt_shared_adam_step is not None
        completed_step_value = float(self._lbt_shared_adam_step.item())
        completed_step = int(completed_step_value)
        if completed_step_value != completed_step:
            raise RuntimeError("AdamWBF16SR Adam step is not integral")
        return (
            [self.state[parameter]["exp_avg"] for parameter in parameters],
            [self.state[parameter]["exp_avg_sq"] for parameter in parameters],
            completed_step,
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError(
                "AdamWBF16SR does not support closures"
            )
        base_seed, stochastic_step = self._validated_phase_metadata()
        parameters, gradients = self._validated_parameters(require_gradients=True)
        (
            first_moments,
            second_moments,
            completed_adam_step,
        ) = self._state_for_update(parameters)
        (
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
        ) = self._validated_hyperparameters()
        if self._lbt_noop_flag is None:
            self._lbt_noop_flag = torch.zeros(
                1, device=self._lbt_sr_device, dtype=torch.int32
            )
        extension = get_extension()
        extension.adamw(
            CHUNK_SIZE,
            self._lbt_noop_flag,
            [gradients, parameters, first_moments, second_moments],
            learning_rate,
            beta1,
            beta2,
            epsilon,
            completed_adam_step + 1,
            stochastic_step,
            weight_decay,
            base_seed,
        )
        assert self._lbt_shared_adam_step is not None
        self._lbt_shared_adam_step.fill_(completed_adam_step + 1)
        self.param_groups[0][STEP_KEY] = stochastic_step + 1
        return None


def _is_flattened_phase_key(key: str) -> bool:
    return any(key.endswith(f".{field}") for field in _PHASE_KEYS)


def checkpoint_optimizer_sr_schema(checkpoint_id: str) -> str:
    """Classify isolated optimizer SR metadata without loading its tensors."""
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(checkpoint_id).read_metadata()
    prefixes_by_field = {
        field: {
            key[: -len(field) - 1]
            for key in metadata.state_dict_metadata
            if key.startswith("optimizer.param_groups.")
            and key.endswith(f".{field}")
        }
        for field in _PHASE_KEYS
    }
    all_prefixes = set().union(*prefixes_by_field.values())
    if not all_prefixes:
        return "missing"
    if all(prefixes == all_prefixes for prefixes in prefixes_by_field.values()):
        return CHECKPOINT_SCHEMA
    return "unknown"


class _OptimizerWithoutPhaseState:
    """DCP destination for an explicitly declared legacy/new SR phase."""

    def __init__(self, optimizer_container) -> None:
        self.optimizer_container = optimizer_container

    def state_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.optimizer_container.state_dict().items()
            if not _is_flattened_phase_key(key)
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        # TorchTitan unflattens according to keys present on each *live*
        # param_group. Remove the new phase keys temporarily; otherwise its
        # unflattener asks the legacy state_dict for fields we deliberately
        # omitted and fails before AdamWBF16SR.load_state_dict can apply the
        # explicit new-phase policy.
        snapshots = [
            {key: group[key] for key in _PHASE_KEYS if key in group}
            for optimizer in self.optimizer_container.optimizers
            for group in optimizer.param_groups
        ]
        groups = [
            group
            for optimizer in self.optimizer_container.optimizers
            for group in optimizer.param_groups
        ]
        for group in groups:
            for key in _PHASE_KEYS:
                group.pop(key, None)
        try:
            self.optimizer_container.load_state_dict(dict(state_dict))
        finally:
            # Successful policy-aware loads re-add a fresh phase themselves.
            # Restore the live metadata only if an earlier unflatten/load phase
            # failed and left the optimizer without it.
            for group, snapshot in zip(groups, snapshots, strict=True):
                for key, value in snapshot.items():
                    group.setdefault(key, value)


def register_with_checkpointer(checkpointer, optimizer_container, logger) -> None:
    """Make the explicit legacy-phase policy work for full DCP resumes.

    DCP normally fails during planning when the live optimizer requests phase
    keys that are absent from an older checkpoint, before
    ``AdamWBF16SR.load_state_dict`` can enforce its policy. This wrapper first
    classifies metadata. Under the explicit ``start_new_phase`` policy only,
    it presents a destination without phase keys; the optimizer then loads its
    ordinary AdamW state and deliberately resets the phase to configured step
    zero. Normal v2 checkpoints follow the unmodified DCP path.
    """
    if getattr(checkpointer, "ft_manager", None) is not None:
        raise RuntimeError(
            "AdamWBF16SR isolated stochastic state is not compatible with "
            "TorchFT replica checkpoints"
        )
    optimizers = getattr(optimizer_container, "optimizers", None)
    if not isinstance(optimizers, list) or not optimizers:
        raise RuntimeError("AdamWBF16SR checkpointer registration requires optimizers")
    if any(not isinstance(optimizer, AdamWBF16SR) for optimizer in optimizers):
        raise RuntimeError(
            "AdamWBF16SR checkpointer registration received a mixed optimizer container"
        )
    original_dcp_load = checkpointer.dcp_load

    def dcp_load_with_optimizer_sr(
        this,
        state_dict,
        checkpoint_id,
        from_hf,
        from_quantized,
    ):
        load_state = state_dict
        starts_new_phase = False
        if "optimizer" in state_dict and not from_hf:
            try:
                schema = checkpoint_optimizer_sr_schema(checkpoint_id)
            except Exception as exc:
                raise RuntimeError(
                    "could not verify AdamWBF16SR checkpoint metadata; refusing "
                    "an ambiguous resume"
                ) from exc
            if schema != CHECKPOINT_SCHEMA:
                policy = os.environ.get(MISSING_POLICY_ENV, "").strip().lower()
                if policy != START_NEW_PHASE_POLICY:
                    raise RuntimeError(
                        "checkpoint has missing, partial, or unknown AdamWBF16SR "
                        "stochastic-phase metadata. Resume is rejected by default. "
                        "For an explicitly non-bitwise continuation only, set "
                        f"{MISSING_POLICY_ENV}={START_NEW_PHASE_POLICY} to load "
                        "ordinary AdamW state and reset the isolated phase to "
                        "configured step zero."
                    )
                load_state = dict(state_dict)
                load_state["optimizer"] = _OptimizerWithoutPhaseState(
                    optimizer_container
                )
                starts_new_phase = True

        result = original_dcp_load(
            load_state,
            checkpoint_id=checkpoint_id,
            from_hf=from_hf,
            from_quantized=from_quantized,
        )
        if starts_new_phase:
            logger.warning(
                "Checkpoint lacked a compatible AdamWBF16SR phase; explicitly "
                "started a new phase at configured step zero. This continuation "
                "is not bitwise-equivalent to the source run."
            )
        return result

    checkpointer.dcp_load = types.MethodType(dcp_load_with_optimizer_sr, checkpointer)
