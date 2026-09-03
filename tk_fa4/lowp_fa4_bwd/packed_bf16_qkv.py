"""Packed QKV projection helpers.

The low-precision attention routes already publish Q, K, and V from one
projection.  This module gives the BF16 control the same projection topology:
one canonical ``qkv`` parameter, one ``F.linear`` call, and view-only native
GQA splits.  The D64 low-precision route also uses the canonical packed
parameter directly, avoiding a full Q/K/V concatenation on every forward and
backward.  Checkpoint conversion is explicit and lossless so packed routes can
start from the exact same model tensors as historical split-Q/K/V controls.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PackedQKVLayout:
    """Logical row layout of one packed GQA QKV projection weight."""

    hidden: int
    q_heads: int
    kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden", self.hidden),
            ("q_heads", self.q_heads),
            ("kv_heads", self.kv_heads),
            ("head_dim", self.head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.q_heads % self.kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads for GQA")

    @property
    def q_width(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def projection_width(self) -> int:
        return self.q_width + 2 * self.kv_width

    @property
    def weight_shape(self) -> tuple[int, int]:
        return self.projection_width, self.hidden

    def require_weight(self, weight: torch.Tensor, name: str = "qkv") -> None:
        if tuple(weight.shape) != self.weight_shape:
            raise ValueError(
                f"{name} weight must have shape {self.weight_shape}, got "
                f"{tuple(weight.shape)}"
            )

    def require_split_weights(
        self,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
    ) -> None:
        expected = {
            "q": (self.q_width, self.hidden),
            "k": (self.kv_width, self.hidden),
            "v": (self.kv_width, self.hidden),
        }
        for name, weight in (
            ("q", q_weight),
            ("k", k_weight),
            ("v", v_weight),
        ):
            if tuple(weight.shape) != expected[name]:
                raise ValueError(
                    f"{name} weight must have shape {expected[name]}, got "
                    f"{tuple(weight.shape)}"
                )


def pack_qkv_weights(
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    layout: PackedQKVLayout,
) -> torch.Tensor:
    """Concatenate canonical model rows in Q, K, V order."""
    layout.require_split_weights(q_weight, k_weight, v_weight)
    if not (q_weight.dtype == k_weight.dtype == v_weight.dtype):
        raise ValueError("Q, K, and V weights must have one dtype")
    if not (q_weight.device == k_weight.device == v_weight.device):
        raise ValueError("Q, K, and V weights must be on one device")
    return torch.cat((q_weight, k_weight, v_weight), dim=0).contiguous()


def unpack_qkv_weight(
    qkv_weight: torch.Tensor,
    layout: PackedQKVLayout,
    *,
    clone: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Q/K/V row slices, optionally with independent storage."""
    layout.require_weight(qkv_weight)
    values = torch.split(
        qkv_weight,
        (layout.q_width, layout.kv_width, layout.kv_width),
        dim=0,
    )
    if clone:
        return tuple(value.clone() for value in values)  # type: ignore[return-value]
    return values  # type: ignore[return-value]


def project_packed_qkv(
    rows: torch.Tensor,
    qkv_weight: torch.Tensor,
    layout: PackedQKVLayout,
    *,
    batch: int,
    sequence: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one linear and split its output into native GQA tensor views."""
    layout.require_weight(qkv_weight)
    if rows.ndim != 2 or tuple(rows.shape) != (
        batch * sequence,
        layout.hidden,
    ):
        raise ValueError(
            "rows must have shape "
            f"{(batch * sequence, layout.hidden)}, got {tuple(rows.shape)}"
        )
    projected = F.linear(rows, qkv_weight).reshape(
        batch,
        sequence,
        layout.projection_width,
    )
    q, k, v = torch.split(
        projected,
        (layout.q_width, layout.kv_width, layout.kv_width),
        dim=-1,
    )
    return (
        q.reshape(batch, sequence, layout.q_heads, layout.head_dim),
        k.reshape(batch, sequence, layout.kv_heads, layout.head_dim),
        v.reshape(batch, sequence, layout.kv_heads, layout.head_dim),
    )


class PackedQKVAttentionWeights(nn.Module):
    """Canonical packed QKV plus output-projection parameters."""

    def __init__(
        self,
        layout: PackedQKVLayout,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        std: float = 0.02,
    ) -> None:
        super().__init__()
        if std < 0.0:
            raise ValueError("std must be non-negative")
        self.layout = layout
        self.qkv = nn.Parameter(
            torch.empty(layout.weight_shape, device=device, dtype=dtype)
        )
        self.o = nn.Parameter(
            torch.empty(
                layout.hidden,
                layout.q_width,
                device=device,
                dtype=dtype,
            )
        )
        # Preserve the historical split-control RNG call boundaries.  With a
        # shared seed this produces exactly the same Q, K, V, and O tensors as
        # three independently allocated projection parameters, while runtime
        # still sees one canonical packed parameter and one projection GEMM.
        q, k, v = unpack_qkv_weight(self.qkv, layout)
        nn.init.normal_(q, mean=0.0, std=std)
        nn.init.normal_(k, mean=0.0, std=std)
        nn.init.normal_(v, mean=0.0, std=std)
        nn.init.normal_(self.o, mean=0.0, std=std)

    @property
    def q(self) -> torch.Tensor:
        """Logical Q view for diagnostics; ``qkv`` remains the parameter."""
        return unpack_qkv_weight(self.qkv, self.layout)[0]

    @property
    def k(self) -> torch.Tensor:
        """Logical K view for diagnostics; ``qkv`` remains the parameter."""
        return unpack_qkv_weight(self.qkv, self.layout)[1]

    @property
    def v(self) -> torch.Tensor:
        """Logical V view for diagnostics; ``qkv`` remains the parameter."""
        return unpack_qkv_weight(self.qkv, self.layout)[2]

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Accept the historical split-Q/K/V schema under strict loading."""
        qkv_key = f"{prefix}qkv"
        split_keys = tuple(f"{prefix}{name}" for name in ("q", "k", "v"))
        split_present = tuple(key in state_dict for key in split_keys)
        if qkv_key in state_dict and any(split_present):
            error_msgs.append(
                f"checkpoint contains both packed and split QKV at {prefix!r}"
            )
        elif any(split_present) and not all(split_present):
            error_msgs.append(
                f"checkpoint contains incomplete split QKV at {prefix!r}"
            )
        elif qkv_key not in state_dict and all(split_present):
            state_dict[qkv_key] = pack_qkv_weights(
                state_dict[split_keys[0]],
                state_dict[split_keys[1]],
                state_dict[split_keys[2]],
                self.layout,
            )
            for key in split_keys:
                del state_dict[key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def _qkv_prefix(key: str, terminal: str) -> str | None:
    if key == terminal:
        return ""
    suffix = f".{terminal}"
    if key.endswith(suffix):
        return key[: -len(terminal)]
    return None


def _copy_state_dict_metadata(
    source: Mapping[str, torch.Tensor],
    target: OrderedDict[str, torch.Tensor],
) -> None:
    metadata = getattr(source, "_metadata", None)
    if metadata is not None:
        target._metadata = metadata.copy()  # type: ignore[attr-defined]


def pack_qkv_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    layout: PackedQKVLayout,
) -> OrderedDict[str, torch.Tensor]:
    """Map every complete ``q``/``k``/``v`` group to one ``qkv`` tensor.

    The input is never mutated.  Groups are recognized by their complete key
    prefix, so both root-level attention state and full-model keys such as
    ``layers.0.attention.weights.q`` are supported.
    """
    groups: dict[str, tuple[str, str, str, str]] = {}
    for key in state_dict:
        prefix = _qkv_prefix(key, "q")
        if prefix is None:
            continue
        k_key = f"{prefix}k"
        v_key = f"{prefix}v"
        qkv_key = f"{prefix}qkv"
        present = (k_key in state_dict, v_key in state_dict)
        if not any(present):
            continue
        if not all(present):
            raise KeyError(
                f"incomplete QKV checkpoint group at prefix {prefix!r}"
            )
        if qkv_key in state_dict:
            raise KeyError(
                f"checkpoint contains both split and packed QKV at {prefix!r}"
            )
        groups[prefix] = (key, k_key, v_key, qkv_key)

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    consumed = {
        consumed_key
        for q_key, k_key, v_key, _qkv_key in groups.values()
        for consumed_key in (k_key, v_key)
    }
    q_to_group = {group[0]: group for group in groups.values()}
    for key, value in state_dict.items():
        if key in consumed:
            continue
        group = q_to_group.get(key)
        if group is None:
            result[key] = value
            continue
        _q_key, k_key, v_key, qkv_key = group
        result[qkv_key] = pack_qkv_weights(
            value, state_dict[k_key], state_dict[v_key], layout
        )
    _copy_state_dict_metadata(state_dict, result)
    return result


def unpack_qkv_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    layout: PackedQKVLayout,
) -> OrderedDict[str, torch.Tensor]:
    """Map every packed ``qkv`` checkpoint tensor back to split Q/K/V."""
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        prefix = _qkv_prefix(key, "qkv")
        if prefix is None:
            result[key] = value
            continue
        split_keys = tuple(f"{prefix}{name}" for name in ("q", "k", "v"))
        if any(split_key in state_dict for split_key in split_keys):
            raise KeyError(
                f"checkpoint contains both packed and split QKV at {prefix!r}"
            )
        q, k, v = unpack_qkv_weight(value, layout, clone=True)
        result[split_keys[0]] = q
        result[split_keys[1]] = k
        result[split_keys[2]] = v
    _copy_state_dict_metadata(state_dict, result)
    return result


def canonical_split_qkv_tensors(
    tensors: Mapping[str, torch.Tensor],
    layout: PackedQKVLayout,
) -> OrderedDict[str, torch.Tensor]:
    """Expose packed named parameters or gradients under split model names.

    Unlike :func:`unpack_qkv_state_dict`, this returns views.  It is intended
    for parameter schemas, gradient diagnostics, and sampled comparisons where
    materializing every full-model QKV tensor would distort memory profiling.
    """
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in tensors.items():
        prefix = _qkv_prefix(key, "qkv")
        if prefix is None:
            result[key] = value
            continue
        split_keys = tuple(f"{prefix}{name}" for name in ("q", "k", "v"))
        if any(split_key in tensors for split_key in split_keys):
            raise KeyError(
                f"tensor mapping contains both packed and split QKV at "
                f"{prefix!r}"
            )
        q, k, v = unpack_qkv_weight(value, layout)
        result[split_keys[0]] = q
        result[split_keys[1]] = k
        result[split_keys[2]] = v
    return result
