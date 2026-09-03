from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tk_fa4.lowp_fa4_bwd import packed_bf16_qkv as packed_module
from tk_fa4.lowp_fa4_bwd.packed_bf16_qkv import (
    PackedQKVAttentionWeights,
    PackedQKVLayout,
    canonical_split_qkv_tensors,
    pack_qkv_state_dict,
    pack_qkv_weights,
    project_packed_qkv,
    unpack_qkv_state_dict,
    unpack_qkv_weight,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
)


def _layout() -> PackedQKVLayout:
    return PackedQKVLayout(
        hidden=12,
        q_heads=4,
        kv_heads=2,
        head_dim=4,
    )


def _split_parameters(
    layout: PackedQKVLayout,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    return (
        torch.nn.Parameter(
            torch.randn(layout.q_width, layout.hidden, dtype=torch.float64)
        ),
        torch.nn.Parameter(
            torch.randn(layout.kv_width, layout.hidden, dtype=torch.float64)
        ),
        torch.nn.Parameter(
            torch.randn(layout.kv_width, layout.hidden, dtype=torch.float64)
        ),
    )


def _apply_pair_rope(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    pairs = tensor.reshape(*tensor.shape[:-1], tensor.shape[-1] // 2, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    cosine = cosine.unsqueeze(2)
    sine = sine.unsqueeze(2)
    return torch.stack(
        (
            first * cosine - second * sine,
            first * sine + second * cosine,
        ),
        dim=-1,
    ).flatten(-2)


def _causal_gqa_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    groups = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(groups, dim=2)
    v = v.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bthd,bshd->bhts", q, k)
    scores = scores / q.shape[-1] ** 0.5
    mask = torch.ones(
        scores.shape[-2:], dtype=torch.bool, device=scores.device
    ).triu(diagonal=1)
    probabilities = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
    return torch.einsum("bhts,bshd->bthd", probabilities, v)


def _split_attention(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    layout: PackedQKVLayout,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    batch, sequence, _hidden = x.shape
    rows = x.reshape(batch * sequence, layout.hidden)
    q = F.linear(rows, q_weight).reshape(
        batch, sequence, layout.q_heads, layout.head_dim
    )
    k = F.linear(rows, k_weight).reshape(
        batch, sequence, layout.kv_heads, layout.head_dim
    )
    v = F.linear(rows, v_weight).reshape(
        batch, sequence, layout.kv_heads, layout.head_dim
    )
    q = _apply_pair_rope(q, cosine, sine)
    k = _apply_pair_rope(k, cosine, sine)
    output = _causal_gqa_core(q, k, v)
    return F.linear(
        output.reshape(batch * sequence, layout.q_width), o_weight
    ).reshape_as(x)


def _packed_attention(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    o_weight: torch.Tensor,
    layout: PackedQKVLayout,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    batch, sequence, _hidden = x.shape
    q, k, v = project_packed_qkv(
        x.reshape(batch * sequence, layout.hidden),
        qkv_weight,
        layout,
        batch=batch,
        sequence=sequence,
    )
    q = _apply_pair_rope(q, cosine, sine)
    k = _apply_pair_rope(k, cosine, sine)
    output = _causal_gqa_core(q, k, v)
    return F.linear(
        output.reshape(batch * sequence, layout.q_width), o_weight
    ).reshape_as(x)


def test_layout_requires_native_gqa_and_exact_shapes() -> None:
    with pytest.raises(ValueError, match="divisible"):
        PackedQKVLayout(hidden=8, q_heads=3, kv_heads=2, head_dim=4)
    with pytest.raises(ValueError, match="hidden must be positive"):
        PackedQKVLayout(hidden=0, q_heads=4, kv_heads=2, head_dim=4)

    layout = _layout()
    with pytest.raises(ValueError, match="q weight"):
        pack_qkv_weights(
            torch.empty(layout.q_width - 1, layout.hidden),
            torch.empty(layout.kv_width, layout.hidden),
            torch.empty(layout.kv_width, layout.hidden),
            layout,
        )


def test_packed_projection_is_one_linear_with_view_only_native_gqa_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1)
    layout = _layout()
    batch, sequence = 2, 3
    rows = torch.randn(
        batch * sequence, layout.hidden, dtype=torch.float64
    )
    q, k, v = _split_parameters(layout)
    qkv = pack_qkv_weights(q, k, v, layout)
    calls = 0
    original_linear = F.linear

    def counted_linear(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_linear(*args, **kwargs)

    monkeypatch.setattr(packed_module.F, "linear", counted_linear)
    packed_q, packed_k, packed_v = project_packed_qkv(
        rows,
        qkv,
        layout,
        batch=batch,
        sequence=sequence,
    )
    assert calls == 1
    assert packed_q.shape == (
        batch,
        sequence,
        layout.q_heads,
        layout.head_dim,
    )
    assert packed_k.shape == packed_v.shape == (
        batch,
        sequence,
        layout.kv_heads,
        layout.head_dim,
    )
    storage_pointer = packed_q.untyped_storage().data_ptr()
    assert packed_k.untyped_storage().data_ptr() == storage_pointer
    assert packed_v.untyped_storage().data_ptr() == storage_pointer
    assert torch.equal(
        packed_q,
        original_linear(rows, q).reshape_as(packed_q),
    )
    assert torch.equal(
        packed_k,
        original_linear(rows, k).reshape_as(packed_k),
    )
    assert torch.equal(
        packed_v,
        original_linear(rows, v).reshape_as(packed_v),
    )


def test_full_causal_gqa_forward_input_grad_and_weight_grads_match() -> None:
    torch.manual_seed(2)
    layout = _layout()
    batch, sequence = 2, 5
    q, k, v = _split_parameters(layout)
    o = torch.nn.Parameter(
        torch.randn(layout.hidden, layout.q_width, dtype=torch.float64)
    )
    qkv = torch.nn.Parameter(
        pack_qkv_weights(q, k, v, layout).detach().clone()
    )
    packed_o = torch.nn.Parameter(o.detach().clone())
    split_x = torch.randn(
        batch, sequence, layout.hidden, dtype=torch.float64, requires_grad=True
    )
    packed_x = split_x.detach().clone().requires_grad_(True)
    angles = torch.randn(
        batch, sequence, layout.head_dim // 2, dtype=torch.float64
    )
    cosine, sine = angles.cos(), angles.sin()
    output_weight = torch.randn_like(split_x)

    split_output = _split_attention(
        split_x, q, k, v, o, layout, cosine, sine
    )
    packed_output = _packed_attention(
        packed_x, qkv, packed_o, layout, cosine, sine
    )
    torch.testing.assert_close(packed_output, split_output, rtol=0.0, atol=0.0)
    (split_output * output_weight).sum().backward()
    (packed_output * output_weight).sum().backward()

    torch.testing.assert_close(packed_x.grad, split_x.grad, rtol=1e-13, atol=1e-13)
    packed_q_grad, packed_k_grad, packed_v_grad = unpack_qkv_weight(
        qkv.grad, layout
    )
    torch.testing.assert_close(packed_q_grad, q.grad, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(packed_k_grad, k.grad, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(packed_v_grad, v.grad, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(packed_o.grad, o.grad, rtol=1e-13, atol=1e-13)


def test_packed_initialization_preserves_split_rng_boundaries() -> None:
    layout = _layout()
    torch.manual_seed(3)
    q = torch.empty(layout.q_width, layout.hidden, dtype=torch.float64)
    k = torch.empty(layout.kv_width, layout.hidden, dtype=torch.float64)
    v = torch.empty(layout.kv_width, layout.hidden, dtype=torch.float64)
    o = torch.empty(layout.hidden, layout.q_width, dtype=torch.float64)
    for tensor in (q, k, v, o):
        torch.nn.init.normal_(tensor, mean=0.0, std=0.02)

    torch.manual_seed(3)
    packed = PackedQKVAttentionWeights(
        layout, dtype=torch.float64, std=0.02
    )
    packed_q, packed_k, packed_v = unpack_qkv_weight(packed.qkv, layout)
    assert torch.equal(packed_q, q)
    assert torch.equal(packed_k, k)
    assert torch.equal(packed_v, v)
    assert torch.equal(packed.o, o)


def test_adamw_is_elementwise_equivalent_for_packed_parameter() -> None:
    torch.manual_seed(4)
    layout = _layout()
    q, k, v = _split_parameters(layout)
    packed = torch.nn.Parameter(
        pack_qkv_weights(q, k, v, layout).detach().clone()
    )
    split_optimizer = torch.optim.AdamW(
        (q, k, v),
        lr=3.0e-3,
        betas=(0.8, 0.95),
        eps=1.0e-9,
        weight_decay=0.1,
        foreach=False,
    )
    packed_optimizer = torch.optim.AdamW(
        (packed,),
        lr=3.0e-3,
        betas=(0.8, 0.95),
        eps=1.0e-9,
        weight_decay=0.1,
        foreach=False,
    )

    for step in range(5):
        generator = torch.Generator().manual_seed(100 + step)
        q_grad = torch.randn(q.shape, generator=generator, dtype=q.dtype)
        k_grad = torch.randn(k.shape, generator=generator, dtype=k.dtype)
        v_grad = torch.randn(v.shape, generator=generator, dtype=v.dtype)
        q.grad, k.grad, v.grad = q_grad, k_grad, v_grad
        packed.grad = pack_qkv_weights(q_grad, k_grad, v_grad, layout)
        split_optimizer.step()
        packed_optimizer.step()
        split_optimizer.zero_grad(set_to_none=True)
        packed_optimizer.zero_grad(set_to_none=True)
        torch.testing.assert_close(
            packed,
            pack_qkv_weights(q, k, v, layout),
            rtol=1e-15,
            atol=1e-15,
        )

    packed_state = packed_optimizer.state[packed]
    for state_name in ("exp_avg", "exp_avg_sq"):
        expected = pack_qkv_weights(
            split_optimizer.state[q][state_name],
            split_optimizer.state[k][state_name],
            split_optimizer.state[v][state_name],
            layout,
        )
        torch.testing.assert_close(
            packed_state[state_name], expected, rtol=1e-15, atol=1e-15
        )
    for parameter in (q, k, v):
        assert torch.equal(
            packed_state["step"], split_optimizer.state[parameter]["step"]
        )


def test_checkpoint_mapping_is_lossless_and_rejects_ambiguous_groups() -> None:
    torch.manual_seed(5)
    layout = _layout()
    q, k, v = _split_parameters(layout)
    prefix = "layers.0.attention.weights."
    split = OrderedDict(
        (
            ("embedding", torch.randn(7, 3)),
            (f"{prefix}q", q.detach().clone()),
            (f"{prefix}k", k.detach().clone()),
            (f"{prefix}v", v.detach().clone()),
            (
                f"{prefix}o",
                torch.randn(layout.hidden, layout.q_width),
            ),
        )
    )
    packed = pack_qkv_state_dict(split, layout)
    assert list(packed) == [
        "embedding",
        f"{prefix}qkv",
        f"{prefix}o",
    ]
    round_trip = unpack_qkv_state_dict(packed, layout)
    assert list(round_trip) == list(split)
    for key in split:
        assert torch.equal(round_trip[key], split[key])
    assert (
        round_trip[f"{prefix}q"].untyped_storage().data_ptr()
        != round_trip[f"{prefix}k"].untyped_storage().data_ptr()
    )

    canonical_views = canonical_split_qkv_tensors(packed, layout)
    assert list(canonical_views) == list(split)
    packed_pointer = packed[f"{prefix}qkv"].untyped_storage().data_ptr()
    assert (
        canonical_views[f"{prefix}q"].untyped_storage().data_ptr()
        == packed_pointer
    )
    assert (
        canonical_views[f"{prefix}k"].untyped_storage().data_ptr()
        == packed_pointer
    )
    assert (
        canonical_views[f"{prefix}v"].untyped_storage().data_ptr()
        == packed_pointer
    )

    incomplete = OrderedDict(
        ((f"{prefix}q", q), (f"{prefix}k", k))
    )
    with pytest.raises(KeyError, match="incomplete"):
        pack_qkv_state_dict(incomplete, layout)
    ambiguous = OrderedDict(
        (
            (f"{prefix}q", q),
            (f"{prefix}k", k),
            (f"{prefix}v", v),
            (f"{prefix}qkv", pack_qkv_weights(q, k, v, layout)),
        )
    )
    with pytest.raises(KeyError, match="both split and packed"):
        pack_qkv_state_dict(ambiguous, layout)


def test_packed_weight_module_strictly_loads_split_checkpoint_views() -> None:
    torch.manual_seed(6)
    layout = _layout()
    q, k, v = _split_parameters(layout)
    o = torch.randn(layout.hidden, layout.q_width, dtype=torch.float64)
    module = PackedQKVAttentionWeights(layout, dtype=torch.float64)

    incompatible = module.load_state_dict(
        OrderedDict(
            (
                ("q", q.detach().clone()),
                ("k", k.detach().clone()),
                ("v", v.detach().clone()),
                ("o", o.clone()),
            )
        ),
        strict=True,
    )

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    loaded_q, loaded_k, loaded_v = unpack_qkv_weight(module.qkv, layout)
    assert torch.equal(loaded_q, q)
    assert torch.equal(loaded_k, k)
    assert torch.equal(loaded_v, v)
    assert torch.equal(module.o, o)


def test_parameter_accounting_and_production_control_are_explicit() -> None:
    layout = _layout()
    packed = PackedQKVAttentionWeights(layout, dtype=torch.float32)
    expected = (
        layout.hidden * layout.q_width
        + 2 * layout.hidden * layout.kv_width
        + layout.hidden * layout.q_width
    )
    assert sum(parameter.numel() for parameter in packed.parameters()) == expected
    assert set(dict(packed.named_parameters())) == {"qkv", "o"}
    q, k, v = packed.q, packed.k, packed.v
    packed_pointer = packed.qkv.untyped_storage().data_ptr()
    assert q.untyped_storage().data_ptr() == packed_pointer
    assert k.untyped_storage().data_ptr() == packed_pointer
    assert v.untyped_storage().data_ptr() == packed_pointer
    assert tuple(q.shape) == (layout.q_width, layout.hidden)
    assert tuple(k.shape) == tuple(v.shape) == (
        layout.kv_width,
        layout.hidden,
    )

    source = BENCHMARK.read_text()
    packed_attention = source.split("class PackedQKVBF16Attention", 1)[1]
    packed_attention = packed_attention.split("def _require_forward_topology", 1)[0]
    assert "project_packed_qkv(" in packed_attention
    assert "_apply_pair_rope(q, *self.rope)" in packed_attention
    assert "_apply_pair_rope(k, *self.rope)" in packed_attention
    assert "flash_attn_func(q, k, v, causal=True)" in packed_attention
    assert '"bf16_cute_packed_qkv_single_linear"' in source
    assert "DEFAULT_BF16_ATTENTION_CONTROL = \"split_qkv_three_linear\"" in source
