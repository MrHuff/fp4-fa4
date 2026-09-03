#!/usr/bin/env python3
"""Run a short paired BF16/low-precision D64 GQA training probe.

This is an attention-block teacher/student diagnostic, not language-model
pretraining.  It exercises the current causal NVFP4-QK/MXFP4-PV forward,
projection-native FP8 dO/statistics publication, direct-TMA D64 GQA backward,
RoPE, learned Q/K/V/output projections, and AdamW updates.  The BF16 and
low-precision students start from identical weights and see identical data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tk_fa4 import (
    b300_prepare_nvfp4_projection_operand,
    b300_project_dout_unified_lowp_nvfp4,
    b300_project_nvfp4,
)
from tk_fa4.lowp_fa4_bwd.profile_gqa_d128_chain import (
    CompiledGqaBackward,
    _inverse_rope_pair_native,
    _load_extension,
    _make_rope,
    _metrics,
)
from tk_fa4.lowp_fa4_bwd.tune_d64_gqa_cute import _load_control


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FORWARD_DIR = REPO_ROOT / "tk_fa4" / "fp4_fa4_fwd"
FLASH_ATTN_ROOT = REPO_ROOT / "flash-attention"
sys.path.insert(0, str(FORWARD_DIR))
try:
    from hao_direct_fp4pv_benchmark import (
        prepare_native_inputs,
        quantize_mxfp4_v,
        quantize_nvfp4_qk,
    )
finally:
    sys.path.pop(0)
sys.path.insert(0, str(FLASH_ATTN_ROOT))
try:
    from flash_attn.cute.interface import flash_attn_func
finally:
    sys.path.pop(0)


@dataclass
class Weights:
    q: torch.nn.Parameter
    k: torch.nn.Parameter
    v: torch.nn.Parameter
    out: torch.nn.Parameter

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.q, self.k, self.v, self.out]

    def bf16(self) -> tuple[torch.Tensor, ...]:
        return tuple(parameter.bfloat16() for parameter in self.parameters())


def _clone_weights(weights: Weights) -> Weights:
    return Weights(
        *(torch.nn.Parameter(value.detach().clone()) for value in weights.parameters())
    )


def _apply_rope(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    pairs = tensor.float().reshape(*tensor.shape[:-1], tensor.shape[-1] // 2, 2)
    x, y = pairs[..., 0], pairs[..., 1]
    cosine_f = cosine.float().unsqueeze(2)
    sine_f = sine.float().unsqueeze(2)
    return torch.stack(
        (x * cosine_f - y * sine_f, x * sine_f + y * cosine_f),
        dim=-1,
    ).flatten(-2).bfloat16()


def _relative_mse(
    actual: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    difference = actual.float() - target.float()
    denominator = target.float().square().mean().clamp_min(1.0e-12)
    loss = difference.square().mean() / denominator
    gradient = difference * (2.0 / (difference.numel() * denominator))
    return loss, gradient


def _gradient_metrics(
    reference: list[torch.Tensor],
    actual: list[torch.Tensor],
) -> dict[str, Any]:
    names = ("q", "k", "v", "out")
    return {
        name: _metrics(ref, candidate)
        for name, ref, candidate in zip(names, reference, actual)
    }


def _weight_metrics(reference: Weights, actual: Weights) -> dict[str, Any]:
    return _gradient_metrics(
        [value.detach() for value in reference.parameters()],
        [value.detach() for value in actual.parameters()],
    )


class LowpRoute:
    def __init__(
        self,
        *,
        extension: Any,
        topology: dict[str, Any],
        sequence: int,
        q_heads: int,
        kv_heads: int,
        hidden: int,
        rope: tuple[torch.Tensor, torch.Tensor],
        loss_scale: float,
    ) -> None:
        self.extension = extension
        self.topology = topology
        self.sequence = sequence
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.hidden = hidden
        self.depth = 64
        self.rope = rope
        self.loss_scale = loss_scale
        self.output = torch.empty(
            1,
            sequence,
            q_heads,
            self.depth,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.lse = torch.empty(
            1,
            q_heads,
            1,
            sequence,
            device="cuda",
            dtype=torch.float32,
        )
        self.q_fp8 = torch.empty(
            1,
            sequence,
            q_heads,
            self.depth,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        self.k_fp8 = torch.empty(
            1,
            sequence,
            kv_heads,
            self.depth,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        self.v_fp8 = torch.empty_like(self.k_fp8)
        self.dout_fp8 = torch.zeros_like(self.q_fp8)
        stats = torch.zeros(1, q_heads, 1, sequence, device="cuda")
        control = _load_control(
            fp8_p_storage="tmem",
            direct_tma_dkdv=True,
        )
        self.backward = CompiledGqaBackward(
            control,
            q=self.q_fp8,
            k=self.k_fp8,
            v=self.v_fp8,
            o_or_sum=stats,
            dout=self.dout_fp8,
            lse_or_scaled_lse=stats,
            q_heads=q_heads,
            kv_heads=kv_heads,
            lowp=True,
            precomputed_stats=True,
            workspace_stats=True,
            scale_softmax=(self.depth**-0.5) / 16.0,
            exp2_degree=1,
            exp2_period=2,
            reuse_quantized_p=False,
            lowp_do_stages=1,
            direct_tma_dkdv=True,
        )

    def _project_qkv(
        self,
        x: torch.Tensor,
        weights: Weights,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_operand = tuple(b300_prepare_nvfp4_projection_operand(x))
        q_weight, k_weight, v_weight, _ = weights.bf16()
        q = b300_project_nvfp4(
            x_operand,
            tuple(b300_prepare_nvfp4_projection_operand(q_weight)),
        ).reshape(1, self.sequence, self.q_heads, self.depth)
        k = b300_project_nvfp4(
            x_operand,
            tuple(b300_prepare_nvfp4_projection_operand(k_weight)),
        ).reshape(1, self.sequence, self.kv_heads, self.depth)
        v = b300_project_nvfp4(
            x_operand,
            tuple(b300_prepare_nvfp4_projection_operand(v_weight)),
        ).reshape(1, self.sequence, self.kv_heads, self.depth)
        q = _apply_rope(q, *self.rope)
        k = _apply_rope(k, *self.rope)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        weights: Weights,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        q, k, v = self._project_qkv(x, weights)
        q_packed, q_scales = quantize_nvfp4_qk(q, 1.0)
        k_packed, k_scales = quantize_nvfp4_qk(k, 1.0)
        v_packed, v_scales = quantize_mxfp4_v(v, mode=0, tile_keys=128)
        prepared = prepare_native_inputs(
            q_packed,
            k_packed,
            v_packed,
            q_scales,
            k_scales,
            v_scales,
            "nvfp4",
            "mxfp4",
            1,
            self.sequence,
            self.q_heads,
            self.depth,
            self.depth,
            kv_heads=self.kv_heads,
            key_tile=128,
        )
        self.extension.forward_hao_direct_fp4pv(
            prepared.q_fp4_bhsd,
            prepared.q_scale_prepared,
            prepared.q_global_scale,
            prepared.k_fp4_bhsd,
            prepared.k_scale_prepared,
            prepared.k_global_scale,
            prepared.v_fp4_bhds,
            prepared.v_scale_prepared,
            self.output,
            self.lse,
            0,
            True,
            True,
        )
        out_matrix = self.output.reshape(self.sequence, -1)
        out_weight = weights.out.bfloat16()
        y = b300_project_nvfp4(
            tuple(b300_prepare_nvfp4_projection_operand(out_matrix)),
            tuple(b300_prepare_nvfp4_projection_operand(out_weight)),
        )
        return y, (q, k, v)

    def gradients(
        self,
        x: torch.Tensor,
        weights: Weights,
        target: torch.Tensor,
        y: torch.Tensor,
        qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        diagnose: bool = False,
    ) -> tuple[float, list[torch.Tensor], dict[str, Any] | None]:
        loss, dy = _relative_mse(y, target)
        dy_scaled = (dy * self.loss_scale).bfloat16().contiguous()
        out_weight = weights.out.bfloat16()
        lse_bsh = self.lse[:, :, 0].permute(0, 2, 1).contiguous()
        self.backward.reset()
        dout_bundle = b300_project_dout_unified_lowp_nvfp4(
            tuple(b300_prepare_nvfp4_projection_operand(dy_scaled)),
            tuple(
                b300_prepare_nvfp4_projection_operand(
                    out_weight.T.contiguous()
                )
            ),
            self.output,
            lse_bsh,
            batch=1,
            seqlen=self.sequence,
            heads=self.q_heads,
            store_bf16=diagnose,
            publish_fp8_backward=True,
            publish_stats=True,
            stats_workspace=self.backward.workspace_torch,
        )
        assert dout_bundle.dout_backward_fp8 is not None
        q, k, v = qkv
        self.q_fp8.copy_((q.float() * 4.0).to(torch.float8_e4m3fn))
        self.k_fp8.copy_((k.float() * 4.0).to(torch.float8_e4m3fn))
        self.v_fp8.copy_((v.float() * 4.0).to(torch.float8_e4m3fn))
        self.dout_fp8.copy_(dout_bundle.dout_backward_fp8)
        self.backward.run(reset=False)
        dq = _inverse_rope_pair_native(self.backward.dq, *self.rope)
        dk = _inverse_rope_pair_native(self.backward.dk, *self.rope)
        # dV is accumulated against the fixed-scale FP8 dO operand and is
        # therefore published in 4x encoded-dO units.  Projection consumers
        # must fold this decode into their input transform.
        dv = (self.backward.dv.float() / 4.0).bfloat16()
        scale = 1.0 / self.loss_scale
        gradients = [
            torch.mm(dq.reshape(self.sequence, -1).T, x).float() * scale,
            torch.mm(dk.reshape(self.sequence, -1).T, x).float() * scale,
            torch.mm(dv.reshape(self.sequence, -1).T, x).float() * scale,
            torch.mm(dy_scaled.T, self.output.reshape(self.sequence, -1)).float()
            * scale,
        ]
        diagnostic = None
        if diagnose:
            assert dout_bundle.dout is not None
            q_reference = q.detach().requires_grad_(True)
            k_reference = k.detach().requires_grad_(True)
            v_reference = v.detach().requires_grad_(True)
            exact_output = flash_attn_func(
                q_reference,
                k_reference,
                v_reference,
                causal=True,
            )
            if isinstance(exact_output, tuple):
                exact_output = exact_output[0]
            exact_output.backward(dout_bundle.dout)
            assert q_reference.grad is not None
            assert k_reference.grad is not None
            assert v_reference.grad is not None
            diagnostic = {
                "exact_attention_output_vs_lowp": _metrics(
                    exact_output.detach(), self.output
                ),
                "dout_projection_vs_bf16": _metrics(
                    torch.mm(dy_scaled, out_weight).reshape_as(self.output),
                    dout_bundle.dout,
                ),
                "dout_fp8_publication_vs_projection": _metrics(
                    dout_bundle.dout,
                    dout_bundle.dout_backward_fp8.float() / 4.0,
                ),
                "attention_gradient_same_qkv_and_dout": {
                    "dq": _metrics(
                        q_reference.grad.detach(), self.backward.dq
                    ),
                    "dk": _metrics(
                        k_reference.grad.detach(), self.backward.dk
                    ),
                    "dv_decoded": _metrics(
                        v_reference.grad.detach(),
                        self.backward.dv.float() / 4.0,
                    ),
                    "dv_raw_encoded_units": _metrics(
                        v_reference.grad.detach(), self.backward.dv
                    ),
                },
            }
        return float(loss), gradients, diagnostic


def _bf16_forward(
    x: torch.Tensor,
    weights: Weights,
    *,
    sequence: int,
    q_heads: int,
    kv_heads: int,
    rope: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    q_weight, k_weight, v_weight, out_weight = weights.bf16()
    q = torch.mm(x, q_weight.T).reshape(1, sequence, q_heads, 64)
    k = torch.mm(x, k_weight.T).reshape(1, sequence, kv_heads, 64)
    v = torch.mm(x, v_weight.T).reshape(1, sequence, kv_heads, 64)
    q = _apply_rope(q, *rope)
    k = _apply_rope(k, *rope)
    out = flash_attn_func(q, k, v, causal=True)
    if isinstance(out, tuple):
        out = out[0]
    return torch.mm(out.reshape(sequence, -1), out_weight.T)


def _make_weights(
    *,
    hidden: int,
    q_heads: int,
    kv_heads: int,
    seed: int,
) -> Weights:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    def weight(rows: int) -> torch.nn.Parameter:
        return torch.nn.Parameter(
            torch.randn(
                rows,
                hidden,
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.02
        )

    return Weights(
        weight(q_heads * 64),
        weight(kv_heads * 64),
        weight(kv_heads * 64),
        weight(hidden),
    )


def _noisy_clone(
    teacher: Weights,
    *,
    relative_noise: float,
    seed: int,
) -> Weights:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    values = []
    for source in teacher.parameters():
        noise = torch.randn(
            source.shape,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        values.append(
            torch.nn.Parameter(
                source.detach() + noise * source.detach().std() * relative_noise
            )
        )
    return Weights(*values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--relative-noise", type=float, default=0.20)
    parser.add_argument("--loss-scale", type=float, default=2.0**16)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--extension",
        type=Path,
        default=Path(
            "/tmp/_C_tk_gb200_causal_s4096_h32_d64."
            "cpython-312-aarch64-linux-gnu.so"
        ),
    )
    parser.add_argument(
        "--module",
        default="_C_tk_gb200_causal_s4096_h32_d64",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("the short training probe requires one visible GPU")
    if (args.sequence, args.q_heads, args.kv_heads, args.hidden) != (
        4096,
        32,
        8,
        2048,
    ):
        raise ValueError("the retained artifact is fixed at S4096/Hq32/Hkv8/K2048")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    extension = _load_extension(args.extension, args.module)
    topology = dict(extension.read_hao_direct_topology())
    expected = {
        "seqlen": args.sequence,
        "heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "dqk": 64,
        "dvo": 64,
        "causal": True,
    }
    for name, value in expected.items():
        if topology[name] != value:
            raise ValueError(f"forward topology {name}={topology[name]} != {value}")
    os.environ["TK_FA4_FP4PV_FWD_CONFIG"] = str(topology["route"])

    rope = _make_rope(args.sequence, 64)
    teacher = _make_weights(
        hidden=args.hidden,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        seed=args.seed + 1,
    )
    initial = _noisy_clone(
        teacher,
        relative_noise=args.relative_noise,
        seed=args.seed + 2,
    )
    bf16_weights = _clone_weights(initial)
    lowp_weights = _clone_weights(initial)
    train_x = (
        torch.randn(args.sequence, args.hidden, device="cuda") * 0.1
    ).bfloat16()
    validation_x = (
        torch.randn(args.sequence, args.hidden, device="cuda") * 0.1
    ).bfloat16()
    with torch.no_grad():
        train_target = _bf16_forward(
            train_x,
            teacher,
            sequence=args.sequence,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            rope=rope,
        )
        validation_target = _bf16_forward(
            validation_x,
            teacher,
            sequence=args.sequence,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            rope=rope,
        )

    route = LowpRoute(
        extension=extension,
        topology=topology,
        sequence=args.sequence,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        hidden=args.hidden,
        rope=rope,
        loss_scale=args.loss_scale,
    )
    bf16_optimizer = torch.optim.AdamW(
        bf16_weights.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    lowp_optimizer = torch.optim.AdamW(
        lowp_weights.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    records: list[dict[str, Any]] = []
    for step in range(1, args.steps + 1):
        started = time.perf_counter()
        bf16_start = torch.cuda.Event(enable_timing=True)
        bf16_end = torch.cuda.Event(enable_timing=True)
        lowp_start = torch.cuda.Event(enable_timing=True)
        lowp_end = torch.cuda.Event(enable_timing=True)

        bf16_start.record()
        bf16_optimizer.zero_grad(set_to_none=True)
        bf16_y = _bf16_forward(
            train_x,
            bf16_weights,
            sequence=args.sequence,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            rope=rope,
        )
        bf16_loss, _ = _relative_mse(bf16_y, train_target)
        bf16_loss.backward()
        bf16_loss_value = float(bf16_loss.detach())
        bf16_gradients = [
            parameter.grad.detach().clone()
            for parameter in bf16_weights.parameters()
            if parameter.grad is not None
        ]
        bf16_optimizer.step()
        bf16_end.record()

        lowp_start.record()
        lowp_optimizer.zero_grad(set_to_none=True)
        lowp_y, qkv = route.forward(train_x, lowp_weights)
        lowp_loss, lowp_gradients, backward_diagnostic = route.gradients(
            train_x,
            lowp_weights,
            train_target,
            lowp_y,
            qkv,
            diagnose=step == 1,
        )
        for parameter, gradient in zip(
            lowp_weights.parameters(), lowp_gradients
        ):
            parameter.grad = gradient

        gradient_quality = _gradient_metrics(bf16_gradients, lowp_gradients)
        lowp_optimizer.step()
        lowp_end.record()
        lowp_end.synchronize()
        bf16_gpu_ms = float(bf16_start.elapsed_time(bf16_end))
        lowp_gpu_ms = float(lowp_start.elapsed_time(lowp_end))
        record = {
            "step": step,
            "bf16_train_relative_mse": bf16_loss_value,
            "lowp_train_relative_mse": lowp_loss,
            "lowp_to_bf16_loss_ratio": lowp_loss / max(bf16_loss_value, 1.0e-30),
            "same_step_output": _metrics(bf16_y.detach(), lowp_y),
            "gradient_quality": gradient_quality,
            "backward_diagnostic": backward_diagnostic,
            "weight_quality_after_update": _weight_metrics(
                bf16_weights, lowp_weights
            ),
            "all_finite": bool(
                math.isfinite(bf16_loss_value)
                and math.isfinite(lowp_loss)
                and all(torch.isfinite(value).all() for value in lowp_gradients)
                and all(
                    torch.isfinite(value).all()
                    for value in lowp_weights.parameters()
                )
            ),
            "timing": {
                "bf16_training_boundary_ms": bf16_gpu_ms,
                "lowp_training_boundary_ms": lowp_gpu_ms,
                "speedup_lowp_over_bf16": bf16_gpu_ms / lowp_gpu_ms,
            },
            "wall_seconds": time.perf_counter() - started,
        }
        records.append(record)
        print(
            f"step={step} bf16={bf16_loss_value:.6e} "
            f"lowp={lowp_loss:.6e} ratio={record['lowp_to_bf16_loss_ratio']:.4f} "
            f"speedup={record['timing']['speedup_lowp_over_bf16']:.3f}x "
            f"finite={record['all_finite']}",
            flush=True,
        )

    with torch.no_grad():
        bf16_validation = _relative_mse(
            _bf16_forward(
                validation_x,
                bf16_weights,
                sequence=args.sequence,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                rope=rope,
            ),
            validation_target,
        )[0]
        lowp_validation_y, _ = route.forward(validation_x, lowp_weights)
        lowp_validation = _relative_mse(
            lowp_validation_y, validation_target
        )[0]

    steady_records = records[1:] if len(records) > 1 else records
    bf16_times = [
        record["timing"]["bf16_training_boundary_ms"]
        for record in steady_records
    ]
    lowp_times = [
        record["timing"]["lowp_training_boundary_ms"]
        for record in steady_records
    ]
    bf16_median = statistics.median(bf16_times)
    lowp_median = statistics.median(lowp_times)
    result = {
        "configuration": {
            "sequence": args.sequence,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": 64,
            "hidden": args.hidden,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "relative_noise": args.relative_noise,
            "loss_scale": args.loss_scale,
            "seed": args.seed,
            "forward_route": topology["route"],
            "backward_route": "fp8_direct_tma_dkdv",
        },
        "records": records,
        "steady_state_timing": {
            "discarded_initial_steps": int(len(records) > 1),
            "samples": len(steady_records),
            "bf16_training_boundary_median_ms": bf16_median,
            "lowp_training_boundary_median_ms": lowp_median,
            "speedup_lowp_over_bf16": bf16_median / lowp_median,
        },
        "final_validation": {
            "bf16_relative_mse": float(bf16_validation),
            "lowp_relative_mse": float(lowp_validation),
            "lowp_to_bf16_ratio": float(lowp_validation / bf16_validation),
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
