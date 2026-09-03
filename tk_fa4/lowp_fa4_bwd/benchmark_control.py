#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics

import torch

from tk_fa4 import _C as tk_ext
from tk_fa4 import _C_b300_lowp_bwd as lowp_ext
from tk_fa4.interface import b300_mha_fwd


def time_rotated(callables, warmup: int, iterations: int):
    names = tuple(callables)
    for i in range(warmup):
        for offset in range(len(names)):
            callables[names[(i + offset) % len(names)]]()
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    for i in range(iterations):
        for offset in range(len(names)):
            name = names[(i + offset) % len(names)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            callables[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=31)
    parser.add_argument("--seed", type=int, default=2026080401)
    parser.add_argument("--qk-std", type=float, default=0.1)
    parser.add_argument("--v-std", type=float, default=0.1)
    parser.add_argument("--dout-std", type=float, default=0.1)
    parser.add_argument("--q-quant-scale", type=float, default=256.0)
    parser.add_argument("--k-quant-scale", type=float, default=256.0)
    parser.add_argument("--ds-quant-scale", type=float, default=4096.0)
    parser.add_argument("--fp4-q-quant-scale", type=float, default=16.0)
    parser.add_argument("--fp4-k-quant-scale", type=float, default=16.0)
    parser.add_argument("--fp4-ds-quant-scale", type=float, default=4096.0)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--compare-split-q",
        action="store_true",
        help="time only forward-log mode 8 and split-Q mode 9",
    )
    parser.add_argument(
        "--compare-dp-formats",
        action="store_true",
        help="time only the retained FP8-dP hybrid and MXFP4-dP candidate",
    )
    parser.add_argument(
        "--compare-winner",
        action="store_true",
        help="validate and time only copied BF16 versus FP4+FP8-dP/dV",
    )
    parser.add_argument(
        "--compare-mixed-dp",
        action="store_true",
        help="validate and time the retained FP8 dP against three-command mixed dP",
    )
    parser.add_argument(
        "--compare-prepacked-v",
        action="store_true",
        help="validate and time inline versus forward-reusable mixed V packing",
    )
    parser.add_argument(
        "--compare-fp8-pipeline",
        action="store_true",
        help="validate and time pure FP8 versus BF16 and FP4+FP8-dP/dV",
    )
    parser.add_argument(
        "--compare-pure",
        action="store_true",
        help="validate and time only copied BF16 versus the pure-FP4 route",
    )
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help="also time fused Q/K FP4 packing and packing+backward",
    )
    parser.add_argument("--dout-tile", type=int, default=-1)
    parser.add_argument(
        "--profile-mode",
        choices=(
            "original",
            "copied",
            "fp8",
            "fp4",
            "fp4_fp8pv",
            "fp4_fp8pv_x32",
            "fp4_fp8pv_x32_reuse_p",
            "fp4_fp8pv_x32_split_dk",
            "fp4_fp8dpdv_x32_split_dk",
            "fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk",
            "fp4_mixed_prepacked_v",
            "fp4_fp8dp_mxfp4dv_x32_split_dk",
            "fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk",
            "fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk",
            "fp4_mxfp4dpdv_forward_log_split_q_x32_split_dk",
            "fp4_pure_mxfp4",
            "fp4_full",
        ),
        default=None,
        help="run only one backward route and skip validation/timing output",
    )
    parser.add_argument("--profile-launches", type=int, default=1)
    parser.add_argument(
        "--stress-launches",
        type=int,
        default=0,
        help="require repeated FP4/FP8-PV launches to remain finite and bounded",
    )
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible GPU")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    shape_qk = (1, args.seqlen, args.heads, 192)
    shape_v = (1, args.seqlen, args.heads, 128)
    q = (torch.randn(shape_qk, device=device) * args.qk_std).to(torch.bfloat16)
    k = (torch.randn(shape_qk, device=device) * args.qk_std).to(torch.bfloat16)
    v = (torch.randn(shape_v, device=device) * args.v_std).to(torch.bfloat16)
    dout = (torch.randn(shape_v, device=device) * args.dout_std).to(torch.bfloat16)
    if args.dout_tile >= 0:
        if args.dout_tile >= args.seqlen // 128:
            raise ValueError("dout-tile is outside the sequence")
        selected_dout = dout[:, args.dout_tile * 128 : (args.dout_tile + 1) * 128].clone()
        dout.zero_()
        dout[:, args.dout_tile * 128 : (args.dout_tile + 1) * 128] = selected_dout
    out, lse = b300_mha_fwd(q, k, v, causal=True, return_lse=True)
    scale = float(192**-0.5)
    q_fp8 = (
        (q.float() * args.q_quant_scale)
        .to(torch.float8_e4m3fn)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    k_fp8 = (
        (k.float() * args.k_quant_scale)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    (
        q_fp4,
        score_q_fp4,
        k_fp4,
        score_k_fp4,
        q_dk_mxfp4,
        k_dq_mxfp4,
        q_dk_nvfp4_scale,
        k_dq_nvfp4_scale,
    ) = (
        lowp_ext.quantize_fp4_dual_qk_blockscale(
            q,
            k,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
        )
    )
    needs_prepacked_v = (
        args.compare_prepacked_v
        or args.profile_mode == "fp4_mixed_prepacked_v"
    )
    mixed_v_prepacked = (
        lowp_ext.prepack_mixed_v(v) if needs_prepacked_v else None
    )
    advanced_heads = {
        8192: frozenset((8, 16)),
        16384: frozenset((4, 8, 16, 32, 64, 128)),
        32768: frozenset((16, 32, 64, 128)),
        65536: frozenset((16, 32, 64, 128)),
    }
    original_fn = tk_ext.b300_mha_bwd_hot_cute16_candidate_internal
    if args.heads in advanced_heads.get(args.seqlen, ()):
        original_fn = getattr(
            tk_ext,
            f"b300_mha_bwd_hot_cute16_candidate_s{args.seqlen}_v382_advanced_long_internal",
            original_fn,
        )

    def original():
        return original_fn(
            q, k, v, out, lse, dout, True, scale, args.seqlen, False
        )

    def copied():
        return lowp_ext.backward_bf16_control(
            q, k, v, out, lse, dout, True, scale, False
        )

    def fp8_native():
        return lowp_ext.backward_fp8_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp8,
            k_fp8,
            args.q_quant_scale,
            args.k_quant_scale,
            args.ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_native():
        return lowp_ext.backward_fp4_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_fp8pv_native():
        return lowp_ext.backward_fp4_fp8pv_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_fp8pv_x32_native():
        return lowp_ext.backward_fp4_fp8pv_x32_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_fp8pv_x32_reuse_p_native():
        return lowp_ext.backward_fp4_fp8pv_x32_reuse_p_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_fp8pv_x32_split_dk_native():
        return lowp_ext.backward_fp4_fp8pv_x32_reuse_p_split_dk_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_fp8dpdv_x32_split_dk_native():
        return lowp_ext.backward_fp4_fp8dpdv_x32_split_dk_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_native():
        return lowp_ext.backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_mixed_prepacked_v_native():
        if mixed_v_prepacked is None:
            raise RuntimeError("mixed V was not prepacked for this route")
        return lowp_ext.backward_fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_prepacked_v_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
            mixed_v_prepacked,
        )

    def fp4_fp8dp_mxfp4dv_x32_split_dk_native():
        return lowp_ext.backward_fp4_fp8dp_mxfp4dv_x32_split_dk_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native():
        return lowp_ext.backward_fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native():
        return lowp_ext.backward_fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_mxfp4dpdv_forward_log_split_q_x32_split_dk_native():
        return lowp_ext.backward_fp4_mxfp4dpdv_forward_log_split_q_x32_split_dk_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_pure_mxfp4_native():
        return lowp_ext.backward_fp4_mxfp4dpdvdsdqdk_forward_log_split_q_x32_native(
            q,
            k,
            v,
            out,
            lse,
            dout,
            q_fp4,
            score_q_fp4,
            k_fp4,
            score_k_fp4,
            q_dk_mxfp4,
            k_dq_mxfp4,
            q_dk_nvfp4_scale,
            k_dq_nvfp4_scale,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    def fp4_quantize():
        return lowp_ext.quantize_fp4_dual_qk_unpacked(
            q,
            k,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
        )

    def fp4_full():
        return lowp_ext.backward_fp4_fused_quant(
            q,
            k,
            v,
            out,
            lse,
            dout,
            args.fp4_q_quant_scale,
            args.fp4_k_quant_scale,
            args.fp4_ds_quant_scale,
            True,
            scale,
            False,
        )

    if args.compare_mixed_dp:
        reference_output = copied()
        winner_output = fp4_fp8dpdv_x32_split_dk_native()
        mixed_output = fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_native()
        torch.cuda.synchronize()
        for route_name, route_output in (
            ("winner", winner_output),
            ("mixed", mixed_output),
        ):
            for name, reference, actual in zip(
                ("dq", "dk", "dv"), reference_output, route_output
            ):
                reference_flat = reference.float().flatten()
                actual_flat = actual.float().flatten()
                difference = actual_flat - reference_flat
                relative_l2 = (
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(reference_flat).clamp_min(1e-12)
                )
                norm_ratio = (
                    torch.linalg.vector_norm(actual_flat)
                    / torch.linalg.vector_norm(reference_flat).clamp_min(1e-12)
                )
                cosine = torch.nn.functional.cosine_similarity(
                    reference_flat, actual_flat, dim=0
                )
                print(
                    f"{route_name}_{name}: "
                    f"finite={bool(torch.isfinite(actual_flat).all())} "
                    f"rel_l2={float(relative_l2):.9g} "
                    f"norm_ratio={float(norm_ratio):.9g} "
                    f"cosine={float(cosine):.9g}"
                )
        samples = time_rotated(
            {
                "winner": fp4_fp8dpdv_x32_split_dk_native,
                "mixed": fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_native,
            },
            args.warmup,
            args.iterations,
        )
        medians = {
            name: statistics.median(values)
            for name, values in samples.items()
        }
        for name, values in samples.items():
            print(
                f"{name}: {medians[name]:.6f} ms "
                f"[{min(values):.6f}, {max(values):.6f}]"
            )
        print(f"mixed/winner={medians['mixed'] / medians['winner']:.6f}")
        return

    if args.compare_prepacked_v:
        bf16_control = copied
        winner_output = fp4_fp8dpdv_x32_split_dk_native()
        inline_output = fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_native()
        inline_repeat_output = (
            fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_native()
        )
        prepacked_output = fp4_mixed_prepacked_v_native()
        reference_output = bf16_control()
        torch.cuda.synchronize()
        for name, reference, winner, inline, inline_repeat, prepacked in zip(
            ("dq", "dk", "dv"),
            reference_output,
            winner_output,
            inline_output,
            inline_repeat_output,
            prepacked_output,
        ):
            reference_flat = reference.float().flatten()
            winner_flat = winner.float().flatten()
            inline_flat = inline.float().flatten()
            inline_repeat_flat = inline_repeat.float().flatten()
            prepacked_flat = prepacked.float().flatten()
            difference = prepacked_flat - inline_flat
            repeat_difference = inline_repeat_flat - inline_flat
            relative_l2 = (
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(inline_flat).clamp_min(1e-12)
            )
            repeat_relative_l2 = (
                torch.linalg.vector_norm(repeat_difference)
                / torch.linalg.vector_norm(inline_flat).clamp_min(1e-12)
            )
            cosine = torch.nn.functional.cosine_similarity(
                reference_flat, prepacked_flat, dim=0
            )
            winner_cosine = torch.nn.functional.cosine_similarity(
                reference_flat, winner_flat, dim=0
            )
            print(
                f"prepacked_{name}: "
                f"finite={bool(torch.isfinite(prepacked_flat).all())} "
                f"vs_inline_max_abs={float(difference.abs().max()):.9g} "
                f"vs_inline_rel_l2={float(relative_l2):.9g} "
                f"inline_repeat_rel_l2={float(repeat_relative_l2):.9g} "
                f"vs_bf16_cosine={float(cosine):.9g} "
                f"winner_vs_bf16_cosine={float(winner_cosine):.9g}"
            )
        samples = time_rotated(
            {
                "bf16_control": bf16_control,
                "fp8_dp_dv_winner": fp4_fp8dpdv_x32_split_dk_native,
                "inline_v": fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_native,
                "prepacked_v": fp4_mixed_prepacked_v_native,
            },
            args.warmup,
            args.iterations,
        )
        prepack_samples = time_rotated(
            {"prepack_v_only": lambda: lowp_ext.prepack_mixed_v(v)},
            args.warmup,
            args.iterations,
        )
        samples.update(prepack_samples)
        medians = {
            name: statistics.median(values)
            for name, values in samples.items()
        }
        for name, values in samples.items():
            print(
                f"{name}: {medians[name]:.6f} ms "
                f"[{min(values):.6f}, {max(values):.6f}]"
            )
        print(
            "prepacked/inline="
            f"{medians['prepacked_v'] / medians['inline_v']:.6f}"
        )
        print(
            "winner/bf16="
            f"{medians['fp8_dp_dv_winner'] / medians['bf16_control']:.6f}"
        )
        print(
            "prepacked/bf16="
            f"{medians['prepacked_v'] / medians['bf16_control']:.6f}"
        )
        print(
            "prepacked/winner="
            f"{medians['prepacked_v'] / medians['fp8_dp_dv_winner']:.6f}"
        )
        return

    if args.compare_winner:
        reference_output = copied()
        winner_output = fp4_fp8dpdv_x32_split_dk_native()
        torch.cuda.synchronize()
        for name, reference, actual in zip(
            ("dq", "dk", "dv"),
            reference_output,
            winner_output,
        ):
            reference_flat = reference.float().flatten()
            actual_flat = actual.float().flatten()
            difference = actual_flat - reference_flat
            relative_l2 = (
                torch.linalg.vector_norm(difference) /
                torch.linalg.vector_norm(reference_flat).clamp_min(1e-12)
            )
            cosine = torch.nn.functional.cosine_similarity(
                reference_flat,
                actual_flat,
                dim=0,
            )
            print(
                f"{name}: finite={bool(torch.isfinite(actual_flat).all())} "
                f"rel_l2={float(relative_l2):.9g} "
                f"cosine={float(cosine):.9g}"
            )
        samples = time_rotated(
            {
                "copied_control": copied,
                "fp4_fp8dpdv_x32_split_dk_native":
                    fp4_fp8dpdv_x32_split_dk_native,
            },
            args.warmup,
            args.iterations,
        )
        medians = {
            name: statistics.median(values)
            for name, values in samples.items()
        }
        for name, values in samples.items():
            print(
                f"{name}: {medians[name]:.6f} ms "
                f"[{min(values):.6f}, {max(values):.6f}]"
            )
        print(
            "winner/copied="
            f"{medians['fp4_fp8dpdv_x32_split_dk_native'] / medians['copied_control']:.6f}"
        )
        return

    if args.compare_fp8_pipeline:
        reference_output = copied()
        fp8_output = fp8_native()
        torch.cuda.synchronize()
        for name, reference, actual in zip(
            ("dq", "dk", "dv"),
            reference_output,
            fp8_output,
        ):
            reference_flat = reference.float().flatten()
            actual_flat = actual.float().flatten()
            difference = actual_flat - reference_flat
            relative_l2 = (
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference_flat).clamp_min(1e-12)
            )
            cosine = torch.nn.functional.cosine_similarity(
                reference_flat,
                actual_flat,
                dim=0,
            )
            print(
                f"fp8_{name}: finite={bool(torch.isfinite(actual_flat).all())} "
                f"rel_l2={float(relative_l2):.9g} "
                f"cosine={float(cosine):.9g}"
            )
        samples = time_rotated(
            {
                "copied_control": copied,
                "fp8_native": fp8_native,
                "fp4_fp8dpdv_x32_split_dk_native":
                    fp4_fp8dpdv_x32_split_dk_native,
            },
            args.warmup,
            args.iterations,
        )
        medians = {
            name: statistics.median(values)
            for name, values in samples.items()
        }
        for name, values in samples.items():
            print(
                f"{name}: {medians[name]:.6f} ms "
                f"[{min(values):.6f}, {max(values):.6f}]"
            )
        print(
            "fp8/copied="
            f"{medians['fp8_native'] / medians['copied_control']:.6f}"
        )
        print(
            "fp8/fp4_fp8dpdv="
            f"{medians['fp8_native'] / medians['fp4_fp8dpdv_x32_split_dk_native']:.6f}"
        )
        return

    if args.profile_mode is not None:
        profile_callables = {
            "original": original,
            "copied": copied,
            "fp8": fp8_native,
            "fp4": fp4_native,
            "fp4_fp8pv": fp4_fp8pv_native,
            "fp4_fp8pv_x32": fp4_fp8pv_x32_native,
            "fp4_fp8pv_x32_reuse_p": fp4_fp8pv_x32_reuse_p_native,
            "fp4_fp8pv_x32_split_dk": fp4_fp8pv_x32_split_dk_native,
            "fp4_fp8dpdv_x32_split_dk":
                fp4_fp8dpdv_x32_split_dk_native,
            "fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk":
                fp4_mixed_fp8_mxfp4dp_fp8dv_x32_split_dk_native,
            "fp4_mixed_prepacked_v": fp4_mixed_prepacked_v_native,
            "fp4_fp8dp_mxfp4dv_x32_split_dk":
                fp4_fp8dp_mxfp4dv_x32_split_dk_native,
            "fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk":
                fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native,
            "fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk":
                fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native,
            "fp4_mxfp4dpdv_forward_log_split_q_x32_split_dk":
                fp4_mxfp4dpdv_forward_log_split_q_x32_split_dk_native,
            "fp4_pure_mxfp4": fp4_pure_mxfp4_native,
            "fp4_full": fp4_full,
        }
        for _ in range(args.profile_launches):
            profile_callables[args.profile_mode]()
        torch.cuda.synchronize()
        return

    if args.compare_pure:
        reference_output = tuple(value.clone() for value in copied())
        pure_output = tuple(value.clone() for value in fp4_pure_mxfp4_native())
        torch.cuda.synchronize()
        for label, reference, candidate in zip(
            ("dq", "dk", "dv"), reference_output, pure_output
        ):
            reference_float = reference.float()
            candidate_float = candidate.float()
            difference = candidate_float - reference_float
            relative_l2 = float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    reference_float.flatten(), candidate_float.flatten(), dim=0
                )
            )
            print(
                f"pure_{label}: finite={bool(torch.isfinite(candidate).all())} "
                f"rel_l2={relative_l2:.9g} cosine={cosine:.9g}"
            )
        samples = time_rotated(
            {"bf16": copied, "pure": fp4_pure_mxfp4_native},
            args.warmup,
            args.iterations,
        )
        medians = {
            name: statistics.median(values) for name, values in samples.items()
        }
        for name, values in samples.items():
            print(
                f"{name}: {medians[name]:.6f} ms "
                f"[{min(values):.6f}, {max(values):.6f}]"
            )
        print(f"pure/bf16={medians['pure'] / medians['bf16']:.6f}")
        return

    if args.compare_dp_formats:
        reference_output = original()
        hybrid_output = (
            fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native()
        )
        mxfp4_output = (
            fp4_mxfp4dpdv_forward_log_split_q_x32_split_dk_native()
        )
        pure_mxfp4_output = fp4_pure_mxfp4_native()
        torch.cuda.synchronize()
        for route_name, route_output in (
            ("fp8_dp", hybrid_output),
            ("mxfp4_dp", mxfp4_output),
            ("pure_mxfp4", pure_mxfp4_output),
        ):
            for label, reference, candidate in zip(
                ("dq", "dk", "dv"),
                reference_output,
                route_output,
            ):
                reference_float = reference.float()
                candidate_float = candidate.float()
                diff = candidate_float - reference_float
                rel_l2 = float(
                    torch.linalg.vector_norm(diff)
                    / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)
                )
                cosine = float(
                    torch.nn.functional.cosine_similarity(
                        reference_float.flatten(),
                        candidate_float.flatten(),
                        dim=0,
                    )
                )
                print(
                    f"{route_name}_{label}: "
                    f"finite={bool(torch.isfinite(candidate).all())} "
                    f"rel_l2={rel_l2:.9g} cosine={cosine:.9g}"
                )
                if (
                    args.diagnostics
                    and route_name == "pure_mxfp4"
                    and label in ("dq", "dk")
                ):
                    norm_ratio = float(
                        torch.linalg.vector_norm(candidate_float)
                        / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)
                    )
                    print(
                        f"{route_name}_{label}_norm_ratio={norm_ratio:.9g}"
                    )
                    for col_start, col_end in ((0, 64), (64, 128), (128, 192)):
                        reference_chunk = reference_float[..., col_start:col_end]
                        candidate_chunk = candidate_float[..., col_start:col_end]
                        chunk_diff = candidate_chunk - reference_chunk
                        chunk_rel = float(
                            torch.linalg.vector_norm(chunk_diff)
                            / torch.linalg.vector_norm(reference_chunk).clamp_min(1e-12)
                        )
                        chunk_cosine = float(
                            torch.nn.functional.cosine_similarity(
                                reference_chunk.flatten(),
                                candidate_chunk.flatten(),
                                dim=0,
                            )
                        )
                        print(
                            f"{route_name}_{label}_cols_{col_start}_{col_end} "
                            f"rel_l2={chunk_rel:.9g} cosine={chunk_cosine:.9g}"
                        )
        format_samples = time_rotated(
            {
                "fp8_dp":
                    fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native,
                "mxfp4_dp":
                    fp4_mxfp4dpdv_forward_log_split_q_x32_split_dk_native,
                "pure_mxfp4": fp4_pure_mxfp4_native,
            },
            args.warmup,
            args.iterations,
        )
        format_medians = {
            name: statistics.median(values)
            for name, values in format_samples.items()
        }
        for name, values in format_samples.items():
            print(
                f"{name}: {format_medians[name]:.6f} ms "
                f"[{min(values):.6f}, {max(values):.6f}]"
            )
        print(
            "mxfp4_dp/fp8_dp="
            f"{format_medians['mxfp4_dp'] / format_medians['fp8_dp']:.6f}"
        )
        print(
            "pure_mxfp4/mxfp4_dp="
            f"{format_medians['pure_mxfp4'] / format_medians['mxfp4_dp']:.6f}"
        )
        return

    if args.compare_split_q:
        split_samples = time_rotated(
            {
                "forward_log":
                    fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native,
                "split_q":
                    fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native,
            },
            args.warmup,
            args.iterations,
        )
        split_medians = {
            name: statistics.median(values)
            for name, values in split_samples.items()
        }
        for name, values in split_samples.items():
            print(
                f"{name}: {split_medians[name]:.6f} ms "
                f"[{min(values):.6f}, {max(values):.6f}]"
            )
        print(
            "split_q/forward_log="
            f"{split_medians['split_q'] / split_medians['forward_log']:.6f}"
        )
        return

    original_out = original()
    copied_out = copied()
    fp8_out = fp8_native()
    fp4_out = fp4_native()
    fp4_fp8pv_out = fp4_fp8pv_native()
    fp4_fp8pv_x32_out = fp4_fp8pv_x32_native()
    fp4_fp8pv_x32_reuse_p_out = fp4_fp8pv_x32_reuse_p_native()
    fp4_fp8pv_x32_split_dk_out = fp4_fp8pv_x32_split_dk_native()
    fp4_fp8dpdv_x32_split_dk_out = fp4_fp8dpdv_x32_split_dk_native()
    fp4_fp8dp_mxfp4dv_x32_split_dk_out = (
        fp4_fp8dp_mxfp4dv_x32_split_dk_native()
    )
    fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_out = (
        fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native()
    )
    fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_out = (
        fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native()
    )
    torch.cuda.synchronize()
    if args.stress_launches > 0:
        stress_routes = (
            ("fp4_fp8pv", fp4_fp8pv_native, fp4_fp8pv_out),
            ("fp4_fp8pv_x32", fp4_fp8pv_x32_native, fp4_fp8pv_x32_out),
            (
                "fp4_fp8pv_x32_reuse_p",
                fp4_fp8pv_x32_reuse_p_native,
                fp4_fp8pv_x32_reuse_p_out,
            ),
            (
                "fp4_fp8pv_x32_split_dk",
                fp4_fp8pv_x32_split_dk_native,
                fp4_fp8pv_x32_split_dk_out,
            ),
            (
                "fp4_fp8dpdv_x32_split_dk",
                fp4_fp8dpdv_x32_split_dk_native,
                fp4_fp8dpdv_x32_split_dk_out,
            ),
            (
                "fp4_fp8dp_mxfp4dv_x32_split_dk",
                fp4_fp8dp_mxfp4dv_x32_split_dk_native,
                fp4_fp8dp_mxfp4dv_x32_split_dk_out,
            ),
            (
                "fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk",
                fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native,
                fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_out,
            ),
            (
                "fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk",
                fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native,
                fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_out,
            ),
        )
        for route_name, route, initial in stress_routes:
            stable_reference = tuple(value.clone() for value in initial)
            stress_max_abs_diff = 0.0
            for launch_idx in range(args.stress_launches):
                repeated = route()
                torch.cuda.synchronize()
                finite = all(
                    bool(torch.isfinite(value).all()) for value in repeated
                )
                max_abs_diff = max(
                    float((value.float() - reference.float()).abs().max())
                    for value, reference in zip(repeated, stable_reference)
                )
                stress_max_abs_diff = max(stress_max_abs_diff, max_abs_diff)
                if not finite or max_abs_diff > 0.125:
                    raise RuntimeError(
                        f"{route_name} instability at launch {launch_idx}: "
                        f"finite={finite} max_abs_diff={max_abs_diff}"
                    )
            print(
                f"{route_name}_stress: launches={args.stress_launches} "
                f"finite=True max_abs_diff={stress_max_abs_diff:.9g}"
            )
    labels = ("dq", "dk", "dv")
    for label, reference, candidate in zip(labels, original_out, copied_out):
        diff = (candidate.float() - reference.float()).abs()
        rel_l2 = float(
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
        )
        print(
            f"{label}: dtype={candidate.dtype} finite={bool(torch.isfinite(candidate).all())} "
            f"max_abs={float(diff.max()):.9g} rel_l2={rel_l2:.9g}"
        )
    for label, reference, candidate in zip(labels, original_out, fp8_out):
        diff = (candidate.float() - reference.float()).abs()
        reference_float = reference.float()
        candidate_float = candidate.float()
        rel_l2 = float(
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(
                reference_float.flatten(),
                candidate_float.flatten(),
                dim=0,
            )
        )
        print(
            f"fp8_{label}: dtype={candidate.dtype} "
            f"finite={bool(torch.isfinite(candidate).all())} "
            f"max_abs={float(diff.max()):.9g} rel_l2={rel_l2:.9g} "
            f"cosine={cosine:.9g} "
            f"norm_ratio={float(torch.linalg.vector_norm(candidate_float) / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)):.9g}"
        )
        if args.diagnostics and label in ("dq", "dk"):
            for depth_start, depth_end in ((0, 96), (96, 192)):
                ref_part = reference_float[..., depth_start:depth_end].flatten()
                cand_part = candidate_float[..., depth_start:depth_end].flatten()
                part_cosine = torch.nn.functional.cosine_similarity(
                    ref_part, cand_part, dim=0
                )
                print(
                    f"fp8_{label}_depth_{depth_start}_{depth_end}_cosine="
                    f"{float(part_cosine):.9g}"
                )
            ref_pairs = reference_float.reshape(
                1, args.seqlen // 256, 2, 128, args.heads, -1
            )
            cand_pairs = candidate_float.reshape_as(ref_pairs)
            for half in (0, 1):
                same = torch.nn.functional.cosine_similarity(
                    ref_pairs[:, :, half].flatten(),
                    cand_pairs[:, :, half].flatten(),
                    dim=0,
                )
                swapped = torch.nn.functional.cosine_similarity(
                    ref_pairs[:, :, 1 - half].flatten(),
                    cand_pairs[:, :, half].flatten(),
                    dim=0,
                )
                print(
                    f"fp8_{label}_keypair_half_{half}_same_cosine={float(same):.9g} "
                    f"swapped_cosine={float(swapped):.9g}"
                )
            if args.dout_tile >= 0 and label == "dq":
                ref_tile_norms = torch.linalg.vector_norm(
                    reference_float.reshape(1, args.seqlen // 128, 128, args.heads, -1),
                    dim=(0, 2, 3, 4),
                )
                cand_tile_norms = torch.linalg.vector_norm(
                    candidate_float.reshape(1, args.seqlen // 128, 128, args.heads, -1),
                    dim=(0, 2, 3, 4),
                )
                ref_top = torch.topk(ref_tile_norms, k=4)
                cand_top = torch.topk(cand_tile_norms, k=4)
                print(
                    "fp8_dq_reference_top_tiles="
                    f"{list(zip(ref_top.indices.tolist(), ref_top.values.tolist()))}"
                )
                print(
                    "fp8_dq_candidate_top_tiles="
                    f"{list(zip(cand_top.indices.tolist(), cand_top.values.tolist()))}"
                )
    for label, reference, candidate in zip(labels, original_out, fp4_out):
        diff = (candidate.float() - reference.float()).abs()
        reference_float = reference.float()
        candidate_float = candidate.float()
        rel_l2 = float(
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(
                reference_float.flatten(),
                candidate_float.flatten(),
                dim=0,
            )
        )
        print(
            f"fp4_{label}: dtype={candidate.dtype} "
            f"finite={bool(torch.isfinite(candidate).all())} "
            f"max_abs={float(diff.max()):.9g} rel_l2={rel_l2:.9g} "
            f"cosine={cosine:.9g} "
            f"norm_ratio={float(torch.linalg.vector_norm(candidate_float) / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)):.9g}"
        )
        if args.diagnostics and label in ("dq", "dk"):
            for depth_start, depth_end in ((0, 96), (96, 192)):
                ref_part = reference_float[..., depth_start:depth_end].flatten()
                cand_part = candidate_float[..., depth_start:depth_end].flatten()
                part_cosine = torch.nn.functional.cosine_similarity(
                    ref_part, cand_part, dim=0
                )
                print(
                    f"fp4_{label}_depth_{depth_start}_{depth_end}_cosine="
                    f"{float(part_cosine):.9g}"
                )
        if args.diagnostics and args.dout_tile >= 0 and label == "dq":
            tile_start = args.dout_tile * 128
            ref_rows = reference_float[:, tile_start : tile_start + 128].reshape(128, -1)
            cand_rows = candidate_float[:, tile_start : tile_start + 128].reshape(128, -1)
            ref_rows = torch.nn.functional.normalize(ref_rows, dim=1)
            cand_rows = torch.nn.functional.normalize(cand_rows, dim=1)
            row_cosines = cand_rows @ ref_rows.T
            best_values, best_rows = row_cosines.max(dim=1)
            print("fp4_dq_best_reference_row_for_candidate")
            print(best_rows.cpu().tolist())
            print("fp4_dq_best_row_cosines")
            print([round(float(value), 5) for value in best_values.cpu()])

    for label, reference, candidate in zip(
        labels,
        original_out,
        fp4_fp8pv_out,
    ):
        diff = (candidate.float() - reference.float()).abs()
        reference_float = reference.float()
        candidate_float = candidate.float()
        rel_l2 = float(
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(
                reference_float.flatten(),
                candidate_float.flatten(),
                dim=0,
            )
        )
        print(
            f"fp4_fp8pv_{label}: dtype={candidate.dtype} "
            f"finite={bool(torch.isfinite(candidate).all())} "
            f"max_abs={float(diff.max()):.9g} rel_l2={rel_l2:.9g} "
            f"cosine={cosine:.9g} "
            f"norm_ratio={float(torch.linalg.vector_norm(candidate_float) / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)):.9g}"
        )

    for route_name, route_output in (
        ("fp4_fp8pv_x32", fp4_fp8pv_x32_out),
        ("fp4_fp8pv_x32_reuse_p", fp4_fp8pv_x32_reuse_p_out),
        ("fp4_fp8pv_x32_split_dk", fp4_fp8pv_x32_split_dk_out),
        ("fp4_fp8dpdv_x32_split_dk", fp4_fp8dpdv_x32_split_dk_out),
        (
            "fp4_fp8dp_mxfp4dv_x32_split_dk",
            fp4_fp8dp_mxfp4dv_x32_split_dk_out,
        ),
        (
            "fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk",
            fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_out,
        ),
        (
            "fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk",
            fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_out,
        ),
    ):
        for label, reference, candidate in zip(
            labels,
            original_out,
            route_output,
        ):
            diff = (candidate.float() - reference.float()).abs()
            reference_float = reference.float()
            candidate_float = candidate.float()
            rel_l2 = float(
                torch.linalg.vector_norm(diff)
                / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    reference_float.flatten(),
                    candidate_float.flatten(),
                    dim=0,
                )
            )
            print(
                f"{route_name}_{label}: dtype={candidate.dtype} "
                f"finite={bool(torch.isfinite(candidate).all())} "
                f"max_abs={float(diff.max()):.9g} rel_l2={rel_l2:.9g} "
                f"cosine={cosine:.9g} "
                f"norm_ratio={float(torch.linalg.vector_norm(candidate_float) / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)):.9g}"
            )
            if args.diagnostics and label == "dv":
                for depth_start, depth_end in ((0, 64), (64, 128)):
                    ref_part = reference_float[..., depth_start:depth_end].flatten()
                    cand_part = candidate_float[..., depth_start:depth_end].flatten()
                    part_norm = torch.linalg.vector_norm(ref_part).clamp_min(1e-12)
                    part_cosine = torch.nn.functional.cosine_similarity(
                        ref_part, cand_part, dim=0
                    )
                    print(
                        f"{route_name}_{label}_depth_{depth_start}_{depth_end}_"
                        f"cosine={float(part_cosine):.9g} "
                        f"norm_ratio={float(torch.linalg.vector_norm(cand_part) / part_norm):.9g}"
                    )

    if args.diagnostics and args.dout_tile >= 0:
        query_start = args.dout_tile * 128
        query_end = query_start + 128
        q_tile = q[:, query_start:query_end].float()
        dout_tile = dout[:, query_start:query_end].float()
        scores = torch.einsum("bqhd,bkhd->bhqk", q_tile, k.float()) * scale
        query_indices = torch.arange(
            query_start, query_end, device=device
        ).view(1, 1, 128, 1)
        key_indices = torch.arange(args.seqlen, device=device).view(1, 1, 1, -1)
        scores.masked_fill_(key_indices > query_indices, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        dp = torch.einsum("bqhv,bkhv->bhqk", dout_tile, v.float())
        delta = (
            dout_tile * out[:, query_start:query_end].float()
        ).sum(dim=-1).permute(0, 2, 1)
        ds = probabilities * (dp - delta.unsqueeze(-1))
        print(
            "software_ds_scaled_amax="
            f"{float((ds * args.ds_quant_scale).abs().amax()):.9g}"
        )
        ds_fp8 = (ds * args.ds_quant_scale).to(torch.float8_e4m3fn).float()
        print("software_ds_fp8_key_by_query_top_left")
        print(ds_fp8[0, 0, :8, :4].transpose(0, 1).cpu())
        software_fp8_dq = torch.einsum(
            "bhqk,bkhd->bqhd", ds_fp8, k_fp8.float()
        ) * (scale / (args.ds_quant_scale * args.k_quant_scale))
        actual_fp8_dq = fp8_out[0][
            :, query_start:query_end
        ].float()
        dq_dense_diff = actual_fp8_dq - software_fp8_dq
        dq_dense_rel = (
            torch.linalg.vector_norm(dq_dense_diff) /
            torch.linalg.vector_norm(software_fp8_dq).clamp_min(1e-12)
        )
        dq_dense_cosine = torch.nn.functional.cosine_similarity(
            software_fp8_dq.flatten(), actual_fp8_dq.flatten(), dim=0
        )
        print(
            "software_fp8_dq_selected_tile "
            f"rel_l2={float(dq_dense_rel):.9g} "
            f"cosine={float(dq_dense_cosine):.9g}"
        )
        key_blocks = torch.arange(
            args.seqlen, device=device
        ).div(128, rounding_mode="floor")
        even_keys = (key_blocks & 1) == 0
        odd_keys = ~even_keys
        software_fp8_dq_even = torch.einsum(
            "bhqk,bkhd->bqhd",
            ds_fp8[..., even_keys],
            k_fp8[:, even_keys].float(),
        ) * (scale / (args.ds_quant_scale * args.k_quant_scale))
        software_fp8_dq_odd = torch.einsum(
            "bhqk,bkhd->bqhd",
            ds_fp8[..., odd_keys],
            k_fp8[:, odd_keys].float(),
        ) * (scale / (args.ds_quant_scale * args.k_quant_scale))
        software_fp8_dq_local = torch.cat(
            (
                software_fp8_dq_even[:, :64],
                software_fp8_dq_odd[:, 64:],
            ),
            dim=1,
        )
        software_fp8_dq_peer = torch.cat(
            (
                software_fp8_dq_odd[:, :64],
                software_fp8_dq_even[:, 64:],
            ),
            dim=1,
        )
        for contribution_name, contribution in (
            ("local", software_fp8_dq_local),
            ("peer", software_fp8_dq_peer),
            ("twice_local", 2.0 * software_fp8_dq_local),
        ):
            contribution_cosine = torch.nn.functional.cosine_similarity(
                contribution.flatten(), actual_fp8_dq.flatten(), dim=0
            )
            contribution_norm = (
                torch.linalg.vector_norm(actual_fp8_dq) /
                torch.linalg.vector_norm(contribution).clamp_min(1e-12)
            )
            print(
                f"software_fp8_dq_{contribution_name} "
                f"cosine={float(contribution_cosine):.9g} "
                f"candidate_over_model_norm={float(contribution_norm):.9g}"
            )
        for query_half in range(2):
            query_slice = slice(query_half * 64, (query_half + 1) * 64)
            candidate_half = actual_fp8_dq[:, query_slice]
            for contribution_name, contribution in (
                ("total", software_fp8_dq),
                ("local", software_fp8_dq_local),
                ("peer", software_fp8_dq_peer),
            ):
                contribution_half = contribution[:, query_slice]
                contribution_cosine = torch.nn.functional.cosine_similarity(
                    contribution_half.flatten(), candidate_half.flatten(), dim=0
                )
                contribution_norm = (
                    torch.linalg.vector_norm(candidate_half) /
                    torch.linalg.vector_norm(contribution_half).clamp_min(1e-12)
                )
                print(
                    f"software_fp8_dq_qhalf{query_half}_{contribution_name} "
                    f"cosine={float(contribution_cosine):.9g} "
                    f"candidate_over_model_norm={float(contribution_norm):.9g}"
                )
        software_rows = torch.nn.functional.normalize(
            software_fp8_dq.reshape(128, -1), dim=1
        )
        actual_rows = torch.nn.functional.normalize(
            actual_fp8_dq.reshape(128, -1), dim=1
        )
        software_row_cosines = actual_rows @ software_rows.T
        software_best_values, software_best_rows = software_row_cosines.max(dim=1)
        print(
            "software_fp8_dq_best_rows="
            f"{software_best_rows.cpu().tolist()}"
        )
        print(
            "software_fp8_dq_best_row_cosine_min_mean="
            f"{float(software_best_values.min()):.9g},"
            f"{float(software_best_values.mean()):.9g}"
        )
        q_fp8_tile = q_fp8[..., query_start:query_end].permute(0, 3, 1, 2).float()
        software_fp8_dk = torch.einsum(
            "bhqk,bqhd->bkhd", ds_fp8, q_fp8_tile
        ) * (scale / (args.ds_quant_scale * args.q_quant_scale))
        software_bf16_dk = torch.einsum(
            "bhqk,bqhd->bkhd", ds, q_tile
        ) * scale
        dense_end = max(0, query_start - 256)
        for name, expected, actual in (
            ("software_bf16_dk", software_bf16_dk, original_out[1].float()),
            ("software_fp8_dk", software_fp8_dk, fp8_out[1].float()),
        ):
            expected_dense = expected[:, :dense_end].flatten()
            actual_dense = actual[:, :dense_end].flatten()
            dense_diff = actual_dense - expected_dense
            dense_rel = torch.linalg.vector_norm(dense_diff) / torch.linalg.vector_norm(
                expected_dense
            ).clamp_min(1e-12)
            dense_cosine = torch.nn.functional.cosine_similarity(
                expected_dense, actual_dense, dim=0
            )
            print(
                f"{name}_dense_prefix={dense_end} rel_l2={float(dense_rel):.9g} "
                f"cosine={float(dense_cosine):.9g}"
            )
    samples = time_rotated(
        {
            "original_v382": original,
            "copied_control": copied,
            "fp8_native": fp8_native,
            "fp4_native": fp4_native,
            "fp4_fp8pv_native": fp4_fp8pv_native,
            "fp4_fp8pv_x32_native": fp4_fp8pv_x32_native,
            "fp4_fp8pv_x32_reuse_p_native":
                fp4_fp8pv_x32_reuse_p_native,
            "fp4_fp8pv_x32_split_dk_native":
                fp4_fp8pv_x32_split_dk_native,
            "fp4_fp8dpdv_x32_split_dk_native":
                fp4_fp8dpdv_x32_split_dk_native,
            "fp4_fp8dp_mxfp4dv_x32_split_dk_native":
                fp4_fp8dp_mxfp4dv_x32_split_dk_native,
            "fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native":
                fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native,
            "fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native":
                fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native,
        },
        args.warmup,
        args.iterations,
    )
    medians = {name: statistics.median(values) for name, values in samples.items()}
    for name, values in samples.items():
        print(
            f"{name}: {medians[name]:.6f} ms "
            f"[{min(values):.6f}, {max(values):.6f}]"
        )
    print(f"copied/original={medians['copied_control'] / medians['original_v382']:.6f}")
    print(f"fp8/original={medians['fp8_native'] / medians['original_v382']:.6f}")
    print(f"fp4/original={medians['fp4_native'] / medians['original_v382']:.6f}")
    print(
        "fp4_fp8pv/original="
        f"{medians['fp4_fp8pv_native'] / medians['original_v382']:.6f}"
    )
    print(
        "fp4_fp8pv/fp4="
        f"{medians['fp4_fp8pv_native'] / medians['fp4_native']:.6f}"
    )
    print(
        "fp4_fp8pv_x32/fp4="
        f"{medians['fp4_fp8pv_x32_native'] / medians['fp4_native']:.6f}"
    )
    print(
        "fp4_fp8pv_x32/fp4_fp8pv="
        f"{medians['fp4_fp8pv_x32_native'] / medians['fp4_fp8pv_native']:.6f}"
    )
    print(
        "fp4_fp8pv_x32_reuse_p/fp4="
        f"{medians['fp4_fp8pv_x32_reuse_p_native'] / medians['fp4_native']:.6f}"
    )
    print(
        "fp4_fp8pv_x32_reuse_p/fp4_fp8pv_x32="
        f"{medians['fp4_fp8pv_x32_reuse_p_native'] / medians['fp4_fp8pv_x32_native']:.6f}"
    )
    print(
        "fp4_fp8pv_x32_split_dk/fp4="
        f"{medians['fp4_fp8pv_x32_split_dk_native'] / medians['fp4_native']:.6f}"
    )
    print(
        "fp4_fp8pv_x32_split_dk/reuse_p="
        f"{medians['fp4_fp8pv_x32_split_dk_native'] / medians['fp4_fp8pv_x32_reuse_p_native']:.6f}"
    )
    print(
        "fp4_fp8dpdv_x32_split_dk/control="
        f"{medians['fp4_fp8dpdv_x32_split_dk_native'] / medians['fp4_fp8pv_x32_split_dk_native']:.6f}"
    )
    print(
        "fp4_fp8dp_mxfp4dv_x32_split_dk/fp8dpdv="
        f"{medians['fp4_fp8dp_mxfp4dv_x32_split_dk_native'] / medians['fp4_fp8dpdv_x32_split_dk_native']:.6f}"
    )
    print(
        "fp4_fp8dp_mxfp4dv_forward_log/baseline_mxfp4="
        f"{medians['fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native'] / medians['fp4_fp8dp_mxfp4dv_x32_split_dk_native']:.6f}"
    )
    print(
        "fp4_fp8dp_mxfp4dv_forward_log_split_q/forward_log="
        f"{medians['fp4_fp8dp_mxfp4dv_forward_log_split_q_x32_split_dk_native'] / medians['fp4_fp8dp_mxfp4dv_forward_log_x32_split_dk_native']:.6f}"
    )
    if args.end_to_end:
        pipeline_samples = time_rotated(
            {
                "fp4_prepack_fused": fp4_quantize,
                "fp4_full": fp4_full,
            },
            args.warmup,
            args.iterations,
        )
        pipeline_medians = {
            name: statistics.median(values)
            for name, values in pipeline_samples.items()
        }
        for name, values in pipeline_samples.items():
            print(
                f"{name}: {pipeline_medians[name]:.6f} ms "
                f"[{min(values):.6f}, {max(values):.6f}]"
            )
        print(
            "fp4_full/copied="
            f"{pipeline_medians['fp4_full'] / medians['copied_control']:.6f}"
        )


if __name__ == "__main__":
    main()
