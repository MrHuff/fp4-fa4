#!/usr/bin/env python3
"""Benchmark the final D128 causal-training boundaries used by the report.

The script keeps three timing scopes separate:

* the prebound attention-core backward (BF16 CuTe FA4 versus native replay),
* the projection-inclusive attention-sublayer backward, and
* the projection-inclusive attention-sublayer forward plus backward.

The low-precision route is the B1/S4096/Hq32/Hkv8/D128 FP8-PV forward with
saved NVFP4 Q/K and the native E5M2-dO replay backward.  Every selected binary
is authenticated before import.  In particular, the projection extension must
be selected through ``TK_FA4_LOWP_BWD_EXTENSION_SOURCE`` before this script is
started; importing a worktree-default binary by accident is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SHAPE = {
    "batch": 1,
    "sequence": 4096,
    "q_heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
}
LOSS_SCALE = 2.0**16
# Match the intended training regime: the synthetic upstream gradient has
# unit standard deviation after the static loss scale is applied.
UPSTREAM_GRADIENT_STD = 1.0 / LOSS_SCALE


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("expected a 64-character SHA256")
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-extension", type=Path, required=True)
    parser.add_argument("--projection-sha256", type=_sha256, required=True)
    parser.add_argument(
        "--projection-bytes", type=_positive_int, required=True
    )
    parser.add_argument("--forward-extension", type=Path, required=True)
    parser.add_argument("--forward-module", required=True)
    parser.add_argument("--forward-sha256", type=_sha256, required=True)
    parser.add_argument("--forward-bytes", type=_positive_int, required=True)
    parser.add_argument("--backward-extension", type=Path, required=True)
    parser.add_argument("--backward-module", required=True)
    parser.add_argument("--backward-sha256", type=_sha256, required=True)
    parser.add_argument("--backward-bytes", type=_positive_int, required=True)
    parser.add_argument("--warmups", type=_positive_int, default=20)
    parser.add_argument("--samples", type=_positive_int, default=101)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _file_identity(
    path: Path,
    expected_sha256: str,
    expected_bytes: int,
    *,
    label: str,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    identity = {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "bytes": resolved.stat().st_size,
    }
    if (
        identity["sha256"] != expected_sha256
        or identity["bytes"] != expected_bytes
    ):
        raise RuntimeError(
            f"{label} identity mismatch: observed {identity}, expected "
            f"SHA256={expected_sha256} bytes={expected_bytes}"
        )
    return identity


def _git_value(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _gpu_inventory() -> list[dict[str, str]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    result = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) == 4:
            result.append(
                dict(zip(("index", "uuid", "pci_bus_id", "name"), fields))
            )
    return result


def _quantiles(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "samples": len(ordered),
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p10_ms": ordered[int(0.10 * (len(ordered) - 1))],
        "p90_ms": ordered[int(0.90 * (len(ordered) - 1))],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "raw_ms": values,
    }


def _rotated_timing(
    callbacks: dict[str, Callable[[], None]],
    *,
    warmups: int,
    samples: int,
    pre_round: Callable[[], None] | None = None,
) -> dict[str, dict[str, Any]]:
    names = list(callbacks)
    events = {
        name: (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for name in names
    }
    elapsed = {name: [] for name in names}
    for round_index in range(warmups + samples):
        if pre_round is not None:
            pre_round()
        torch.cuda.synchronize()
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            start, end = events[name]
            start.record()
            callbacks[name]()
            end.record()
            end.synchronize()
            if round_index >= warmups:
                elapsed[name].append(float(start.elapsed_time(end)))
    return {name: _quantiles(values) for name, values in elapsed.items()}


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_f = left.detach().float().reshape(-1)
    right_f = right.detach().float().reshape(-1)
    denominator = left_f.norm() * right_f.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float(torch.dot(left_f, right_f) / denominator)


def _comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_f = left.detach().float()
    right_f = right.detach().float()
    left_norm = left_f.norm().clamp_min(1.0e-20)
    return {
        "cosine": _cosine(left_f, right_f),
        "relative_l2": float((right_f - left_f).norm() / left_norm),
        "norm_ratio": float(right_f.norm() / left_norm),
        "left_finite": bool(torch.isfinite(left_f).all()),
        "right_finite": bool(torch.isfinite(right_f).all()),
    }


def _zero_module_gradients(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.grad is not None:
            parameter.grad.zero_()


def _require_released(lowp: Any) -> None:
    state = lowp._forward_workspace.publication_state
    if state.in_flight_generation is not None:
        raise RuntimeError("low-precision workspace remained in flight")


def main() -> None:
    args = _parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("expose exactly one GPU to this benchmark")
    torch.cuda.set_device(0)

    projection_identity = _file_identity(
        args.projection_extension,
        args.projection_sha256,
        args.projection_bytes,
        label="projection extension",
    )
    selected_projection = os.environ.get(
        "TK_FA4_LOWP_BWD_EXTENSION_SOURCE"
    )
    if selected_projection is None:
        raise RuntimeError(
            "set TK_FA4_LOWP_BWD_EXTENSION_SOURCE to the authenticated "
            "projection extension before starting Python"
        )
    if Path(selected_projection).expanduser().resolve() != Path(
        projection_identity["path"]
    ):
        raise RuntimeError(
            "TK_FA4_LOWP_BWD_EXTENSION_SOURCE does not select the declared "
            "projection extension"
        )
    forward_identity = _file_identity(
        args.forward_extension,
        args.forward_sha256,
        args.forward_bytes,
        label="forward extension",
    )
    backward_identity = _file_identity(
        args.backward_extension,
        args.backward_sha256,
        args.backward_bytes,
        label="backward extension",
    )

    # These imports intentionally occur only after the projection path and all
    # three file identities have been checked.
    from tk_fa4.lowp_fa4_bwd import benchmark_causal_backward_matrix as matrix
    from tk_fa4.lowp_fa4_bwd import benchmark_llama12b_e2e as benchmark
    import tk_fa4.interface as interface

    loaded_projection = Path(interface._C_b300_lowp_bwd.__file__).resolve()
    if loaded_projection != Path(projection_identity["path"]):
        raise RuntimeError(
            f"loaded projection {loaded_projection} does not match declared "
            f"projection {projection_identity['path']}"
        )

    config = benchmark.config_from_model_preset(
        "llama3.1-8b", batch=1, sequence=4096, layers=1
    )
    rope = benchmark._make_llama3_rope(config)
    forward, forward_topology = benchmark._load_forward(
        Path(forward_identity["path"]), args.forward_module, config
    )
    backward = benchmark._load_extension(
        Path(backward_identity["path"]), args.backward_module
    )
    loaded_backward = benchmark._require_authenticated_native_tk_extension(
        backward
    )
    if (
        loaded_backward["sha256"] != backward_identity["sha256"]
        or loaded_backward["bytes"] != backward_identity["bytes"]
    ):
        raise RuntimeError("loaded backward extension identity changed")

    runtime = benchmark.LowpAttentionRuntime(
        config,
        rope,
        forward_extension=forward,
        forward_topology=forward_topology,
        loss_scale=LOSS_SCALE,
        gradient_global_scale=2.0**-8,
        projection_dgrad="nvfp4",
        qkv_projection_format="e4m3",
        output_projection_format="e4m3",
        experimental_native_nvfp4_projection_out=False,
        experimental_fused_attention_rmsnorm_nvfp4=False,
        backward_exp2_degree=1,
        backward_exp2_period=0,
        backward_fp8_ds_lift=16,
        backward_reuse_quantized_p=False,
        backward_forward_mx_probability_replay=False,
        backward_forward_mx_probability_scale_handoff=False,
        backward_match_forward_operands=False,
        per_block_qk_scales=True,
        experimental_split_v_backward=False,
        experimental_output_shared_split_v=False,
        experimental_d128_mxfp4_v_backward=False,
        backward_probability_correction=1.0,
        q_quant_scale=2.25,
        k_quant_scale=2.0,
        projection_weight_scale_2d=True,
        v_mxfp4_scale_2d=False,
        adaptive_qk_weight_scales=False,
        native_tk_d128_backward_extension=backward,
        native_tk_d128_native_score_backward=True,
        native_tk_d128_v509_e5m2_dout_backward=True,
    )

    torch.manual_seed(args.seed)
    lowp = benchmark.LowpAttention(config, runtime)
    torch.manual_seed(args.seed)
    bf16 = benchmark.PackedQKVBF16Attention(config, rope)
    with torch.no_grad():
        bf16.weights.qkv.copy_(
            torch.cat(
                (
                    lowp.weights.q,
                    lowp.weights.k,
                    lowp.weights.v,
                ),
                dim=0,
            )
        )
        bf16.weights.o.copy_(lowp.weights.o)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + 1)
    x_base = (
        torch.randn(
            (config.batch, config.sequence, config.hidden),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(0.02)
        .bfloat16()
    )
    grad_base = (
        torch.randn(
            x_base.shape,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(UPSTREAM_GRADIENT_STD)
        .bfloat16()
    )

    # One paired numerical check also materializes persistent gradient buffers
    # before they are zeroed outside every timed region.
    x_bf16 = x_base.detach().clone().requires_grad_(True)
    x_lowp = x_base.detach().clone().requires_grad_(True)
    y_bf16 = bf16(x_bf16)
    y_lowp = lowp(x_lowp)
    y_bf16.backward(grad_base)
    y_lowp.backward(grad_base)
    torch.cuda.synchronize()
    _require_released(lowp)
    if bf16.weights.qkv.grad is None:
        raise RuntimeError("packed BF16 QKV weight gradient was not materialized")
    bf16_q_grad, bf16_k_grad, bf16_v_grad = torch.split(
        bf16.weights.qkv.grad,
        (config.q_width, config.kv_width, config.kv_width),
        dim=0,
    )
    module_numerics = {
        "output": _comparison(y_bf16, y_lowp),
        "input_gradient": _comparison(x_bf16.grad, x_lowp.grad),
        "q_weight_gradient": _comparison(
            bf16_q_grad, lowp.weights.q.grad
        ),
        "k_weight_gradient": _comparison(
            bf16_k_grad, lowp.weights.k.grad
        ),
        "v_weight_gradient": _comparison(
            bf16_v_grad, lowp.weights.v.grad
        ),
        "o_weight_gradient": _comparison(
            bf16.weights.o.grad, lowp.weights.o.grad
        ),
    }

    prepared_backward: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def prepare_module_backward() -> None:
        prepared_backward.clear()
        for name, module in (("bf16", bf16), ("replay", lowp)):
            _zero_module_gradients(module)
            x = x_base.detach().clone().requires_grad_(True)
            output = module(x)
            prepared_backward[name] = (x, output)

    def bf16_module_backward() -> None:
        prepared_backward["bf16"][1].backward(grad_base)

    def lowp_module_backward() -> None:
        prepared_backward["replay"][1].backward(grad_base)
        _require_released(lowp)

    module_backward = _rotated_timing(
        {
            "bf16_fa4": bf16_module_backward,
            "forward_payload_replay": lowp_module_backward,
        },
        warmups=args.warmups,
        samples=args.samples,
        pre_round=prepare_module_backward,
    )

    def run_bf16_combined() -> None:
        _zero_module_gradients(bf16)
        x = x_base.detach().clone().requires_grad_(True)
        bf16(x).backward(grad_base)

    def run_lowp_combined() -> None:
        _zero_module_gradients(lowp)
        x = x_base.detach().clone().requires_grad_(True)
        lowp(x).backward(grad_base)
        _require_released(lowp)

    module_combined = _rotated_timing(
        {
            "bf16_fa4": run_bf16_combined,
            "forward_payload_replay": run_lowp_combined,
        },
        warmups=args.warmups,
        samples=args.samples,
    )

    # Refresh one exact production publication after the module brackets.
    sink: list[dict[str, torch.Tensor | None]] = []
    runtime.forward_diagnostic_sink = sink
    _zero_module_gradients(lowp)
    capture_x = x_base.detach().clone().requires_grad_(True)
    lowp(capture_x).backward(grad_base)
    torch.cuda.synchronize()
    runtime.forward_diagnostic_sink = None
    _require_released(lowp)
    if len(sink) != 1:
        raise RuntimeError(f"expected one forward diagnostic, got {len(sink)}")
    diagnostic = sink[0]
    required_diagnostic = (
        "q_backward_fp8",
        "k_backward_fp8",
        "v_backward_fp8",
        "attention_output",
        "lse",
    )
    if any(diagnostic.get(name) is None for name in required_diagnostic):
        raise RuntimeError("forward diagnostic omitted a required tensor")
    q_fp8 = diagnostic["q_backward_fp8"]
    k_fp8 = diagnostic["k_backward_fp8"]
    v_fp8 = diagnostic["v_backward_fp8"]
    attention_output = diagnostic["attention_output"]
    lse = diagnostic["lse"]
    assert isinstance(q_fp8, torch.Tensor)
    assert isinstance(k_fp8, torch.Tensor)
    assert isinstance(v_fp8, torch.Tensor)
    assert isinstance(attention_output, torch.Tensor)
    assert isinstance(lse, torch.Tensor)

    dy_operand = tuple(
        benchmark.b300_prepare_nvfp4_projection_operand_scaled(
            grad_base.reshape(config.sequence, config.hidden).contiguous(),
            runtime.loss_scale,
        )
    )
    out_weight_backward = tuple(
        benchmark.b300_prepare_nvfp4_projection_weight(
            lowp.weights.o.T.contiguous()
        )
    )

    def publish_e5m2() -> Any:
        return benchmark.b300_project_dout_unified_lowp_nvfp4_v509_e5m2(
            dy_operand,
            out_weight_backward,
            attention_output,
            lse,
            stats_workspace=runtime.backward.workspace_torch,
            dq_clear=runtime.backward.dq,
        )

    publication = publish_e5m2()
    runtime.bind_backward_inputs(
        q_fp8,
        k_fp8,
        v_fp8,
        publication.dout_backward_e5m2,
        native_score_workspace=lowp._forward_workspace.outputs,
    )

    # The BF16 denominator is the established prebound CuTe FA4 backward on
    # BF16 decodes of the represented gradient operands. Its score path is
    # BF16; the replay route deliberately reconstructs scores from saved
    # NVFP4 bytes, so this is a matched shape rather than identical score data.
    matrix_runtime = matrix._load_runtime(REPO_ROOT)
    controls = matrix.ControlCache(matrix_runtime)
    shape = matrix.Shape(**SHAPE)
    shape.validate()
    q_bf16 = q_fp8.float().mul_(0.25).bfloat16().contiguous()
    k_bf16 = k_fp8.float().mul_(0.25).bfloat16().contiguous()
    v_bf16 = v_fp8.float().mul_(0.25).bfloat16().contiguous()
    dout_bf16 = (
        publication.dout_backward_e5m2.float()
        .mul_(0.25)
        .bfloat16()
        .contiguous()
    )
    bf16_output, bf16_lse = matrix_runtime.flash_attention(
        q_bf16, k_bf16, v_bf16, causal=True, return_lse=True
    )
    if bf16_lse.ndim == 3:
        bf16_lse = bf16_lse.unsqueeze(2).contiguous()
    elif bf16_lse.ndim == 4 and bf16_lse.shape[2] == 1:
        bf16_lse = bf16_lse.contiguous()
    else:
        raise RuntimeError(f"unexpected BF16 LSE shape {bf16_lse.shape}")
    state = matrix.RepresentedState(
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        dout_fp8=publication.dout_backward_e5m2,
        q_bf16=q_bf16,
        k_bf16=k_bf16,
        v_bf16=v_bf16,
        dout_bf16=dout_bf16,
        output_bf16=bf16_output,
        lse_bh1s=bf16_lse,
        direct_dpsum=torch.empty_like(bf16_lse),
        direct_lse_log2=torch.empty_like(bf16_lse),
    )
    bf16_route = matrix._build_bf16(
        matrix_runtime, controls, shape, state
    )

    saved_dstat = runtime.backward.dstat.clone()
    zero_dout = torch.zeros_like(publication.dout_backward_e5m2)
    runtime.backward.dstat.zero_()
    runtime.bind_backward_inputs(
        q_fp8,
        k_fp8,
        v_fp8,
        zero_dout,
        native_score_workspace=lowp._forward_workspace.outputs,
    )
    runtime.backward.run(reset=False)
    torch.cuda.synchronize()
    zero_dout_counts = {
        name: int(torch.count_nonzero(getattr(runtime.backward, name)))
        for name in ("dq", "dk", "dv")
    }
    if any(zero_dout_counts.values()):
        raise RuntimeError(
            f"zero-dO exact-zero gate failed: {zero_dout_counts}"
        )
    runtime.backward.dstat.copy_(saved_dstat)
    runtime.bind_backward_inputs(
        q_fp8,
        k_fp8,
        v_fp8,
        publication.dout_backward_e5m2,
        native_score_workspace=lowp._forward_workspace.outputs,
    )

    held_publication: list[Any] = [publication]

    def run_bf16_core() -> None:
        bf16_route.backward.run(reset=True)

    def run_replay_core() -> None:
        runtime.backward.run(reset=False)

    def run_publisher_and_replay() -> None:
        bundle = publish_e5m2()
        runtime.bind_backward_inputs(
            q_fp8,
            k_fp8,
            v_fp8,
            bundle.dout_backward_e5m2,
            native_score_workspace=lowp._forward_workspace.outputs,
        )
        runtime.backward.run(reset=False)
        held_publication[0] = bundle

    isolated_backward = _rotated_timing(
        {
            "bf16_cute_fa4": run_bf16_core,
            "replay_core": run_replay_core,
            "publisher_plus_replay": run_publisher_and_replay,
        },
        warmups=args.warmups,
        samples=args.samples,
    )
    torch.cuda.synchronize()
    isolated_checks = {
        "zero_dout_nonzero_counts": zero_dout_counts,
        "replay_outputs_finite": {
            name: bool(torch.isfinite(getattr(runtime.backward, name)).all())
            for name in ("dq", "dk", "dv")
        },
        "replay_outputs_nonzero": {
            name: int(torch.count_nonzero(getattr(runtime.backward, name))) > 0
            for name in ("dq", "dk", "dv")
        },
        "bf16_outputs_finite": {
            name: bool(torch.isfinite(getattr(bf16_route.backward, name)).all())
            for name in ("dq", "dk", "dv")
        },
    }

    result = {
        "schema": "tkfa4.d128_v509_report_boundaries.v1",
        "configuration": {
            "shape": SHAPE,
            "warmups": args.warmups,
            "samples": args.samples,
            "seed": args.seed,
            "loss_scale": LOSS_SCALE,
            "upstream_gradient_std": UPSTREAM_GRADIENT_STD,
            "scaled_upstream_gradient_std": (
                LOSS_SCALE * UPSTREAM_GRADIENT_STD
            ),
            "timing": (
                "CUDA events; provider order rotates each round; gradients "
                "and graph preparation occur outside backward-only events"
            ),
            "isolated_bf16_denominator": (
                "BF16 score/softmax on BF16 decodes of represented E4M3 "
                "Q/K/V and represented E5M2 dO; shape matched, score bytes "
                "not identical to saved-NVFP4 replay"
            ),
        },
        "boundaries": {
            "isolated_backward": {
                "bf16_cute_fa4": (
                    "prebound CuTe BF16 causal FA4 backward including its "
                    "required reset/stat preprocessing"
                ),
                "replay_core": (
                    "prebound native saved-NVFP4 score replay with E4M3 "
                    "Q/K/V, E5M2 dO, and clearing output entrypoint"
                ),
                "publisher_plus_replay": (
                    "fused output-projection dO/stat publisher, host binding, "
                    "and replay core; dy/weight operand packing prepared"
                ),
            },
            "module_backward": (
                "packed QKV and output projection backward, attention "
                "backward, publications, inverse RoPE, dgrad and wgrad; "
                "forward executed before the timing event"
            ),
            "module_forward_backward": (
                "projection-inclusive attention sublayer forward plus "
                "backward; excludes external RMSNorm, residual, optimizer, "
                "and language-model loss"
            ),
        },
        "artifacts": {
            "projection": projection_identity,
            "forward": forward_identity,
            "backward": backward_identity,
        },
        "source": {
            "repo": str(REPO_ROOT),
            "head": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "dirty": bool(_git_value("status", "--porcelain")),
            "harness": _file_identity(
                Path(__file__),
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                Path(__file__).stat().st_size,
                label="benchmark harness",
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_device": {
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": torch.cuda.get_device_properties(
                    0
                ).total_memory,
                "multiprocessor_count": torch.cuda.get_device_properties(
                    0
                ).multi_processor_count,
            },
            "physical_inventory": _gpu_inventory(),
        },
        "contracts": {
            "forward_topology": dict(forward_topology),
            "backward": runtime.backward.contract(),
            "forward_dispatch": runtime.forward_dispatch_contract(),
            "workspace": lowp.forward_workspace_contract(),
            "bf16_control": {
                "name": bf16_route.name,
                "policy": bf16_route.policy,
                "provenance": bf16_route.control_provenance,
            },
        },
        "checks": {
            "module_numerics": module_numerics,
            "isolated": isolated_checks,
        },
        "timings": {
            "isolated_backward": isolated_backward,
            "module_backward": module_backward,
            "module_forward_backward": module_combined,
        },
        "comparisons": {
            "isolated_replay_core_speedup_vs_bf16": (
                isolated_backward["bf16_cute_fa4"]["median_ms"]
                / isolated_backward["replay_core"]["median_ms"]
            ),
            "isolated_publisher_plus_replay_speedup_vs_bf16": (
                isolated_backward["bf16_cute_fa4"]["median_ms"]
                / isolated_backward["publisher_plus_replay"]["median_ms"]
            ),
            "module_backward_speedup_vs_bf16": (
                module_backward["bf16_fa4"]["median_ms"]
                / module_backward["forward_payload_replay"]["median_ms"]
            ),
            "module_forward_backward_speedup_vs_bf16": (
                module_combined["bf16_fa4"]["median_ms"]
                / module_combined["forward_payload_replay"]["median_ms"]
            ),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
