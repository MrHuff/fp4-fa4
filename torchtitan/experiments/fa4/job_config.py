#
# Copyright (c) 2025-2026 Graphcore Ltd. All rights reserved.
#
"""TorchTitan configuration extension for the reproduced FA4 experiments."""

from dataclasses import dataclass, field
from typing import Literal

from torchtitan.config.job_config import Model as TorchTitanModel
from torchtitan.config.job_config import Training as TorchTitanTraining
from torchtitan.models.llama3.model.args import RoPEScalingArgs


@dataclass
class Model(TorchTitanModel):
    """Model overrides needed to authenticate the fixed Llama geometries."""

    rope_theta: float | None = None
    max_seq_len: int | None = None
    rope_scaling_args: RoPEScalingArgs = field(default_factory=RoPEScalingArgs)


@dataclass
class Training(TorchTitanTraining):
    """Legacy switches read by the measured fail-closed adapter."""

    enable_fp32_master_params: bool = False
    enable_cce: bool = False
    compile: bool = False
    mixed_precision_reduce: Literal["bfloat16", "float32"] = "bfloat16"


@dataclass
class FA4:
    enabled: bool = True
    # Training-loop facilities are off by default.  They are selected by the
    # generated FA4 recipe and require the FA4-specific trainer entry point.
    cuda_data_prefetch: bool = False
    fail_on_nonfinite_metrics: bool = False
    scan_nonfinite_gradients: bool = False
    gradient_diagnostics_topk: int = 0
    mode: Literal["softmax", "softcap", "sigmoid_attention"] = "softmax"
    audit_coefficients: bool = False
    softcap: float = 50.0
    softcap_degree: int = 3
    softcap_backend: str = "cute"
    softcap_backward_mode: str = "algebraic"
    sigmoid_variant: Literal["poly", "sfu"] = "poly"
    sigmoid_sfu_freq: int = 16
    sigmoid_sfu_res: int = 0
    sigmoid_sfu_freq_bwd: int | None = None
    sigmoid_sfu_res_bwd: int | None = None
    sigmoid_backward_mode: Literal["algebraic", "direct"] = "algebraic"
    sigmoid_bias: float | None = None
    sigmoid_poly_backend: str = "cute"
    sigmoid_qk_norm: bool = True

    # Source and binary identities are deliberately explicit.  The adapter
    # refuses an incomplete or mismatched build instead of choosing a nearby
    # extension from PYTHONPATH.
    exact_source_root: str = ""
    exact_runtime_source_sha256: str = ""
    exact_flash_attn_root: str = ""
    exact_flash_attn_source_sha256: str = ""
    exact_cutlass_dsl_root: str = ""
    exact_cutlass_dsl_version: str = "4.5.2"
    exact_cutlass_dsl_native_sha256: str = ""
    exact_artifact_profile: str = ""
    exact_forward_extension: str = ""
    exact_forward_module: str = ""
    exact_forward_sha256: str = ""
    exact_forward_batch_size: int = 1
    exact_pv_format: Literal["e4m3_fp8", "mxfp4_e8m0_block32"] | None = "e4m3_fp8"
    exact_learned_projection_format: Literal["e4m3", "nvfp4"] = "e4m3"
    exact_d128_represented_qk_backward: bool = False
    exact_d128_native_score_backward: bool = False
    exact_d128_e5m2_dout_backward: bool = False
    exact_mx_v_publication: Literal[
        "retained_split",
        "output_shared_split",
        "shared_d32xs32_forward_anchors",
    ] = "retained_split"
    exact_backward_extension: str = ""
    exact_backward_sha256: str = ""
    exact_native_tk_d64_backward_extension: str = ""
    exact_native_tk_d64_backward_module: str = ""
    exact_native_tk_d64_backward_sha256: str = ""
    exact_native_tk_d64_backward_bytes: int = 0
    exact_native_tk_d128_backward_extension: str = ""
    exact_native_tk_d128_backward_module: str = ""
    exact_native_tk_d128_backward_sha256: str = ""
    exact_native_tk_d128_backward_bytes: int = 0
    exact_backward_control_source: str = ""
    exact_backward_control_sha256: str = ""
    exact_backward_control_bytes: int = 0
    exact_allow_fp32_master_shadows: bool = False


@dataclass
class SplineMLP:
    """Activation selected by the historical SFU-B1 converter chain.

    The published continuation path carries the native PyTorch activations.
    The separate experimental spline extension was not part of the recovered
    source closure and is therefore not exposed as a runnable choice.
    """

    activation_impl: Literal["native_silu", "native_gelu"] = "native_silu"


@dataclass
class JobConfig:
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)
    fa4: FA4 = field(default_factory=FA4)
    spline_mlp: SplineMLP = field(default_factory=SplineMLP)


__all__ = ["FA4", "JobConfig", "Model", "SplineMLP", "Training"]
