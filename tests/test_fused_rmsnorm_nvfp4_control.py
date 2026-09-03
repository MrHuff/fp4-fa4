import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
INTERFACE = ROOT / "tk_fa4" / "interface.py"
PACKAGE = ROOT / "tk_fa4" / "__init__.py"
CUDA = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"
HEADER = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "rmsnorm_nvfp4_quantize.cuh"
MAKEFILE = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "Makefile"


def _function(path: Path, name: str):
    tree = ast.parse(path.read_text())
    node = next(
        candidate
        for candidate in tree.body
        if isinstance(candidate, ast.FunctionDef) and candidate.name == name
    )
    namespace = {"Config": SimpleNamespace}
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _config(**overrides: int) -> SimpleNamespace:
    values = {
        "batch": 16,
        "sequence": 4096,
        "hidden": 2048,
        "head_dim": 64,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _valid_gate_kwargs() -> dict[str, object]:
    return {
        "enabled": True,
        "qkv_projection_format": "nvfp4",
        "experimental_native_nvfp4_projection_out": True,
        "projection_weight_scale_2d": True,
    }


def test_fused_gate_is_opt_in_and_accepts_only_the_b16_native_route() -> None:
    require = _function(E2E, "_require_fused_attention_rmsnorm_nvfp4")
    require(
        _config(batch=1, sequence=128, hidden=96, head_dim=32),
        enabled=False,
        qkv_projection_format="e4m3",
        experimental_native_nvfp4_projection_out=False,
        projection_weight_scale_2d=False,
    )
    require(_config(), **_valid_gate_kwargs())
    source = E2E.read_text()
    assert "experimental_fused_attention_rmsnorm_nvfp4: bool = False" in source
    runtime = source.split("class LowpAttentionRuntime:", 1)[1].split(
        "class _LowpAttentionFunction", 1
    )[0]
    assert "_require_fused_attention_rmsnorm_nvfp4(" in runtime


@pytest.mark.parametrize(
    ("config_overrides", "keyword", "value"),
    (
        ({"batch": 1}, None, None),
        ({"sequence": 2048}, None, None),
        ({"hidden": 4096}, None, None),
        ({"head_dim": 128}, None, None),
        ({}, "qkv_projection_format", "e4m3"),
        ({}, "experimental_native_nvfp4_projection_out", False),
        ({}, "projection_weight_scale_2d", False),
    ),
)
def test_fused_gate_rejects_every_unauthenticated_boundary(
    config_overrides: dict[str, int],
    keyword: str | None,
    value: object,
) -> None:
    require = _function(E2E, "_require_fused_attention_rmsnorm_nvfp4")
    kwargs = _valid_gate_kwargs()
    if keyword is not None:
        kwargs[keyword] = value
    with pytest.raises(
        ValueError,
        match="experimental fused attention RMSNorm NVFP4 requires",
    ):
        require(_config(**config_overrides), **kwargs)


def test_public_api_preserves_the_native_operand_abi_and_saved_state() -> None:
    interface = INTERFACE.read_text()
    wrapper = interface.split("def b300_prepare_nvfp4_projection_operand_rmsnorm(", 1)[
        1
    ].split("def b300_prepare_e4m3_projection_operand(", 1)[0]
    assert "_C_b300_lowp_bwd.quantize_nvfp4_projection_operand_rmsnorm" in wrapper
    assert "gamma must be contiguous CUDA BF16 [K]" in wrapper
    assert "epsilon must be finite and positive" in wrapper
    assert "positive M divisible by 128 and K=2048" in wrapper
    assert "len(prepared) != 5" in wrapper
    for expected in (
        "((rows, columns // 2), torch.float4_e2m1fn_x2)",
        "torch.float8_e4m3fn",
        "((1,), torch.float32)",
        "((rows,), torch.float32)",
        "((rows, columns), torch.bfloat16)",
    ):
        assert expected in wrapper

    package = PACKAGE.read_text()
    assert package.count("b300_prepare_nvfp4_projection_operand_rmsnorm") == 2
    assert package.count("b300_rmsnorm_backward") == 2


def test_fused_backward_uses_two_cuda_stages_without_fp32_matrix_temps() -> None:
    interface = INTERFACE.read_text()
    wrapper = interface.split("def b300_rmsnorm_backward(", 1)[1].split(
        "def b300_prepare_e4m3_projection_operand(", 1
    )[0]
    assert "_C_b300_lowp_bwd.rmsnorm_backward_bf16" in wrapper
    assert "[M, 2048]" in wrapper
    assert "fused RMSNorm backward must return two tensors" in wrapper

    e2e = E2E.read_text()
    backward = e2e.split("def backward(", 1)[1].split(
        "class LowpAttention(nn.Module)", 1
    )[0]
    stage = backward.split(
        'with _stage("lowp/bwd/attention_rmsnorm"):', 1
    )[1].split("dx = dx_matrix.reshape", 1)[0]
    assert "b300_rmsnorm_backward(" in stage
    for old_full_matrix_operation in (
        "raw_rows.float()",
        "dx_matrix.float()",
        "inv_rms.float().unsqueeze(1)",
    ):
        assert old_full_matrix_operation not in stage

    cuda = CUDA.read_text()
    host = cuda.split("std::vector<at::Tensor> rmsnorm_backward_bf16(", 1)[
        1
    ].split("std::vector<at::Tensor> quantize_nvfp4_projection_weight(", 1)[0]
    assert host.count("<<<") == 2
    assert "rmsnorm_backward_partial_kernel<<<" in host
    assert "rmsnorm_backward_gamma_finalize_kernel<<<" in host
    assert host.count("CUDACHECK(cudaGetLastError());") == 2
    assert "cudaStreamSynchronize" not in host
    assert "return {input_gradient, gamma_gradient};" in host


def test_cuda_host_path_is_two_kernels_plus_an_async_amax_clear() -> None:
    cuda = CUDA.read_text()
    host = cuda.split(
        "std::vector<at::Tensor> " "quantize_nvfp4_projection_operand_rmsnorm(",
        1,
    )[1].split("std::vector<at::Tensor> rmsnorm_backward_bf16(", 1)[0]
    assert host.count("<<<") == 2
    assert "cudaMemsetAsync(" in host
    assert "rmsnorm_bf16_amax_kernel<<<" in host
    assert "quantize_from_amax_kernel<<<" in host
    assert host.count("CUDACHECK(cudaGetLastError());") == 2
    assert "cudaStreamSynchronize" not in host
    for separate_launch in (
        "absmax_kernel<<<",
        "divide_kernel<<<",
        "fp8_nan_fixup_kernel<<<",
    ):
        assert separate_launch not in host
    assert "return {packed, scales, global_scale, inv_rms, normalized};" in host
    assert "input.size(0) > 0" in host
    assert "RMSNORM_BACKWARD_COLUMNS" in host
    assert '"quantize_nvfp4_projection_operand_rmsnorm"' in cuda
    assert "&quantize_nvfp4_projection_operand_rmsnorm" in cuda
    assert '#include "rmsnorm_nvfp4_quantize.cuh"' in cuda
    assert "rmsnorm_nvfp4_quantize.cuh" in MAKEFILE.read_text()


def test_device_path_rounds_before_amax_and_preserves_scale_swizzle() -> None:
    header = HEADER.read_text()
    assert "const __grid_constant__ nvfp4_quantize::globals g" in header
    assert "__fmul_rn(value, row_inv_rms)" in header
    assert "__fmul_rn(unit, weight)" in header
    assert "__float2bfloat16_rn(scaled)" in header
    assert "fabsf(__bfloat162float(output))" in header
    assert "global_amax[0] / 2688.0f" in header
    assert "sanitize_e4m3_scale(scale_registers" in header
    assert "(tid % 32) * 16 + (tid / 32) * 4" in header
    assert "tma::store_async(g.A_sc" in header


def test_decoder_bypasses_only_attention_norm_on_the_opt_in_route() -> None:
    source = E2E.read_text()
    decoder = source.split("class DecoderLayer(nn.Module):", 1)[1].split(
        "class Llama12B(nn.Module):", 1
    )[0]
    forward = decoder.split("def forward(self, x: torch.Tensor)", 1)[1]
    assert "if self.fused_attention_rmsnorm_nvfp4:" in forward
    assert "self.attention(x, self.attention_norm.weight)" in forward
    assert "self.attention(self.attention_norm(x))" in forward
    assert "self.mlp(self.ffn_norm(x))" in forward
    assert 'with _stage("lowp/fwd/attention_rmsnorm_nvfp4_pack")' in source
    assert 'with _stage("lowp/bwd/attention_rmsnorm")' in source


def test_autograd_saves_raw_norm_state_and_returns_gamma_gradient() -> None:
    source = E2E.read_text()
    function = source.split("class _LowpAttentionFunction", 1)[1].split(
        "class LowpAttention", 1
    )[0]
    assert "raw_rows," in function
    assert "attention_norm_weight," in function
    assert "inv_rms," in function
    assert "dx_matrix, dattention_norm_weight = b300_rmsnorm_backward(" in function
    assert "rows = prepared_rows[4]" in function
    assert "dx,\n            dattention_norm_weight," in function
    assert 'default="nvfp4"' in source
    assert '"--experimental-fused-attention-rmsnorm-nvfp4"' in source
