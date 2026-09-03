from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from tk_fa4.lowp_fa4_bwd.projection_quantization_reference import (
    transpose_prepared_nvfp4_weight_reference,
)


ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_llama12b_e2e.py"
CUDA = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"
INTERFACE = ROOT / "tk_fa4" / "interface.py"
PACKAGE = ROOT / "tk_fa4" / "__init__.py"
VALIDATOR = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "validate_gqa_d64_projection_boundaries.py"
)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    assert codes.ndim == 2 and codes.shape[1] % 2 == 0
    codes = codes.to(torch.uint8)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def _unpack_codes(payload: torch.Tensor) -> torch.Tensor:
    payload = payload.contiguous().view(torch.uint8)
    return torch.stack(
        (payload & 0x0F, payload >> 4),
        dim=-1,
    ).reshape(payload.shape[0], payload.shape[1] * 2)


def _scale_index(row: int, group: int, page_columns: int) -> int:
    page_row = row // 128
    page_column = group // 4
    offset = (row % 32) * 16 + ((row % 128) // 32) * 4 + group % 4
    return (page_row * page_columns + page_column) * 512 + offset


def _publish_true_2d_scales(
    tile_values: torch.Tensor,
    *,
    rows: int,
    columns: int,
) -> torch.Tensor:
    assert tuple(tile_values.shape) == (rows // 16, columns // 16)
    pages = torch.empty(
        rows // 128,
        columns // 64,
        512,
        dtype=torch.uint8,
    )
    flat = pages.view(-1)
    for row in range(rows):
        for group in range(columns // 16):
            flat[_scale_index(row, group, columns // 64)] = tile_values[
                row // 16,
                group,
            ]
    return pages


def test_reference_transposes_packed_codes_and_scale_tile_grid_exactly() -> None:
    rows, columns = 128, 256
    codes = (
        torch.arange(rows * columns, dtype=torch.int64)
        .reshape(rows, columns)
        .remainder(16)
        .to(torch.uint8)
    )
    source_tiles = (
        torch.arange(
            (rows // 16) * (columns // 16),
            dtype=torch.int64,
        )
        .reshape(rows // 16, columns // 16)
        .add(1)
        .to(torch.uint8)
    )
    payload = _pack_codes(codes)
    scales = _publish_true_2d_scales(
        source_tiles,
        rows=rows,
        columns=columns,
    )
    global_scale = torch.tensor([0.125], dtype=torch.float32)

    transposed = transpose_prepared_nvfp4_weight_reference(
        (payload, scales, global_scale)
    )

    assert transposed[2] is global_scale
    assert tuple(transposed[0].shape) == (columns, rows // 2)
    assert tuple(transposed[1].shape) == (
        columns // 128,
        rows // 64,
        512,
    )
    assert torch.equal(_unpack_codes(transposed[0]), codes.T)
    transposed_scale_bytes = transposed[1].view(-1)
    for row in range(columns):
        for group in range(rows // 16):
            actual = transposed_scale_bytes[
                _scale_index(row, group, rows // 64)
            ]
            expected = source_tiles[group, row // 16]
            assert actual.item() == expected.item()

    round_trip = transpose_prepared_nvfp4_weight_reference(transposed)
    assert torch.equal(round_trip[0].view(torch.uint8), payload)
    assert torch.equal(round_trip[1].view(torch.uint8), scales)
    assert round_trip[2] is global_scale

    typed_transposed = transpose_prepared_nvfp4_weight_reference(
        (
            payload.view(torch.float4_e2m1fn_x2),
            scales.view(torch.float8_e4m3fn),
            global_scale,
        )
    )
    assert typed_transposed[0].dtype == torch.float4_e2m1fn_x2
    assert typed_transposed[1].dtype == torch.float8_e4m3fn
    assert torch.equal(
        typed_transposed[0].view(torch.uint8),
        transposed[0],
    )
    assert torch.equal(
        typed_transposed[1].view(torch.uint8),
        transposed[1],
    )


def test_reference_rejects_non_2d_replicated_weight_scales() -> None:
    rows = columns = 128
    payload = torch.zeros(rows, columns // 2, dtype=torch.uint8)
    tile_values = torch.ones(rows // 16, columns // 16, dtype=torch.uint8)
    scales = _publish_true_2d_scales(
        tile_values,
        rows=rows,
        columns=columns,
    )
    scales.view(-1)[_scale_index(1, 0, columns // 64)] = 2

    with pytest.raises(ValueError, match="not replicated per 16x16 tile"):
        transpose_prepared_nvfp4_weight_reference(
            (payload, scales, torch.ones(1))
        )


def test_autograd_reuses_caller_owned_dual_layout_for_d64_and_d128() -> None:
    source = E2E.read_text()
    module = ast.parse(source)
    function_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "_LowpAttentionFunction"
    )
    forward = next(
        node
        for node in function_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    backward = next(
        node
        for node in function_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "backward"
    )
    direct_assignment = next(
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "use_direct_dual_out_weight"
            for target in node.targets
        )
    )
    direct_policy = ast.unparse(direct_assignment.value)
    assert direct_policy == (
        "_uses_direct_dual_output_weight_prep(runtime)"
    )
    eligibility = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_uses_direct_dual_output_weight_prep"
    )
    eligibility_source = ast.unparse(eligibility)
    assert "projection_weight_scale_2d" in eligibility_source
    assert "head_dim" not in eligibility_source
    assert "pv_format" not in eligibility_source
    dual_calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_prepare_direct_dual_output_weight"
    ]
    assert len(dual_calls) == 1

    forward_source = ast.get_source_segment(source, forward)
    backward_source = ast.get_source_segment(source, backward)
    assert forward_source is not None and backward_source is not None
    assert (
        "ctx.output_weight_backward_operand = out_weight_backward_operand"
        in forward_source
    )
    assert "*out_weight_backward_operand" not in forward_source
    assert (
        "out_weight_backward_operand = ctx.output_weight_backward_operand"
        in backward_source
    )
    assert "if out_weight_backward_operand is None:" in backward_source
    assert "out_weight.T.contiguous()" in backward_source
    assert backward_source.count("out_weight.T.contiguous()") == 1
    assert "lowp/bwd/output_weight_transpose_nvfp4_pack" in backward_source

    result_assignments = [
        node
        for node in ast.walk(backward)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "result"
            for target in node.targets
        )
        and isinstance(node.value, ast.Tuple)
    ]
    assert any(len(node.value.elts) == 10 for node in result_assignments)
    finish = backward_source.index("publication_state.finish_backward")
    result_return = backward_source.index("return result")
    assert finish < result_return


def test_cuda_dual_prep_is_one_quantization_plus_one_tiled_transpose() -> None:
    source = CUDA.read_text()
    dual = source.split(
        "std::vector<at::Tensor> quantize_nvfp4_projection_weight_dual",
        1,
    )[1].split(
        "std::vector<at::Tensor> "
        "quantize_nvfp4_projection_operand_scaled",
        1,
    )[0]
    assert dual.count(
        "quantize_nvfp4_projection_operand_impl(input, 1.0, true)"
    ) == 1
    assert "input.transpose" not in dual
    assert "const dim3 grid(columns / kTile, rows / kTile);" in dual
    assert "transpose_prepared_nvfp4_weight_kernel<<<" in dual
    assert dual.count("forward[2]") == 2

    kernel = source.split(
        "void transpose_prepared_nvfp4_weight_kernel",
        1,
    )[1].split(
        "std::vector<at::Tensor> quantize_nvfp4_projection_weight_dual",
        1,
    )[0]
    assert "constexpr int kTile = 64;" in kernel
    assert "__shared__ uint8_t payload_tile[kTile][kTile + 1];" in kernel
    assert "__shared__ uint8_t scale_tile[kScaleTiles];" in kernel
    assert "prepared_nvfp4_scale_index" in kernel
    assert '"quantize_nvfp4_projection_weight_dual"' in source


def test_dual_prep_api_is_fail_closed_and_public() -> None:
    interface = INTERFACE.read_text()
    package = PACKAGE.read_text()
    function = interface.split(
        "def b300_prepare_nvfp4_projection_weight_dual(",
        1,
    )[1].split(
        "def b300_prepare_nvfp4_projection_operand_scaled(",
        1,
    )[0]
    assert "both matrix dimensions must be divisible by 128" in function
    assert "expected_dtypes" in function
    assert "not output.is_contiguous()" in function
    assert "forward[2].data_ptr() != transpose[2].data_ptr()" in function
    assert "B300DualNVFP4ProjectionWeight" in package
    assert "b300_prepare_nvfp4_projection_weight_dual" in package


def test_d64_validator_checks_dual_bytes_against_both_oracles() -> None:
    source = VALIDATOR.read_text()
    assert (
        'DUAL_WEIGHT_QUANTIZATION_SYMBOL = '
        '"quantize_nvfp4_projection_weight_dual"'
    ) in source
    assert '"forward_bytes_equal_independent_quantization"' in source
    assert '"transpose_bytes_equal_independent_quantization"' in source
    assert '"transpose_bytes_equal_storage_reference"' in source
    assert '"global_scale_storage_shared"' in source
    assert "dual NVFP4 weight preparation invariants failed" in source
