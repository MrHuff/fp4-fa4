import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORWARD_MATRIX = (
    ROOT / "tk_fa4" / "lowp_fa4_bwd" / "benchmark_causal_forward_matrix.py"
)
D64_PROJECTION_PROFILE = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "profile_gqa_d64_paired_projection.py"
)
D64_BOUNDARY_VALIDATOR = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "validate_gqa_d64_projection_boundaries.py"
)


def _nvfp4_helpers_called_on(path: Path, tensor_name: str) -> list[str]:
    tree = ast.parse(path.read_text())
    helpers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        argument = node.args[0]
        if not isinstance(function, ast.Name) or not isinstance(argument, ast.Name):
            continue
        if function.id.startswith("b300_prepare_nvfp4_projection_"):
            if argument.id == tensor_name:
                helpers.append(function.id)
    return helpers


def test_d64_forward_matrix_uses_true_2d_learned_weight_scaling() -> None:
    assert _nvfp4_helpers_called_on(FORWARD_MATRIX, "qkv_weight") == [
        "b300_prepare_nvfp4_projection_weight",
        "b300_prepare_nvfp4_projection_weight",
    ]
    assert "b300_prepare_nvfp4_projection_operand" in _nvfp4_helpers_called_on(
        FORWARD_MATRIX, "rows"
    )
    source = FORWARD_MATRIX.read_text()
    assert 'if projection_format == "nvfp4"' in source
    assert 'else "e4m3_per_output_channel"' in source


def test_d64_backward_profile_uses_true_2d_learned_weight_scaling() -> None:
    assert _nvfp4_helpers_called_on(
        D64_PROJECTION_PROFILE, "projection_weight"
    ) == ["b300_prepare_nvfp4_projection_weight"]
    assert "b300_prepare_nvfp4_projection_operand" in _nvfp4_helpers_called_on(
        D64_PROJECTION_PROFILE, "reference_gradient"
    )


def test_d64_boundary_validator_covers_weight_and_route_invariants() -> None:
    source = D64_BOUNDARY_VALIDATOR.read_text()
    assert "b300_prepare_nvfp4_projection_weight(qkv_weight)" in source
    assert "b300_prepare_nvfp4_projection_operand(qkv_weight)" not in source
    assert '"decoded_transpose_bitwise_equal"' in source
    assert '"q_backward_fp8_equal"' in source
    assert '"k_backward_fp8_equal"' in source
    assert '"v_backward_fp8_equal"' in source
    assert '"route_publication_provenance"' in source
    assert '"v_mxfp4_payload_shape"' in source
    assert '"v_forward_fp8_shape"' in source
    assert "normal = project(False)" in source
    assert "interleaved = project(True)" in source
