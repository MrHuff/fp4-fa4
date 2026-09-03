from __future__ import annotations

import ast
from pathlib import Path

import torch

from tk_fa4.lowp_fa4_bwd.validate_experimental_split_v_publication import (
    V_SCALE_VALID_INDICES,
    _byte_comparison,
    _contract_view,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (
    ROOT
    / "tk_fa4"
    / "lowp_fa4_bwd"
    / "validate_experimental_split_v_publication.py"
)


def _module() -> ast.Module:
    return ast.parse(VALIDATOR.read_text())


def _literal_assignment(name: str) -> object:
    for node in _module().body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment {name}")


def test_validator_covers_every_unchanged_publication() -> None:
    assert _literal_assignment("FORWARD_FIELDS") == (
        "q_payload",
        "q_scales",
        "q_global_scale",
        "k_payload",
        "k_scales",
        "k_global_scale",
        "v_payload",
        "v_scales",
    )
    assert _literal_assignment("BACKWARD_QK_FIELDS") == (
        "q_backward_e4m3",
        "k_backward_e4m3",
    )


def test_validator_launches_baseline_split_and_direct_controls() -> None:
    source = VALIDATOR.read_text()
    assert "baseline = _project(" in source
    assert "split = _project(" in source
    assert "direct = _project(" in source
    assert source.count("represented_backward=True") == 2
    assert source.count("per_block_qk_scales=True") == 2
    assert source.count("experimental_split_v_backward=True") == 1
    assert source.count("represented_backward=False") == 1
    assert source.count("per_block_qk_scales=False") == 1


def test_direct_control_is_accepted_only_after_v_contract_alignment() -> None:
    source = VALIDATOR.read_text()
    assert 'for field in ("v_payload", "v_scales", "v_backward_e4m3")' in source
    assert '"direct_control_has_aligned_forward_mxv_contract"' in source
    assert 'for field in ("v_payload", "v_scales")' in source
    assert '"split_backward_v_matches_direct_accumulator_publication"' in source


def test_validator_pins_the_loaded_extension_and_split_symbol() -> None:
    source = VALIDATOR.read_text()
    assert 'parser.add_argument("--expected-projection-extension", type=Path)' in source
    assert "if not hasattr(extension, SPLIT_V_SYMBOL):" in source
    assert '"sha256": _sha256(path)' in source


def test_byte_comparison_reports_cpu_byte_equality_and_shape_mismatch() -> None:
    reference = torch.tensor([[1, 2, 3]], dtype=torch.uint8)
    equal = _byte_comparison(reference, reference.clone())
    unequal = _byte_comparison(
        reference,
        torch.tensor([[1, 9, 3]], dtype=torch.uint8),
    )
    wrong_shape = _byte_comparison(
        reference,
        torch.tensor([1, 2], dtype=torch.uint8),
    )
    assert equal["equal"] is True
    assert equal["mismatches"] == 0
    assert unequal["equal"] is False
    assert unequal["mismatches"] == 1
    assert wrong_shape["equal"] is False
    assert wrong_shape["shape_equal"] is False


def test_v_scale_contract_view_omits_only_d64_padding() -> None:
    page = torch.arange(512, dtype=torch.int64).reshape(1, 512)
    view = _contract_view("v_scales", page)
    expected = tuple(
        lane * 16 + group * 4 + quarter
        for lane in range(32)
        for group in range(2)
        for quarter in range(4)
    )
    assert V_SCALE_VALID_INDICES == expected
    assert view.shape == (1, 256)
    assert tuple(view[0].tolist()) == expected
    assert _contract_view("q_scales", page) is page
