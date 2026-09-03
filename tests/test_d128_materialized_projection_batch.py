from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tk_fa4 import interface


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "tk_fa4" / "lowp_fa4_bwd" / "lowp_fa4_bwd.cu"


def _tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.bfloat16)


def _authenticated_materialized_operands(
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch in (2, 4):
        return (
            torch.empty(
                (batch, 4096, 32, 128),
                dtype=torch.bfloat16,
                device="meta",
            ),
            torch.empty(
                (batch, 4096, 8, 128),
                dtype=torch.bfloat16,
                device="meta",
            ),
            torch.empty(
                (batch, 4096, 8, 128),
                dtype=torch.bfloat16,
                device="meta",
            ),
        )
    return (
        _tensor((batch, 2, 4, 128)),
        _tensor((batch, 2, 2, 128)),
        _tensor((batch, 2, 2, 128)),
    )


def _call_projection(
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    **kwargs: object,
) -> object:
    return interface.b300_project_gqa_d128_hierarchical_qkv_gradient_nvfp4(
        dq,
        dk,
        dv,
        (object(), object(), object()),
        object(),
        object(),
        **kwargs,
    )


@pytest.fixture
def projection_stub(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    calls: list[tuple[object, ...]] = []

    def project(*args: object) -> list[object]:
        calls.append(args)
        return ["projected", "payload", "scales"]

    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        SimpleNamespace(
            project_gqa_d128_hierarchical_qkv_gradient_nvfp4=project,
        ),
    )
    return calls


@pytest.fixture
def tile_pack_stub(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    calls: list[tuple[object, ...]] = []

    def pack(*args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(
        interface,
        "_C_b300_lowp_bwd",
        SimpleNamespace(
            pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles=pack,
        ),
    )
    return calls


def _call_tile_pack(batch: int) -> None:
    interface.b300_pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(
        _tensor((batch, 2, 4, 128)),
        _tensor((batch, 2, 2, 128)),
        _tensor((batch, 2, 2, 128)),
        torch.empty(1),
        torch.empty((batch, 2, 64), dtype=torch.int32),
        (object(), object()),
        torch.empty((batch, 4, 1), dtype=torch.int32),
        row_tile_begin=0,
        row_tile_end=1,
        col_tile_begin=0,
        col_tile_end=8,
    )


@pytest.mark.parametrize("batch", (1, 2, 4))
def test_materialized_projection_accepts_only_authenticated_batches(
    projection_stub: list[tuple[object, ...]],
    batch: int,
) -> None:
    dq, dk, dv = _authenticated_materialized_operands(batch)

    assert _call_projection(dq, dk, dv) == "projected"
    assert len(projection_stub) == 1


@pytest.mark.parametrize("batch", (0, 3, 5))
def test_materialized_projection_rejects_unauthenticated_batches(
    projection_stub: list[tuple[object, ...]],
    batch: int,
) -> None:
    dq = _tensor((batch, 2, 4, 128))
    dk = _tensor((batch, 2, 2, 128))
    dv = _tensor((batch, 2, 2, 128))

    with pytest.raises(ValueError, match="batch 1, 2, or 4|positive"):
        _call_projection(dq, dk, dv)
    assert projection_stub == []


def test_hierarchical_projection_remains_batch_one_only(
    projection_stub: list[tuple[object, ...]],
) -> None:
    dq = _tensor((2, 2, 4, 2, 128))
    dk = _tensor((2, 2, 2, 128))
    dv = _tensor((2, 2, 2, 128))

    with pytest.raises(ValueError, match="hierarchical.*batch 1"):
        _call_projection(dq, dk, dv)
    assert projection_stub == []


def test_hierarchical_projection_retains_batch_one(
    projection_stub: list[tuple[object, ...]],
) -> None:
    dq = _tensor((2, 1, 4, 2, 128))
    dk = _tensor((1, 2, 2, 128))
    dv = _tensor((1, 2, 2, 128))

    assert _call_projection(dq, dk, dv) == "projected"
    assert len(projection_stub) == 1


def test_projection_rejects_mismatched_materialized_rows(
    projection_stub: list[tuple[object, ...]],
) -> None:
    dq = _tensor((2, 1, 4, 128))
    dk = _tensor((2, 2, 2, 128))
    dv = _tensor((2, 2, 2, 128))

    with pytest.raises(ValueError, match="matching dK/dV"):
        _call_projection(dq, dk, dv)
    assert projection_stub == []


@pytest.mark.parametrize("batch", (2, 4))
def test_materialized_b2_b4_rejects_non_8b_geometry(
    projection_stub: list[tuple[object, ...]],
    batch: int,
) -> None:
    with pytest.raises(ValueError, match="B2/B4.*S4096/Hq32/Hkv8"):
        _call_projection(
            _tensor((batch, 256, 4, 128)),
            _tensor((batch, 256, 2, 128)),
            _tensor((batch, 256, 2, 128)),
        )
    assert projection_stub == []


def test_tile_ready_pack_retains_batch_one(
    tile_pack_stub: list[tuple[object, ...]],
) -> None:
    _call_tile_pack(1)
    assert len(tile_pack_stub) == 1


def test_tile_ready_pack_rejects_materialized_batch_two_before_extension(
    tile_pack_stub: list[tuple[object, ...]],
) -> None:
    with pytest.raises(ValueError, match="authenticated only for batch 1"):
        _call_tile_pack(2)
    assert tile_pack_stub == []


def test_native_projection_guard_is_materialized_b2_b4_only() -> None:
    source = CUDA_SOURCE.read_text()
    function = source.split(
        "project_gqa_d128_hierarchical_qkv_gradient_nvfp4(", 1
    )[1].split(
        "void pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(", 1
    )[0]

    assert "batch == 1 || batch == 2 || batch == 4" in function
    assert "(batch != 2 && batch != 4)" in function
    assert "sequence == 4096" in function
    assert "q_heads == 32" in function
    assert "kv_heads == 8" in function
    assert "hidden == 4096" in function
    assert (
        "hierarchical D128 GQA projection currently requires B=1" in function
    )
    assert "tile-ready D128 GQA pack currently requires batch 1" not in function


def test_tile_ready_pack_is_explicitly_batch_one_in_python_and_cuda() -> None:
    python = Path(interface.__file__).read_text().split(
        "def b300_pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(", 1
    )[1].split("\ndef ", 1)[0]
    cuda = CUDA_SOURCE.read_text().split(
        "void pack_gqa_d128_hierarchical_qkv_gradient_nvfp4_tiles(", 1
    )[1].split("\nvoid ", 1)[0]

    assert 'int(dk.shape[0]) != 1' in python
    assert 'int(dv.shape[0]) != 1' in python
    assert "batch == 1" in cuda
    assert "authenticated only for batch 1" in cuda
