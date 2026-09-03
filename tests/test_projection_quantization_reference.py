import torch

from tk_fa4.lowp_fa4_bwd.projection_quantization_reference import (
    _quantize_e2m1_rne,
    _quantize_e2m1_stochastic,
    fake_quantize_mxfp4,
    fake_quantize_mxfp4_v_1d,
    fake_quantize_nvfp4,
)


def test_nvfp4_weight_quantization_is_transpose_consistent() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(32, 48, generator=generator)
    for selector in ("static6", "static4", "adaptive_mae", "adaptive_mse"):
        row = fake_quantize_nvfp4(
            weight,
            block_shape=(16, 16),
            selector=selector,
        )
        column = fake_quantize_nvfp4(
            weight.T.contiguous(),
            block_shape=(16, 16),
            selector=selector,
        )
        torch.testing.assert_close(row.values.T, column.values, rtol=0, atol=0)
        torch.testing.assert_close(
            row.block_scales.T,
            column.block_scales,
            rtol=0,
            atol=0,
        )

def test_adaptive_nvfp4_never_worsens_its_tile_objective() -> None:
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(32, 32, generator=generator)
    static = fake_quantize_nvfp4(
        weight,
        block_shape=(16, 16),
        selector="static6",
    ).values.float()
    adaptive_mae = fake_quantize_nvfp4(
        weight,
        block_shape=(16, 16),
        selector="adaptive_mae",
    ).values.float()
    adaptive_mse = fake_quantize_nvfp4(
        weight,
        block_shape=(16, 16),
        selector="adaptive_mse",
    ).values.float()
    assert (adaptive_mae - weight).abs().sum() <= (static - weight).abs().sum()
    assert (adaptive_mse - weight).square().sum() <= (
        static - weight
    ).square().sum()


def test_mxfp4_2d_weight_quantization_is_transpose_consistent() -> None:
    generator = torch.Generator().manual_seed(13)
    weight = torch.randn(64, 96, generator=generator)
    for mode in ("ceil", "rte", "dense"):
        row = fake_quantize_mxfp4(
            weight,
            block_shape=(32, 32),
            scale_mode=mode,
        )
        column = fake_quantize_mxfp4(
            weight.T.contiguous(),
            block_shape=(32, 32),
            scale_mode=mode,
        )
        torch.testing.assert_close(row.values.T, column.values, rtol=0, atol=0)
        torch.testing.assert_close(
            row.block_scales.T,
            column.block_scales,
            rtol=0,
            atol=0,
        )


def test_e2m1_rne_uses_even_code_at_every_midpoint() -> None:
    midpoints = torch.tensor((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0))
    expected = torch.tensor((0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0))
    torch.testing.assert_close(
        _quantize_e2m1_rne(midpoints),
        expected,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _quantize_e2m1_rne(-midpoints),
        -expected,
        rtol=0,
        atol=0,
    )


def test_e2m1_stochastic_rounding_is_unbiased_between_neighbors() -> None:
    generator = torch.Generator().manual_seed(20260826)
    values = torch.full((200_000,), 0.125)
    rounded = _quantize_e2m1_stochastic(values, generator=generator)
    assert set(rounded.unique().tolist()) == {0.0, 0.5}
    assert abs(float(rounded.mean()) - 0.125) < 0.002


def test_mxfp4_v_1d_groups_sequence_and_requires_explicit_sr_key() -> None:
    tensor = torch.zeros(2, 64, 3, 4)
    tensor[:, :32] = 1.0
    tensor[:, 32:] = 2.0
    result = fake_quantize_mxfp4_v_1d(tensor)
    assert result.values.shape == tensor.shape
    assert result.block_scales.shape == (2, 3, 4, 2)
    assert result.diagnostics["block_shape"] == [1, 32]
    assert result.diagnostics["rounding"] == "rne"

    try:
        fake_quantize_mxfp4_v_1d(tensor, rounding="stochastic")
    except ValueError as error:
        assert "explicit generator" in str(error)
    else:
        raise AssertionError("stochastic V QDQ accepted no generator")
