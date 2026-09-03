from __future__ import annotations

import os

import pytest
import torch

from tk_fa4.lowp_fa4_bwd.cute_d128_mxfp4_dp_prototype import (
    D128GqaDpGeometry,
    compile,
    prepare_scale_pages,
    quantize_backward_mxfp4_reference,
)


def test_geometry_requires_native_scale_pages_and_valid_gqa() -> None:
    with pytest.raises(ValueError, match="divisible by 128"):
        D128GqaDpGeometry(batch=1, sequence=192, q_heads=8, kv_heads=2)
    with pytest.raises(ValueError, match="divisible by kv_heads"):
        D128GqaDpGeometry(batch=1, sequence=128, q_heads=8, kv_heads=3)


def test_prepare_scale_pages_reorders_and_repeats_only_v_scales() -> None:
    geometry = D128GqaDpGeometry(
        batch=2,
        sequence=256,
        q_heads=8,
        kv_heads=2,
    )
    dout = torch.arange(
        2 * 2 * 8 * 512,
        dtype=torch.int64,
    ).remainder_(251).to(torch.uint8).reshape(2, 2, 8, 512)
    v = torch.arange(
        2 * 2 * 2 * 512,
        dtype=torch.int64,
    ).remainder_(241).to(torch.uint8).reshape(2, 2, 2, 512)

    dout_mma, v_mma = prepare_scale_pages(geometry, dout, v)

    assert dout_mma.shape == (2, 2, 4, 2, 512)
    assert v_mma.shape == (2, 2, 4, 2, 512)
    for batch in range(geometry.batch):
        for kv_head in range(geometry.kv_heads):
            for group_head in range(geometry.group_size):
                q_head = kv_head * geometry.group_size + group_head
                torch.testing.assert_close(
                    dout_mma[batch, kv_head, group_head],
                    dout[batch, :, q_head],
                )
                torch.testing.assert_close(
                    v_mma[batch, kv_head, group_head],
                    v[batch, :, kv_head],
                )


def test_reference_quantizer_emits_producer_shapes() -> None:
    source = torch.linspace(-0.25, 0.25, 2 * 128 * 3 * 128).reshape(
        2,
        128,
        3,
        128,
    )
    payload, scales, represented, raw_operand = quantize_backward_mxfp4_reference(
        source,
        scale_rows=1,
        scale_selector="mse_1d",
    )
    assert payload.shape == (2, 128, 3, 64)
    assert payload.dtype == torch.uint8
    assert scales.shape == (2, 1, 3, 512)
    assert scales.dtype == torch.uint8
    assert represented.shape == source.shape
    assert raw_operand.shape == source.shape
    assert torch.isfinite(represented).all()
    assert torch.isfinite(raw_operand).all()


@pytest.mark.skipif(
    os.environ.get("TK_FA4_RUN_CUTE_D128_MXDP_SMOKE") != "1",
    reason="set TK_FA4_RUN_CUTE_D128_MXDP_SMOKE=1 for the SM100 compile smoke",
)
@pytest.mark.parametrize("batch", (1, 2))
def test_sm100_cute_d128_gqa_dp_matches_quantized_reference(batch: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("the prototype requires SM100")

    torch.manual_seed(20260827 + batch)
    device = torch.device("cuda")
    geometry = D128GqaDpGeometry(
        batch=batch,
        sequence=128,
        q_heads=8,
        kv_heads=2,
    )
    dout_source = torch.randn(
        batch,
        128,
        8,
        128,
        device=device,
    ).mul_(0.03)
    v_source = torch.randn(
        batch,
        128,
        2,
        128,
        device=device,
    ).mul_(0.03)
    dout_fp4, dout_scales, _, dout_raw = (
        quantize_backward_mxfp4_reference(
            dout_source,
            scale_rows=32,
            scale_selector="rte",
        )
    )
    v_fp4, v_scales, _, v_raw = quantize_backward_mxfp4_reference(
        v_source,
        scale_rows=1,
        scale_selector="mse_1d",
    )
    dout_scales_mma, v_scales_mma = prepare_scale_pages(
        geometry,
        dout_scales,
        v_scales,
    )

    extension = compile(
        geometry,
        dout_fp4,
        v_fp4,
        dout_scales_mma,
        v_scales_mma,
    )
    actual = extension(
        dout_fp4,
        v_fp4,
        dout_scales_mma,
        v_scales_mma,
    )
    torch.cuda.synchronize()

    v_gqa = (
        v_raw.permute(0, 2, 1, 3)
        .repeat_interleave(geometry.group_size, dim=1)
    )
    # tcgen05 sees each width-six operand at 6x its represented value, so
    # this is represented dP x36.  The active backward must convert it to its
    # existing centering domain as raw_dP_x36 * (16 / 36) - dPsum_x16.
    expected_raw_dp_x36 = torch.matmul(
        dout_raw.permute(0, 2, 1, 3),
        v_gqa.transpose(-1, -2),
    )
    expected_raw_dp_x36 = expected_raw_dp_x36.to(torch.float8_e4m3fn).float()
    torch.testing.assert_close(
        actual.float(),
        expected_raw_dp_x36,
        rtol=0,
        atol=0,
    )
