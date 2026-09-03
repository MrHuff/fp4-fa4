# Saturated 8B D128 causal-GQA bracket (2026-08-29)

This receipt records a matched single-GB200, batch-2, sequence-4096
Llama-3.1-8B bracket.  All three routes start from the same serialized BF16
checkpoint and use the same synthetic token stream, AdamW configuration, and
torch-compiled cut-cross-entropy.  Each timing contains three warm-up updates
and twenty measured updates.

The low-precision routes use native ThunderKittens forward/projection kernels
and the generated CuTe DSL causal-GQA backward.  They do **not** use a native
ThunderKittens D128 backward.  This distinction is important: the recovered
native D128 causal-GQA implementation is a legacy generic fallback and is not
competitive with CuTe.

## Result

| Route | Step p50 | Tokens/s | Useful BF16-equivalent MFU | Decoder forward p50 | Backward p50 | Speedup vs BF16 |
|---|---:|---:|---:|---:|---:|---:|
| Packed BF16 + BF16 CuTe FA4 | 488.690 ms | 16,763.18 | 35.948% | 136.820 ms | 279.746 ms | 1.000x |
| NVFP4-QK + FP8-PV | 435.798 ms | 18,797.70 | 40.311% | 119.913 ms | 242.602 ms | 1.121x |
| NVFP4-QK + MXFP4-PV | 434.760 ms | 18,842.58 | 40.407% | 119.606 ms | 241.638 ms | 1.124x |

MXFP4-PV was 1.0024x faster than FP8-PV end to end and 1.0026x faster in the
measured decoder forward.  Those differences are smaller than the per-run
step coefficient of variation (about 0.48%), so this bracket establishes a
tie, not a statistically defensible MX win.  The backward implementations are
identical; their 0.4% timing difference is noise.

Peak allocated memory was 112.11 GiB for BF16 and 110.65 GiB for both
low-precision routes.  Peak reserved memory was 129.40 GiB and 125.93 GiB,
respectively.  B2 therefore exercises a substantially fuller model than the
earlier B1 tests while remaining below the 180-GiB gate.

## What the routes actually publish

Both low-precision routes use NVFP4 Q/K projection outputs with row-by-K16
two-dimensional scale geometry.  Their backward consumes the same E4M3
projection-accumulator Q/K/V representation and the same CuTe DSL control
(generated-source SHA256
`cfbd3ad27e5188d39c475abc238b57b5331fc7e631054a7075c7993150c70764`).

The FP8 route can use its E4M3 V for both forward PV and backward.  The MX route
uses MXFP4 V for forward but retains an E4M3 projection-accumulator V for
backward (`publication_path=retained_dual_v`).  That extra publication is the
remaining structural reason an isolated MXFP4-PV kernel advantage need not
appear end to end.

## Short-run numerics

All updates and gradients were finite.  Held-out losses were:

| Route | Initial | Final | Change | Final delta vs BF16 |
|---|---:|---:|---:|---:|
| BF16 | 12.570267 | 12.561865 | -0.008402 | 0 |
| FP8-PV | 12.583249 | 12.583220 | -0.000030 | +0.021355 |
| MXFP4-PV | 12.598454 | 12.578766 | -0.019688 | +0.016901 |

These are twenty-update synthetic-token diagnostics, not pretraining
convergence evidence.  They are useful for detecting NaNs and gross route
breakage only.  The sub-percent FP8/MX timing decision and loss curves both
need longer replicated runs on the intended dataset.

## Provenance

- Source commit: `6530af2551984154bc5d97f6b76eb37c9dca1af8`
- Hardware: one NVIDIA GB200, SM100, 189,471 MiB reported HBM
- PyTorch: `2.9.0a0+145a3a7bda.nv25.10`; CUDA `13.0`
- Shared initial checkpoint SHA256:
  `3dc0e3dc151b4caeb70a3871f44b5ae73ba0aec577ea2752fac18e5b3cc9bdf1`
- BF16/FP8/MX JSON SHA256:
  `85d0c19234140c58505bb612f1534ded50c397ed578fdfc40f9f84c42540671d`,
  `86b985895378e81bc8f14ed7ecce3dd38f535879d4de3da70ebec007c4000ddc`,
  `a39ea99ac742df01e0acbe4b98d45aaee91ae6e52d33a56363dae416967c1cf5`
- FP8 forward image SHA256:
  `8c3ca848c6524347b99a69cb24b80c86d680b83e84c2e49c6e39be8b16118aba`
- MX forward image SHA256:
  `38d709f0b2b789664dee53dc1a9edec7bb8dc5000dadb8a47989a5b8a4faeb9d`
- Projection image SHA256:
  `58cedf0225ab368432d6474741d92c35a9e6f1f97a2dde1625d720db404b128f`

The full machine-readable outputs remain in
`/tmp/fa4_8b_b2_cute_20260829/results/` on the measurement host.  Two attempted
longer repeats changed the warm-up/update identity and were correctly rejected
by the harness's exact-reference gate; they are excluded from this receipt.
