# Causal MXFP4 effective-range ablation

## Protocol

- GPU: GB200
- Shape: B1, S4096, Hq32, Hkv8, causal GQA
- Head dimensions: D64 and D128
- QK: NVFP4
- PV: MXFP4
- Seeds: 0, 1, 2, 3
- Reference: BF16 attention output

P4 selects an E8M0 amplitude targeting an E2M1 payload maximum of 4. The
`log2(1.5)` scale shift is folded into the existing E8M0 rounding offset, so
it adds no scale-selection instruction. Its one-`FFMA2` direct exp2 map is
refitted to the positive half of the `[-4, 4]` E2M1 range:

```text
max(0, 1.05184257 * x + 1.59894716)
```

The highest `3.5 -> 4` rounding boundary is constrained exactly, and the fit
does not reach the `5 -> 6` boundary over the intended normalized range.

V4 uses the same E8M0 target convention and explicitly caps payload magnitude
at code 4. The V6 implementation was checked byte-for-byte against the CUDA
quantizer for both payload and scale layout. V4 was checked to emit zero code-6
payloads.

## Four-seed accuracy

| D | P/V range | Cosine | Relative L2 | RMSE |
|---:|:---:|---:|---:|---:|
| 64 | 6/6 | **0.956097** | **0.295628** | **0.020189** |
| 64 | 4/6 | 0.953473 | 0.302160 | 0.020635 |
| 64 | 6/4 | 0.955778 | 0.298207 | 0.020365 |
| 64 | 4/4 | 0.953169 | 0.303965 | 0.020758 |
| 128 | 6/6 | **0.949346** | **0.320394** | **0.021890** |
| 128 | 4/6 | 0.947056 | 0.323379 | 0.022094 |
| 128 | 6/4 | 0.949170 | 0.323417 | 0.022097 |
| 128 | 4/4 | 0.946834 | 0.325620 | 0.022247 |

## Matched timing

Long seed-0 runs used 50 ms warmup and 300 ms measurement windows.

| D | P/V range | Time (ms) |
|---:|:---:|---:|
| 64 | 6/6 | 0.148224 |
| 64 | 4/6 | **0.148032** |
| 64 | 6/4 | 0.148480 |
| 64 | 4/4 | 0.148544 |
| 128 | 6/6 | 0.145408 |
| 128 | 4/6 | 0.145408 |
| 128 | 6/4 | 0.145408 |
| 128 | 4/4 | 0.145408 |

The D64 timing spread is measurement noise; P4 no longer carries the extra
addition present in the initial diagnostic implementation.

## V-only reconstruction audit

| D | V range | Code-6 count | Cosine | Relative L2 | RMSE |
|---:|:---:|---:|---:|---:|---:|
| 64 | 6 | 187391 | 0.991914 | 0.128766 | 0.128764 |
| 64 | 4 | 0 | 0.991605 | 0.129865 | 0.129863 |
| 128 | 6 | 373436 | 0.991908 | 0.128826 | 0.128759 |
| 128 | 4 | 0 | 0.991610 | 0.129810 | 0.129742 |

## Conclusion

The max-4 hypothesis does not improve this causal random-normal benchmark.
P4 and V4 each reduce accuracy independently, and P4/V4 compounds the loss.
The experiment is now a valid zero-overhead P ablation and a strict V payload
ablation, so the result is not explained by the stale max-6 affine fit or by
V payloads silently retaining code 6.
