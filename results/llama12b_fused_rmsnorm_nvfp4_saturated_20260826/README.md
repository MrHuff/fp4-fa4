# Fused RMSNorm to native NVFP4 at saturated Llama-1.2B

This result closes the forward-boundary issue identified in
`results/llama12b_b16_forward_factorial_20260826.md`.  At the exact
B16/S4096/H2048/D64 shape, attention RMSNorm and native-NVFP4 activation
preparation are now one explicit route, and RMSNorm backward no longer
materializes full-sized FP32 intermediates through eager PyTorch autograd.

The headline result is a **1.1109x p50 full-step speedup** for the fused
MXFP4-PV route relative to the otherwise matched unfused MXFP4-PV route.
After this fix, fused MXFP4-PV and fused FP8-PV have essentially identical
whole-step throughput; MX retains the expected small decoder-forward lead.

## What the format names mean

“E4M3,” “NVFP4 QK,” and “FP8-PV” describe different boundaries and must not
be treated as mutually exclusive names for a complete training route.

- Both fused routes use native E2M1 NVFP4 activation and learned-weight
  operands for the QKV projection, row-by-K16 activation scales, true 16-by-16
  learned-weight scales, and native-NVFP4 Q/K forward publication.
- The MX route consumes causal-interleaved `mxfp4_e8m0_block32` V in the
  forward P x V contraction.
- The FP8 route consumes E4M3 V in the forward P x V contraction.
- MX additionally publishes projection-accumulator E4M3 V for backward.  The
  FP8 route can use its E4M3 publication for the same backward boundary.
- Represented NVFP4 Q/K values are carried in E4M3-typed containers for
  backward; the container type does not turn those represented values into an
  E4M3 forward-QK route.

The earlier “E4M3 MX” slowdown was therefore a composite-path problem.  The
isolated factorial measured MX attention 85.8--86.3 us faster than FP8, but
the old E4M3 QKV publisher added 101.7 us for MX and erased that gain.  The
native-NVFP4 publisher reduced that incremental publication cost to 37.6 us,
leaving the prepared native-NVFP4+MX pair 49.2 us faster than native
NVFP4+FP8.

QKV projection, RoPE, Q/K publication, and V publication were already fused
inside the projection epilogue.  This change addresses the remaining boundary
immediately before that GEMM.

## Implementation

The opt-in `--experimental-fused-attention-rmsnorm-nvfp4` route is deliberately
fail-closed to B16, S4096, H2048, D64, caller-owned native-NVFP4 QKV
publication, and 2D learned-weight scaling.

Forward uses two asynchronous CUDA stages:

1. one CTA per row computes RMSNorm, retains `inv_rms`, publishes the BF16
   normalized rows required by QKV weight-gradient calculation, and reduces a
   matrix-wide amax; and
2. the native TMA packer produces the exact E2M1 payload, row-by-K16 E4M3 scale
   pages, and global decode scalar from that BF16 publication.

Backward uses one H2048-specialized pass to compute BF16 `dx` and per-CTA FP32
`dgamma` partials across 16 rows, followed by a deterministic small reduction
of the partials.  It uses no atomics, stream synchronization, or full MxH FP32
temporary matrices.

## Isolated component gate

`results/fused_rmsnorm_nvfp4_component_20260826.json` is a create-only,
authenticated two-seed result.  Each provider used eight warmups and 100
rotating/interleaved CUDA-event samples on one GB200.  The control is eager
FP32 RMSNorm followed by the same exact native-NVFP4 packer; backward uses the
closed-form FP32 derivative and the identical saved `inv_rms`.

| Boundary | Eager mean / p50 (us) | Fused mean / p50 (us) | Mean / p50 speedup |
|---|---:|---:|---:|
| Forward, seed 20260826 | 1525.39 / 1522.42 | 293.31 / 291.60 | 5.201x / 5.221x |
| Forward, seed 20260827 | 1524.95 / 1524.19 | 294.52 / 291.55 | 5.178x / 5.228x |
| Backward, seed 20260826 | 3043.85 / 3041.57 | 482.68 / 480.67 | 6.306x / 6.328x |
| Backward, seed 20260827 | 3044.87 / 3044.27 | 484.20 / 482.13 | 6.288x / 6.314x |

The backward control in this component table is the explicit closed-form
formula, not a timed production BF16 autograd graph.  It isolates the
mathematical RMSNorm boundary with identical saved state.  The saturated
comparison below is the evidence for the production custom-autograd path.

For both seeds, packing the fused BF16 normalized publication separately
produced byte-identical NVFP4 payload, scale pages, and global decode.  The
normalized publication has relative L2 8.84e-6 and 1.01e-5 versus eager;
`inv_rms` relative L2 is below 3.6e-8.  Backward `dx` relative L2 is 3.35e-6
and 3.04e-6.  `dgamma` is bitwise exact for one seed and has relative L2
1.67e-7 for the other.  Every finite/accuracy gate passed.

The component JSON is 105,507 bytes with SHA256
`399356556026d7406ad5a5a29f194588be9ba71305ba9c0d8f1f8b942f0511c0`.
The final exact-shape-guarded extension used by that component run is
22,976,552 bytes with SHA256
`09198a39ccdde33ffa119dcacef12113d6caaac45c5cea1be4c2966808170eac`.
The harness records clean implementation commit
`40a17150dcd5f0ce48eebc148f456dd005113f93` and its own SHA256
`c25c30e15ca9070d78df4532affe3a2b18e63aaa0ab17016dafa90407f109d2f`.

## Saturated end-to-end result

All runs use one GB200, the same 16-layer Llama-1.2B checkpoint, the same
packed Dolma token stream, B16 x S4096 (65,536 tokens/update), torch-compiled
MLCE, three warmups, and 20 measured updates.  Two independent processes were
run per route.  The MX comparison order was U1-F1-F2-U2 so the fused pair is
bracketed by unfused controls.  The table pools 40 measured updates per route.

| Route | Step mean / p50 (ms) | Decoder mean / p50 | Backward mean / p50 | tok/s mean / p50 | Useful MFU mean / p50 | Peak alloc / reserved (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| MX unfused | 598.950 / 599.423 | 173.262 / 173.042 | 389.551 / 389.734 | 109,419 / 109,332 | 39.974% / 39.943% | 155.529 / 167.869 |
| MX fused | 541.139 / 539.578 | 148.253 / 147.910 | 356.529 / 355.091 | 121,120 / 121,458 | 44.249% / 44.373% | 135.527 / 147.617 |
| FP8 fused | 540.928 / 539.186 | 148.791 / 148.525 | 355.302 / 353.911 | 121,166 / 121,546 | 44.266% / 44.405% | 135.527 / 147.648 |

The useful-MFU column is the harness's BF16-equivalent arithmetic estimate
with a 2.25-PFLOP/s denominator.  It is not profiler-measured SM utilization.

Fusing the MX route saves 57.811 ms at the pooled mean and 59.845 ms at p50,
for 1.1068x and 1.1109x step ratios.  Mean decoder forward falls by 25.009 ms
and mean backward by 33.022 ms.  The backward reduction is expected: the
unfused graph leaves RMSNorm backward to eager PyTorch, while the fused graph
uses the new two-stage CUDA RMSNorm backward.  Low-precision FA4 backward is
unchanged.

Allocated HBM falls by 20.001 GiB and reserved HBM by 20.252 GiB.  The two
unfused p50 repeats differ by 0.023%; the fused repeats differ by 0.053%.

## MXFP4-PV versus FP8-PV

With the same fused RMSNorm and native-NVFP4 QK preparation, MX decoder
forward is 0.538 ms faster at the pooled mean (0.615 ms at p50).  The whole
step is nevertheless a tie: MX is 0.211 ms slower at the mean and 0.392 ms at
p50, only 0.039% and 0.073%, respectively.  The per-run mean difference
changes sign.

The serialized backward-contract objects are byte-for-byte identical, with
SHA256 `18e6f8e38125fb007edd92c442860883908bae09c1144477c0070c4a6bcc9528`.
The 1.227-ms pooled aggregate backward delta is within repeat variation and
does not establish route-specific backward work.  The defensible conclusion
is **MX forward is faster; current end-to-end throughput is tied**.

## Numerical scope

All 138 warmup and measured records are finite.  The fixed BF16 reference
moves from heldout loss 12.166168 to 8.049294.  The two fused MX runs finish at
7.968902 and 7.967290; the fused FP8 runs finish at 8.150694 and 8.122597.
Those are only 23 optimizer updates per run, including warmups, and do not
establish a convergence or accuracy ranking.

At the shared initial state, fused MX versus fused FP8 has sampled-logit cosine
0.852680 and relative L2 0.54289.  FP8 is also closer to BF16 at initialization
than MX.  MX's lower short-run loss is encouraging, but it is not evidence that
MX is the more accurate long-run route.  A long pretraining run remains the
appropriate numerical test.

## Provenance

All six raw JSON documents are retained in this directory.  Their hashes are:

- `mx_unfused_u1.json`: `5e51a5c315f85520c31e014392a0e1311f405abf6b9344fa4dfb8a13b9428c04`
- `mx_unfused_u2.json`: `f0de81bb45b6c9f4de38cd201ad964719e9c76ef12e71e13f7e7da8b04ec2f33`
- `mx_fused_f1.json`: `cc375775ae88f8d68d32ff970f752ad562d111cbc943cf87078e8c6f24897741`
- `mx_fused_f2.json`: `83479c63b102b47c30557d6784cc80004d2d88d96fd10360a19d0838c41961f1`
- `fp8_fused_p1.json`: `54d0884874787198ac0b425d1f09cccf6ad0e109d600f167376b9981d012f965`
- `fp8_fused_p2.json`: `aa79b26f57eea5b298250ac2d7a86638d9facee4dc8d8c42f4252546ef7db953`

The common checkpoint SHA256 is
`2760f5eb47fd0241317dfd69bd0e2d906909d948d81a5a93f0fd371944f0d2bc`;
the packed token stream identity is
`0e7c735ad8794429330a23dada1a2cd26d3abe955ce4c46d31e40e161c55fd16`.
The six saturated runs used the pre-guard fusion/projection extension at
22,976,216 bytes, SHA256
`63bb4a1f5092592a7e837ee8e3a9d6795f5c6970960e47f59fddfdc46d962f56`;
the only later source change tightens its host-side shape guard.
The MX and FP8 forward extensions have SHA256
`cc06fe4337fdc3a7c900f81d68fabc4a8e0c375ea536fbe6405754237a393717`
and `88d81d3783e5aa80f0e9cf259a2ea7c935da4c2a5dc3ba1868e63f802a2c6208`.
The common backward control has SHA256
`cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1`.

No external profiler counters were collected for this final bracket.  The raw
results retain sampled final tensors but no complete final checkpoint, and
torch-compiled MLCE records logical full logits without proving logit-buffer
elision.

## Reproduce the isolated component

Build the extension serially from `tk_fa4/lowp_fa4_bwd`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 -B \
  tk_fa4/lowp_fa4_bwd/benchmark_fused_rmsnorm_nvfp4.py \
  --extension-source /absolute/path/to/_C_b300_lowp_bwd.so \
  --extension-sha256 <sha256> --extension-bytes <bytes> \
  --output /new/path/fused_rmsnorm_nvfp4_component.json
```

The harness refuses to overwrite an existing output, authenticates the
extension before import, after import, and after timing, gates on SM100, and
returns status 2 after writing the evidence if any correctness check fails.
