# B16 causal QKV-projection x PV-format forward factorial (2026-08-26)

This experiment isolates the forward boundary that was ambiguous in the
saturated end-to-end runs. It crosses two learned QKV projection formats
(E4M3 and native NVFP4) with two causal PV formats (E4M3 FP8 and MXFP4) at the
Llama-1.2B B16/S4096/Hq32/Hkv8/D64 shape on one GB200.

The same deterministic BF16 post-RMSNorm activation and packed learned-QKV
weight are used by all four cases in a process. Projection binders are
authenticated before timing and then use unchecked out-parameter symbols.
The measured forward executes no backward kernel, but every projection route
publishes the training-compatible Q/K/V operands required by the represented
backward path.

Projection publication workspaces, attention output, and LSE are caller-owned.
This establishes a caller-owned CUDA output API contract for the projection,
attention, and prepared-combined boundaries; transient allocator activity was
not traced, so allocation freedom is not claimed. Operand preparation uses
allocating functional APIs, and the full-combined boundary includes those
calls.

## Two-run result

Two independent processes used seeds 20260826 and 20260827. Each process ran
eight warm-ups and 400 CUDA-event samples per case per stage, with provider
position rotated and balanced. Values below are the arithmetic mean of the
two per-process device-time means.

| QKV projection | PV | Prep (us) | Projection + publication (us) | Attention (us) | Prepared projection + attention (us) | Full combined (us) |
|---|---|---:|---:|---:|---:|---:|
| E4M3 | FP8 | 136.259 | 937.295 | 1,939.388 | 2,874.029 | 3,009.301 |
| E4M3 | MXFP4 | 136.371 | 1,039.040 | 1,853.578 | 2,892.317 | 3,023.696 |
| native NVFP4 | FP8 | 157.051 | 848.927 | 1,939.936 | 2,787.056 | 2,944.768 |
| native NVFP4 | MXFP4 | 152.542 | 886.553 | 1,853.674 | 2,737.832 | 2,894.887 |

The causal interpretation is direct:

- MX attention is 85.8--86.3 us faster than FP8 attention regardless of QKV
  projection format.
- With the old E4M3 QKV path, MX publication costs 101.7 us more and overcomes
  that attention saving. The prepared pair is 18.3 us slower than FP8.
- With native NVFP4 QKV, MX publication costs only 37.6 us more, so the
  prepared pair is 49.2 us faster than FP8, a 1.01798x speed ratio.
- Native NVFP4 + MXFP4-PV is 1.05643x faster than E4M3 + MXFP4-PV and 1.04975x
  faster than E4M3 + FP8-PV at the prepared boundary.
- Including the current allocating preparation APIs, native NVFP4 + MXFP4-PV
  remains 1.01723x faster than native NVFP4 + FP8-PV and 1.03952x faster than
  E4M3 + FP8-PV.

Both runs preserve the ordering: the native MX-versus-FP8 prepared deltas are
-48.949 and -49.497 us, while the E4M3 deltas are +20.207 and +16.369 us.
Peak allocation/reservation was 4.725/5.707 GiB in each process.

## Numerical gate

An untimed BF16 learned-QKV projection plus causal PyTorch SDPA reference uses
the same rows, learned weight, and RoPE as the low-precision cases. The harness
fails closed unless every output is finite, cosine is at least 0.95, and
relative L2 is at most 0.35. All four cases passed in both processes.

| QKV projection | PV | Mean cosine vs BF16 | Mean relative L2 vs BF16 | Gate |
|---|---|---:|---:|---|
| E4M3 | FP8 | 0.992963 | 0.118483 | pass |
| E4M3 | MXFP4 | 0.980037 | 0.198823 | pass |
| native NVFP4 | FP8 | 0.972752 | 0.232608 | pass |
| native NVFP4 | MXFP4 | 0.960384 | 0.279146 | pass |

The pairwise low-precision diagnostics are also finite. Across the two seeds,
MX-versus-FP8 attention-output cosine is approximately 0.986 for both QKV
formats, with relative L2 approximately 0.167. Native-NVFP4-versus-E4M3 QKV
output cosine is approximately 0.967 with FP8-PV and 0.947 with MXFP4-PV. LSE
cosine exceeds 0.999994 in every pairwise comparison. These gates establish
bounded one-layer forward error for this deterministic input; they do not
establish pretraining convergence.

## Provenance

The projection/backward extension is 22,903,528 bytes with SHA256
`c3b3ba4e1c19d37d1ebc441d0487ca898035bc5cbcbc7422007e8c022df6a3d6`.
The FP8-PV and MXFP4-PV forward extensions have SHA256
`88d81d3783e5aa80f0e9cf259a2ea7c935da4c2a5dc3ba1868e63f802a2c6208`
and `cc06fe4337fdc3a7c900f81d68fabc4a8e0c375ea536fbe6405754237a393717`.
All three loaded paths and post-load hashes matched their preload receipts in
both processes.

The final same-GPU raw 400-sample JSON files are:

- `fa4_b16_forward_factorial_v2_samegpu_s400_seed20260826.json`: 844,797
  bytes, SHA256
  `51e9ad10862cedc0baa5f44d18bf97d44318011e9085c6a5a460885fe9286b2b`.
- `fa4_b16_forward_factorial_v2_samegpu_s400_seed20260827.json`: 844,768
  bytes, SHA256
  `7d9945c1856f7734c9bd63f38e88df60cf37c7fe2e54b662818b344d09704a04`.

The compact machine-readable summary is
`results/llama12b_b16_forward_factorial_20260826.json`.

Reproduce with `benchmark_b16_forward_factorial.py`, the authenticated
extensions above, `--warmups 8 --samples 400`, the recorded correctness
thresholds, and one of the recorded seeds.
