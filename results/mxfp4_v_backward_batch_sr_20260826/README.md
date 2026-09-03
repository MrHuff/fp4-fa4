# MXFP4 backward-V batch and stochastic-rounding screen

This single-GB200 diagnostic tests whether the error from reusing forward
MXFP4 V in backward is rescued by larger physical batches or true stochastic
E2M1 rounding.  It uses one nested fixed 16-example bank, the production
E4M3 QKV projection publisher, causal GQA with Hq32/Hkv8/D64, S256, and a
readable exact attention Jacobian.  It is a numerical screen, not a kernel or
end-to-end timing result.

The reference is the direct projection-accumulator E4M3 V used by the deployed
FP8-PV and MXFP4-PV backward.  `native MX RNE` is the actual forward MXFP4 V
payload lifted to E4M3 by the production publisher.  `SR draw` QDQs direct
E4M3 V with the production 1x32 E8M0 selector and independently randomized,
unbiased E2M1 rounding.  The current backward ABI still consumes E4M3, so SR
is explicitly a decoded-MX-to-E4 numerical proxy rather than a packed-MX
kernel.

## Result

| Physical B | Native V rel-L2 | nonzero V zeroed | native-RNE dWq rel-L2 | native-RNE dWk rel-L2 | mean single-SR dWq rel-L2 | mean single-SR dWk rel-L2 | 16-key mean dWq rel-L2 | 16-key mean dWk rel-L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.14% | 8.92% | 10.85% | 10.86% | 15.37% | 15.45% | 4.44% | 4.46% |
| 2 | 11.15% | 8.92% | 10.91% | 10.86% | 15.46% | 15.42% | 4.46% | 4.47% |
| 4* | 11.12% | 8.93% | 10.77% | 10.69% | 15.31% | 15.25% | 4.36% | 4.35% |
| 8 | 11.12% | 8.92% | 10.78% | 10.75% | 15.23% | 15.25% | 4.37% | 4.37% |
| 16 | 11.11% | 8.90% | 10.77% | 10.73% | 15.25% | 15.28% | 4.36% | 4.37% |

`*` B4 is a diagnostic projection launch, not one of the production runtime's
authenticated batches.  B1/B2/B8/B16 are authenticated.  Every nested prefix
from B1 through B16 was byte-identical for Q, K, direct E4M3 V, and represented
MX V, so the flat error is not a batch-indexing defect.

Physical batching does not average the native-MX gradient error away: dWq/dWk
remain about 10.7--10.9% rel-L2 from B1 to B16.  One true-SR realization is
worse at about 15.3%.  Averaging the same fixed batch over 16 independent SR
keys reaches about 4.4%, confirming that the proxy is approximately unbiased
in expectation, but an ordinary optimizer step receives one realization, not
16 repeated backward evaluations.  SR also leaves the zeroed-nonzero fraction
near 9%; it randomizes which small values survive rather than eliminating
zeroing.

As required by the attention derivative, changing V affects dP, dS, dQ, dK,
and the Q/K projection gradients.  dV and the V-weight gradient are exactly
unchanged in the deterministic comparison.  The small ~1e-7 SR-mean dV
difference is only FP32 accumulation order while averaging 16 identical dV
tensors.

## Decision

Larger physical batches and naive V-only stochastic rounding do not justify
switching the deployed backward away from its shared direct-E4M3 V.  Keep the
same optimized backward for FP8-PV and MXFP4-PV.  Future native-MX backward
work should start from the repaired, representation-consistent P/V/stat
contract and a real packed-MX consumer; it should not be justified by the
earlier batch-1 failures or by this decoded proxy.

The create-only machine-readable receipt is
`mxfp4_v_backward_batch_sr_b1_b16.json`.  Reproduce it with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
TK_FA4_LOWP_BWD_EXTENSION_SOURCE=\
tk_fa4/_C_b300_lowp_bwd.cpython-312-aarch64-linux-gnu.so \
python -B tk_fa4/lowp_fa4_bwd/screen_mxfp4_v_backward_batch_sr.py \
  --gpu 0 --batches 1 2 4 8 16 --sequence 256 --hidden 512 \
  --sr-draws 16 --seed 20260826 \
  --expected-projection-extension \
    tk_fa4/_C_b300_lowp_bwd.cpython-312-aarch64-linux-gnu.so \
  --output \
    results/mxfp4_v_backward_batch_sr_20260826/\
mxfp4_v_backward_batch_sr_b1_b16.json
```
