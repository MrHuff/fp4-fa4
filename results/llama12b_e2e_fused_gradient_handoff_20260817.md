# Llama-1.2B fused gradient-handoff result

Date: 2026-08-17

## Outcome

The isolated FP4/FP8 gains propagate after fusing the dominant assembled-layer
boundary.  With BF16 and low precision alternated in the same process, the
full low-precision training step takes 78.021 ms versus 85.242 ms for BF16
CuTe: 1.0925x throughput and an 8.471% step-time reduction.

| Complete 1.2B step | BF16 CuTe | Fused low precision | Ratio |
|---|---:|---:|---:|
| Forward | 28.746 ms | 22.653 ms | 1.269x faster |
| Backward | 45.500 ms | 44.594 ms | 1.020x faster |
| Fused AdamW | 10.728 ms | 10.728 ms | 1.000x |
| Full step | 85.242 ms | 78.021 ms | 1.0925x throughput |
| Tokens/s | 48,051 | 52,498 | 1.0925x |
| Useful MFU at 2.25 PFLOP/s | 17.555% | 19.179% | +1.625 pp |

The comparison holds both 1,235,814,400-parameter models on GPU 0 and
alternates route order for eight measured rounds.  This removes the large
clock/device-state bias observed when all BF16 steps ran before all
low-precision steps.  Median AdamW times are equal, which is a useful check
that the two routes saw comparable device state.

## Boundary diagnosis and fusion

The pre-fusion assembled-layer profile measured approximately:

- low-precision attention backward: 358.6 us
- inverse RoPE + dV decode + loss-scale conversion + QKV concatenation:
  470.8 us
- QKV weight gradient: 146.0 us
- output weight gradient: 98.8 us

Thus the materialization chain after attention cost more than attention
backward itself.  The isolated kernel benchmarks excluded that chain because
they began and ended with already prepared operands.

The new D64 GQA handoff performs inverse RoPE, the direct-TMA dV x4 decode,
loss-scale conversion, and direct publication of the
`[all dQ | all dK | all dV]` projection-gradient matrix in one CUDA pass.
Against the previous PyTorch graph it is bit-for-bit identical and takes
65.0 us versus 492.8 us, a 7.58x local speedup.  Across 16 layers, removing
that boundary is what changes backward from slower than BF16 to slightly
faster than BF16.

## Numerical status

The handoff itself is exact relative to the previous BF16 materialization, so
it introduces no additional error.  The aggressive FP4 forward configuration
still has the previously measured convergence problem: on the repeated
synthetic batch, the low-precision loss falls much more slowly than BF16.
This establishes a real end-to-end performance win, but the forward
approximation/quantization policy still needs numerical work before making a
training-convergence claim.

## Artifacts

- Alternating measurement:
  `llama12b_e2e_interleaved_fused_gradient_handoff_s4096_20260817.json`
- Alternating harness:
  `../tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e_interleaved.py`
- Full model harness:
  `../tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py`
- Boundary profiler:
  `../tk_fa4/lowp_fa4_bwd/profile_llama12b_e2e_boundaries.py`

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:flash-attention \
python3 tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e_interleaved.py \
  --rounds 8 \
  --output \
    results/llama12b_e2e_interleaved_fused_gradient_handoff_s4096_20260817.json
```
