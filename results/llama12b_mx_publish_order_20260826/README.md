# B16 split-V MX/E4 publication-order probe

This probe changes only the order of two existing projection-epilogue
consumers for the deployed split-V MX route.  Direct E4M3 backward V is issued
before forward MXFP4 V instead of after it.  It adds no synchronization,
restaging, allocation, dispatcher path, or new quantizer.  The intent is to
let the E4M3 stores drain while the independent MX gather/amax/scale/pack work
executes.

The fixed shape is the saturated Llama-1.2B attention boundary: B16, S4096,
H2048, Hq32/Hkv8, D64 on one GB200.  Each receipt uses 8 warmups and 400
balanced rotating-provider samples with the same seed and the same FP8/MX
forward extensions.

## Result

| Native-NVFP4 QKV boundary | baseline `08734ab` | reordered `abd96216` | change |
|---|---:|---:|---:|
| FP8 projection/publication mean | 851.217 us | 850.652 us | -0.565 us |
| MX projection/publication mean | 885.035 us | 881.093 us | -3.942 us |
| MX publication premium, mean | 33.818 us | 30.441 us | **-3.377 us** |
| MX publication premium, provider-p50 difference | 30.640 us | 26.752 us | **-3.888 us** |
| MX attention saving, mean | 87.633 us | 87.609 us | -0.024 us |
| MX prepared-boundary saving, mean | 51.276 us | 55.500 us | **+4.224 us** |
| MX prepared-boundary saving, provider-p50 difference | 54.048 us | 58.992 us | **+4.944 us** |

An independent 50,000-draw bootstrap over the paired per-cycle provider
differences gives a 95% interval of [-5.65, -1.13] us for the change in mean
MX publication premium (99.87% of draws favor the reorder).  The corresponding
prepared-boundary improvement is [1.79, 6.65] us (99.95% of draws favor the
reorder).  These intervals measure run-to-run sample uncertainty, not a claim
that device clock or software-environment drift is impossible.

The win is real but small.  About 30.4 us of MX publication premium remains;
the reorder does not remove the intrinsic causal gather, row amax, BF16-to-F32
scale, and E2M1 pack.  At 16 decoder layers, the measured component improvement
is only on the order of 0.05--0.07 ms per step before downstream overlap.

## Correctness and provenance

Both factorial receipts pass every BF16-output gate.  The dedicated split-V
validator additionally proves that the reorder leaves forward Q/K/MX-V and
backward Q/K byte-identical, while split backward V remains byte-identical to
the independent direct-accumulator E4M3 control.

- Baseline projection/backward binary: SHA256
  `08734ab13795d0182089ed523b3779375deda63a89473e9db10ce7360de3a576`,
  24,061,504 bytes.
- Reordered binary: SHA256
  `abd96216925acda6042df36dcb45dbacfdab24becff6f0a379911c176c775054`,
  23,995,968 bytes.
- FP8 forward: `88d81d3783e5aa80f0e9cf259a2ea7c935da4c2a5dc3ba1868e63f802a2c6208`.
- MX forward: `cc06fe4337fdc3a7c900f81d68fabc4a8e0c375ea536fbe6405754237a393717`.

Raw receipts:

- `baseline_08734ab_b16_factorial_n400.json`
- `candidate_abd962_b16_factorial_n400.json`
- `split_v_validator_abd962.json`

## Decision

Accept the reorder.  It is publication-byte neutral and statistically reduces
the target overhead, although it is not large enough to change the overall
training conclusion by itself.  The next gate is a fresh saturated optimizer-
step FP8/MX bracket using this exact binary; only that bracket may support an
end-to-end speed claim.
