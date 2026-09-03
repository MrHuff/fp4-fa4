# D128 shared-tile MXFP4 V publication (2026-08-30)

## Decision

The projection now quantizes each D32xS32 V tile once and publishes the same
E2M1 code matrix in the two physical orientations required by forward PV and
backward dP. A register-only 32x32 nibble transpose creates the row-major
backward payload, and the same four E8M0 tile anchors are written under the
forward and backward scale swizzles. The B2/S4096/H4096/Hq32/Hkv8/D128 GB200
gate found zero code, anchor, scale-replication, forward-reference, or
route-neutral Q/K mismatches.

The exact four-anchor v507 consumer is a correct negative performance result.
It matches its represented oracle (dQ cosine 0.999991, dK cosine effectively
1.0), preserves dV bitwise, and passes the exact-zero gate, but costs 581.632
us without clears versus 485.376 us for v501. Its earlier score launch recovers
about 39 us from v502, but the remaining scale-TMEM restage and alias schedule
still make it unsuitable for training.

The practical candidate is therefore explicit shared-tile publication plus
the v503 common-row consumer. v503 collapses the four D32 anchors to their
maximum during its existing shared-memory restage; it is not an exact
four-anchor consumer. Only 63,359 of 8,388,608 E2M1 codes changed in this
screen, and the resulting V remained at 0.999897 cosine to the direct
four-anchor decode. The represented-gradient oracle passed, dV remained
bitwise identical, and exact-zero dO produced exact-zero gradients.

In a self-conditioned ABBA/BAAB composed slice, retained E4M3 backward-V
publication plus v501 measured 758.256 us, while the shared one-quantization
producer plus v503 measured 719.872 us: a 38.384 us/layer raw saving and
1.0533x speedup.

The subsequent frozen-source 8B B2 full-model A/control/B gate passed. All
three routes used the same GB200, initial checkpoint, synthetic-token policy,
torch-compiled cut-cross-entropy, projection binary, three warmups, and twenty
measured updates. The two shared-MX arms averaged 433.422 ms p50 and 435.105 ms
sustained, versus 434.262 ms and 435.547 ms for FP8-PV + v501. This is a
1.00194x p50 and 1.00102x sustained whole-step speedup: repeatable, but too
small to call material.

| Route | Step p50 | Sustained | Decoder forward p50 | Backward p50 | Tokens/s |
|---|---:|---:|---:|---:|---:|
| NVFP4-QK + FP8-PV + v501 | 434.262 ms | 435.547 ms | 121.057 ms | 240.297 ms | 18,864.18 |
| NVFP4-QK + shared MXFP4-PV + v503, A/B mean | 433.422 ms | 435.105 ms | 118.805 ms | 240.889 ms | 18,900.74 |

Shared MX saves 2.252 ms in decoder forward but pays 0.592 ms in backward;
CE, clipping, optimizer time, and ordinary run variability dilute the result.
Every measured update and gradient was finite. Initial global gradient-norm
ratios versus the historical matched BF16 sample were 0.9207 for FP8 and
0.9306--0.9309 for shared MX. These are short synthetic diagnostics, not
pretraining convergence evidence. The full-model execution gate is closed;
real-data long training remains required.

## ABI boundary

The shared producer uses D32xS32 forward anchors, not the legacy rowwise
producer contract. Runtime selection must carry that scale-policy tag and
pair it explicitly with either:

- v507 for exact four-anchor diagnostics only; or
- v503 for the selected common-row approximation.

It must never silently infer the consumer from the common payload and scale
tensor shapes. The selected runtime additionally authenticates the exact
projection type and checked/unchecked callables, requires the exact first-use
authenticated caller workspace, retains that workspace while bound, and
rejects cross-runtime backward sharing for this producer-specific ABI.

The durable summary is
`shared_tile_mx_gate_summary_20260830.json`. It records the authenticated
binary identities and SHA-256 identities of the complete ephemeral raw
receipts.
