# FP4 FlashAttention experiment

This package is the portable integration boundary for causal FP4
FlashAttention in TorchTitan. It contains the exact fail-closed attention
adapter and BF16 comparator from the measured training integration, together
with the fused BF16 stochastic-rounding AdamW route and its checkpoint state.
Cluster submission, object-store synchronization, W&B integration, and CCE are
not part of this package.

## Intended design

The first supported target is Llama-style grouped-query attention with causal
masking, head dimension 128, sequence length 4096, and per-device batch size 4
on NVIDIA Blackwell SM100 GPUs.

The low-precision release candidate uses:

- NVFP4 Q/K in the attention score product, with row-by-K16 two-dimensional
  scales;
- E4M3 FP8 for the probability/value (P/V) product;
- native backward score replay with E4M3 Q/K/V inputs and E5M2 output gradients;
- NVFP4 learned Q/K/V/O projections; and
- BF16 FlashAttention as the matched control.

MXFP4 P/V with E8M0 block-32 scales is retained as a named diagnostic route
with the same NVFP4 learned projections used by the performance matrix.
Separate long diagnostics with E4M3 and NVFP4 learned projections both
diverged, so MXFP4 P/V is not a supported convergence recipe.

The artifact layer also exposes E4M3 learned-projection controls for both P/V
formats. Together with the two NVFP4 learned-projection routes, this preserves
the complete 2-by-2 diagnostic matrix. The E4M3 routes are reproducibility
controls, not additional release candidates.

## Integration rules

1. Keep format selection explicit in configuration and record it in every
   result receipt.
2. Validate tensor shape, layout, device, pointer lifetime, and runtime ABI at
   the Python boundary.
3. Stop on unsupported shapes.  Do not silently fall back to a different
   numerical method while reporting the requested route.
4. Keep unsafe implementations disabled until their correctness and liveness
   gates pass.
5. Separate kernel timing, complete forward-plus-backward timing, and end-to-end
   training throughput.  Each measures a different part of the system.
6. Keep data access, checkpoint storage, schedulers, and credentials outside
   this package.

## Modules

- `exact_lowp_attention.py`: exact NVFP4-QK/FP8-or-MX-PV adapter, learned
  projection integration, route authentication, and native backward wiring.
- `fa4_attention.py`: generic FA4 wrapper and authenticated BF16 comparator.
- `converters.py`: BF16 storage conversion, safe large-tensor initialization,
  the fused-MLP and native-activation stages used by the recovered SFU-B1
  chain, and the optional FP32-master parameter stage.
- `data.py` and `validator.py`: FA4-specific loader settings and fail-closed
  configuration validation.
- `optimizer/`: fused BF16 stochastic-rounding AdamW implementation, CUDA
  source, provider receipt, and checkpointed stochastic phase.
- `checkpoint.py`: opt-in Gloo metadata group and optimizer resume hook.
- `trainer.py`: opt-in checkpoint-aligned pinned-memory CUDA lookahead,
  nonfinite guards, and bounded per-parameter gradient diagnostics.
- `train.py`: the FA4-specific TorchTitan entry point used by rendered recipes.
- `job_config.py`: typed FA4 configuration extension.
- `train_spec.py`: the exact 1.2B/D64 and 8B/D128 Llama model geometries.

The adapter accepts only the measured sequence, head dimensions, local
batches, topology, conversion order, and binary identities. Nearby shapes do
not fall back silently. See `configs/fa4/README.md` for launch preparation and
`release/manifest.json` for source pins and remaining GPU validation blockers.
New configs use the format-neutral converter name `fa4_exact_lowp_attention`;
the older `fa4_exact_nvfp4_qk_fp8_pv` name remains registered only so archived
configs continue to load.

The 1.235B/D64 geometry, schema-v3 B16 artifact profiles, source builders, and
TorchTitan dispatch are present. They make a new source-built D64 run
configurable; they do not reproduce the unavailable historical `cd57` CuTe
control or establish a validated release route until clean-clone GB200
numerical/build, DDP16 save/fresh-resume, and long-horizon data gates pass.
