# Kernel and experiment source map

This document maps the paper's reader-facing method names to the exact source
that implements them. It is intentionally stricter than a directory tour: the
non-causal forward results and the causal-training results come from different
source epochs, and the two epochs must not be substituted for one another.

The source pins and tree hashes are recorded in
`release/SOURCE_PROVENANCE.md`. The commands and availability of external
inputs are recorded in `release/EXPERIMENT_MATRIX.md`. This file answers the
narrower question: **which code implements each mathematical and experimental
route?**

Status terms used below:

- **Release route**: included in the portable build and authenticated
  TorchTitan adapter.
- **Operator-only**: the kernel and measurement path are present, but the
  TorchTitan training manifest does not admit that exact shape.
- **Historical source snapshot**: the source that produced an older paper
  result is preserved separately from the final causal tree.
- **Diagnostic**: retained to explain an ablation or rejected design; it is
  not selected by the release recipe.
- **Disabled**: source is preserved, but public dispatch stops before entering
  the implementation because a correctness, liveness, or performance gate was
  not cleared.

## Routes at a glance

| Reader-facing route | Attention operands | Backward | Supported use |
| --- | --- | --- | --- |
| **Direct-P (Fast)** | NVFP4 Q/K and MXFP4 P/V; direct approximate conversion from normalized log scores to E2M1 probability codes | None | Historical non-causal inference evaluation |
| **Direct-P (Accurate)** | The same formats with a row anchor and represented-probability normalization | None | Historical non-causal inference evaluation |
| **HAO FP8-P/V comparator** | HAO's NVFP4 Q/K and E4M3 FP8 P/V | None | Pinned non-causal comparator |
| **Causal FP8-P/V** | NVFP4 Q/K with row-by-K16 two-dimensional scales and E4M3 FP8 P/V | Reconstructs scores from saved quantized Q/K, scales, and log-sum-exp | Release candidate at B1 and B4; B2 is operator-only |
| **Causal MXFP4-P/V** | The same NVFP4 Q/K and MXFP4/E8M0-block32 P/V | The same backward, using a separately published E4M3 V view | Diagnostic training route at B1 and B4; B2 is operator-only |
| **E4M3-Projection FP8-P/V** | E4M3 learned Q/K/V/O projections, then the Causal FP8-P/V attention route | Reconstructs scores from saved quantized Q/K, scales, and log-sum-exp | Projection-precision control at B1 and B4; B2 is operator-only |
| **E4M3-Projection MXFP4-P/V** | E4M3 learned Q/K/V/O projections, then the Causal MXFP4-P/V attention route | The same backward from saved quantized operands | Preserved divergence diagnostic at B1 and B4; B2 is operator-only |
| **BF16 FA4 control** | BF16 Q/K/V | BF16 FA4 | Release control |

The four composed low-precision controls cross two learned-projection formats
with two attention P/V formats. They intentionally use the **same backward
kernel**. Changing projection or P/V format does not select a different
backward schedule in the release recipe. The MXFP4 forward route publishes an
E4M3 V view for backward; the experimental attempts to consume MXFP4 V
directly are listed separately below.

## Historical non-causal forward

The paper's non-causal operator grid, fixed-input model evaluations, and
Direct-P conclusions must be traced to the historical snapshot below, not to
the newer root `tk_fa4` tree.

### Direct-P (Fast and Accurate)

The complete historical kernel source is under
`reproduction/snapshots/forward_cfc06dad`. The main implementation closure is:

| Role | Exact source |
| --- | --- |
| Build configuration | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/Makefile.hao_direct_fp4pv` |
| Translation unit | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_candidate.cu` |
| Compile-time route and layout | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_config.inc` |
| QK, pipeline, P publication, and PV mainloop | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_kernel.inc` |
| Direct probability-code construction | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_softmax_reader.inc` |
| Host and Python bindings | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_host.inc` |
| Shared host checks and tensor plumbing | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/shared_host_helpers.inc` |
| Shared score/PV helpers | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/fwd_configs.inc`; `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/fwd_device_helpers.inc`; `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/stage2_ex2_alu_helpers.cuh`; `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/depth1_upstream_mxfp4_fp8pv_kernel.inc` |
| BF16 reference included in the same extension | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/upstream_mxfp4_fp8pv_bf16_baseline.inc` |
| MXFP4 V preparation | `reproduction/snapshots/forward_cfc06dad/TK_quantisation/mxfp4_v3/Makefile`; `reproduction/snapshots/forward_cfc06dad/TK_quantisation/mxfp4_v3/tk_quantize.cu`; `reproduction/snapshots/forward_cfc06dad/TK_quantisation/mxfp4_v3/mxfp4_v3_quantize.cuh` |
| Single-extension benchmark | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_benchmark.py` |
| Matched shape-grid build, run, and receipt writer | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_comprehensive_suite.py` |

The `nvmx-fast` and `nvmx-accurate` selections in
`hao_comprehensive_suite.py` are the executable definitions of the two public
policies. This is preferable to reconstructing their many compile-time flags
from prose.

### HAO comparators

The HAO implementation is vendored at a pinned revision under
`third_party/hao_flash_attention_fp4`. Its forward comparator closure is:

| Role | Exact source |
| --- | --- |
| HAO SM100 FP4/FP8 forward kernel | `third_party/hao_flash_attention_fp4/flash_attn/cute/flash_fwd_sm100_fp4.py` |
| HAO public dispatch | `third_party/hao_flash_attention_fp4/flash_attn/cute/interface.py` |
| HAO input factory and benchmark utilities | `third_party/hao_flash_attention_fp4/flash_attn/cute/benchmarks/bench_fp4.py` |
| Matched HAO BF16 and NVFP4-QK/FP8-PV runner | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_native_reference_benchmark.py` |
| Compatibility changes applied to the vendored source | `patches/hao_flash_attention_fp4_9b0abef_compat.patch` |

The Direct-P suite passes the same HAO-created tensors to the local kernels and
these comparator paths. HAO's source is a comparator, not a hidden dependency
of the Direct-P CUDA extension.

### Non-causal model and reconstruction experiments

These drivers consume extensions built by the historical shape-grid suite.
They do not define a second copy of the attention kernel.

| Paper experiment | Measurement and validation source |
| --- | --- |
| ViT classification, BERT masked-language modeling, and SST-2 | `tk_fa4/fp4_fa4_fwd/downstream_provider_suite.py`; `tk_fa4/fp4_fa4_fwd/eval_regular_attention.py`; `tk_fa4/fp4_fa4_fwd/eval_bert_mlm_attention.py`; `tk_fa4/fp4_fa4_fwd/eval_bert_sequence_classification.py` |
| ViT-MAE reconstruction | `tk_fa4/fp4_fa4_fwd/eval_vit_mae_reconstruction.py` |
| Wan2.1 fixed-input evaluation and route calibration | `tk_fa4/fp4_fa4_fwd/build_wan_nv_mx_bundle.py`; `tk_fa4/fp4_fa4_fwd/eval_wan_video.py`; `tk_fa4/fp4_fa4_fwd/eval_wan_affine_routes.py`; `tk_fa4/fp4_fa4_fwd/validate_wan_joint_routes.py` |
| GB200 summary analysis | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/analyze_hao_comprehensive.py` |
| B300 forward aggregation | `results/fp4_fa4_b300_tuning_20260802/build_summary.py`; kernel sources are the root `tk_fa4/fp4_fa4_fwd/Makefile.hao_direct_fp4pv` and its `hao_direct_fp4pv_*` closure |

The public model and dataset assets needed by these drivers are enumerated in
`release/EXPERIMENT_MATRIX.md`. Some historical Wan and B300 raw inputs are not
in the repository, so those old byte-identical acquisitions remain
receipt-only even though the kernel and analysis source are present.

### Historical backward prototypes from the same epoch

The `cfc06dad` snapshot now preserves the backward work that existed beside
the non-causal forward study. Its entry point is
`reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_bwd/fp4_fa4_bwd.cu`,
its build file is the adjacent `Makefile`, and the implementation is split
across the 10 `bwd_*.inc` files in that directory. The shared local closure is
`b300_bwd_fa4.cuh`, `b300_causal/bf16_b300_mha_causal_fp4.cu`, and
`b300_common.cuh`. The 12-file subtree is byte-exact to historical Git tree
`dd35ecca9db03cdbb063af7e4b3762438b9d5cca`.

This family is preserved for source continuity. It has no surviving portable
receipt that supports a result in the paper, and it must not be substituted
for the later backward implementation that reconstructs scores from saved
quantized Q/K, scales, and log-sum-exp. Its explicit catalog entry
is `historical_cfc06dad_backward_prototypes`; intermediate revisions in the
later native-backward tree are accounted for by
`release/legacy_backward_makefiles.txt`.

## Causal D128 forward

The causal source is the root `tk_fa4` export. Both routes implement causal
grouped-query attention at S4096/Hq32/Hkv8/D128 and are compiled separately for
local batches 1, 2, and 4.

### Causal FP8-P/V

| Role | Exact source |
| --- | --- |
| Isolated builder | `tk_fa4/lowp_fa4_bwd/build_causal_gqa_fp8pv_forward.py` |
| Causal/GQA source transformation | `tk_fa4/lowp_fa4_bwd/causal_gqa_fp8pv_forward.patch` |
| Base build configuration | `tk_fa4/fp4_fa4_fwd/Makefile.hao_direct` |
| Translation unit | `tk_fa4/fp4_fa4_fwd/hao_direct_candidate.cu` |
| Route configuration | `tk_fa4/fp4_fa4_fwd/hao_direct_config.inc` |
| QK, causal pipeline, P publication, and FP8 PV | `tk_fa4/fp4_fa4_fwd/hao_direct_kernel.inc`; `tk_fa4/fp4_fa4_fwd/hao_direct_softmax_reader.inc` |
| Host and Python bindings | `tk_fa4/fp4_fa4_fwd/hao_direct_host.inc`; `tk_fa4/fp4_fa4_fwd/shared_host_helpers.inc` |
| Shared device helpers | `tk_fa4/fp4_fa4_fwd/fwd_configs.inc`; `tk_fa4/fp4_fa4_fwd/fwd_device_helpers.inc`; `tk_fa4/fp4_fa4_fwd/stage2_ex2_alu_helpers.cuh`; `tk_fa4/fp4_fa4_fwd/depth1_upstream_mxfp4_fp8pv_kernel.inc` |
| Clean release build | `tools/build_fa4.py` with target `fp8-forward` |
| Runtime route and autograd composition | `torchtitan/experiments/fa4/exact_lowp_attention.py` |
| Binary and source authentication | `torchtitan/experiments/fa4/artifacts.py` |
| Isolated forward checks | `tk_fa4/lowp_fa4_bwd/benchmark_causal_forward_boundaries.py`; `tk_fa4/lowp_fa4_bwd/validate_causal_gqa_fp8pv_batch.py` |
| Source/ABI regression checks | `tests/test_causal_gqa_fp8pv_forward_patch.py`; `tests/test_causal_forward_boundaries.py`; `tests/test_causal_forward_fixed_route_fastpath.py` |

The builder copies the base forward tree to an isolated temporary directory,
applies the checked patch there, and compiles the result. It does not edit the
checked-in base source.

### Causal MXFP4-P/V

| Role | Exact source |
| --- | --- |
| Isolated allowlisted builder | `tk_fa4/lowp_fa4_bwd/build_causal_gqa_d128_mxfp4pv_forward.py` |
| Base build configuration | `tk_fa4/fp4_fa4_fwd/Makefile.hao_direct_fp4pv` |
| Translation unit | `tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_candidate.cu` |
| Route configuration | `tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_config.inc` |
| QK, causal pipeline, direct P publication, and MXFP4 PV | `tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_kernel.inc`; `tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_softmax_reader.inc` |
| Host and Python bindings | `tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_host.inc`; `tk_fa4/fp4_fa4_fwd/shared_host_helpers.inc` |
| Standalone MXFP4 preparation | `TK_quantisation/mxfp4_v3/Makefile`; `TK_quantisation/mxfp4_v3/tk_quantize.cu`; `TK_quantisation/mxfp4_v3/mxfp4_v3_quantize.cuh` |
| Clean release build | `tools/build_fa4.py` with targets `mxfp4-quantizer` and `mx-forward` |
| Runtime route and autograd composition | `torchtitan/experiments/fa4/exact_lowp_attention.py` |
| Binary and source authentication | `torchtitan/experiments/fa4/artifacts.py` |
| Isolated authentication and timing | `tk_fa4/lowp_fa4_bwd/authenticate_causal_gqa_mxfp4pv_forward.py`; `tk_fa4/lowp_fa4_bwd/benchmark_causal_forward_boundaries.py` |
| Source/ABI regression checks | `tests/test_mxfp4_forward_artifact_authentication.py`; `tests/test_causal_forward_boundaries.py`; `tests/test_d128_forward_publication.py` |

The release builder selects the safe 32-row anchor and represented-denominator
policy. Faster unsafe policies remain in the source for analysis but are not
selected. This route is a throughput and numerical diagnostic: the observed
longer 8B training trajectories diverged, so it is not advertised as a
convergent pre-training recipe.

## Projection and operand publication

Both causal routes use one projection extension. It fuses learned QKV
projection output handling, rotary positional embedding, row-by-K16 NVFP4 Q/K
publication, backward-oriented E4M3 Q/K/V publication, optional MXFP4 V
publication, and E5M2 output-gradient/statistic publication.

| Role | Exact source |
| --- | --- |
| Build configuration | `tk_fa4/lowp_fa4_bwd/Makefile` |
| Translation unit and Python bindings | `tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu` |
| Projection epilogue and publication ownership | `tk_fa4/lowp_fa4_bwd/projection_fp4_epilogue.cuh` |
| D128 row-by-K16 QKV quantization | `tk_fa4/lowp_fa4_bwd/gqa_d128_hierarchical_qkv_nvfp4_quantize.cuh`; `tk_fa4/lowp_fa4_bwd/hierarchical_qkv_nvfp4_quantize.cuh` |
| Fused inverse-RoPE handoff | `tk_fa4/lowp_fa4_bwd/inverse_rope_nvfp4_quantize.cuh` |
| Fused RMSNorm/activation publication helper | `tk_fa4/lowp_fa4_bwd/rmsnorm_nvfp4_quantize.cuh` |
| Gradient-to-projection consumer | `tk_fa4/lowp_fa4_bwd/dq_projection_consumer.cuh` |
| Experimental E4M3-to-MXFP4 V conversion | `tk_fa4/lowp_fa4_bwd/e4m3_to_mxfp4_v.cuh` |
| Underlying TK NVFP4 primitive | `ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_quantize.cuh` |
| Clean release build | `tools/build_fa4.py` with target `projection-publisher` |
| TorchTitan integration | `torchtitan/experiments/fa4/exact_lowp_attention.py` |
| Reference math | `tk_fa4/lowp_fa4_bwd/projection_quantization_reference.py` |
| GPU validation | `tk_fa4/lowp_fa4_bwd/validate_projection_fp4_epilogue.py`; `tk_fa4/lowp_fa4_bwd/validate_unified_qkv_projection_rope.py`; `tk_fa4/lowp_fa4_bwd/validate_gqa_d128_projection_boundaries.py`; `tk_fa4/native_gqa_tk_bwd/validate_v509_fused_projection_publisher.py` |
| Source/ABI regression checks | `tests/test_projection_quantization_reference.py`; `tests/test_d128_materialized_projection_batch.py`; `tests/test_d128_native_nvfp4_qkv_out_abi.py`; `tests/test_projection_mxfp4_v_epilogue.py`; `tests/test_v509_e5m2_dout_fused_publication.py` |

The projection extension is part of the timed projection-inclusive and
end-to-end boundaries. Standalone attention-core timing prepares these operands
before the timer by design.

## Backward from saved quantized operands

The backward pass reconstructs scores from the exact NVFP4 Q/K payload, its
row- and column-oriented scales, and the saved log-sum-exp statistic. It uses
represented E4M3 Q/K/V for gradient matrix products and E5M2 for dO, whose
wider exponent range avoids the severe small-gradient underflow observed with
E4M3 dO.

| Role | Exact source |
| --- | --- |
| Internal source label | `v509` |
| Exact-batch build files | `tk_fa4/native_gqa_tk_bwd/Makefile.v509`; `tk_fa4/native_gqa_tk_bwd/Makefile.v509_b2`; `tk_fa4/native_gqa_tk_bwd/Makefile.v509_b4` |
| Translation unit | `tk_fa4/native_gqa_tk_bwd/v509_d128_gqa_nvfp4_score_e4m3_qkv_e5m2_dout_b1_exact_s4096_experimental_bshd.cu` |
| Mainloop and epilogue | `tk_fa4/native_gqa_tk_bwd/v509_d128_gqa_nvfp4_score_e4m3_qkv_e5m2_dout_b1_exact_s4096_experimental_bshd.cuh` |
| E4M3-by-E5M2 descriptor microkernel | `tk_fa4/native_gqa_tk_bwd/e5m2_dout_mixed_mma_microgate_20260831.cuh` |
| Shared native-backward pipeline | `tk_fa4/native_gqa_tk_bwd/native_gqa_tk_bwd_pipelined.cuh` |
| Inherited owner, score, and gradient stages | The exact transitive header list is declared in each `Makefile.v509*`; it includes the `v420`, `v421`, `v429`, `v431`, `v433`, `v436`, `v437`, and `v438` headers in `tk_fa4/native_gqa_tk_bwd` |
| Clean release build | `tools/build_fa4.py` with target `v509-backward` |
| Shape- and ABI-checking Python runtime | `tk_fa4/lowp_fa4_bwd/native_tk_d128_nvfp4_score_e5m2_dout_backward.py` |
| TorchTitan autograd/runtime adapter | `torchtitan/experiments/fa4/exact_lowp_attention.py` |
| Isolated and projection-inclusive benchmark | `tk_fa4/lowp_fa4_bwd/benchmark_v509_report_boundaries.py` |
| Batched address/raster gate | `tk_fa4/native_gqa_tk_bwd/validate_v509_batched_isolation.py` |
| Fused publisher gate | `tk_fa4/native_gqa_tk_bwd/validate_v509_fused_projection_publisher.py` |
| E5M2 descriptor and producer gates | `tk_fa4/native_gqa_tk_bwd/validate_e5m2_dout_mixed_mma_microgate_20260831.py`; `tk_fa4/native_gqa_tk_bwd/validate_e5m2_dout_producer_microgate_20260831.py` |
| Source/ABI regression checks | `tests/test_native_tk_d128_nvfp4_score_e5m2_dout_backward.py`; `tests/test_v509_e5m2_dout_source_contract.py`; `tests/test_v509_e5m2_runtime_wiring.py`; `tests/test_validate_v509_batched_isolation.py` |

The separately compiled B1, B2, and B4 artifacts share one implementation but
bind the batch raster at compile time. All three are available for operator
tests. The portable TorchTitan manifest admits B1 and B4 for training; B2 is
kept operator-only until the training adapter has a separately authenticated
integration gate.

## BF16 FA4 control

The matched control uses the pinned FlashAttention CuTe-DSL implementation. It
does not compile through `tools/build_fa4.py`; the CuTe runtime specializes it
for the requested shape.

| Role | Exact source |
| --- | --- |
| BF16 public dispatch | `flash-attention/flash_attn/cute/interface.py` |
| BF16 forward | `flash-attention/flash_attn/cute/flash_fwd.py`; `flash-attention/flash_attn/cute/flash_fwd_sm100.py` |
| BF16 backward | `flash-attention/flash_attn/cute/flash_bwd.py`; `flash-attention/flash_attn/cute/flash_bwd_sm100.py`; `flash-attention/flash_attn/cute/flash_bwd_preprocess.py`; `flash-attention/flash_attn/cute/flash_bwd_postprocess.py` |
| Authenticated TorchTitan comparator adapter | `torchtitan/experiments/fa4/fa4_attention.py` |
| Configuration and integration checks | `tests/unit_tests/test_fa4_reproduction.py` |

The submodule also contains the disabled direct-FP4 experiment described
below. Its presence does not change the BF16 route selected by the comparator
adapter.

## End-to-end training experiment stack

The kernel routes above are connected to TorchTitan through the following
source. This is the complete model-side closure specific to the paper rather
than the whole upstream TorchTitan framework.

| Role | Exact source |
| --- | --- |
| Route selection, authenticated artifacts, autograd | `torchtitan/experiments/fa4/exact_lowp_attention.py`; `torchtitan/experiments/fa4/fa4_attention.py`; `torchtitan/experiments/fa4/artifacts.py` |
| FA4 configuration schema | `torchtitan/experiments/fa4/job_config.py` |
| 1.235B/D64 and 8.03B/D128 model geometry | `torchtitan/experiments/fa4/train_spec.py` |
| Model conversion and BF16 storage handling | `torchtitan/experiments/fa4/converters.py` |
| Dataset registration | `torchtitan/experiments/fa4/data.py` |
| Checkpoint hooks | `torchtitan/experiments/fa4/checkpoint.py` |
| Fused BF16 stochastic-rounding AdamW | `torchtitan/experiments/fa4/optimizer/build.py`; `torchtitan/experiments/fa4/optimizer/fused_adamw_bf16_sr.py`; `torchtitan/experiments/fa4/optimizer/optimizer_sr_state.py`; `torchtitan/experiments/fa4/optimizer/csrc/adamw_bf16_sr/adamw_bf16_sr.cu`; `torchtitan/experiments/fa4/optimizer/csrc/adamw_bf16_sr/binding.cpp` |
| Credential-free config renderer | `tools/render_fa4_training_config.py` |
| Local D128 B1/B2/B4 full-update harness | `tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py` |
| D64 B16 forward and saturated-update harnesses | `tk_fa4/lowp_fa4_bwd/benchmark_b16_forward_factorial.py`; `tk_fa4/lowp_fa4_bwd/benchmark_llama12b_saturated.py` |
| Attention/projection boundary harness | `tk_fa4/lowp_fa4_bwd/benchmark_v509_report_boundaries.py` |
| Integration tests | `tests/unit_tests/test_fa4_reproduction.py`; `tests/unit_tests/test_fa4_d64_runtime_profile.py`; `tests/test_d64_release_recipe.py`; `tests/test_llama12b_batched_exact_runtime.py`; `tests/test_lowp_attention_workspace_lifecycle.py`; `tests/test_packed_lowp_qkv_control.py` |

The renderer creates a portable **new-run** recipe. The historical distributed
curve used a private, unshuffled MosaicML Streaming snapshot whose immutable
shard manifest and exact sample order are not in this repository. Therefore
the committed historical curve can be redrawn from its receipt, but it cannot
yet be reproduced byte for byte from raw data. This is a data-input gap, not a
missing kernel.

## Portable D64 backward and historical lineage

The native D64 owner schedule used in early 1.2B work is preserved in full:

| Role | Exact source |
| --- | --- |
| Internal source label | `v416` |
| Build file | `tk_fa4/native_gqa_tk_bwd/Makefile.v416` |
| Translation unit | `tk_fa4/native_gqa_tk_bwd/v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds.cu` |
| Mainloop and epilogue | `tk_fa4/native_gqa_tk_bwd/v416_d64_gqa_e4m3_production_bshd_dq_first_vec2_ds.cuh` |
| Shared pipeline and predecessor headers | `tk_fa4/native_gqa_tk_bwd/native_gqa_tk_bwd_pipelined.cuh`; `tk_fa4/native_gqa_tk_bwd/v387_d64_gqa_e4m3_async_pipeline.cuh`; `tk_fa4/native_gqa_tk_bwd/v386_d64_gqa_e4m3_k128q128_halfcols.cuh`; `tk_fa4/native_gqa_tk_bwd/v385_d64_gqa_e4m3_k128q128.cuh` |
| Runtime adapter | `tk_fa4/lowp_fa4_bwd/native_tk_d64_backward.py` |
| Isolated matrix driver | `tk_fa4/lowp_fa4_bwd/benchmark_causal_backward_matrix.py` |
| Portable build and manifest | `tools/build_fa4.py` profile `llama1p2b-d64-b16`; `torchtitan/experiments/fa4/artifacts.py` |
| TorchTitan dispatch | `torchtitan/experiments/fa4/exact_lowp_attention.py` |
| Runtime regression checks | `tests/test_native_tk_d64_backward_runner.py`; `tests/test_causal_backward_matrix.py`; `tests/unit_tests/test_fa4_d64_runtime_profile.py`; `tests/test_d64_release_recipe.py` |

The builder now emits a distinct schema-v3 D64/B16 profile with BF16 control,
E4M3-projection FP8-P/V, E4M3-projection MXFP4-P/V, shared projection
publisher, and native-v416 manifests. The adapter checks the profile, exact
B16/S4096/Hq32/Hkv8/D64 geometry, module, bytes, and SHA256 before dispatch; a
D128 v509 image cannot satisfy the contract. This closes the source and binary
wiring, not the hardware release gates: a clean-clone GB200 build, distributed
save/fresh-resume smoke, and long-horizon public-data run are still pending.

The exact commit-era v416 delta is preserved under
`reproduction/snapshots/d64_v416_713819d`; the earlier real-token harness delta
is under `reproduction/snapshots/d64_training_cd59dda`. Their manifests bind
the source files to historical Git blobs and SHA256 values. The exact cd57
CuTe control used by the earlier short trajectory is unavailable and must not
be substituted silently.

## Preserved diagnostic backward paths

These routes explain the causal design history. They are not selected by the
release build or TorchTitan training configuration.

| Reader-facing diagnostic | Internal source mapping | Runtime and validation | Release status |
| --- | --- | --- | --- |
| **Represented-FP8 backward precursor** | `v501`: `tk_fa4/native_gqa_tk_bwd/Makefile.v501`; `tk_fa4/native_gqa_tk_bwd/v501_d128_gqa_e4m3_unified_best_route_production_bshd.cu`; `tk_fa4/native_gqa_tk_bwd/v501_d128_gqa_e4m3_unified_best_route_production_bshd.cuh` | `tk_fa4/lowp_fa4_bwd/native_tk_d128_backward.py`; `tk_fa4/lowp_fa4_bwd/validate_native_tk_d128_backward.py`; `tests/test_native_tk_d128_backward_runner.py` | Historical systems prototype only |
| **Common-row MX-V backward** | `v503`: `tk_fa4/native_gqa_tk_bwd/Makefile.v503`; `tk_fa4/native_gqa_tk_bwd/v503_d128_gqa_mxfp4v_rowscale_e4m3do_b2_s4096_owner4_experimental_bshd.cu`; `tk_fa4/native_gqa_tk_bwd/v503_d128_gqa_mxfp4v_rowscale_e4m3do_b2_s4096_owner4_experimental_bshd.cuh` | `tk_fa4/lowp_fa4_bwd/native_tk_d128_mxfp4_v_backward.py`; `tests/test_native_tk_d128_mxfp4_v_backward_runner.py`; `tests/test_d128_mxfp4_v_e2e_wiring.py` | Diagnostic B2/S4096 route; not a throughput claim |
| **Direct Common-Row MX Publication** | `v506`: `tk_fa4/native_gqa_tk_bwd/Makefile.v506`; `tk_fa4/native_gqa_tk_bwd/v506_d128_gqa_mxfp4v_commonrow_e4m3do_b2_s4096_owner4_experimental_bshd.cu`; `tk_fa4/native_gqa_tk_bwd/v506_d128_gqa_mxfp4v_commonrow_e4m3do_b2_s4096_owner4_experimental_bshd.cuh`; projection publication is in `tk_fa4/lowp_fa4_bwd/projection_fp4_epilogue.cuh` | `tk_fa4/lowp_fa4_bwd/authenticate_d128_mx_backward_v_publication.py`; `tests/test_experimental_d128_direct_common_rowscale_contract.py` | Disabled after missing its performance gate |
| **Four-Anchor Shared-Tile MX-V backward** | `v507`: `tk_fa4/native_gqa_tk_bwd/Makefile.v507`; `tk_fa4/native_gqa_tk_bwd/v507_d128_gqa_mxfp4v_sharedtile_e4m3do_b2_s4096_owner4_experimental_bshd.cu`; `tk_fa4/native_gqa_tk_bwd/v507_d128_gqa_mxfp4v_sharedtile_e4m3do_b2_s4096_owner4_experimental_bshd.cuh` | `tk_fa4/lowp_fa4_bwd/authenticate_d128_shared_tile_mx_publication.py`; `tests/test_native_tk_d128_v507_source_contract.py`; `tests/test_d128_shared_tile_mx_interface.py`; `tests/test_mxfp4_shared_tile_transpose_contract.py` | Correct diagnostic, too slow for release |
| **E4M3-dO backward precursor** | `v508`: `tk_fa4/native_gqa_tk_bwd/Makefile.v508`; `tk_fa4/native_gqa_tk_bwd/v508_d128_gqa_nvfp4_score_e4m3_gradient_b1_exact_s4096_experimental_bshd.cu`; `tk_fa4/native_gqa_tk_bwd/v508_d128_gqa_nvfp4_score_e4m3_gradient_b1_exact_s4096_experimental_bshd.cuh` | `tk_fa4/lowp_fa4_bwd/native_tk_d128_nvfp4_score_backward.py`; `tests/test_native_tk_d128_nvfp4_score_backward_runner.py` | Superseded because E4M3 dO underflowed too aggressively |
| **Dense-Score E4M3/E5M2 backward** | `v510`, local-only commit `aa021504`; exact overlay under `reproduction/snapshots/v510_aa021504/` | Snapshot includes its runtime, validator, source-contract tests, patch, and SHA256 inventory | Preserved unpromoted negative branch; exact rejection receipt unavailable |

The common-row runtime explicitly authenticates only its named consumer and
rejects the four-anchor binary. This prevents shape-compatible but
semantically different MXFP4 V layouts from being exchanged accidentally.

## Disabled direct CuTe FP4-QK backward

The repository preserves a separate direct-FP4-QK CuTe experiment. It is not
the retained backward from saved quantized operands and is not used by any
reported D128 training result.

| Role | Exact source |
| --- | --- |
| Native FP4-QK backward implementation | `flash-attention/flash_attn/cute/fp4_flash_bwd_sm100.py` |
| Public dispatch and fail-closed D128 guard | `flash-attention/flash_attn/cute/interface.py` |
| SM100 matrix descriptors | `flash-attention/flash_attn/cute/mma_sm100_desc.py` |
| Recoverable overlay against its original base | `patches/flash_attention_fp4_runtime_overlay_9743edaf_20260831.patch` |
| Overlay file hashes and safety constraint | `patches/flash_attention_fp4_runtime_overlay_9743edaf_20260831.manifest.json` |
| Source-contract checks | `tests/test_cute_d128_mxfp4_dp_prototype.py`; `tests/test_d128_gqa_mxfp4_v_dp_patch.py` |

The native D128 two-CTA schedule can fail to make progress. Public D128
dispatch therefore stops and directs callers to the verified BF16 bridge. The
source remains available for research and review, but enabling it would create
a new, unsupported experiment rather than reproduce the paper.

## Build and validation entry points

For the supported D128 causal closure, `tools/build_fa4.py` is the single build
entry point. It emits the projection publisher, standalone MXFP4 quantizer,
both forward variants, and exact-batch backward binaries that consume saved
quantized operands under one explicit absolute build root. It also writes
authenticated artifact manifests
consumed by `torchtitan/experiments/fa4/artifacts.py`.

The principal measurement boundaries are:

| Boundary | Exact driver |
| --- | --- |
| Non-causal forward kernel grid | `reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_comprehensive_suite.py` |
| Causal forward kernel | `tk_fa4/lowp_fa4_bwd/benchmark_causal_forward_boundaries.py` |
| Causal backward, projection-inclusive backward, and projection-inclusive forward-plus-backward | `tk_fa4/lowp_fa4_bwd/benchmark_v509_report_boundaries.py` |
| Single-GPU 8B complete update | `tk_fa4/lowp_fa4_bwd/benchmark_llama12b_e2e.py` |
| Distributed 8B pre-training | `tools/render_fa4_training_config.py`; `python -m torchtitan.experiments.fa4.train` |
| Portable topology/NCCL preflight | `tools/fa4_nccl_preflight.py` |
| Portable one- or multi-node launch | `scripts/fa4/run_torchrun.sh` |
| Deterministic plots, tables, and paper | `tools/reproduce_fa4_paper.py` |

No prebuilt project `.so` is a scientific source of truth in this release.
The build wrapper, CUDA/CuTe source, authenticated adapters, and tests above are
the source of truth; binaries must be rebuilt and hashed in the target
environment.
