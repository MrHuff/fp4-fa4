# Public-tree sanitization

This file records the deterministic public-surface transformation applied to
the parentless release candidate derived from private release commit
`4a55ec4a4ab96de8a2b70baa380b37e5d57e32ca`. The transformation removes
operational identity and assets that are unnecessary for scientific
reproduction. It does not change shapes, seeds, samples, token coordinates,
timings, losses, gradient norms, throughput, MFU, source commits, runtime
hashes, raw-input content hashes, route labels, or claim boundaries.

## Receipt and script normalization

The current-paper surface uses scientific arm labels in place of scheduler
job IDs and experiment-tracker run IDs or names:

| Public arm label | Scientific meaning |
| --- | --- |
| `bf16_b4_control` | B4 BF16 FA4 control |
| `nvfp4_projection_fp8_pv_b4` | B4 NVFP4 projections with FP8 P/V |
| `e4m3_projection_mxfp4_pv_b4` | B4 E4M3 projections with MXFP4 P/V |
| `nvfp4_projection_mxfp4_pv_b4` | B4 NVFP4 projections with MXFP4 P/V |
| `e4m3_projection_fp8_pv_b1` | B1 E4M3 projections with FP8 P/V |
| `e4m3_projection_mxfp4_pv_b1` | B1 E4M3 projections with MXFP4 P/V |
| `nvfp4_projection_fp8_pv_b1` | B1 NVFP4 projections with FP8 P/V |
| `nvfp4_projection_mxfp4_pv_b1` | B1 NVFP4 projections with MXFP4 P/V |

The following eight files were normalized. The private-source and public
SHA256 values authenticate the transformation without redistributing the
removed identifiers.

| File | Private source SHA256 | Public SHA256 |
| --- | --- | --- |
| `results/fp4_fa4_technical_report_v2_20260819/export_llama8b_b4_snapshot.py` | `b53446d2b211d231621faea51408aef80dc2d8670d487fc824cded2c5056c815` | `0a0739ec371316fc9295acd3c25929388f2153faa9b1dd2988c9b820051af438` |
| `results/fp4_fa4_technical_report_v2_20260819/fetch_llama8b_training_curves.py` | `9dc07f2eaae1dfd47b729190631ffbfbb5beab04a50c66ad3b7e5c45e55a0637` | `8a3ea9e2e7a7b62f2fa727b0e3b4768b10f2fb58fcd7d0ca8d1cc02d9bafe4c8` |
| `results/fp4_fa4_technical_report_v2_20260819/plot_causal_training.py` | `7416ada80081c7dd25c5034e4e1524874a577f6df2c178e999b909212ff349ba` | `88dba23aebb251b722f1a315d26d03514f759efc031eae493fa0641893c70b9d` |
| `results/fp4_fa4_technical_report_v2_20260819/receipts/llama8b_b4_matched_snapshot_20260902T1121Z.json` | `65407758a64e1d6235ae34bdf97259f6f7bef265225e0e74684464a139b824d7` | `c7001d4a1f2278e78771448d16de466c03fa2dc578e8434184f12c9fd1d77ce3` |
| `results/fp4_fa4_technical_report_v2_20260819/receipts/llama8b_b4_matched_snapshot_20260902T1358Z.json` | `81d3c964d1e14e974de9b73b9502856d5e541605e870eeb5c529d4488fbbc628` | `0ed4b988db3a0805d520b0d41e241d224e0ad43e2258bf993625d85c1af2f0da` |
| `results/fp4_fa4_technical_report_v2_20260819/receipts/llama8b_b4_w64_launch_check_20260902.json` | `f3d6798db8d31476f515591ab4f2b567b39f22d6cc3f133811ea282e205451bc` | `f652ea07c34048e9180629737dc000933e481e88856e7c64ee87f148eea21063` |
| `results/fp4_fa4_technical_report_v2_20260819/receipts/llama8b_training_curves_20260901.json` | `9bebd7374a894fd848596058a087561d6214e63fd91e52932294d85c8d2a9fe0` | `eecf78acd3cccd20f2cfae57cd8e4b9f6a79da12ecd3a4537e10f40b73591ca1` |
| `results/fp4_fa4_technical_report_v2_20260819/receipts/v509_four_arm_cutoff_20260831T2209Z.json` | `5c4b1b6bc40d613e13ece228f55be7429b1ad79ff096323f7a5fe7097245d6f0` | `b8e19765627e4f40d262578f9b59614f61f7eff76b35a3467b9d82b5f2784fc2` |

The completed-run exporter and receipt were designed to be credential-free
at creation and therefore require no public-tree rewrite. They are
byte-identical to the reviewed private-paper versions:

| File | Reviewed SHA256 |
| --- | --- |
| `results/fp4_fa4_technical_report_v2_20260819/fetch_llama8b_b4_complete.py` | `ccb2e51bc207175c35238875df8758d04185fa4da6b7cba1c7635fb11a7282dc` |
| `results/fp4_fa4_technical_report_v2_20260819/receipts/llama8b_b4_completed_20260903.json` | `36272a35bd95c3138425e7330403f94d87e40ddd2109cdcb2bcf5e2b21c1c55e` |

Specifically, the transformation:

- removes `job_id`, tracker `run_id`, tracker `run_name`, and the private
  default tracker project from the frozen receipts and their validators;
- replaces identifier-keyed diagnostic maps with the public arm labels above;
- generalizes managed-cluster, object-store, and metric-source wording while
  retaining worker topology, checkpoint object counts, measured bubbles, and
  hashes;
- makes future metric fetching take an uncommitted `--source-map` with schema
  `tkfa4.metric_sources.v1`, rather than embedding a project or run locator;
  the generated public receipt records only the source-map SHA256; and
- updates the manuscript's receipt hashes and the matched-snapshot validator
  to the normalized public artifacts.

A second public-only hardening pass removes the remaining operational envelope
from the matched snapshot. It deletes per-source worker indices and filenames,
byte counts, source-log and download-manifest hashes, exact download times, and
scheduler- or tracker-specific state labels. Generic history-deduplication and
four-rank agreement statistics remain because they describe the scientific
data checks, but their field names no longer name a private service or log
source. The plotted loss, gradient, throughput, model-flop-utilization, token,
and route payloads are unchanged. In particular, the canonical SHA256 of the
MXFP4 divergence payload remains
`440a59110e1f6cd06ed23cabb2b433bf7b2bd4782eabd7c585ca48a41831d606`.

The source map is an operator-provided input of the form
`{"schema":"tkfa4.metric_sources.v1","runs":{"e4_fp8":"...", ...}}`.
Its four values are full tracker run locators. A source map that contains
service-side identifiers must not be committed.

## Hostname normalization

Nineteen JSON receipts contained the same benchmark pod hostname. Only the
value of a `hostname` field was changed, from the private machine name to
`benchmark-host-redacted`. The GPU model, GPU memory, CUDA and driver versions,
host platform, Python version, timing samples, and numerical measurements were
left unchanged.

The 19 files are:

- `results/causal_isolated_matrix_20260820/backward/d64_replay_affine_floor_classifier_s4096_gpu2.json`
- `results/causal_isolated_matrix_20260820/backward/d64_replay_all_lane_exact_stable_projection_s4096.json`
- `results/causal_isolated_matrix_20260820/backward/d64_replay_poly_exp2_degree2_s4096_gpu1.json`
- `results/causal_isolated_matrix_20260820/forward/s4096_h32_kv8_d64.json`
- `results/causal_isolated_matrix_20260820/forward_boundaries_d4q01_split_v_s4096.json`
- `results/causal_isolated_matrix_20260820/forward_matrix_d4all_focus/shapes/s4096_h32_kv8_d64/result.json`
- `results/causal_isolated_matrix_20260820/forward_matrix_d4q01/shapes/s4096_h32_kv8_d64/result.json`
- `results/causal_isolated_matrix_20260820/forward_matrix_d4q01_full/shapes/s1024_h32_kv8_d64/result.json`
- `results/causal_isolated_matrix_20260820/forward_matrix_d4q01_full/shapes/s2048_h32_kv8_d64/result.json`
- `results/causal_isolated_matrix_20260820/forward_matrix_d4q01_full/shapes/s4096_h16_kv4_d64/result.json`
- `results/causal_isolated_matrix_20260820/forward_matrix_d4q01_full/shapes/s4096_h32_kv8_d64/result.json`
- `results/causal_isolated_matrix_20260820/forward_matrix_d4q01_full/shapes/s4096_h64_kv16_d64/result.json`
- `results/causal_isolated_matrix_20260820/forward_matrix_d4q01_full/shapes/s512_h32_kv8_d64/result.json`
- `results/causal_isolated_matrix_20260820/forward_matrix_d4q01_full/shapes/s8192_h32_kv8_d64/result.json`
- `results/causal_isolated_matrix_20260820/forward_matrix_d4q01_s16384/shapes/s16384_h32_kv8_d64/result.json`
- `results/llama12b_mx_publish_order_20260826/baseline_08734ab_b16_factorial_n400.json`
- `results/llama12b_mx_publish_order_20260826/candidate_abd962_b16_factorial_n400.json`
- `results/lowp_fa4_bwd_d128_fused_stitch_20260822/llama31_8b_s4096_causal_forward_fixed_qk_consumer_audited.json`
- `results/lowp_fa4_bwd_d128_fused_stitch_20260822/llama31_8b_s4096_causal_forward_perblock_qk_consumer_audited.json`

## Removed publication assets

Two directories containing 22 proprietary OTF files (4,498,384 bytes total)
were removed:

- `results/fp4_fa4_technical_report_20260728/fonts/`
- `results/fp4_fa4_technical_report_v2_20260819/fonts/`

Their combined path-and-content inventory SHA256 was
`b742529d7f8db04ca139d35739b1cfe0bdbc7d4cd25b59c6210e0565fd423062`.
The two legacy class files now select Latin Modern fonts from the TeX
distribution. The current paper already used open TeX fonts and did not load
either removed directory.

Two unused assets from the vendored HAO comparator were removed:

- `third_party/hao_flash_attention_fp4/assets/fa4_paper.pdf`, 8,896,004 bytes,
  SHA256 `9b269854c3fe3f66db6fd607b122578fd3f453d13d4142c67b48892c2d2ad78c`;
- `third_party/hao_flash_attention_fp4/assets/flashattn_banner.jpg`, 322,273
  bytes, SHA256
  `dbf9e1e910446414035e90c05bd7cb5932e390c438fd0622c04d2172d39ac63b`.

The banner reference was removed from the vendored README. Neither asset is
loaded by the comparator code or by the paper.

## Removed binaries

Four generated SM100a cubins (257,016 bytes total) were removed from
`tk_fa4/results/cute_dsl_bwd_dump_20260505T162123Z/`. Their SHA256 values were:

- `44e273960843cebdf95542667940c9cfcbb77f5a168253a04ddced8cbc8c1966`
- `e819e7acf3e3d7fea77782d93695597c6839e0b3d2ecefac563781cb8603247a`
- `58bd4955665ba32a640dc308dfff2f0cca77ecf939ea291dc0239c2f94cbad5c`
- `3719e4caf21da4f41f57322852b85ac0da20ddbf8ed2441e68e04851c9d18111`

The directory still contains all four PTX dumps and the available annotated
SASS dump. Kernel sources and build tooling remain in the release, so public
binaries must be rebuilt for the target GPU.

## Release-assembly boundary

This transform deliberately does not edit `release/manifest.json`,
`release/source_files.sha256`, the root license, third-party notices, README,
or verifier/tests. The parentless-release assembly step must regenerate the
tree inventory and any tree-derived hashes after all concurrent public edits
land, then run the verifier against that final index. The private source tree
remains the authority for the pre-normalization hashes above.
