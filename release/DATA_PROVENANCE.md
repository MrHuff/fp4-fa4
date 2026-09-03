# Data, tokenizer, and training provenance

This document separates three different reproducibility claims:

- **Exact and present** means the required bytes are in the repository.
- **Exact identity, input absent** means the expected digest is known, but the
  external asset is not redistributed. A user-supplied copy can be accepted
  only after it passes the verifier.
- **Identity incomplete** means the historical name and loader settings are
  known, but at least one immutable shard or order identity was never captured.
  A new run can replace that evidence; it cannot be called an exact
  reproduction of the historical run.

The machine-readable record is
[`release/data_manifest.json`](data_manifest.json). It deliberately contains
no credentials or private object-store locator.

## Exact Llama tokenizer

The historical D64 and D128 runs used the same Llama 3.1 tokenizer. The exact
four-file payload was recovered and independently matched to the production
asset-tar identity. We do not copy those files into this repository because
their redistribution terms require review.

| File below `Llama-3.1-8B/` | Bytes | SHA256 |
| --- | ---: | --- |
| `original/tokenizer.model` | 2,183,982 | `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` |
| `special_tokens_map.json` | 73 | `462d91939dbc37178aa5a3eae7068d1990ccc92e09f288cc71f42cdf139d69cc` |
| `tokenizer.json` | 9,085,658 | `76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa` |
| `tokenizer_config.json` | 50,500 | `8004530facf809ac432114de2a4dcc65fcb632da5ec16d666091aeb6a2ee444a` |

The regular-file tree has SHA256
`ba9162eb542cf6445c6a1c9cf997dc176458b1dcfa127aad434b563ec5d94718`.
Its canonical deterministic PAX tar is 11,325,440 bytes with SHA256
`5f78fe6ca69a64aa2ccf5d07bd9a6e33d29da33bd0441416d31b1cf2274c1ab5`.
The verifier reconstructs both identities in a stream and does not write a tar.

A surviving local Hugging Face ref names
`meta-llama/Llama-3.1-8B` commit
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`. That is an **unverified
association**, not the acceptance criterion: no surviving receipt binds that
ref to these exact four bytes. Obtain the files under the upstream license and
run:

```bash
python tools/verify_fa4_data.py \
  --tokenizer-dir /path/to/Llama-3.1-8B
```

If the supplied directory is a complete model snapshot with extra files, use
`--allow-extra-tokenizer-files`. The command then authenticates only the four
files selected for the historical asset archive.

## Matched 8B SlimPajama training

### What is exact

The completed matched B4 comparison is bound to these source identities:

- training integration commit
  `e7db209b0c7017c415fdd66e04e85f96ae24f276`;
- TorchTitan commit `20b3de7585696c327bd5aa9f9627f0300abdbf9d`;
- FA4 runtime commit `4590537f1479e1a7e847f2783e9ab7aa7f11b975`;
- source archive SHA256
  `20af94e0899a56fcd1eb6b8dae9a75217012631d72554f75612b35bcb84b4181`;
- runtime bundle SHA256
  `e0cef3469d9203169e0497152fa83dc9d6f12a5c1c1bbcbb9c0fd43edc58281c`.

The exact historical loader, tokenizer adapter, checkpoint/resume logic, and
trainer are materialized under
[`reproduction/snapshots/training_e7db209b`](../reproduction/snapshots/training_e7db209b/README.md).
All seven Python files are byte-identical to the source revision. The adjacent
training TOML is intentionally sanitized: its two private `dataset_path`
values, and only those values, are replaced with
`__HISTORICAL_SLIMPAJAMA_MDS_ROOT__`. The manifest records both the historical
SHA256 and the sanitized snapshot SHA256. The original private locator is not
part of the release.

The exact model is the untied-embedding, 8,030,261,248-parameter
`8B_llama3_blog` flavor: 32 layers, hidden size 4096, feed-forward size 14336,
32 query heads, eight key/value heads, head dimension 128, and vocabulary
128256. It uses sequence length 4096, RoPE theta 500000, and Llama 3 scaling
factor 8 with low/high frequency factors 1/4 and original context 8192.

The matched launch used 64 GPUs as 16 four-GPU workers. Local batch four and
four gradient-accumulation steps give physical global batch 256 and effective
global batch 1024, or 4,194,304 tokens per update. The target is 23,842 updates
(100,000,595,968 tokens). The optimizer is fused stateless AdamW with BF16
parameters and moments, stochastic BF16 parameter writes, peak learning rate
`3e-4`, betas 0.9/0.95, epsilon `1e-8`, weight decay 0.1, and gradient norm
limit 1. The schedule warms up for 2,000 updates and then uses cosine decay to
one percent of peak. Standard dense BF16 cross entropy is compiled with
TorchInductor; both cut-cross-entropy paths are disabled.

Validation runs for 16 steps every 298 updates. Synchronous distributed
checkpoints are written every 239 updates, retain the latest three, and export
FP32. The model seed is 42. The full field-level recipe and source-file hashes
are in `data_manifest.json`.

The authoritative completed-comparison receipt is:

- `receipts/llama8b_b4_completed_20260903.json`, SHA256
  `36272a35bd95c3138425e7330403f94d87e40ddd2109cdcb2bcf5e2b21c1c55e`.

It records one checkpoint-selected logical trajectory per healthy arm. Each
trajectory reached the 100,000,595,968-token target and contains 954 training
rows and 81 validation rows. The throughput comparison includes all 874 common
post-warmup reports: medians are 21,852.6656 tokens/s/GPU for BF16 and
24,302.9730 tokens/s/GPU for NVFP4-projection/FP8-P/V, a 1.1121285x ratio of
medians. Final same-update validation losses are 2.3048148155 and 2.3948404789.
These are single trajectories, not repeated trials.

Two additional receipts retain launch and negative-control evidence:

- `receipts/llama8b_b4_w64_launch_check_20260902.json`, SHA256
  `f652ea07c34048e9180629737dc000933e481e88856e7c64ee87f148eea21063`;
- `receipts/llama8b_b4_matched_snapshot_20260902T1358Z.json`, SHA256
  `0ed4b988db3a0805d520b0d41e241d224e0ad43e2258bf993625d85c1af2f0da`,
  used only for the separate MXFP4-P/V divergence diagnostic.

All three paths are below
`results/fp4_fa4_technical_report_v2_20260819/`.
They are the public normalized forms described in
[`PUBLIC_SANITIZATION.md`](PUBLIC_SANITIZATION.md); scientific fields and raw
input hashes are unchanged, while scheduler and tracker identifiers are
omitted.

### What prevents an exact fresh reproduction

The historical data source was a custom MosaicML Streaming Dataset (MDS)
version of SlimPajama, not the Hugging Face streaming adapter now provided for
portable new runs. The known loader contract is:

| Setting | Train | Validation |
| --- | --- | --- |
| Split | `train` | `validation` |
| Shuffle | false | false |
| Batching | stratified | stratified |
| Loader workers per rank | 8 | 1 |
| Prefetch factor | 8 | 8 |
| Canonical nodes | 32 | 32 |

The loader appends beginning/end tokens to tokenized documents, keeps a
checkpointed token buffer, and slices 4097 tokens into an input and next-token
target of length 4096. That implementation is now present and hash-bound. The
historical configuration is hash-bound, with the explicit two-field redaction
described above.

What is missing is decisive: neither split has a retained `index.json` digest
or immutable object version, a shard inventory with byte hashes, or per-update
batch fingerprints. Production had batch fingerprinting disabled, and the
recovered worker logs contain no batch-hash records. `shuffle=false` does not
repair this gap because MDS partitioning and row order depend on the missing
index and shard identities.

Consequently:

- the committed receipts can regenerate the completed healthy curves and the
  separate MXFP4-P/V diagnostic;
- a public SlimPajama run can produce new, well-pinned evidence;
- that run must not be described as the exact historical trajectory; and
- the historical claim becomes exactly replayable only if the missing MDS
  identities are recovered or if the paper replaces it with a fully receipted
  new run.

The recovery audit also found the exact credential-free W&B history exports
for BF16, FP8 P/V, and MXFP4 P/V. They remain outside this repository. Their
expected hashes are recorded in `data_manifest.json`. The associated worker
logs require private-endpoint scrubbing before publication, so the manifest
records their identities but does not treat them as release assets yet.

## D64 and short real-data gates

### Dolma3 Longmino prefix

The 256-update 1.2B/D64 and 8B/D128 short gate used physical rows 0--511 of
the Dolma3 Longmino `len-8-16k` MDS stream, without shuffling. The canonical
materialized JSONL has:

- 512 rows and 21,911,537 bytes;
- SHA256
  `860b33924dffd53f4c20b80abbcee96e1bf09c3c313290c15ea3a6ee418269ce`;
- 66 exact duplicates removed, leaving 446 documents; and
- a record-hash aggregate of
  `521e3e34c60323de5eebfe3e09a95432b020560f87f2ca7d7ef165845ca5e08e`.

Its upstream MDS `index.json` is also pinned by SHA256, byte size, object
version, ETag, schema, shard count, and declared document count in the machine
manifest. The JSONL itself is not redistributed. Verify a recovered or
rematerialized copy with:

```bash
python tools/verify_fa4_data.py \
  --dolma3-prefix-jsonl /path/to/dolma3-longmino-len-8-16k-first512.jsonl
```

The 1.2B model has 1,235,814,400 parameters, 16 layers, hidden size 2048,
feed-forward size 8192, 32 query heads, eight key/value heads, head dimension
64, and tied embeddings. Every route used batch one, sequence 4096, seed
20260818, learning rate `1e-4`, no gradient clipping, and the same 256 packed
training batches. Exact document-order and packed-token hashes are in the
manifest.

All three 1.2B routes completed. The scheduler reported failure only because
the submitted post-run check expected 512 documents instead of the correct
446 after deduplication. The corrected manifest is committed but was never
submitted; this distinction is retained in the evidence record.

That short experiment used the historical source now preserved under
`reproduction/snapshots/d64_training_cd59dda` and a CuTe backward control that
predates native v416. The exact control is identified as 220,876 bytes with
SHA256
`cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1`,
but its source bytes are unavailable. The retained loss comparison therefore
must not be presented as a v416 convergence result.

### Portable 50B-token D64 replacement

The new `llama1p2b-d64-50b` profile fixes the same 1.235B tied-embedding
topology at B16/S4096/Hq32/Hkv8/D64. It uses 16 data-parallel ranks, physical
and effective global batch 256, gradient accumulation one, and 47,684 updates
for 50,000,297,984 tokens. The peak learning rate is `0.00048828125`; warmup
and checkpoint intervals are 954 updates; validation runs for 16 steps every
262 updates at local batch 16. It compiles ordinary dense cross entropy and
keeps CCE disabled.

This is a replacement recipe, not a reconstructed historical launch. It uses
the release's fused BF16 stochastic-rounding AdamW; the older 1.2B templates
used ordinary AdamW. It also consumes a user-supplied, exhaustively manifested
public SlimPajama snapshot. The historical SlimPajama MDS train/validation
indices, shard byte hashes, and row order were never retained, so the release
cannot claim a byte-identical token stream. The exact Llama tokenizer identity
above remains required unless the renderer's explicit nonhistorical-tokenizer
mode is selected for a smoke test.

### Two different DCLM corpora

Do not conflate the completed public-prefix ablation with the missing canonical
DCLM reconstruction.

The completed engineering ablation used the first 20,000 records from
`allenai/dolmino-mix-1124` at revision
`a319f19eef1e257417b11ea8c30da266ae175557`, object
`data/dclm/0000/dclm-0000.json.zst`. The materialized JSONL is 150,441,082
bytes with SHA256
`7c9e3b55b3c1abbfb79412037c7c0f80ba6e16571ed677d1dd6bc4eb14d16e0f`.
The compressed-object hash was not retained, so the exact JSONL digest remains
the final acceptance test:

```bash
python tools/verify_fa4_data.py \
  --dclm-a319-prefix-jsonl /path/to/dolmino-dclm-a319-first20000.jsonl
```

The requested canonical 2K reconstruction instead expects one 469,413,680-byte
Arrow file with SHA256
`846edebc43fa909016d6499240e2fb8101b8fd23bf6d9e07a69be347f708be93`.
It has 88,669 rows, 88,638 unique rows, and columns `text`, `id`, and
`source_ds`. Its exact source-object key remains unresolved and the file is not
present. The training-token, validation-token, and document-order hashes are
known and recorded in the manifest. A C4 corpus or the public first-20,000
JSONL must not be substituted. If the Arrow file is recovered, authenticate it
with:

```bash
python tools/verify_fa4_data.py \
  --canonical-dclm-arrow /path/to/dolma-dclm-data-00000.arrow
```

## Other external paper inputs

Several forward-quality evaluations name public assets but do not pin immutable
revisions or all input bytes. Their existing results remain historical
receipts until those identities are added or the measurements are rerun.

| Experiment | Recorded input | Missing exact identity |
| --- | --- | --- |
| ViT on CIFAR-10 | `nateraw/vit-base-patch16-224-cifar10`; `uoft-cs/cifar10` test | Model revision/weights and dataset revision/bytes |
| BERT MLM | `google-bert/bert-base-uncased`; `Salesforce/wikitext`, `wikitext-2-raw-v1` test | Model revision/weights and dataset revision/bytes |
| SST-2 | `textattack/bert-base-uncased-SST-2`; GLUE/SST-2 validation | Model revision/weights and dataset revision/bytes |
| ViT-MAE reconstruction | `facebook/vit-mae-base`; 100 named COCO 2017 validation images | Model revision/weights and selected image hashes |
| Wan 2.1 fixed-input evaluation | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` and `Wan-AI/Wan2.1-T2V-14B-Diffusers` | Checkpoint revisions/weights and some table inputs |

The synthetic HAO and causal timing grids have no external dataset dependency;
their seeds, tensor shapes, kernels, and timing boundaries belong in the
experiment manifests rather than this data manifest.

## Release completion checklist

The data side of the release is complete only when all paper evidence satisfies
one of these conditions:

1. the exact external bytes are legally redistributable and included with
   hashes;
2. an acquisition step pins an immutable upstream revision and the verifier
   authenticates the resulting bytes; or
3. the manuscript labels the evidence as a historical receipt and does not
   claim an exact fresh reproduction.

For the main 8B training curves, condition 3 is the current state. The matched
BF16 and FP8-P/V trajectories completed their declared 100-billion-token
schedule, but the public release contains one trajectory per arm and the
normalized receipt rather than the raw hosted histories. A byte-identical
public rerun still needs a pinned data snapshot, deterministic packing and rank
partitioning, and enabled batch fingerprints; independent seeds are needed for
uncertainty claims. The MXFP4-P/V trajectory is a documented failure
diagnostic, not the retained training recipe.
