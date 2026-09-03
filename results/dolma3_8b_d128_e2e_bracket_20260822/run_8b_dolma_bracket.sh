#!/bin/bash
set -Eeuo pipefail

run_root=/tmp/fa4_8b_e2e_f3db4bf_20260822
repo=/workspace/codebases/pv/fp4_matmul_worktrees/dolmino_d4all_calibrated_20260820
python=/tmp/fa4-cutlass452base-venv.ddzKw2/bin/python
corpus=${run_root}/data/dolma3-longmino-len-8-16k-first512.jsonl
tokenizer=/tmp/fa4-dolma3-d64-assets.QZwFvk/assets/tokenizer.json
projection=/tmp/_C_b300_lowp_bwd_d128_stitch_20260822.cpython-312-aarch64-linux-gnu.so
mx_forward=/tmp/codex-d128-causal-20260821/_C_tk_causal_gqa_nvfp4_mxfp4pv_accurate_s4096h32kv8d128_v1.cpython-312-aarch64-linux-gnu.so
fp8_forward=/tmp/codex-d128-causal-20260821/_C_tk_causal_gqa_nvfp4_fp8pv_exact_s4096h32kv8d128_v1.cpython-312-aarch64-linux-gnu.so
mx_module=_C_tk_causal_gqa_nvfp4_mxfp4pv_accurate_s4096h32kv8d128_v1
fp8_module=_C_tk_causal_gqa_nvfp4_fp8pv_exact_s4096h32kv8d128_v1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUTE_DSL_CACHE_DIR=${run_root}/cache/cute
export TORCHINDUCTOR_CACHE_DIR=${run_root}/cache/inductor
export TK_FA4_LOWP_BWD_EXTENSION_SOURCE=${projection}
export PYTHONPATH=${repo}:${repo}/flash-attention
export MALLOC_ARENA_MAX=2
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

mkdir -p "${run_root}/results" "${CUTE_DSL_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}"

test "$(git -C "${repo}" rev-parse HEAD)" = \
  d61663fe6976faecaec587886f2091487c01bb7e
test "$(git -C "${repo}" diff --binary --no-ext-diff HEAD -- | wc -c)" = 0
printf '%s  %s\n' \
  860b33924dffd53f4c20b80abbcee96e1bf09c3c313290c15ea3a6ee418269ce \
  "${corpus}" \
  76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa \
  "${tokenizer}" \
  c0e5ce51f69e7c4da3fb29c212fdf19716d5a172d5414695d923c8b83170f514 \
  "${projection}" \
  63c43e5cde3af4e9cde82aad1d667221a7ae77dd6271eb653fd202509707b77a \
  "${mx_forward}" \
  f9f67026148c355b3b90026861fc25f3b6b7edccf2d6254703d5ddc4164c3d9e \
  "${fp8_forward}" | sha256sum --check -

common_args=(
  --model-preset llama3.1-8b --layers 32
  --rounds 256 --training-batches 256
  --validation-batches 8 --eval-every 64
  --seed 20260818 --learning-rate 0.0001
  --corpus "${corpus}" --tokenizer "${tokenizer}"
  --train-fraction 0.8 --rope-theta 500000 --rope-factor 8
  --expected-backward-extension "${projection}"
  --mx-extension "${mx_forward}" --mx-module "${mx_module}"
  --fp8-extension "${fp8_forward}" --fp8-module "${fp8_module}"
  --backward-gain 1 --mx-backward-gain 1 --fp8-backward-gain 1
  --q-quant-scale 2.25 --k-quant-scale 2.0
  --qk-scale-refresh-every 0
  --backward-exp2-degree 1 --backward-exp2-period 0
  --mx-backward-reuse-quantized-p
  --fp8-backward-reuse-quantized-p
  --no-mx-backward-match-forward-operands
  --no-fp8-backward-match-forward-operands
  --no-mx-per-block-qk-scales --no-fp8-per-block-qk-scales
  --no-mx-experimental-split-v-backward
  --no-mx-backward-forward-probability-replay
  --no-mx-backward-forward-probability-scale-handoff
  --projection-weight-scaling 2d --v-mxfp4-scaling 1d
  --mx-qkv-projection-format nvfp4
  --fp8-qkv-projection-format nvfp4
)

run_arm() {
  local label=$1
  local route=$2
  local output=${run_root}/results/${label}.json
  local progress=${run_root}/results/${label}.progress.json
  local log=${run_root}/results/${label}.log
  printf '=== %s START %s ===\n' "${label}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee "${log}"
  timeout --signal=TERM --kill-after=120s 60m \
    taskset -c 0 "${python}" -u \
      "${repo}/tk_fa4/lowp_fa4_bwd/train_llama12b_real_tokens.py" \
      "${common_args[@]}" --routes "${route}" \
      --progress-output "${progress}" --progress-every 16 \
      --output "${output}" 2>&1 | tee -a "${log}"
  printf '=== %s COMPLETE %s ===\n' "${label}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "${log}"
}

run_arm mx-a nvfp4_qk_mxfp4_pv
run_arm bf16 bf16_cute
run_arm fp8 nvfp4_qk_fp8_pv_exact
run_arm mx-b nvfp4_qk_mxfp4_pv

cd "${repo}"
"${python}" tk_fa4/lowp_fa4_bwd/merge_real_token_route_results.py \
  --bf16 "${run_root}/results/bf16.json" \
  --mx "${run_root}/results/mx-a.json" \
  --fp8 "${run_root}/results/fp8.json" \
  --output "${run_root}/results/merged-mx-a.json"
"${python}" tk_fa4/lowp_fa4_bwd/merge_real_token_route_results.py \
  --bf16 "${run_root}/results/bf16.json" \
  --mx "${run_root}/results/mx-b.json" \
  --fp8 "${run_root}/results/fp8.json" \
  --output "${run_root}/results/merged-mx-b.json"

printf 'ALL 8B DOLMA ARMS COMPLETE %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
