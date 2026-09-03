#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
result_dir="${RESULT_DIR:-${repo}/results/causal_nv_mx_vs_bf16_fa4_20260815/gb200}"
build_dir="${repo}/tk_fa4/fp4_fa4_fwd"
suffix=$(python3-config --extension-suffix)

mkdir -p "${result_dir}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/workspace/codebases/flash-attention-fp4:${PYTHONPATH:-}"

cases=(
  "s2048_h32_d64 2048 32 64"
  "s4096_h32_d64 4096 32 64"
  "s8192_h32_d64 8192 32 64"
  "s4096_h64_d64 4096 64 64"
  "s2048_h32_d128 2048 32 128"
  "s4096_h32_d128 4096 32 128"
  "s8192_h32_d128 8192 32 128"
  "s16384_h32_d128 16384 32 128"
  "s4096_h64_d128 4096 64 128"
  "s8192_h64_d128 8192 64 128"
)

cd "${build_dir}"
for case_spec in "${cases[@]}"; do
  read -r label seqlen heads dim <<< "${case_spec}"
  module="_C_tk_gb200_causal_${label}"
  extension="/tmp/${module}${suffix}"

  make -B -f Makefile.hao_direct_fp4pv -j1 \
    GPU=B200 \
    HAO_SEQ_LEN="${seqlen}" \
    HAO_HEADS="${heads}" \
    HAO_KV_HEADS=8 \
    HAO_HEAD_DIM="${dim}" \
    HAO_NUM_SM=152 \
    HAO_CAUSAL=1 \
    HAO_FP4PV_MX_POLICY=fast \
    OUT="${extension}" \
    MODULE="${module}" \
    > "${result_dir}/${label}_build.log" 2>&1

  /usr/local/cuda-13.0/bin/cuobjdump --dump-resource-usage "${extension}" \
    > "${result_dir}/${label}_resources.txt"

  fold_args=()
  if [[ "${dim}" == 128 ]]; then
    fold_args+=(--nv-qk-fold-k64-scales both)
    fold_args+=(--nv-qk-fold-scale-select mse)
  fi

  for repeat in 0 1 2; do
    leakage_args=()
    if [[ "${repeat}" == 0 ]]; then
      leakage_args+=(--causal-leakage-check)
    fi
    python3 -u hao_direct_fp4pv_benchmark.py \
      --extension "${extension}" \
      --extension-module "${module}" \
      --qk-format nvfp4 \
      --pv-format mxfp4 \
      "${fold_args[@]}" \
      --causal \
      "${leakage_args[@]}" \
      --skip-hao-fp4 \
      --summary-only \
      --warmup-ms 100 \
      --rep-ms 300 \
      --seed 20260815 \
      > "${result_dir}/${label}_repeat${repeat}.json" \
      2> "${result_dir}/${label}_repeat${repeat}.stderr.log"
  done

  echo "completed ${label}"
done
