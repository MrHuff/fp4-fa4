#!/usr/bin/env bash

set -euo pipefail

readonly RUN_ROOT="/workspace/codebases/pv/fp4_matmul_worktrees/causal_matrix_20260820"
readonly PYTHON_BIN="/workspace/codebases/poly_stuff/.venv-poly/bin/python"
readonly CORPUS="/tmp/dolmino-dclm-a319f19-dclm0000-prefix20000.jsonl"
readonly TOKENIZER="/workspace/codebases/poly_stuff/low-precision-functions/low-bits-training/assets/hf/Meta-Llama-3.1-8B/tokenizer.json"
readonly BACKWARD_EXTENSION="${RUN_ROOT}/tk_fa4/_C_b300_lowp_bwd.cpython-312-aarch64-linux-gnu.so"
readonly REPLAY_PATCH="${RUN_ROOT}/tk_fa4/lowp_fa4_bwd/d64_gqa_forward_mx_probability_replay.patch"
readonly MX_EXTENSION="/tmp/codex-d4all-pscale-replay-20260820/_C_cfwd_mx_d4all_pscale_s4096h32kv8d64_20260820.cpython-312-aarch64-linux-gnu.so"
readonly FP8_EXTENSION="/tmp/codex-causal-forward-matrix-d4q01-full-20260820/_C_cfwd_fp8exact0_for_d4q01_s4096h32kv8d64_d4q01_full_20260820.cpython-312-aarch64-linux-gnu.so"
readonly RESULT_DIR="${RUN_ROOT}/results/causal_isolated_matrix_20260820/training"
readonly RESULT_BASENAME="dolmino_dclm_a319_2k_d4all_replay_auto_bwd_seed20260818_20260820"
readonly OUTPUT="${RESULT_DIR}/${RESULT_BASENAME}.json"
readonly PARTIAL_OUTPUT="${RESULT_DIR}/${RESULT_BASENAME}.partial.json"
readonly LOG="${RESULT_DIR}/${RESULT_BASENAME}.log"
readonly STATUS="${RESULT_DIR}/${RESULT_BASENAME}.status.json"
readonly STATUS_TMP="${STATUS}.tmp"
readonly PHYSICAL_GPU=3
readonly GPU_UUID="GPU-60db269d-3281-dae0-547b-e0d4ecd95e06"

# Public source used to create CORPUS:
#   repository: allenai/dolmino-mix-1124
#   revision: a319f19eef1e257417b11ea8c30da266ae175557
#   object: data/dclm/0000/dclm-0000.json.zst
#   selection: first 20,000 newline-delimited records
# This is a fresh-data DCLM run, not a token-identical replay of the lost
# 469,413,680-byte canonical Arrow subshard.
#
# MX uses the historically stable causal d4all probability representation,
# publishes its final packed E8M0 scale words, and replays that representation
# in the current optimized backward. FP8-PV uses the retained exact forward.
# The omitted backward period selects the verified S4096 D64 automatic policy.

mkdir -p "${RESULT_DIR}"
for path in "${OUTPUT}" "${PARTIAL_OUTPUT}" "${LOG}" "${STATUS}" "${STATUS_TMP}"; do
    if [[ -e "${path}" ]]; then
        echo "refusing to overwrite ${path}" >&2
        exit 2
    fi
done

check_sha256() {
    local expected="$1"
    local path="$2"
    printf '%s  %s\n' "${expected}" "${path}" | sha256sum --check --status
}

check_sha256 "1ca8cd1f52a9be3dc2877fcfe6e1d845b091ed9b58a75e1c77816e6a1409b718" \
    "${RUN_ROOT}/tk_fa4/lowp_fa4_bwd/train_llama12b_real_tokens.py"
check_sha256 "aeed2603d40290b815218cc77142ddacda0c734384429f26c0d4a6a200fbe884" \
    "${BACKWARD_EXTENSION}"
check_sha256 "d58fa8f26a32e2ef3e2734c9a1d2651639070dc0decb350fdb71d0847162111e" \
    "${REPLAY_PATCH}"
check_sha256 "62618c7869d4656f762f14d0e09465ba4c753af29ed10dde21af82ca3c722e78" \
    "${MX_EXTENSION}"
check_sha256 "fba5d54ed080d5808342058bfa2c781d9ff55cc8c5e37e373235e72e0a1e70c3" \
    "${FP8_EXTENSION}"
check_sha256 "7c9e3b55b3c1abbfb79412037c7c0f80ba6e16571ed677d1dd6bc4eb14d16e0f" \
    "${CORPUS}"
check_sha256 "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa" \
    "${TOKENIZER}"

if ! git -C "${RUN_ROOT}" diff --quiet --; then
    echo "refusing to launch from a tracked-dirty worktree" >&2
    exit 3
fi

mapfile -t active_gpu_pids < <(
    nvidia-smi -i "${PHYSICAL_GPU}" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | sed '/^[[:space:]]*$/d'
)
if ((${#active_gpu_pids[@]})); then
    echo "physical GPU ${PHYSICAL_GPU} is no longer idle" >&2
    exit 4
fi

readonly STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
readonly GIT_HEAD="$(git -C "${RUN_ROOT}" rev-parse HEAD)"
jq -n \
    --arg state "running" \
    --arg started_utc "${STARTED_UTC}" \
    --argjson runner_pid "${BASHPID}" \
    --argjson physical_gpu "${PHYSICAL_GPU}" \
    --arg gpu_uuid "${GPU_UUID}" \
    --arg git_head "${GIT_HEAD}" \
    --arg corpus "${CORPUS}" \
    --arg corpus_sha256 "7c9e3b55b3c1abbfb79412037c7c0f80ba6e16571ed677d1dd6bc4eb14d16e0f" \
    --arg train_token_sha256 "ca015efd3423c24a471099f1db18cc532c4eb25216a3183e125a398e2ccaf737" \
    --arg validation_token_sha256 "821b34a143eec2480ebe6619e7a87842d710665d503422401419aa1c37b60c4a" \
    '{
        state: $state,
        started_utc: $started_utc,
        runner_pid: $runner_pid,
        physical_gpu: $physical_gpu,
        gpu_uuid: $gpu_uuid,
        git_head: $git_head,
        corpus: $corpus,
        corpus_sha256: $corpus_sha256,
        expected_train_token_sha256: $train_token_sha256,
        expected_validation_token_sha256: $validation_token_sha256
    }' >"${STATUS_TMP}"
mv "${STATUS_TMP}" "${STATUS}"

cd "${RUN_ROOT}" || exit 5
set +e
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" \
PYTHONPATH="${RUN_ROOT}" \
PYTHONUNBUFFERED=1 \
"${PYTHON_BIN}" -u -B \
    tk_fa4/lowp_fa4_bwd/train_llama12b_real_tokens.py \
    --layers 16 \
    --rounds 2000 \
    --training-batches 2000 \
    --validation-batches 8 \
    --eval-every 50 \
    --seed 20260818 \
    --learning-rate 0.0001 \
    --routes \
        bf16_cute \
        nvfp4_qk_mxfp4_pv \
        nvfp4_qk_fp8_pv_exact \
    --corpus "${CORPUS}" \
    --tokenizer "${TOKENIZER}" \
    --train-fraction 0.8 \
    --rope-theta 500000.0 \
    --rope-factor 32.0 \
    --expected-backward-extension "${BACKWARD_EXTENSION}" \
    --mx-extension "${MX_EXTENSION}" \
    --mx-module _C_cfwd_mx_d4all_pscale_s4096h32kv8d64_20260820 \
    --fp8-extension "${FP8_EXTENSION}" \
    --fp8-module _C_cfwd_fp8exact0_for_d4q01_s4096h32kv8d64_d4q01_full_20260820 \
    --backward-gain 1.0 \
    --q-quant-scale 2.25 \
    --k-quant-scale 2.0 \
    --qk-scale-refresh-every 0 \
    --backward-exp2-degree 2 \
    --no-mx-backward-reuse-quantized-p \
    --mx-backward-match-forward-operands \
    --no-fp8-backward-reuse-quantized-p \
    --fp8-backward-match-forward-operands \
    --mx-per-block-qk-scales \
    --fp8-per-block-qk-scales \
    --mx-qkv-projection-format e4m3 \
    --fp8-qkv-projection-format e4m3 \
    --mx-experimental-split-v-backward \
    --projection-weight-scaling 2d \
    --v-mxfp4-scaling 1d \
    --mx-backward-forward-probability-replay \
    --mx-backward-forward-probability-scale-handoff \
    --output "${PARTIAL_OUTPUT}" \
    2>&1 | tee "${LOG}"
readonly TRAIN_STATUS="${PIPESTATUS[0]}"
set -e

if ((TRAIN_STATUS != 0)); then
    jq -n \
        --arg state "failed" \
        --arg started_utc "${STARTED_UTC}" \
        --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --argjson exit_code "${TRAIN_STATUS}" \
        '{
            state: $state,
            started_utc: $started_utc,
            finished_utc: $finished_utc,
            exit_code: $exit_code
        }' >"${STATUS_TMP}"
    mv "${STATUS_TMP}" "${STATUS}"
    exit "${TRAIN_STATUS}"
fi

if ! jq -e '
    .schema == "llama12b_real_tokens_training_v3"
    and .configuration.layers == 16
    and .configuration.rounds == 2000
    and .configuration.training_batches == 2000
    and .configuration.parameter_count >= 1200000000
    and .configuration.parameter_count < 1300000000
    and .configuration.gradient_clip_norm == null
    and .configuration.backward_exp2_degree == 1
    and .configuration.backward_exp2_period == 2
    and .configuration.mx_backward_forward_probability_replay == true
    and .configuration.mx_backward_forward_probability_scale_handoff == true
    and .configuration.mx_forward_topology.p_scale_publication_supported == true
    and .configuration.mx_forward_extension.sha256 == "62618c7869d4656f762f14d0e09465ba4c753af29ed10dde21af82ca3c722e78"
    and .configuration.fp8_forward_extension.sha256 == "fba5d54ed080d5808342058bfa2c781d9ff55cc8c5e37e373235e72e0a1e70c3"
    and .configuration.train_tokens.sha256 == "ca015efd3423c24a471099f1db18cc532c4eb25216a3183e125a398e2ccaf737"
    and .configuration.validation_tokens.sha256 == "821b34a143eec2480ebe6619e7a87842d710665d503422401419aa1c37b60c4a"
    and (.records | keys | sort) == (["bf16_cute", "nvfp4_qk_fp8_pv_exact", "nvfp4_qk_mxfp4_pv"] | sort)
    and (.records | all(.[]; length == 2000))
    and (.records | all(.[]; all(.[];
        .finite == true
        and .failure_stage == null
        and .gradient_preclip_total_norm == null
        and .gradient_was_clipped == false
        and .gradient_clip_error == null
    )))
    and (.validation_history | length == 41)
    and (.routes | all(.[]; .training.all_steps_finite == true))
    and .source.git.tracked_dirty == false
' "${PARTIAL_OUTPUT}" >/dev/null; then
    jq -n \
        --arg state "artifact_validation_failed" \
        --arg started_utc "${STARTED_UTC}" \
        --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{
            state: $state,
            started_utc: $started_utc,
            finished_utc: $finished_utc,
            exit_code: 6
        }' >"${STATUS_TMP}"
    mv "${STATUS_TMP}" "${STATUS}"
    exit 6
fi

mv "${PARTIAL_OUTPUT}" "${OUTPUT}"
readonly OUTPUT_SHA256="$(sha256sum "${OUTPUT}" | awk '{print $1}')"
jq -n \
    --arg state "complete" \
    --arg started_utc "${STARTED_UTC}" \
    --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg output "${OUTPUT}" \
    --arg output_sha256 "${OUTPUT_SHA256}" \
    '{
        state: $state,
        started_utc: $started_utc,
        finished_utc: $finished_utc,
        exit_code: 0,
        output: $output,
        output_sha256: $output_sha256
    }' >"${STATUS_TMP}"
mv "${STATUS_TMP}" "${STATUS}"
