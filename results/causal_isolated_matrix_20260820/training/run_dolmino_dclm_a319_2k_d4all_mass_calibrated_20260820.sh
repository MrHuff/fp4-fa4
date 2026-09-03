#!/usr/bin/env bash

set -euo pipefail

readonly RUN_ROOT="/workspace/codebases/pv/fp4_matmul_worktrees/dolmino_d4all_calibrated_20260820"
readonly PYTHON_BIN="/workspace/codebases/poly_stuff/.venv-poly/bin/python"
readonly TRAINER="${RUN_ROOT}/tk_fa4/lowp_fa4_bwd/train_llama12b_real_tokens.py"
readonly CORPUS="/tmp/dolmino-dclm-a319f19-dclm0000-prefix20000.jsonl"
readonly TOKENIZER="/workspace/codebases/poly_stuff/low-precision-functions/low-bits-training/assets/hf/Meta-Llama-3.1-8B/tokenizer.json"
readonly BACKWARD_EXTENSION="/tmp/codex-aug19-stable-bwd-rebuild-20260820/tk_fa4/_C_b300_lowp_bwd.cpython-312-aarch64-linux-gnu.so"
readonly BACKWARD_CONTROL="/tmp/codex-aug19-stable-bwd-rebuild-20260820/generated/fmha_bwd_d64_gqa_aug19_exact.py"
readonly MX_EXTENSION="/tmp/codex-d4all-pscale-replay-20260820/_C_cfwd_mx_d4all_pscale_s4096h32kv8d64_20260820.cpython-312-aarch64-linux-gnu.so"
readonly FP8_EXTENSION="/tmp/codex-causal-forward-matrix-d4q01-full-20260820/_C_cfwd_fp8exact0_for_d4q01_s4096h32kv8d64_d4q01_full_20260820.cpython-312-aarch64-linux-gnu.so"
readonly CALIBRATION="${RUN_ROOT}/results/causal_isolated_matrix_20260820/training/d4all_probability_mass_calibration_seed20260818_20260821.json"
readonly MX_BACKWARD_GAIN="0.9768740549154875"
readonly PHYSICAL_GPU=3
readonly EXPECTED_GPU_UUID="GPU-60db269d-3281-dae0-547b-e0d4ecd95e06"
readonly RESULT_DIR="${RUN_ROOT}/results/causal_isolated_matrix_20260820/training"
readonly RESULT_STEM="dolmino_dclm_a319_2k_d4all_mass_calibrated_noreplay_seed20260818_20260820_v3"
readonly OUTPUT="${RESULT_DIR}/${RESULT_STEM}.json"
readonly PARTIAL_OUTPUT="${RESULT_DIR}/${RESULT_STEM}.partial.json"
readonly LOG="${RESULT_DIR}/${RESULT_STEM}.log"
readonly STATUS="${RESULT_DIR}/${RESULT_STEM}.status.json"
readonly STATUS_TMP="${STATUS}.tmp"
readonly LAUNCHER_RELATIVE="results/causal_isolated_matrix_20260820/training/run_dolmino_dclm_a319_2k_d4all_mass_calibrated_20260820.sh"

check_file() {
    local expected_sha256="$1"
    local expected_bytes="$2"
    local path="$3"
    if [[ ! -f "${path}" || -L "${path}" ]]; then
        echo "required artifact is not a regular non-symlink file: ${path}" >&2
        exit 10
    fi
    if ! printf '%s  %s\n' "${expected_sha256}" "${path}" \
        | sha256sum --check --status; then
        echo "SHA256 mismatch for ${path}" >&2
        exit 11
    fi
    if [[ "$(stat -c '%s' "${path}")" != "${expected_bytes}" ]]; then
        echo "byte-size mismatch for ${path}" >&2
        exit 12
    fi
}

check_file "e29baf2e253325268a2777a05765d28da43b825ca1e2dec6936fda3bff8e1b7b" 57969 "${TRAINER}"
check_file "7c9e3b55b3c1abbfb79412037c7c0f80ba6e16571ed677d1dd6bc4eb14d16e0f" 150441082 "${CORPUS}"
check_file "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa" 9085658 "${TOKENIZER}"
check_file "6480ed82250766c6448fb424a2eba5c8b7d57c68e99c58ea8913ef71c7c7e65e" 17220528 "${BACKWARD_EXTENSION}"
check_file "ad3df8751d83a6055a714099e86ff519f4bd1097551ac99ad8f756df97f8d4a0" 219076 "${BACKWARD_CONTROL}"
check_file "62618c7869d4656f762f14d0e09465ba4c753af29ed10dde21af82ca3c722e78" 1953512 "${MX_EXTENSION}"
check_file "fba5d54ed080d5808342058bfa2c781d9ff55cc8c5e37e373235e72e0a1e70c3" 1813632 "${FP8_EXTENSION}"
check_file "d96b9c08836c8f272385736b3ecabf6aeb75cd05efae333490623d0014accd01" 4551 "${CALIBRATION}"

if [[ "$(git -C "${RUN_ROOT}/flash-attention" rev-parse HEAD)" \
    != "9743edaf3227a25f6afc4fa7be8b5e8498610553" ]]; then
    echo "flash-attention is not checked out at its pinned submodule commit" >&2
    exit 13
fi
if [[ "$(git -C "${RUN_ROOT}/qutlass" rev-parse HEAD)" \
    != "406e86fb2d7df436e94f825bcda8e59b1a7250a6" ]]; then
    echo "qutlass is not checked out at its pinned submodule commit" >&2
    exit 14
fi
if [[ "$(git -C "${RUN_ROOT}/qutlass/third_party/cutlass" rev-parse HEAD)" \
    != "b2ca083d2bb96c41d9b3c5a930637c641f6669bf" ]]; then
    echo "qutlass CUTLASS is not checked out at its pinned commit" >&2
    exit 15
fi
if [[ ! -f "${RUN_ROOT}/qutlass/third_party/cutlass/examples/python/CuTeDSL/blackwell/fmha_bwd.py" ]]; then
    echo "the pinned CuTe backward control source is missing" >&2
    exit 16
fi

if [[ "$(jq -r '.aggregate.inverse_mean' "${CALIBRATION}")" != "${MX_BACKWARD_GAIN}" ]]; then
    echo "calibration artifact does not select ${MX_BACKWARD_GAIN}" >&2
    exit 17
fi
if ! git -C "${RUN_ROOT}" ls-files --error-unmatch \
    "${LAUNCHER_RELATIVE}" >/dev/null 2>&1; then
    echo "refusing to launch before the launcher is committed" >&2
    exit 18
fi
if [[ -n "$(git -C "${RUN_ROOT}" status --porcelain --untracked-files=all)" ]]; then
    echo "refusing to launch from a dirty worktree" >&2
    exit 19
fi

mkdir -p "${RESULT_DIR}"
for path in "${OUTPUT}" "${PARTIAL_OUTPUT}" "${LOG}" "${STATUS}" "${STATUS_TMP}"; do
    if [[ -e "${path}" ]]; then
        echo "refusing to overwrite ${path}" >&2
        exit 20
    fi
done

mapfile -t active_gpu_pids < <(
    nvidia-smi -i "${PHYSICAL_GPU}" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
)
if ((${#active_gpu_pids[@]})); then
    echo "physical GPU ${PHYSICAL_GPU} is not idle" >&2
    exit 21
fi
IFS=',' read -r gpu_uuid gpu_free_mib < <(
    nvidia-smi -i "${PHYSICAL_GPU}" --query-gpu=uuid,memory.free \
        --format=csv,noheader,nounits | tr -d ' '
)
if [[ "${gpu_uuid}" != "${EXPECTED_GPU_UUID}" || "${gpu_free_mib}" -lt 180000 ]]; then
    echo "GPU identity/free-memory gate failed: ${gpu_uuid}, ${gpu_free_mib} MiB" >&2
    exit 22
fi
readonly host_available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if [[ "${host_available_kib}" -lt 268435456 ]]; then
    echo "host available-memory gate failed: ${host_available_kib} KiB" >&2
    exit 23
fi

readonly STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
readonly GIT_HEAD="$(git -C "${RUN_ROOT}" rev-parse HEAD)"
jq -n \
    --arg state "running" \
    --arg started_utc "${STARTED_UTC}" \
    --argjson runner_pid "${BASHPID}" \
    --argjson physical_gpu "${PHYSICAL_GPU}" \
    --arg gpu_uuid "${EXPECTED_GPU_UUID}" \
    --arg git_head "${GIT_HEAD}" \
    --arg mx_backward_gain "${MX_BACKWARD_GAIN}" \
    --arg calibration "${CALIBRATION}" \
    '{state: $state, started_utc: $started_utc, runner_pid: $runner_pid,
      physical_gpu: $physical_gpu, gpu_uuid: $gpu_uuid, git_head: $git_head,
      mx_backward_gain: ($mx_backward_gain | tonumber),
      calibration: $calibration}' >"${STATUS_TMP}"
mv "${STATUS_TMP}" "${STATUS}"

cd "${RUN_ROOT}"
set +e
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" \
TK_FA4_LOWP_BWD_EXTENSION_SOURCE="${BACKWARD_EXTENSION}" \
PYTHONPATH="${RUN_ROOT}" \
PYTHONUNBUFFERED=1 \
"${PYTHON_BIN}" -u -B "${TRAINER}" \
    --layers 16 \
    --rounds 2000 \
    --training-batches 2000 \
    --validation-batches 8 \
    --eval-every 50 \
    --seed 20260818 \
    --learning-rate 0.0001 \
    --routes bf16_cute nvfp4_qk_mxfp4_pv nvfp4_qk_fp8_pv_exact \
    --corpus "${CORPUS}" \
    --tokenizer "${TOKENIZER}" \
    --train-fraction 0.8 \
    --rope-theta 500000.0 \
    --rope-factor 32.0 \
    --expected-backward-extension "${BACKWARD_EXTENSION}" \
    --backward-control-source "${BACKWARD_CONTROL}" \
    --backward-control-sha256 "ad3df8751d83a6055a714099e86ff519f4bd1097551ac99ad8f756df97f8d4a0" \
    --backward-control-bytes 219076 \
    --mx-extension "${MX_EXTENSION}" \
    --mx-module _C_cfwd_mx_d4all_pscale_s4096h32kv8d64_20260820 \
    --fp8-extension "${FP8_EXTENSION}" \
    --fp8-module _C_cfwd_fp8exact0_for_d4q01_s4096h32kv8d64_d4q01_full_20260820 \
    --backward-gain 1.0 \
    --mx-backward-gain "${MX_BACKWARD_GAIN}" \
    --fp8-backward-gain 1.0 \
    --q-quant-scale 2.25 \
    --k-quant-scale 2.0 \
    --qk-scale-refresh-every 0 \
    --backward-exp2-degree 2 \
    --backward-exp2-period 0 \
    --no-mx-backward-reuse-quantized-p \
    --mx-backward-match-forward-operands \
    --no-fp8-backward-reuse-quantized-p \
    --fp8-backward-match-forward-operands \
    --mx-per-block-qk-scales \
    --fp8-per-block-qk-scales \
    --mx-qkv-projection-format e4m3 \
    --fp8-qkv-projection-format e4m3 \
    --no-mx-experimental-split-v-backward \
    --projection-weight-scaling 2d \
    --v-mxfp4-scaling 1d \
    --output "${PARTIAL_OUTPUT}" \
    2>&1 | tee "${LOG}"
readonly TRAIN_STATUS="${PIPESTATUS[0]}"
set -e

if ((TRAIN_STATUS != 0)); then
    jq -n --arg state "failed" --arg started_utc "${STARTED_UTC}" \
        --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --argjson exit_code "${TRAIN_STATUS}" \
        '{state: $state, started_utc: $started_utc,
          finished_utc: $finished_utc, exit_code: $exit_code}' \
        >"${STATUS_TMP}"
    mv "${STATUS_TMP}" "${STATUS}"
    exit "${TRAIN_STATUS}"
fi

if ! jq -e --argjson mx_gain "${MX_BACKWARD_GAIN}" '
    .schema == "llama12b_real_tokens_training_v3"
    and .configuration.layers == 16
    and .configuration.rounds == 2000
    and .configuration.training_batches == 2000
    and .configuration.parameter_count >= 1200000000
    and .configuration.parameter_count < 1300000000
    and .configuration.mx_backward_gain == $mx_gain
    and .configuration.fp8_backward_gain == 1.0
    and .configuration.mx_backward_forward_probability_replay == false
    and .configuration.mx_backward_forward_probability_scale_handoff == false
    and .configuration.mx_forward_topology.mx_mode23_native_density == 4
    and .configuration.mx_forward_topology.mx_mode23_native_quarter_mask == 15
    and .configuration.mx_forward_topology.pv_format == "mxfp4_e8m0_block32"
    and .configuration.fp8_forward_topology.pv_format == "e4m3_fp8"
    and (.routes | length == 3)
    and (.routes | all(.training.all_steps_finite == true))
    and (.validation_history | length == 41)
    and (.validation_history | all(.routes | all(.mean_loss | isfinite)))
' "${PARTIAL_OUTPUT}" >/dev/null; then
    jq -n --arg state "failed_validation" \
        --arg started_utc "${STARTED_UTC}" \
        --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{state: $state, started_utc: $started_utc,
          finished_utc: $finished_utc, exit_code: 24}' >"${STATUS_TMP}"
    mv "${STATUS_TMP}" "${STATUS}"
    exit 24
fi

mv "${PARTIAL_OUTPUT}" "${OUTPUT}"
readonly OUTPUT_SHA256="$(sha256sum "${OUTPUT}" | awk '{print $1}')"
jq -n --arg state "complete" --arg started_utc "${STARTED_UTC}" \
    --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg output "${OUTPUT}" --arg output_sha256 "${OUTPUT_SHA256}" \
    '{state: $state, started_utc: $started_utc,
      finished_utc: $finished_utc, exit_code: 0,
      output: $output, output_sha256: $output_sha256}' >"${STATUS_TMP}"
mv "${STATUS_TMP}" "${STATUS}"
