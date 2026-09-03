#!/usr/bin/env bash

set -euo pipefail

readonly RUN_ROOT="/workspace/codebases/pv/fp4_matmul_worktrees/causal_matrix_20260820"
readonly PYTHON_BIN="/workspace/codebases/poly_stuff/.venv-poly/bin/python"
readonly CORPUS="/tmp/dolmino-dclm-a319f19-dclm0000-prefix20000.jsonl"
readonly TOKENIZER="/workspace/codebases/poly_stuff/low-precision-functions/low-bits-training/assets/hf/Meta-Llama-3.1-8B/tokenizer.json"
readonly BACKWARD_EXTENSION="/tmp/codex-aug19-stable-bwd-rebuild-20260820/tk_fa4/_C_b300_lowp_bwd.cpython-312-aarch64-linux-gnu.so"
readonly BACKWARD_EXTENSION_SHA256="6480ed82250766c6448fb424a2eba5c8b7d57c68e99c58ea8913ef71c7c7e65e"
readonly BACKWARD_EXTENSION_BYTES=17220528
readonly BACKWARD_CONTROL_SOURCE="/tmp/codex-aug19-stable-bwd-rebuild-20260820/generated/fmha_bwd_d64_gqa_aug19_exact.py"
readonly BACKWARD_CONTROL_SHA256="ad3df8751d83a6055a714099e86ff519f4bd1097551ac99ad8f756df97f8d4a0"
readonly BACKWARD_CONTROL_BYTES=219076
readonly MX_EXTENSION="/tmp/codex-d4all-pscale-replay-20260820/_C_cfwd_mx_d4all_pscale_s4096h32kv8d64_20260820.cpython-312-aarch64-linux-gnu.so"
readonly FP8_EXTENSION="/tmp/codex-causal-forward-matrix-d4q01-full-20260820/_C_cfwd_fp8exact0_for_d4q01_s4096h32kv8d64_d4q01_full_20260820.cpython-312-aarch64-linux-gnu.so"
readonly RESULT_DIR="${RUN_ROOT}/results/causal_isolated_matrix_20260820/training"
readonly RESULT_BASENAME="dolmino_dclm_a319_d4all_replay_aug19_bwd_d2p0_nosplit_3step_seed20260818_20260820"
readonly OUTPUT="${RESULT_DIR}/${RESULT_BASENAME}.json"
readonly PARTIAL_OUTPUT="${RESULT_DIR}/${RESULT_BASENAME}.partial.json"
readonly LOG="${RESULT_DIR}/${RESULT_BASENAME}.log"
readonly STATUS="${RESULT_DIR}/${RESULT_BASENAME}.status.json"
readonly STATUS_TMP="${STATUS}.tmp"
readonly LAUNCHER_RELATIVE="results/causal_isolated_matrix_20260820/training/run_dolmino_dclm_a319_d4all_replay_aug19_bwd_3step_20260820.sh"
readonly PHYSICAL_GPU=3
readonly GPU_UUID="GPU-60db269d-3281-dae0-547b-e0d4ecd95e06"

# Fresh public DCLM data, not a token-identical replay of the lost Arrow shard:
#   allenai/dolmino-mix-1124@a319f19eef1e257417b11ea8c30da266ae175557
#   data/dclm/0000/dclm-0000.json.zst, first 20,000 records.
# The forward artifacts are the current retained exact FP8 route and rebuilt
# D4ALL scale-publication route. Backward uses the recovered Aug19 extension
# and authenticated precomposed control, explicit native EX2 (d2/p0), and the
# represented MXFP4 V path rather than experimental split-V.

check_regular_file() {
    local path="$1"
    if [[ ! -f "${path}" || -L "${path}" ]]; then
        echo "required artifact is not a regular non-symlink file: ${path}" >&2
        exit 1
    fi
}

check_sha256() {
    local expected="$1"
    local path="$2"
    if ! printf '%s  %s\n' "${expected}" "${path}" \
        | sha256sum --check --status; then
        echo "SHA256 mismatch for ${path}" >&2
        exit 1
    fi
}

check_bytes() {
    local expected="$1"
    local path="$2"
    local actual
    actual="$(stat -c '%s' "${path}")"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "byte-size mismatch for ${path}: ${actual} != ${expected}" >&2
        exit 1
    fi
}

for artifact in \
    "${BACKWARD_EXTENSION}" \
    "${BACKWARD_CONTROL_SOURCE}" \
    "${MX_EXTENSION}" \
    "${FP8_EXTENSION}" \
    "${CORPUS}" \
    "${TOKENIZER}"; do
    check_regular_file "${artifact}"
done
check_sha256 "e29baf2e253325268a2777a05765d28da43b825ca1e2dec6936fda3bff8e1b7b" \
    "${RUN_ROOT}/tk_fa4/lowp_fa4_bwd/train_llama12b_real_tokens.py"
check_sha256 "${BACKWARD_EXTENSION_SHA256}" "${BACKWARD_EXTENSION}"
check_bytes "${BACKWARD_EXTENSION_BYTES}" "${BACKWARD_EXTENSION}"
check_sha256 "${BACKWARD_CONTROL_SHA256}" "${BACKWARD_CONTROL_SOURCE}"
check_bytes "${BACKWARD_CONTROL_BYTES}" "${BACKWARD_CONTROL_SOURCE}"
check_sha256 "62618c7869d4656f762f14d0e09465ba4c753af29ed10dde21af82ca3c722e78" \
    "${MX_EXTENSION}"
check_sha256 "fba5d54ed080d5808342058bfa2c781d9ff55cc8c5e37e373235e72e0a1e70c3" \
    "${FP8_EXTENSION}"
check_sha256 "7c9e3b55b3c1abbfb79412037c7c0f80ba6e16571ed677d1dd6bc4eb14d16e0f" \
    "${CORPUS}"
check_sha256 "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa" \
    "${TOKENIZER}"

if ! git -C "${RUN_ROOT}" ls-files --error-unmatch \
    "${LAUNCHER_RELATIVE}" >/dev/null 2>&1; then
    echo "refusing to launch before this launcher is committed" >&2
    exit 2
fi
if ! git -C "${RUN_ROOT}" diff --quiet --; then
    echo "refusing to launch from a tracked-dirty worktree" >&2
    exit 3
fi

mkdir -p "${RESULT_DIR}"
for path in "${OUTPUT}" "${PARTIAL_OUTPUT}" "${LOG}" "${STATUS}" "${STATUS_TMP}"; do
    if [[ -e "${path}" ]]; then
        echo "refusing to overwrite ${path}" >&2
        exit 4
    fi
done

mapfile -t active_gpu_pids < <(
    nvidia-smi -i "${PHYSICAL_GPU}" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | sed '/^[[:space:]]*$/d'
)
if ((${#active_gpu_pids[@]})); then
    echo "physical GPU ${PHYSICAL_GPU} is no longer idle" >&2
    exit 5
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
    --arg backward_extension "${BACKWARD_EXTENSION}" \
    --arg backward_extension_sha256 "${BACKWARD_EXTENSION_SHA256}" \
    --arg backward_control_source "${BACKWARD_CONTROL_SOURCE}" \
    --arg backward_control_sha256 "${BACKWARD_CONTROL_SHA256}" \
    '{
        state: $state,
        started_utc: $started_utc,
        runner_pid: $runner_pid,
        physical_gpu: $physical_gpu,
        gpu_uuid: $gpu_uuid,
        git_head: $git_head,
        backward_extension: $backward_extension,
        backward_extension_sha256: $backward_extension_sha256,
        backward_control_source: $backward_control_source,
        backward_control_sha256: $backward_control_sha256
    }' >"${STATUS_TMP}"
mv "${STATUS_TMP}" "${STATUS}"

cd "${RUN_ROOT}" || exit 6
set +e
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" \
TK_FA4_LOWP_BWD_EXTENSION_SOURCE="${BACKWARD_EXTENSION}" \
PYTHONPATH="${RUN_ROOT}" \
PYTHONUNBUFFERED=1 \
"${PYTHON_BIN}" -u -B \
    tk_fa4/lowp_fa4_bwd/train_llama12b_real_tokens.py \
    --layers 16 \
    --rounds 3 \
    --training-batches 3 \
    --validation-batches 1 \
    --eval-every 1 \
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
    --backward-control-source "${BACKWARD_CONTROL_SOURCE}" \
    --backward-control-sha256 "${BACKWARD_CONTROL_SHA256}" \
    --backward-control-bytes "${BACKWARD_CONTROL_BYTES}" \
    --mx-extension "${MX_EXTENSION}" \
    --mx-module _C_cfwd_mx_d4all_pscale_s4096h32kv8d64_20260820 \
    --fp8-extension "${FP8_EXTENSION}" \
    --fp8-module _C_cfwd_fp8exact0_for_d4q01_s4096h32kv8d64_d4q01_full_20260820 \
    --backward-gain 1.0 \
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
        '{state: $state, started_utc: $started_utc,
          finished_utc: $finished_utc, exit_code: $exit_code}' \
        >"${STATUS_TMP}"
    mv "${STATUS_TMP}" "${STATUS}"
    exit "${TRAIN_STATUS}"
fi

if ! jq -e \
    --arg backward_extension "${BACKWARD_EXTENSION}" \
    --arg backward_extension_sha256 "${BACKWARD_EXTENSION_SHA256}" \
    --arg control_source "${BACKWARD_CONTROL_SOURCE}" \
    --arg control_sha256 "${BACKWARD_CONTROL_SHA256}" \
    --argjson control_bytes "${BACKWARD_CONTROL_BYTES}" '
    .configuration.backward_control_provenance as $control
    | .schema == "llama12b_real_tokens_training_v3"
    and .configuration.layers == 16
    and .configuration.rounds == 3
    and .configuration.training_batches == 3
    and .configuration.parameter_count >= 1200000000
    and .configuration.parameter_count < 1300000000
    and .configuration.gradient_clip_norm == null
    and .configuration.backward_extension.path == $backward_extension
    and .configuration.backward_extension.sha256 == $backward_extension_sha256
    and .configuration.backward_exp2_degree == 2
    and .configuration.backward_exp2_period == 0
    and .configuration.backward_exp2_requested_degree == 2
    and .configuration.backward_exp2_requested_period == 0
    and .configuration.backward_exp2_policy.mode == "explicit"
    and $control.mode == "precomposed"
    and $control.source.path == $control_source
    and $control.source.sha256 == $control_sha256
    and $control.source.bytes == $control_bytes
    and $control.required_constants.TK_DIRECT_TMA_DKDV == true
    and $control.required_constants.TK_FP8_P_STORAGE == "tmem"
    and $control.required_constants.TK_DETACHED_FP8_P_TMEM == false
    and $control.required_runtime_policy.owner_fused_dq_scale == false
    and (.configuration.backward_control_route_provenance
        | all(.[]; . == $control))
    and .configuration.mx_experimental_split_v_backward == false
    and .configuration.mx_projection_publication_topology.v_backward_source
        == "represented_mxfp4_codes"
    and .configuration.mx_backward_forward_probability_replay == true
    and .configuration.mx_backward_forward_probability_scale_handoff == true
    and .configuration.mx_probability_replay_provenance.control_mode
        == "precomposed"
    and .configuration.mx_probability_replay_provenance.control_source
        == $control
    and .configuration.mx_probability_replay_provenance.patch == null
    and .configuration.mx_probability_replay_provenance.generated_control.sha256
        == $control_sha256
    and .configuration.mx_probability_replay_provenance.generated_control.bytes
        == $control_bytes
    and .configuration.mx_forward_topology.causal == true
    and .configuration.mx_forward_topology.causal_interleaved_kv == true
    and .configuration.mx_forward_topology.pv_format == "mxfp4_e8m0_block32"
    and .configuration.mx_forward_topology.mx_pwl_exp2_mode == 23
    and .configuration.mx_forward_topology.mx_mode23_native_density == 4
    and .configuration.mx_forward_topology.mx_mode23_native_quarter_mask == 15
    and .configuration.mx_forward_topology.mx_mode23_native_stage_mask == 3
    and .configuration.mx_forward_topology.p_scale_publication_supported == true
    and .configuration.mx_forward_extension.sha256
        == "62618c7869d4656f762f14d0e09465ba4c753af29ed10dde21af82ca3c722e78"
    and .configuration.fp8_forward_extension.sha256
        == "fba5d54ed080d5808342058bfa2c781d9ff55cc8c5e37e373235e72e0a1e70c3"
    and .configuration.train_tokens.sha256
        == "a5c9cfa199bf5f0b2dd19126777771fc555028222e31a3905148e9fff3c9e5ff"
    and .configuration.validation_tokens.sha256
        == "9ef18a6217bead20fb7775d594a006a57eb6aefe1490bdd75e6fad50bd73e87a"
    and (.records | keys | sort)
        == (["bf16_cute", "nvfp4_qk_fp8_pv_exact", "nvfp4_qk_mxfp4_pv"] | sort)
    and (.records | all(.[]; length == 3))
    and (.records | all(.[]; all(.[];
        .finite == true
        and .failure_stage == null
        and .gradient_preclip_total_norm == null
        and .gradient_was_clipped == false
        and .gradient_clip_error == null
    )))
    and (.validation_history | length == 4)
    and (.validation_history | all(.[];
        .routes | all(.[]; (.mean_loss | isfinite))))
    and (.routes | all(.[]; .training.all_steps_finite == true))
    and .source.git.tracked_dirty == false
' "${PARTIAL_OUTPUT}" >/dev/null; then
    jq -n \
        --arg state "artifact_validation_failed" \
        --arg started_utc "${STARTED_UTC}" \
        --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{state: $state, started_utc: $started_utc,
          finished_utc: $finished_utc, exit_code: 7}' >"${STATUS_TMP}"
    mv "${STATUS_TMP}" "${STATUS}"
    exit 7
fi

mv "${PARTIAL_OUTPUT}" "${OUTPUT}"
readonly OUTPUT_SHA256="$(sha256sum "${OUTPUT}" | awk '{print $1}')"
jq -n \
    --arg state "complete" \
    --arg started_utc "${STARTED_UTC}" \
    --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg output "${OUTPUT}" \
    --arg output_sha256 "${OUTPUT_SHA256}" \
    '{state: $state, started_utc: $started_utc,
      finished_utc: $finished_utc, exit_code: 0,
      output: $output, output_sha256: $output_sha256}' >"${STATUS_TMP}"
mv "${STATUS_TMP}" "${STATUS}"
