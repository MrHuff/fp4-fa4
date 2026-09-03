#!/usr/bin/env bash

set -euo pipefail

readonly RUN_ROOT="/workspace/codebases/pv/fp4_matmul_worktrees/dolmino_d4all_calibrated_20260820"
readonly PYTHON_BIN="/workspace/codebases/poly_stuff/.venv-poly/bin/python"
readonly TRAINER="${RUN_ROOT}/tk_fa4/lowp_fa4_bwd/train_llama12b_real_tokens.py"
readonly CORPUS="/tmp/dolmino-dclm-a319f19-dclm0000-prefix20000.jsonl"
readonly TOKENIZER="/workspace/codebases/poly_stuff/low-precision-functions/low-bits-training/assets/hf/Meta-Llama-3.1-8B/tokenizer.json"
readonly BACKWARD_EXTENSION="/tmp/codex-aug19-stable-bwd-rebuild-20260820/tk_fa4/_C_b300_lowp_bwd.cpython-312-aarch64-linux-gnu.so"
readonly BACKWARD_CONTROL="/tmp/codex-mx-replay-alllane-exact-20260820/v2/fmha_bwd_d64_gqa_aug19_exact.py"
readonly MX_EXTENSION="/tmp/codex-d4all-pscale-replay-20260820/_C_cfwd_mx_d4all_pscale_s4096h32kv8d64_20260820.cpython-312-aarch64-linux-gnu.so"
readonly FP8_EXTENSION="/tmp/codex-causal-forward-matrix-d4q01-full-20260820/_C_cfwd_fp8exact0_for_d4q01_s4096h32kv8d64_d4q01_full_20260820.cpython-312-aarch64-linux-gnu.so"
readonly ALL_LANE_PATCH="${RUN_ROOT}/tk_fa4/lowp_fa4_bwd/d64_gqa_forward_mx_probability_replay_all_lane_exact.patch"
readonly ALL_LANE_GATE="${RUN_ROOT}/results/causal_isolated_matrix_20260820/backward/d64_replay_all_lane_exact_stable_projection_s4096.json"
readonly PHYSICAL_GPU=3
readonly EXPECTED_GPU_UUID="GPU-60db269d-3281-dae0-547b-e0d4ecd95e06"
readonly RESULT_DIR="${RUN_ROOT}/results/causal_isolated_matrix_20260820/training"
readonly RESULT_STEM="dolmino_dclm_a319_2k_d4all_alllane_replay_seed20260818_20260820"
readonly OUTPUT="${RESULT_DIR}/${RESULT_STEM}.json"
readonly PARTIAL_OUTPUT="${RESULT_DIR}/${RESULT_STEM}.partial.json"
readonly PROGRESS="${RESULT_DIR}/${RESULT_STEM}.progress.json"
readonly LOG="${RESULT_DIR}/${RESULT_STEM}.log"
readonly STATUS="${RESULT_DIR}/${RESULT_STEM}.status.json"
readonly STATUS_TMP="${STATUS}.tmp"
readonly LAUNCHER_RELATIVE="results/causal_isolated_matrix_20260820/training/run_dolmino_dclm_a319_2k_d4all_alllane_replay_20260820.sh"
readonly BACKWARD_CONTROL_SHA256="cd57e3360082abe4bad7560c51a7793a4e9bfd4d16efc1259b92ce20238b99e1"
readonly BACKWARD_CONTROL_BYTES=220876
readonly DRIFT_WARNING="0.08"
readonly DRIFT_FAILURE="0.15"
readonly DRIFT_WINDOW=32
readonly DRIFT_PATIENCE=3
readonly DRIFT_MINIMUM_UPDATES=256

# The selected control is exact with respect to the authenticated Aug19 replay
# math, but distributes replay conversion across all lanes. Its isolated S4096
# gate measured 694.656 -> 624.864 us backward and 916.032 -> 826.464 us for
# reset+producer+backward while staying within measured atomic repeat noise.

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

check_file "227c38e5aa6e4451eb7ea8c3cc8136741325cd50a5152b9ecf0f8af8446037f3" 70967 "${TRAINER}"
check_file "7c9e3b55b3c1abbfb79412037c7c0f80ba6e16571ed677d1dd6bc4eb14d16e0f" 150441082 "${CORPUS}"
check_file "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa" 9085658 "${TOKENIZER}"
check_file "6480ed82250766c6448fb424a2eba5c8b7d57c68e99c58ea8913ef71c7c7e65e" 17220528 "${BACKWARD_EXTENSION}"
check_file "${BACKWARD_CONTROL_SHA256}" "${BACKWARD_CONTROL_BYTES}" "${BACKWARD_CONTROL}"
check_file "62618c7869d4656f762f14d0e09465ba4c753af29ed10dde21af82ca3c722e78" 1953512 "${MX_EXTENSION}"
check_file "fba5d54ed080d5808342058bfa2c781d9ff55cc8c5e37e373235e72e0a1e70c3" 1813632 "${FP8_EXTENSION}"
check_file "8f80c4e97b70c2f3e818c45086892fe2d2fd45b1d96e1e894a77f09b32d4f765" 8424 "${ALL_LANE_PATCH}"
check_file "57d8ae841f00649422c391b7c80fee1a649357f388493a5c00d4207a14429acc" 57792 "${ALL_LANE_GATE}"

if ! jq -e --arg control_sha256 "${BACKWARD_CONTROL_SHA256}" '
    .passed == true
    and .provenance.controls.optimized.sha256 == $control_sha256
    and .timing.backward_only_including_reset.comparison.baseline_over_optimized_speedup > 1.0
    and .timing.training_order_reset_producer_backward.comparison.baseline_over_optimized_speedup > 1.0
    and (.checks | all(.[]; . == true))
' "${ALL_LANE_GATE}" >/dev/null; then
    echo "the selected all-lane replay gate is not a passing speedup" >&2
    exit 13
fi

if [[ "$(git -C "${RUN_ROOT}/flash-attention" rev-parse HEAD)" \
    != "9743edaf3227a25f6afc4fa7be8b5e8498610553" ]]; then
    echo "flash-attention is not checked out at its pinned commit" >&2
    exit 14
fi
if [[ "$(git -C "${RUN_ROOT}/qutlass" rev-parse HEAD)" \
    != "406e86fb2d7df436e94f825bcda8e59b1a7250a6" ]]; then
    echo "qutlass is not checked out at its pinned commit" >&2
    exit 15
fi
if [[ "$(git -C "${RUN_ROOT}/qutlass/third_party/cutlass" rev-parse HEAD)" \
    != "b2ca083d2bb96c41d9b3c5a930637c641f6669bf" ]]; then
    echo "qutlass CUTLASS is not checked out at its pinned commit" >&2
    exit 16
fi
if [[ ! -f "${RUN_ROOT}/qutlass/third_party/cutlass/examples/python/CuTeDSL/blackwell/fmha_bwd.py" ]]; then
    echo "the pinned CuTe backward source is missing" >&2
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
for path in \
    "${OUTPUT}" \
    "${PARTIAL_OUTPUT}" \
    "${PROGRESS}" \
    "${LOG}" \
    "${STATUS}" \
    "${STATUS_TMP}"; do
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
    --arg backward_control "${BACKWARD_CONTROL}" \
    --arg backward_control_sha256 "${BACKWARD_CONTROL_SHA256}" \
    --argjson backward_control_bytes "${BACKWARD_CONTROL_BYTES}" \
    --argjson drift_warning "${DRIFT_WARNING}" \
    --argjson drift_failure "${DRIFT_FAILURE}" \
    '{state: $state, started_utc: $started_utc, runner_pid: $runner_pid,
      physical_gpu: $physical_gpu, gpu_uuid: $gpu_uuid, git_head: $git_head,
      backward_control: {path: $backward_control,
        sha256: $backward_control_sha256, bytes: $backward_control_bytes},
      drift_gate: {warning_threshold: $drift_warning,
        failure_threshold: $drift_failure}}' >"${STATUS_TMP}"
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
    --backward-control-sha256 "${BACKWARD_CONTROL_SHA256}" \
    --backward-control-bytes "${BACKWARD_CONTROL_BYTES}" \
    --mx-extension "${MX_EXTENSION}" \
    --mx-module _C_cfwd_mx_d4all_pscale_s4096h32kv8d64_20260820 \
    --fp8-extension "${FP8_EXTENSION}" \
    --fp8-module _C_cfwd_fp8exact0_for_d4q01_s4096h32kv8d64_d4q01_full_20260820 \
    --backward-gain 1.0 \
    --mx-backward-gain 1.0 \
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
    --mx-backward-forward-probability-replay \
    --mx-backward-forward-probability-scale-handoff \
    --progress-output "${PROGRESS}" \
    --progress-every 32 \
    --mx-loss-drift-window "${DRIFT_WINDOW}" \
    --mx-loss-drift-warning-threshold "${DRIFT_WARNING}" \
    --mx-loss-drift-failure-threshold "${DRIFT_FAILURE}" \
    --mx-loss-drift-failure-patience "${DRIFT_PATIENCE}" \
    --mx-loss-drift-minimum-updates "${DRIFT_MINIMUM_UPDATES}" \
    --diagnostic-on-drift-warning \
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
        --arg progress "${PROGRESS}" \
        '{state: $state, started_utc: $started_utc,
          finished_utc: $finished_utc, exit_code: $exit_code,
          progress: $progress}' >"${STATUS_TMP}"
    mv "${STATUS_TMP}" "${STATUS}"
    exit "${TRAIN_STATUS}"
fi

if ! jq -e \
    --arg git_head "${GIT_HEAD}" \
    --arg backward_extension "${BACKWARD_EXTENSION}" \
    --arg backward_control "${BACKWARD_CONTROL}" \
    --arg backward_control_sha256 "${BACKWARD_CONTROL_SHA256}" \
    --argjson backward_control_bytes "${BACKWARD_CONTROL_BYTES}" \
    --argjson drift_warning "${DRIFT_WARNING}" \
    --argjson drift_failure "${DRIFT_FAILURE}" '
    .configuration.backward_control_provenance as $control
    | .schema == "llama12b_real_tokens_training_v3"
    and .source.git.head == $git_head
    and .source.git.tracked_dirty == false
    and .configuration.layers == 16
    and .configuration.rounds == 2000
    and .configuration.training_batches == 2000
    and .configuration.validation_batches == 8
    and .configuration.parameter_count >= 1200000000
    and .configuration.parameter_count < 1300000000
    and .configuration.gradient_clip_norm == null
    and .configuration.backward_extension.path == $backward_extension
    and .configuration.backward_extension.sha256 == "6480ed82250766c6448fb424a2eba5c8b7d57c68e99c58ea8913ef71c7c7e65e"
    and .configuration.backward_exp2_degree == 2
    and .configuration.backward_exp2_period == 0
    and .configuration.backward_exp2_requested_degree == 2
    and .configuration.backward_exp2_requested_period == 0
    and .configuration.backward_exp2_policy.mode == "explicit"
    and $control.mode == "precomposed"
    and $control.source.path == $backward_control
    and $control.source.sha256 == $backward_control_sha256
    and $control.source.bytes == $backward_control_bytes
    and $control.required_constants.TK_DIRECT_TMA_DKDV == true
    and $control.required_constants.TK_FP8_P_STORAGE == "tmem"
    and $control.required_constants.TK_DETACHED_FP8_P_TMEM == false
    and $control.required_runtime_policy.owner_fused_dq_scale == false
    and (.configuration.backward_control_route_provenance
        | all(.[]; . == $control))
    and .configuration.mx_experimental_split_v_backward == false
    and .configuration.mx_backward_gain == 1.0
    and .configuration.fp8_backward_gain == 1.0
    and .configuration.mx_backward_reuse_quantized_p == false
    and .configuration.fp8_backward_reuse_quantized_p == false
    and .configuration.mx_effective_backward_match_forward_operands == true
    and .configuration.fp8_effective_backward_match_forward_operands == true
    and .configuration.mx_per_block_qk_scales == true
    and .configuration.fp8_per_block_qk_scales == true
    and .configuration.mx_qkv_projection_format == "e4m3"
    and .configuration.fp8_qkv_projection_format == "e4m3"
    and .configuration.projection_weight_scaling == "2d"
    and .configuration.v_mxfp4_scaling == "1d"
    and .configuration.mx_projection_publication_topology.v_backward_source
        == "represented_mxfp4_codes"
    and .configuration.fp8_projection_publication_topology.v_backward_source
        == "projection_accumulator_e4m3"
    and .configuration.mx_backward_forward_probability_replay == true
    and .configuration.mx_backward_forward_probability_scale_handoff == true
    and .configuration.mx_backward_forward_probability_replay_requested == true
    and .configuration.mx_backward_forward_probability_scale_handoff_requested == true
    and .configuration.mx_probability_replay_provenance.control_mode
        == "precomposed"
    and .configuration.mx_probability_replay_provenance.control_source
        == $control
    and .configuration.mx_probability_replay_provenance.patch == null
    and .configuration.mx_probability_replay_provenance.generated_control.sha256
        == $backward_control_sha256
    and .configuration.mx_probability_replay_provenance.generated_control.bytes
        == $backward_control_bytes
    and .configuration.mx_forward_topology.causal == true
    and .configuration.mx_forward_topology.causal_interleaved_kv == true
    and .configuration.mx_forward_topology.pv_format == "mxfp4_e8m0_block32"
    and .configuration.mx_forward_topology.mx_pwl_exp2_mode == 23
    and .configuration.mx_forward_topology.mx_mode23_native_density == 4
    and .configuration.mx_forward_topology.mx_mode23_native_quarter_mask == 15
    and .configuration.mx_forward_topology.mx_mode23_native_stage_mask == 3
    and .configuration.mx_forward_topology.p_scale_publication_supported == true
    and .configuration.fp8_forward_topology.causal == true
    and .configuration.fp8_forward_topology.pv_format == "e4m3_fp8"
    and .configuration.mx_forward_extension.sha256
        == "62618c7869d4656f762f14d0e09465ba4c753af29ed10dde21af82ca3c722e78"
    and .configuration.fp8_forward_extension.sha256
        == "fba5d54ed080d5808342058bfa2c781d9ff55cc8c5e37e373235e72e0a1e70c3"
    and .configuration.corpus_sha256
        == "7c9e3b55b3c1abbfb79412037c7c0f80ba6e16571ed677d1dd6bc4eb14d16e0f"
    and .configuration.tokenizer_sha256
        == "76e48799b099d43365bd24ccd8ecc5aedac831718da780552f03b0a6eb4412aa"
    and .configuration.train_tokens.sha256
        == "ca015efd3423c24a471099f1db18cc532c4eb25216a3183e125a398e2ccaf737"
    and .configuration.validation_tokens.sha256
        == "821b34a143eec2480ebe6619e7a87842d710665d503422401419aa1c37b60c4a"
    and .configuration.mx_loss_drift_gate.window == 32
    and .configuration.mx_loss_drift_gate.warning_threshold == $drift_warning
    and .configuration.mx_loss_drift_gate.failure_threshold == $drift_failure
    and .configuration.mx_loss_drift_gate.failure_patience == 3
    and .configuration.mx_loss_drift_gate.minimum_updates == 256
    and .configuration.diagnostic_on_drift_warning == true
    and .configuration.progress_every == 32
    and (.records | keys | sort)
        == (["bf16_cute", "nvfp4_qk_fp8_pv_exact", "nvfp4_qk_mxfp4_pv"] | sort)
    and (.records | all(.[]; length == 2000))
    and (.records | all(.[]; all(.[];
        .finite == true
        and .failure_stage == null
        and .gradient_preclip_total_norm == null
        and .gradient_was_clipped == false
        and .gradient_clip_error == null
    )))
    and (.validation_history | length == 41)
    and (.validation_history | all(.[];
        .routes | all(.[]; (.mean_loss | isfinite))))
    and (.routes | all(.[];
        .training.all_steps_finite == true
        and .timing.timing_fallback_used == false
        and .timing.timed_records > 0))
    and .drift_gate.observed_rounds == 2000
    and .drift_gate.failed == false
' "${PARTIAL_OUTPUT}" >/dev/null; then
    jq -n \
        --arg state "artifact_validation_failed" \
        --arg started_utc "${STARTED_UTC}" \
        --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{state: $state, started_utc: $started_utc,
          finished_utc: $finished_utc, exit_code: 24}' >"${STATUS_TMP}"
    mv "${STATUS_TMP}" "${STATUS}"
    exit 24
fi

if ! jq -e '
    .schema == "llama12b_real_tokens_training_progress_v1"
    and .state == "complete"
    and .last_complete_round == 1999
    and (.route_record_counts | all(.[]; . == 2000))
    and .drift_gate.observed_rounds == 2000
    and .drift_gate.failed == false
' "${PROGRESS}" >/dev/null; then
    jq -n \
        --arg state "progress_validation_failed" \
        --arg started_utc "${STARTED_UTC}" \
        --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{state: $state, started_utc: $started_utc,
          finished_utc: $finished_utc, exit_code: 25}' >"${STATUS_TMP}"
    mv "${STATUS_TMP}" "${STATUS}"
    exit 25
fi

mv "${PARTIAL_OUTPUT}" "${OUTPUT}"
readonly OUTPUT_SHA256="$(sha256sum "${OUTPUT}" | awk '{print $1}')"
readonly PROGRESS_SHA256="$(sha256sum "${PROGRESS}" | awk '{print $1}')"
jq -n \
    --arg state "complete" \
    --arg started_utc "${STARTED_UTC}" \
    --arg finished_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg output "${OUTPUT}" \
    --arg output_sha256 "${OUTPUT_SHA256}" \
    --arg progress "${PROGRESS}" \
    --arg progress_sha256 "${PROGRESS_SHA256}" \
    '{state: $state, started_utc: $started_utc,
      finished_utc: $finished_utc, exit_code: 0,
      output: $output, output_sha256: $output_sha256,
      progress: $progress, progress_sha256: $progress_sha256}' \
    >"${STATUS_TMP}"
mv "${STATUS_TMP}" "${STATUS}"
