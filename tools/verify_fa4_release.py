#!/usr/bin/env python3
"""Validate the source-complete FP4 FlashAttention reproduction checkout.

The verifier is intentionally CPU-only and uses the Python standard library.
It authenticates source pins, submodule revisions, key materialized files, the
checked source inventory, route boundaries, and credential hygiene. It does not
claim that CUDA code builds or that a GPU measurement matches a paper value.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "release" / "manifest.json"
INVENTORY_PATH = ROOT / "release" / "source_files.sha256"

EXPECTED_PINS = {
    "torchtitan": "20b3de7585696c327bd5aa9f9627f0300abdbf9d",
    "historical_training_integration": "e7db209b0c7017c415fdd66e04e85f96ae24f276",
    "causal_fa4_kernels": "4590537f1479e1a7e847f2783e9ab7aa7f11b975",
    "d64_training_source_epoch": "cd59dda37ebf22e0d77b9c9d6851ec164b86e3af",
    "d64_v416_source_epoch": "713819d730369ad9e73ded1aedbc301c261f1130",
    "dense_score_diagnostic": "aa02150404418859e33d1ff99fb46543244b9b70",
    "historical_noncausal_forward": "cfc06dadf684279f657ab66254a3a074be4ee3a9",
    "hao_comparator": "9b0abefdbbbe4d0da1d4e0c7aa128e3338c4b247",
    "technical_report": "4c394504998f653aa702d030c5f98864dcf34c75",
}
EXPECTED_PUBLICATION_AUTHORIZATION = {
    "status": "confirmed_by_authorized_owner",
    "confirmed_date": "2026-09-03",
    "scope": ["project source", "manuscript", "retained publication assets"],
    "outbound_license": "Apache-2.0",
    "copyright_notice": "Copyright (c) 2026 Graphcore Ltd.",
}
EXPECTED_PUBLIC_ROOT = "edde9bcbc5567fe4e69b4faa2c36e667764410e4"
EXPECTED_PUBLIC_HISTORY_POLICY = "parentless_root_with_ordinary_descendants"
EXPECTED_DEPENDENCIES = {
    "ThunderKittens": "9ee85b4afcdea1478b4dda8bb01f8907ab7edb0b",
    "SageAttention": "681004015c42b8ac543302235652e618ac66f966",
    "flash-attention": "b531f67557b8213db339492cd1629e721776f758",
    "flash-attention/csrc/cutlass": "7127592069c2fe01b041e174ba4345ef9b279671",
    "qutlass": "406e86fb2d7df436e94f825bcda8e59b1a7250a6",
    "qutlass/third_party/cutlass": "b2ca083d2bb96c41d9b3c5a930637c641f6669bf",
    "cutlass": "acb45938e9cb3e4db8c1d75155b63d31791e0e5d",
}
EXPECTED_CUTE_OVERLAY = {
    "schema": "fa4_flash_attention_runtime_overlay_patch_v1",
    "base_commit": "9743edaf3227a25f6afc4fa7be8b5e8498610553",
    "patch": {
        "path": "patches/flash_attention_fp4_runtime_overlay_9743edaf_20260831.patch",
        "bytes": 112207,
        "sha256": "bc8caf8cd3c860d2bf958a96113a4b97a7987b2350bfed7f54337f0b9ac0cb8a",
    },
    "files": [
        {
            "path": "flash_attn/cute/fp4_flash_bwd_sm100.py",
            "bytes": 289268,
            "sha256": "0e3c152ebcd0c2bf1ef0edc76fa108c0bb04c497d76c056749dbb57b1ed293f2",
            "insertions": 1234,
            "deletions": 209,
        },
        {
            "path": "flash_attn/cute/interface.py",
            "bytes": 159112,
            "sha256": "13a1edbd711ae29141fceb69c54a8a93bc18384511792cebf3ee433ff220cd75",
            "insertions": 189,
            "deletions": 20,
        },
        {
            "path": "flash_attn/cute/mma_sm100_desc.py",
            "bytes": 16101,
            "sha256": "86efe9315696b7bdb7bfa915c7946e301d3e88d089964cb4f27a74c43d604d09",
            "insertions": 43,
            "deletions": 21,
        },
    ],
    "constraints": {
        "d128_public_dispatch": "fail_closed_to_verified_bf16_bridge",
        "native_d128_two_cta_schedule": "may_hang",
        "provenance_marker_excluded": ".lbt_flash_attn_commit",
    },
}
EXPECTED_HAO_NON_CODE_OMISSIONS = [
    ".humanize/bitlesson.md",
    ".humanize/rlcr/2026-05-22_02-40-44/goal-tracker.md",
    ".humanize/rlcr/2026-05-22_02-40-44/round-0-contract.md",
    ".humanize/rlcr/2026-05-22_02-40-44/round-0-summary.md",
    "assets/fa4_paper.pdf",
    "assets/flashattn_banner.jpg",
    "assets/flashattn_banner.pdf",
]
EXPECTED_MATERIALIZED_TREES = {
    "tk_fa4": "edf60a5703e298fd7e7f8c49e8b1541bf68e7a89",
    "TK_quantisation": "e26f1b83d85f9805dcbe726afe2b464450cac84c",
    "baseline_kernels": "5242c6d77a09cbd415b6d11e100657e0810aa4dd",
    "fused_ops": "efb0668033a9eeee8a95b21759664d6d49f5decc",
    "qutlass_binding": "32332ed7b0d26971bb0873647d154eb8fdc6aa65",
    "results": "38f0afdb92b8726db42902eefa7927281f415acc",
    "reproduction/snapshots/forward_cfc06dad/TK_quantisation": (
        "9a0a63b1aa98ca4e377d0fd867b0b764e19d8b4d"
    ),
    "reproduction/snapshots/forward_cfc06dad/tk_fa4": (
        "1dcefd373495bd9fbcd2ca39331daa45fde77132"
    ),
}
EXPECTED_EXPORT_UPSTREAM_TREES = {
    "tk_fa4": "6aafb4201ad6ae618d3724b851680a3c0ec13eb3",
    "TK_quantisation": "c6454d9524e1cd521427411b0dd2a199d0822a25",
    "baseline_kernels": "5242c6d77a09cbd415b6d11e100657e0810aa4dd",
    "fused_ops": "efb0668033a9eeee8a95b21759664d6d49f5decc",
    "qutlass_binding": "32332ed7b0d26971bb0873647d154eb8fdc6aa65",
    "results": "8f90e08e2988a2e3d4684f74f46cfe5011eb18e9",
}
EXPECTED_SHAPE = {
    "architecture": "NVIDIA Blackwell SM100",
    "causal": True,
    "head_dimension": 128,
    "supported_batches": [1, 2, 4],
    "paper_primary_batch": 4,
    "sequence_length": 4096,
    "query_heads": 32,
    "key_value_heads": 8,
}
EXPECTED_PROFILES = [
    {
        "id": "llama8b-d128",
        "artifact_profiles": [
            "llama8b-d128-b1",
            "llama8b-d128-b2",
            "llama8b-d128-b4",
        ],
        **EXPECTED_SHAPE,
        "validation_state": (
            "historical_receipts_present_fresh_clone_gpu_validation_pending"
        ),
    },
    {
        "id": "llama1p2b-d64",
        "artifact_profiles": ["llama1p2b-d64-b16"],
        "architecture": "NVIDIA Blackwell SM100",
        "causal": True,
        "head_dimension": 64,
        "supported_batches": [16],
        "paper_primary_batch": 16,
        "sequence_length": 4096,
        "query_heads": 32,
        "key_value_heads": 8,
        "validation_state": (
            "cpu_contract_validated_fresh_clone_gpu_and_ddp16_pending"
        ),
    },
]
EXPECTED_ROUTES = {
    "bf16_fa4": "reference_control",
    "nvfp4_qk_fp8_pv": "release_candidate",
    "nvfp4_qk_mxfp4_pv": "diagnostic_only",
    "e4m3_proj_nvfp4_qk_fp8_pv": "diagnostic_only",
    "e4m3_proj_nvfp4_qk_mxfp4_pv": "diagnostic_only",
    "d64_bf16_fa4": "reference_control",
    "d64_e4m3_proj_nvfp4_qk_fp8_pv_v416": "release_candidate",
    "d64_e4m3_proj_nvfp4_qk_mxfp4_pv_v416": "diagnostic_only",
}
EXPECTED_D64_SOURCE_EPOCHS = {
    "d64_training_source_epoch": {
        "parent": "1f3cae064fd3bd5c72c713f7dcea53c4b073952d",
        "commit_tree": "046066c1fd54f1f79fe363fbc4a38a37a495060c",
        "tk_fa4_tree": "592fc6dfda76eb561e6b7f8fbfd040648c1a1c40",
        "path": "reproduction/snapshots/d64_training_cd59dda",
        "manifest_sha256": (
            "1e2e00cea3f4be69364f35f97bb4de19893b54ca8a3ae3f124d638ea292a6a76"
        ),
        "inventory_sha256": (
            "123620a79bf3cbc00c246ea3c849b7d6bb014b9bc7cb30d96f75a4ba3dc02b26"
        ),
        "patch_sha256": (
            "82b1212a6d5a94792e9fbc03b499b88cad1fbac066ca572e25072a0c9a8a6d9e"
        ),
        "source_tree_sha256": (
            "7ccb006810b753ab689801efc84e5705a9b614c771d70d5fbcc304971f9e9447"
        ),
    },
    "d64_v416_source_epoch": {
        "parent": "abd3f33104ac885434f1d6136ab5100361de51ee",
        "commit_tree": "4133134eb97b45910962f513fc5d3f71b6f0d1cd",
        "tk_fa4_tree": "8221a0d371a2c2307725b12d3cd0f287d1989ae7",
        "path": "reproduction/snapshots/d64_v416_713819d",
        "manifest_sha256": (
            "470c495f52adbaabdd2cb3734a4d17f5340beab3a60cc9ad0858f0ea42b5dbc2"
        ),
        "inventory_sha256": (
            "574b28e4a73fa4f2eca54f32e8dd004fe1a944f17cf284ec3796e9a9f3ef3f47"
        ),
        "patch_sha256": (
            "bda6edd3f120d770ca201965549681e59af13381db25123234ae113414a689ec"
        ),
        "source_tree_sha256": (
            "69d31f3ab7e7586374fbea3a73dd357bca922e1fb696504761301102399c2118"
        ),
    },
}
REQUIRED_BLOCKERS = {
    "integration_validation_pending",
    "gpu_build_validation_pending",
    "missing_natural_captures",
    "evidence_raw_archives_missing",
    "training_data_identity_incomplete",
    "single_trajectory_and_raw_history_limit",
    "historical_input_pipeline_parity_pending",
    "portable_profile_coverage_incomplete",
    "runtime_dependency_lock_incomplete",
    "historical_runtime_policy_unresolved",
    "batch_hbm_diagnostics_pending",
}
REQUIRED_PATHS = (
    "CMakeLists.txt",
    "CONTINUATION.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "NOTICE",
    "LICENSES/README.md",
    "LICENSES/Apache-2.0.txt",
    "LICENSES/CUTLASS-7127592-b2ca083-LICENSE.txt",
    "LICENSES/CUTLASS-7127592-b2ca083-python-LICENSE.txt",
    "LICENSES/CUTLASS-acb4593-LICENSE.txt",
    "LICENSES/CUTLASS-acb4593-python-LICENSE.txt",
    "LICENSES/FlashAttention-b531f67-LICENSE.txt",
    "LICENSES/NVIDIA-CuTeDSL-EULA.txt",
    "LICENSES/SageAttention-6810040-LICENSE.txt",
    "LICENSES/ThunderKittens-9ee85b4-LICENSE.txt",
    "LICENSES/TorchTitan-20b3de7-LICENSE.txt",
    "LICENSES/TransformerEngine-06b44b8-LICENSE.txt",
    "LICENSES/qutlass-406e86f-LICENSE.txt",
    "THIRD_PARTY_NOTICES.md",
    "docs/development.md",
    "release/SCIENTIFIC_STATE.md",
    "release/NEXT_EXPERIMENTS.md",
    "release/PUBLIC_EXPORT_POLICY.md",
    "release/PUBLIC_SANITIZATION.md",
    "release/routes.json",
    "release/routes.schema.json",
    "benchmarks/bench_nvfp4_gemm.cu",
    "benchmarks/bench_grouped_nvfp4_gemm.cu",
    "patches/flash_attention_fp4_runtime_overlay_9743edaf_20260831.manifest.json",
    "patches/flash_attention_fp4_runtime_overlay_9743edaf_20260831.patch",
    "tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_candidate.cu",
    "tk_fa4/lowp_fa4_bwd/lowp_fa4_bwd.cu",
    "tk_fa4/native_gqa_tk_bwd/Makefile.v509_b4",
    "tk_fa4/native_gqa_tk_bwd/v509_d128_gqa_nvfp4_score_e4m3_qkv_e5m2_dout_b1_exact_s4096_experimental_bshd.cu",
    "TK_quantisation/mxfp4_v3/mxfp4_v3_quantize.cuh",
    "reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_comprehensive_suite.py",
    "reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_bwd/fp4_fa4_bwd.cu",
    "reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_bwd/bwd_host_dispatch.inc",
    "reproduction/snapshots/d64_training_cd59dda/README.md",
    "reproduction/snapshots/d64_training_cd59dda/SHA256SUMS",
    "reproduction/snapshots/d64_training_cd59dda/manifest.json",
    "reproduction/snapshots/d64_training_cd59dda/cd59dda3_source.patch",
    "reproduction/snapshots/d64_v416_713819d/README.md",
    "reproduction/snapshots/d64_v416_713819d/SHA256SUMS",
    "reproduction/snapshots/d64_v416_713819d/manifest.json",
    "reproduction/snapshots/d64_v416_713819d/713819d7_source.patch",
    "release/D64_REPRODUCTION.md",
    "release/LEGACY_LINEAGE.md",
    "release/legacy_backward_makefiles.txt",
    "third_party/hao_flash_attention_fp4/flash_attn/cute/flash_fwd_sm100_fp4.py",
    "results/fp4_fa4_technical_report_v2_20260819/main.tex",
    "results/fp4_fa4_technical_report_v2_20260819/main.pdf",
    "results/fp4_fa4_technical_report_v2_20260819/plot_causal_training.py",
    "results/fp4_fa4_technical_report_v2_20260819/SUBMISSION.md",
    "results/fp4_fa4_technical_report_v2_20260819/prepare_submission_archives.py",
    "results/fp4_fa4_technical_report_v2_20260819/receipts/causal_d128_report_boundaries_20260901.json",
    "reproduction/snapshots/v510_aa021504/README.md",
    "reproduction/snapshots/v510_aa021504/SHA256SUMS",
    "reproduction/snapshots/v510_aa021504/aa021504.patch",
    "reproduction/snapshots/v510_aa021504/tk_fa4/native_gqa_tk_bwd/Makefile.v510",
    "reproduction/snapshots/v510_aa021504/tk_fa4/native_gqa_tk_bwd/v510_d128_gqa_e4m3_score_qkv_e5m2_dout_b1_exact_s4096_experimental_bshd.cu",
    "reproduction/snapshots/v510_aa021504/tk_fa4/native_gqa_tk_bwd/v510_d128_gqa_e4m3_score_qkv_e5m2_dout_b1_exact_s4096_experimental_bshd.cuh",
    "scripts/fa4/run_torchrun.sh",
    "tools/fa4_dataset_manifest.py",
    "tools/fa4_nccl_preflight.py",
    "tools/render_fa4_training_config.py",
    "tools/verify_fa4_data.py",
    "tools/verify_fa4_training_config.py",
    "torchtitan/experiments/fa4/artifacts.py",
    "torchtitan/experiments/fa4/checkpoint.py",
    "torchtitan/experiments/fa4/converters.py",
    "torchtitan/experiments/fa4/data.py",
    "torchtitan/experiments/fa4/exact_lowp_attention.py",
    "torchtitan/experiments/fa4/fa4_attention.py",
    "torchtitan/experiments/fa4/job_config.py",
    "torchtitan/experiments/fa4/optimizer/csrc/adamw_bf16_sr/adamw_bf16_sr.cu",
    "torchtitan/experiments/fa4/optimizer/csrc/adamw_bf16_sr/binding.cpp",
    "torchtitan/experiments/fa4/optimizer/fused_adamw_bf16_sr.py",
    "torchtitan/experiments/fa4/optimizer/optimizer_sr_state.py",
    "torchtitan/experiments/fa4/train.py",
    "torchtitan/experiments/fa4/train_spec.py",
    "torchtitan/experiments/fa4/trainer.py",
    "torchtitan/experiments/fa4/validator.py",
)
EXPECTED_FILE_HASHES = {
    "reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_comprehensive_suite.py": "70f15536ce20e2ccac1151827cf7a0b8ae89a9651a3b4e46c9c5899135a0b43a",
    "reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_bwd/fp4_fa4_bwd.cu": "cdc926fdf359ae7790eb662cb5f67efef9f4a5416089487ba51d7b696555de5b",
    "reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_bwd/bwd_host_dispatch.inc": "dc32667a20627212bfcce0dce5f819710ae8d3a7fb0778232c4a2c27a9a86be8",
    "reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_kernel.inc": "a388a4386c1e96ce987d4d898de902d481f27e1e97be5eac86882745ee9e6259",
    "reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_direct_fp4pv_softmax_reader.inc": "4e12f612e1e1966adc0ee5d68e3d1d25604f2dc7bf2d147c92b19e1d313db8da",
    "reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_direct_kernel.inc": "bd0be8ff877c9ae3a64c38cd9b9bf318d8898c37be52a015d7ab40b2f012989d",
    "reproduction/snapshots/forward_cfc06dad/tk_fa4/fp4_fa4_fwd/hao_direct_softmax_reader.inc": "dd5b4235eeb8cf7c02a72800a178c904a2ac15fb0a950dd37e0501ffa6e98049",
    "third_party/hao_flash_attention_fp4/flash_attn/cute/flash_fwd_sm100_fp4.py": "00c6856075c55bae3d53c1d49c913a8b3b17cd997cd7634714defb1533034037",
    "third_party/hao_flash_attention_fp4/flash_attn/cute/benchmarks/bench_fp4.py": "5ca604ed90ebc4e1ffe7f606fc97f3c774d2ac3869bea9092237d450cd2f85a4",
    "patches/historical_hao_suite_cfc06dad.patch": "8301b554e3912e6fd24735a5607ea5b35e5f55986397ad3b2c42b0adadc72a1b",
    "patches/hao_flash_attention_fp4_9b0abef_compat.patch": "448aac4ea9eea45517259de3c315de3f9062189243febb62439de42c4e799ea5",
    "patches/flash_attention_fp4_runtime_overlay_9743edaf_20260831.manifest.json": "3559f8402156ed08f4c873592e3189f5919e3136107ca41898a3db9ed4ada315",
    "patches/flash_attention_fp4_runtime_overlay_9743edaf_20260831.patch": "bc8caf8cd3c860d2bf958a96113a4b97a7987b2350bfed7f54337f0b9ac0cb8a",
    "reproduction/snapshots/v510_aa021504/aa021504.patch": "704fe124c17891ba3eb1f072532aad8a6958fde859c86bf91e74fc22c3179a37",
    "reproduction/snapshots/v510_aa021504/SHA256SUMS": "ee51c433b371c5c5c7cbad9d052599cb3714df84203b0d891623195d7450713b",
    "results/fp4_fa4_technical_report_v2_20260819/main.pdf": "4dc4a8db31b184150c3e5613da1de4c9e87c776c4772334b9f0a9996c55d53e0",
}

EXPECTED_DEVELOPMENT_ROUTES = {
    "bf16_fa4_control": "reference_control",
    "direct_p_fast_nv_qk_mx_pv": "historical_replay",
    "direct_p_accurate_nv_qk_mx_pv": "historical_replay",
    "hao_nvfp4_qk_fp8_pv_comparator": "pinned_comparator",
    "causal_nvfp4_qk_fp8_pv_v509": "release_candidate",
    "causal_nvfp4_qk_safe_mxfp4_pv_v509": "diagnostic",
    "causal_e4m3_projection_fp8_pv_v509": "diagnostic",
    "causal_e4m3_projection_mxfp4_pv_v509": "diagnostic",
    "causal_d64_e4m3_projection_fp8_pv_v416": "release_candidate",
    "causal_d64_e4m3_projection_mxfp4_pv_v416": "diagnostic",
    "causal_mxfp4_pv_shift16_unsafe": "disabled_fail_closed",
    "v416_d64_represented_e4m3_backward": "release_candidate",
    "v501_represented_e4m3_backward": "diagnostic",
    "v503_common_row_mxfp4_v_backward": "diagnostic",
    "v506_direct_common_row_mxfp4_v_publication": "disabled_fail_closed",
    "v507_four_anchor_mxfp4_v_backward": "diagnostic",
    "v508_native_score_e4m3_dout_backward": "disabled_fail_closed",
    "v509_quantized_qk_replay": "release_candidate",
    "v510_dense_e4m3_score_e5m2_dout_snapshot": "preserved_unpromoted",
    "historical_cfc06dad_backward_prototypes": "preserved_unpromoted",
    "direct_cute_fp4_qk_backward_d128": "disabled_fail_closed",
}
EXPECTED_CLONE_AUDIT = {
    "status": "passed_for_audited_commit",
    "scope": "historical_private_candidate",
    "audited_commit": "f04ed49bfbe9820c09f34a5f622d18998e873467",
    "receipt": "release/audits/remote_clone_f04ed49b_20260902.json",
}
EXPECTED_CLONE_AUDIT_TREES = {
    "tk_fa4": "6aafb4201ad6ae618d3724b851680a3c0ec13eb3",
    "results": "d7e3a0bbf95a261b02ddeb8a86b07324089727d5",
}
EXPECTED_CLONE_AUDIT_PDF = (
    "6997c94204241b45a13f11877f473f09e6f595655c208969f473da380c31f640"
)

SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "AWS access-key identifier": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "assigned cloud or service secret": re.compile(
        rb"(?i)\b(?:aws_secret_access_key|aws_session_token|wandb_api_key)"
        rb"\s*[=:]\s*['\"]?[^\s'\"]+"
    ),
    "GitHub token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "fine-grained GitHub token": re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "JSON Web Token": re.compile(
        rb"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    ),
    "assigned bearer or access token": re.compile(
        rb"(?i)\b(?:authorization|bearer_token|access_token|hf_token)"
        rb"\s*[=:]\s*['\"]?(?:Bearer\s+)?[A-Za-z0-9._~+/=-]{20,}"
    ),
    "credential-bearing HTTPS URL": re.compile(
        rb"(?i)\bhttps?://[^/\s:@]+:[^@\s/]+@[^\s'\"<>]+"
    ),
    "AWS presigned HTTPS parameter": re.compile(
        rb"(?i)[?&]X-Amz-(?:Credential|Signature)="
    ),
    "object-store URI": re.compile(rb"(?i)\bs3://[^\s'\"<>]+"),
}


class VerificationError(RuntimeError):
    """A release invariant was not satisfied."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _expected_public_history() -> dict[str, str]:
    return {
        "root_commit": EXPECTED_PUBLIC_ROOT,
        "policy": EXPECTED_PUBLIC_HISTORY_POLICY,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict[str, Any]:
    try:
        value = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read manifest: {exc}") from exc
    _require(isinstance(value, dict), "manifest root must be an object")
    return value


def _verify_manifest(manifest: dict[str, Any]) -> None:
    _require(manifest.get("schema_version") == 2, "unsupported schema_version")
    project = manifest.get("project")
    _require(isinstance(project, dict), "project must be an object")
    _require(project.get("name") == "fp4-fa4", "unexpected project name")
    _require(
        project.get("release_state") == "source_complete_validation_pending",
        "unexpected release state",
    )
    _require(
        project.get("visibility") in {"private", "public"},
        "release visibility must be private or public",
    )
    _require(
        project.get("publication_authorization") == EXPECTED_PUBLICATION_AUTHORIZATION,
        "publication authorization or outbound license changed",
    )
    if project.get("visibility") == "public":
        _require(
            project.get("public_history") == _expected_public_history(),
            "public history root or update policy changed",
        )
    _require(
        project.get("offline_clone_audit") == EXPECTED_CLONE_AUDIT,
        "offline clone audit declaration changed",
    )

    pins = manifest.get("source_pins")
    _require(isinstance(pins, dict), "source_pins must be an object")
    for name, expected_commit in EXPECTED_PINS.items():
        record = pins.get(name)
        _require(isinstance(record, dict), f"missing source pin: {name}")
        commit = record.get("commit")
        _require(commit == expected_commit, f"incorrect source pin: {name}")
        _require(
            isinstance(commit, str)
            and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
            f"invalid commit syntax: {name}",
        )

    for name, expected in EXPECTED_D64_SOURCE_EPOCHS.items():
        record = pins[name]
        for field, expected_value in expected.items():
            _require(
                record.get(field) == expected_value,
                f"D64 source epoch {name} changed field {field}",
            )
        snapshot_root = ROOT / expected["path"]
        _require(
            _sha256(snapshot_root / "manifest.json") == expected["manifest_sha256"],
            f"D64 source epoch manifest changed: {name}",
        )
        _require(
            _sha256(snapshot_root / "SHA256SUMS") == expected["inventory_sha256"],
            f"D64 source epoch inventory changed: {name}",
        )

    dense_score = pins["dense_score_diagnostic"]
    _require(
        dense_score.get("parent") == "5d8512f9ce1a36c2fdd6475ef75f327e213a7c45",
        "dense-score diagnostic parent changed",
    )
    _require(
        dense_score.get("commit_tree") == "ca712fd4547dd50d776fe157f217eeabf53cc847",
        "dense-score diagnostic commit tree changed",
    )
    _require(
        dense_score.get("tk_fa4_tree") == "5966739ff26dcfa9512f307422e5ffaac731a13f",
        "dense-score diagnostic tk_fa4 tree changed",
    )
    _require(
        dense_score.get("path") == "reproduction/snapshots/v510_aa021504",
        "dense-score diagnostic path changed",
    )
    _require(
        dense_score.get("patch_sha256")
        == EXPECTED_FILE_HASHES["reproduction/snapshots/v510_aa021504/aa021504.patch"],
        "dense-score diagnostic patch declaration changed",
    )
    _require(
        dense_score.get("inventory_sha256")
        == EXPECTED_FILE_HASHES["reproduction/snapshots/v510_aa021504/SHA256SUMS"],
        "dense-score diagnostic inventory declaration changed",
    )

    exports = manifest.get("source_exports")
    _require(isinstance(exports, list), "source_exports must be an array")
    exports_by_path = {
        item.get("path"): item for item in exports if isinstance(item, dict)
    }
    _require(
        len(exports_by_path) == len(exports),
        "source_exports contains a duplicate or malformed path",
    )
    _require(
        set(exports_by_path) == set(EXPECTED_EXPORT_UPSTREAM_TREES),
        "source export set changed",
    )
    for path, expected_upstream_tree in EXPECTED_EXPORT_UPSTREAM_TREES.items():
        record = exports_by_path.get(path)
        _require(isinstance(record, dict), f"missing source export: {path}")
        _require(
            record.get("upstream_tree") == expected_upstream_tree,
            f"manifest upstream tree changed: {path}",
        )
        _require(
            record.get("materialized_export_tree") == EXPECTED_MATERIALIZED_TREES[path],
            f"manifest materialized tree changed: {path}",
        )

    historical = pins["historical_noncausal_forward"]
    _require(
        historical.get("upstream_tk_fa4_tree")
        == "33312c0d36a221b5d6a20b8a3a3a79d2cd7cff42",
        "historical upstream tk_fa4 tree changed",
    )
    _require(
        historical.get("materialized_tk_fa4_tree")
        == EXPECTED_MATERIALIZED_TREES[
            "reproduction/snapshots/forward_cfc06dad/tk_fa4"
        ],
        "historical materialized tk_fa4 tree changed",
    )
    _require(
        historical.get("historical_backward_tree")
        == "dd35ecca9db03cdbb063af7e4b3762438b9d5cca",
        "historical backward tree changed",
    )
    _require(
        {
            "upstream_tk_fa4_paths": historical.get("upstream_tk_fa4_paths"),
            "materialized_source_development_paths": historical.get(
                "materialized_source_development_paths"
            ),
            "byte_exact_source_development_paths": historical.get(
                "byte_exact_source_development_paths"
            ),
            "excluded_duplicate_result_paths": historical.get(
                "excluded_duplicate_result_paths"
            ),
        }
        == {
            "upstream_tk_fa4_paths": 150,
            "materialized_source_development_paths": 126,
            "byte_exact_source_development_paths": 125,
            "excluded_duplicate_result_paths": 24,
        },
        "historical source path accounting changed",
    )
    _require(
        historical.get("replay_patch_sha256")
        == EXPECTED_FILE_HASHES["patches/historical_hao_suite_cfc06dad.patch"],
        "historical replay patch declaration changed",
    )
    _require(
        historical.get("materialized_suite_sha256")
        == EXPECTED_FILE_HASHES[
            "reproduction/snapshots/forward_cfc06dad/tk_fa4/"
            "fp4_fa4_fwd/hao_comprehensive_suite.py"
        ],
        "historical replay suite declaration changed",
    )
    _require(
        historical.get("materialized_quantization_tree")
        == EXPECTED_MATERIALIZED_TREES[
            "reproduction/snapshots/forward_cfc06dad/TK_quantisation"
        ],
        "historical quantization tree declaration changed",
    )
    hao = pins["hao_comparator"]
    _require(
        hao.get("compatibility_patch_sha256")
        == EXPECTED_FILE_HASHES["patches/hao_flash_attention_fp4_9b0abef_compat.patch"],
        "HAO compatibility patch declaration changed",
    )
    _require(
        hao.get("patched_kernel_sha256")
        == EXPECTED_FILE_HASHES[
            "third_party/hao_flash_attention_fp4/flash_attn/cute/"
            "flash_fwd_sm100_fp4.py"
        ],
        "HAO kernel declaration changed",
    )
    _require(
        hao.get("patched_benchmark_sha256")
        == EXPECTED_FILE_HASHES[
            "third_party/hao_flash_attention_fp4/flash_attn/cute/benchmarks/"
            "bench_fp4.py"
        ],
        "HAO benchmark declaration changed",
    )
    _require(
        hao.get("excluded_non_code_paths") == EXPECTED_HAO_NON_CODE_OMISSIONS,
        "HAO non-code omission declaration changed",
    )
    report = pins["technical_report"]
    _require(
        report.get("results_tree") == EXPECTED_EXPORT_UPSTREAM_TREES["results"],
        "technical-report upstream tree changed",
    )
    _require(
        report.get("materialized_export_tree")
        == EXPECTED_MATERIALIZED_TREES["results"],
        "technical-report materialized tree changed",
    )
    paper_path = "results/fp4_fa4_technical_report_v2_20260819/main.pdf"
    _require(
        report.get("pdf_sha256") == EXPECTED_FILE_HASHES[paper_path],
        "technical-report PDF declaration changed",
    )

    _require(
        manifest.get("dependency_pins") == EXPECTED_DEPENDENCIES,
        "dependency pins changed",
    )
    _require(
        manifest.get("supported_shape") == EXPECTED_SHAPE, "supported shape changed"
    )
    _require(
        manifest.get("supported_profiles") == EXPECTED_PROFILES,
        "supported profile matrix changed",
    )

    routes = manifest.get("route_matrix")
    _require(isinstance(routes, list), "route_matrix must be an array")
    _require(
        all(
            isinstance(route, dict) and isinstance(route.get("id"), str)
            for route in routes
        ),
        "route_matrix contains a malformed route",
    )
    route_levels = {
        route.get("id"): route.get("support_level")
        for route in routes
        if isinstance(route, dict)
    }
    _require(
        len(route_levels) == len(routes), "route_matrix contains duplicate route IDs"
    )
    _require(route_levels == EXPECTED_ROUTES, "route matrix or support level changed")

    _require(
        manifest.get("development_route_catalog")
        == {
            "path": "release/routes.json",
            "schema": "release/routes.schema.json",
            "legacy_backward_inventory": "release/legacy_backward_makefiles.txt",
        },
        "development route catalog declaration changed",
    )

    blockers = manifest.get("blockers")
    _require(isinstance(blockers, list), "blockers must be an array")
    blocker_ids = {
        blocker.get("id")
        for blocker in blockers
        if isinstance(blocker, dict) and isinstance(blocker.get("id"), str)
    }
    _require(
        len(blocker_ids) == len(blockers),
        "blockers contains a duplicate or malformed ID",
    )
    _require(
        REQUIRED_BLOCKERS <= blocker_ids, "one or more required blockers are missing"
    )

    release_files = manifest.get("release_files")
    _require(isinstance(release_files, list), "release_files must be an array")
    _require(
        len(release_files) == len(set(release_files)),
        "release_files contains duplicates",
    )
    for relative_name in release_files:
        _require(isinstance(relative_name, str), "release file path must be a string")
        _require(
            (ROOT / relative_name).is_file(), f"missing release file: {relative_name}"
        )


def _run_git(*args: str, cwd: Path | None = None) -> str:
    repository = ROOT if cwd is None else cwd
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise VerificationError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _git_bytes(revision_and_path: str, *, cwd: Path | None = None) -> bytes:
    repository = ROOT if cwd is None else cwd
    result = subprocess.run(
        ["git", "show", revision_and_path],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise VerificationError(f"git show {revision_and_path} failed: {detail}")
    return result.stdout


def _is_ancestor(commit: str) -> bool:
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if exists.returncode != 0:
        return False
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        detail = result.stderr.strip() or result.stdout.strip()
        raise VerificationError(
            f"git merge-base --is-ancestor {commit} HEAD failed: {detail}"
        )
    return result.returncode == 0


def _verify_history_boundary(manifest: dict[str, Any]) -> str:
    """Authenticate private ancestry or the pinned public-root lineage.

    The public root deliberately cannot prove private historical objects
    through Git ancestry: those objects must not be present in its repository.
    Once published, ordinary commits may descend from that parentless root.
    Historical source identities remain authenticated by the materialized
    tree, hashes, and provenance records checked elsewhere in this verifier.
    """

    project = manifest["project"]
    visibility = project["visibility"]
    base = manifest["source_pins"]["torchtitan"]["commit"]
    audited_commit = project["offline_clone_audit"]["audited_commit"]

    if visibility == "private":
        _require(_is_ancestor(base), "TorchTitan base is not an ancestor of HEAD")
        _require(
            _is_ancestor(audited_commit),
            "clone-audited commit is not an ancestor of HEAD",
        )
        return "private_history"

    public_history = project.get("public_history")
    _require(
        public_history == _expected_public_history(),
        "public history root or update policy changed",
    )
    public_root = EXPECTED_PUBLIC_ROOT
    _require(
        _is_ancestor(public_root),
        "pinned parentless public root is not an ancestor of HEAD",
    )
    root_record = _run_git("rev-list", "--parents", "-n", "1", public_root).split()
    _require(
        root_record == [public_root],
        "pinned public root must remain parentless",
    )
    reachable_roots = set(_run_git("rev-list", "--max-parents=0", "--all").splitlines())
    _require(
        reachable_roots == {public_root},
        "public history must have exactly the pinned parentless root",
    )
    _require(
        not _is_ancestor(base) and not _is_ancestor(audited_commit),
        "public export must not retain private project ancestry",
    )
    unreachable = _run_git("fsck", "--full", "--no-reflogs", "--unreachable")
    _require(
        not unreachable,
        "public export contains unreachable root-repository objects; verify a clean clone",
    )
    return "public_export"


def _verify_git_state(manifest: dict[str, Any]) -> None:
    worktree_state = _run_git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    _require(not worktree_state, "release worktree or a submodule is not clean")

    history_mode = _verify_history_boundary(manifest)

    for path, expected_tree in EXPECTED_MATERIALIZED_TREES.items():
        actual_tree = _run_git("rev-parse", f"HEAD:{path}")
        _require(actual_tree == expected_tree, f"materialized tree changed: {path}")

    for path in (
        "ThunderKittens",
        "SageAttention",
        "flash-attention",
        "qutlass",
        "cutlass",
    ):
        fields = _run_git("ls-files", "--stage", "--", path).split()
        _require(
            len(fields) >= 3 and fields[0] == "160000",
            f"{path} is not a staged gitlink",
        )
        _require(fields[1] == EXPECTED_DEPENDENCIES[path], f"incorrect gitlink: {path}")
        checkout = ROOT / path
        _require((checkout / ".git").exists(), f"submodule is not initialized: {path}")
        _require(
            _run_git("rev-parse", "HEAD", cwd=checkout) == EXPECTED_DEPENDENCIES[path],
            f"checked-out submodule revision changed: {path}",
        )
        _require(
            not _run_git(
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
                cwd=checkout,
            ),
            f"submodule worktree is not clean: {path}",
        )

    nested = {
        "flash-attention/csrc/cutlass": (ROOT / "flash-attention", "csrc/cutlass"),
        "qutlass/third_party/cutlass": (ROOT / "qutlass", "third_party/cutlass"),
    }
    for name, (repo, path) in nested.items():
        fields = _run_git("ls-tree", "HEAD", "--", path, cwd=repo).split()
        _require(
            len(fields) >= 3 and fields[0] == "160000",
            f"missing nested gitlink: {name}",
        )
        _require(
            fields[2] == EXPECTED_DEPENDENCIES[name],
            f"incorrect nested gitlink: {name}",
        )
        checkout = repo / path
        _require(
            (checkout / ".git").exists(), f"nested submodule is not initialized: {name}"
        )
        _require(
            _run_git("rev-parse", "HEAD", cwd=checkout) == EXPECTED_DEPENDENCIES[name],
            f"checked-out nested submodule revision changed: {name}",
        )
        _require(
            not _run_git(
                "status", "--porcelain=v1", "--untracked-files=all", cwd=checkout
            ),
            f"nested submodule worktree is not clean: {name}",
        )

    audit_path = ROOT / EXPECTED_CLONE_AUDIT["receipt"]
    try:
        audit = json.loads(
            audit_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read clone audit receipt: {exc}") from exc
    _require(isinstance(audit, dict), "clone audit receipt must be an object")
    _require(
        audit.get("schema") == "fa4_release_clone_audit_v1",
        "clone audit schema changed",
    )
    _require(
        audit.get("audited_commit") == EXPECTED_CLONE_AUDIT["audited_commit"],
        "clone audit commit does not match its manifest declaration",
    )
    _require(
        audit.get("visibility_at_audit") == "private", "clone audit was not private"
    )
    audited_commit = EXPECTED_CLONE_AUDIT["audited_commit"]
    checks = audit.get("checks")
    _require(isinstance(checks, dict), "clone audit checks must be an object")
    for check in (
        "remote_head_matches",
        "root_and_nested_gitlinks_match",
        "source_inventory_passed",
        "release_verifier_passed",
        "offline_paper_reproduction_passed",
        "offline_reproduction_left_clean_tree",
    ):
        _require(checks.get(check) is True, f"clone audit check did not pass: {check}")
    paper_path = "results/fp4_fa4_technical_report_v2_20260819/main.pdf"
    _require(
        checks.get("pdf_sha256") == EXPECTED_CLONE_AUDIT_PDF,
        "clone audit PDF digest changed",
    )
    _require(
        checks.get("source_inventory_records") == 3729,
        "clone audit inventory count changed",
    )

    if history_mode == "private_history":
        for path, expected_tree in EXPECTED_CLONE_AUDIT_TREES.items():
            _require(
                _run_git("rev-parse", f"{audited_commit}:{path}") == expected_tree,
                f"clone-audited historical tree changed: {path}",
            )
        audited_pdf_sha256 = hashlib.sha256(
            _git_bytes(f"{audited_commit}:{paper_path}")
        ).hexdigest()
        _require(
            audited_pdf_sha256 == EXPECTED_CLONE_AUDIT_PDF,
            "clone-audited historical PDF bytes changed",
        )
        audited_inventory = _git_bytes(
            f"{audited_commit}:release/source_files.sha256"
        ).splitlines()
        _require(
            checks.get("source_inventory_records") == len(audited_inventory),
            "clone audit inventory count does not match its audited commit",
        )


def _verify_sources() -> None:
    for relative_name in REQUIRED_PATHS:
        _require(
            (ROOT / relative_name).is_file(),
            f"missing source/evidence path: {relative_name}",
        )
    _require(
        (ROOT / "LICENSE").read_bytes()
        == (ROOT / "LICENSES/Apache-2.0.txt").read_bytes(),
        "root LICENSE must contain the canonical Apache-2.0 payload",
    )
    _require(
        (ROOT / "NOTICE").read_text(encoding="utf-8")
        == "FP4 FlashAttention\nCopyright (c) 2026 Graphcore Ltd.\n",
        "root NOTICE copyright line changed",
    )
    excluded_binary = (
        ROOT / "TK_quantisation/nvfp4_v5/_tk_quant_v5.cpython-312-aarch64-linux-gnu.so"
    )
    _require(
        not excluded_binary.exists(),
        "archived quantizer binary must be rebuilt, not shipped",
    )
    historical_binary = (
        ROOT
        / "reproduction/snapshots/forward_cfc06dad/TK_quantisation/nvfp4_v5"
        / "_tk_quant_v5.cpython-312-aarch64-linux-gnu.so"
    )
    _require(
        not historical_binary.exists(),
        "historical quantizer binary must be rebuilt, not shipped",
    )
    historical_root = ROOT / "reproduction/snapshots/forward_cfc06dad"
    historical_tk_paths = [
        path for path in (historical_root / "tk_fa4").rglob("*") if path.is_file()
    ]
    _require(
        len(historical_tk_paths) == 126,
        "historical cfc06dad snapshot must contain 126 source/development paths",
    )
    for dependency in ("ThunderKittens", "SageAttention"):
        link = historical_root / dependency
        _require(link.is_symlink(), f"historical {dependency} link is absent")
        _require(
            link.readlink() == Path(f"../../../{dependency}"),
            f"historical {dependency} link target changed",
        )
    scheduler_specs = tuple((ROOT / "results").glob("**/*.yaml")) + tuple(
        (ROOT / "results").glob("**/*.yml")
    )
    _require(
        not scheduler_specs,
        "internal Volt scheduler specifications must not be shipped",
    )
    for relative_name, expected in EXPECTED_FILE_HASHES.items():
        path = ROOT / relative_name
        _require(path.is_file(), f"missing authenticated file: {relative_name}")
        actual = _sha256(path)
        _require(
            actual == expected,
            f"SHA256 mismatch: {relative_name}: {actual} != {expected}",
        )


def _verify_cute_overlay() -> None:
    """Authenticate the recovered CuTe-DSL overlay in its pinned submodule.

    The patch file alone is not enough: this also proves that the checked-out
    FlashAttention child contains the exact three recovered files and no
    different diff shape relative to the recovered base commit.
    """

    manifest_path = (
        ROOT
        / "patches/flash_attention_fp4_runtime_overlay_9743edaf_20260831.manifest.json"
    )
    overlay = _load_json_object(manifest_path, "CuTe overlay manifest")
    _require(overlay == EXPECTED_CUTE_OVERLAY, "CuTe overlay manifest changed")

    patch = EXPECTED_CUTE_OVERLAY["patch"]
    patch_path = ROOT / patch["path"]
    _require(
        patch_path.stat().st_size == patch["bytes"], "CuTe overlay patch size changed"
    )
    _require(
        _sha256(patch_path) == patch["sha256"], "CuTe overlay patch digest changed"
    )

    checkout = ROOT / "flash-attention"
    base_commit = EXPECTED_CUTE_OVERLAY["base_commit"]
    head_commit = EXPECTED_DEPENDENCIES["flash-attention"]
    _require(
        _run_git("rev-parse", "HEAD", cwd=checkout) == head_commit,
        "CuTe overlay checkout is not the pinned child commit",
    )
    _run_git("cat-file", "-e", f"{base_commit}^{{commit}}", cwd=checkout)
    _run_git("merge-base", "--is-ancestor", base_commit, head_commit, cwd=checkout)

    expected_numstat: dict[str, tuple[int, int]] = {}
    for record in EXPECTED_CUTE_OVERLAY["files"]:
        relative_name = record["path"]
        path = checkout / relative_name
        _require(path.is_file(), f"missing recovered CuTe source: {relative_name}")
        _require(
            path.stat().st_size == record["bytes"],
            f"recovered CuTe source size changed: {relative_name}",
        )
        _require(
            _sha256(path) == record["sha256"],
            f"recovered CuTe source digest changed: {relative_name}",
        )
        expected_numstat[relative_name] = (
            record["insertions"],
            record["deletions"],
        )

    names = list(expected_numstat)
    output = _run_git(
        "diff",
        "--numstat",
        f"{base_commit}..{head_commit}",
        "--",
        *names,
        cwd=checkout,
    )
    actual_numstat: dict[str, tuple[int, int]] = {}
    for line in output.splitlines():
        fields = line.split("\t", 2)
        _require(len(fields) == 3, f"malformed CuTe overlay numstat: {line}")
        insertions, deletions, relative_name = fields
        _require(
            insertions.isdigit() and deletions.isdigit(),
            f"CuTe overlay unexpectedly contains a binary diff: {relative_name}",
        )
        actual_numstat[relative_name] = (int(insertions), int(deletions))
    _require(actual_numstat == expected_numstat, "CuTe overlay diff shape changed")


def _verify_hosting_surface(manifest: dict[str, Any]) -> None:
    if manifest["project"]["visibility"] != "public":
        return

    workflow_root = ROOT / ".github" / "workflows"
    workflows = []
    if workflow_root.is_dir():
        workflows = sorted(
            path.relative_to(ROOT).as_posix()
            for path in workflow_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".yml", ".yaml"}
        )
    _require(
        not workflows,
        "public export must not ship executable GitHub Actions workflows: "
        + ", ".join(workflows),
    )

    codeowners = [
        relative_name
        for relative_name in (
            "CODEOWNERS",
            ".github/CODEOWNERS",
            "docs/CODEOWNERS",
        )
        if (ROOT / relative_name).exists()
    ]
    _require(
        not codeowners,
        "public export must not inherit CODEOWNERS: " + ", ".join(codeowners),
    )


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _verify_legacy_backward_inventory() -> None:
    inventory_path = ROOT / "release/legacy_backward_makefiles.txt"
    declared = inventory_path.read_text(encoding="utf-8").splitlines()
    _require(
        declared and all(line and line == line.strip() for line in declared),
        "legacy backward inventory has a blank or malformed line",
    )
    _require(
        len(declared) == len(set(declared)) == 80,
        "legacy backward inventory must contain 80 unique paths",
    )
    observed = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tk_fa4/native_gqa_tk_bwd").glob("Makefile.v*")
        if path.is_file()
    )
    _require(
        sorted(declared) == observed,
        "legacy backward Makefile set changed; update the reviewed inventory",
    )
    for relative_name in declared:
        relative = Path(relative_name)
        _require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and (ROOT / relative).is_file(),
            f"legacy backward entry is absent or unsafe: {relative_name}",
        )


def _verify_route_catalog() -> None:
    catalog_path = ROOT / "release/routes.json"
    schema_path = ROOT / "release/routes.schema.json"
    catalog = _load_json_object(catalog_path, "development route catalog")
    schema = _load_json_object(schema_path, "development route schema")
    _require(catalog.get("schema_version") == 1, "unexpected route schema version")
    _require(
        catalog.get("$schema") == "./routes.schema.json",
        "route catalog does not name its local schema",
    )
    _require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        "route schema draft changed",
    )

    routes = catalog.get("routes")
    _require(isinstance(routes, list), "route catalog routes must be an array")
    _require(
        all(isinstance(route, dict) for route in routes),
        "route catalog contains a malformed entry",
    )
    by_id = {
        route.get("id"): route for route in routes if isinstance(route.get("id"), str)
    }
    _require(len(by_id) == len(routes), "route catalog has duplicate/malformed IDs")
    observed = {name: route.get("status") for name, route in by_id.items()}
    _require(
        observed == EXPECTED_DEVELOPMENT_ROUTES,
        "development route set or status changed",
    )

    human_names: set[str] = set()
    for route_id, route in by_id.items():
        human_name = route.get("human_name")
        _require(
            isinstance(human_name, str) and human_name.strip(),
            f"route has no human name: {route_id}",
        )
        _require(
            human_name not in human_names, f"duplicate route human name: {human_name}"
        )
        human_names.add(human_name)
        selection = route.get("selection")
        _require(
            isinstance(selection, dict), f"route has no selection policy: {route_id}"
        )
        for collection in (
            "source_paths",
            "build_paths",
            "benchmark_paths",
            "test_paths",
        ):
            paths = route.get(collection)
            _require(
                isinstance(paths, list), f"{route_id}.{collection} must be an array"
            )
            for relative_name in paths:
                _require(
                    isinstance(relative_name, str),
                    f"{route_id}.{collection} has a non-string path",
                )
                relative = Path(relative_name)
                _require(
                    not relative.is_absolute() and ".." not in relative.parts,
                    f"{route_id}.{collection} has an unsafe path: {relative_name}",
                )
                _require(
                    (ROOT / relative).is_file(),
                    f"{route_id}.{collection} path is absent: {relative_name}",
                )
        evidence = route.get("evidence")
        _require(isinstance(evidence, list), f"{route_id}.evidence must be an array")
        for record in evidence:
            _require(isinstance(record, dict), f"{route_id} has malformed evidence")
            relative_name = record.get("path")
            _require(
                isinstance(relative_name, str), f"{route_id} evidence path is missing"
            )
            relative = Path(relative_name)
            _require(
                not relative.is_absolute()
                and ".." not in relative.parts
                and (ROOT / relative).is_file(),
                f"{route_id} evidence path is absent or unsafe: {relative_name}",
            )

    _require(
        by_id["causal_nvfp4_qk_fp8_pv_v509"]["selection"]["torchtitan_dispatch"]
        == "release_candidate",
        "FP8-P/V training candidate is no longer selected explicitly",
    )
    _require(
        by_id["v510_dense_e4m3_score_e5m2_dout_snapshot"]["selection"][
            "torchtitan_dispatch"
        ]
        == "not_connected",
        "v510 must remain disconnected from TorchTitan",
    )
    _require(
        by_id["direct_cute_fp4_qk_backward_d128"]["selection"]["torchtitan_dispatch"]
        == "forced_safe_fallback",
        "direct D128 CuTe FP4-QK must remain fail-closed",
    )


def _verify_v510_snapshot() -> None:
    snapshot = ROOT / "reproduction/snapshots/v510_aa021504"
    inventory = snapshot / "SHA256SUMS"
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        inventory.read_text(encoding="utf-8").splitlines(), 1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\0]+)", line)
        _require(match is not None, f"invalid v510 inventory line {line_number}")
        expected, relative_name = match.groups()
        relative = Path(relative_name)
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"unsafe v510 inventory path: {relative_name}",
        )
        _require(relative_name not in seen, f"duplicate v510 path: {relative_name}")
        seen.add(relative_name)
        records.append((expected, relative_name))
    _require(len(records) == 16, "v510 snapshot inventory must contain 16 records")
    for expected, relative_name in records:
        path = snapshot / relative_name
        _require(path.is_file(), f"v510 snapshot file is absent: {relative_name}")
        _require(
            _sha256(path) == expected,
            f"v510 snapshot SHA256 mismatch: {relative_name}",
        )


def _load_inventory() -> list[tuple[str, str]]:
    _require(INVENTORY_PATH.is_file(), "missing release/source_files.sha256")
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        INVENTORY_PATH.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\0]+)", line)
        _require(match is not None, f"invalid inventory line {line_number}")
        digest, relative_name = match.groups()
        _require(
            relative_name not in seen, f"duplicate inventory path: {relative_name}"
        )
        _require(
            not Path(relative_name).is_absolute(),
            f"absolute inventory path: {relative_name}",
        )
        _require(
            ".." not in Path(relative_name).parts,
            f"escaping inventory path: {relative_name}",
        )
        seen.add(relative_name)
        records.append((digest, relative_name))
    _require(records, "source inventory is empty")
    return records


def _verify_inventory_and_secrets() -> None:
    inventory_check = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "tools" / "generate_fa4_source_inventory.py"),
            "--check",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(
        inventory_check.returncode == 0,
        "source inventory is not exhaustive or current; run "
        "tools/generate_fa4_source_inventory.py",
    )
    records = _load_inventory()
    for expected, relative_name in records:
        path = ROOT / relative_name
        _require(
            path.is_file() or path.is_symlink(),
            f"inventory file missing: {relative_name}",
        )
        if path.is_symlink():
            actual = hashlib.sha256(path.readlink().as_posix().encode()).hexdigest()
            payload = path.readlink().as_posix().encode()
        else:
            actual = _sha256(path)
            payload = path.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(payload):
                raise VerificationError(f"{relative_name} contains forbidden {label}")
        _require(actual == expected, f"inventory SHA256 mismatch: {relative_name}")


def main() -> int:
    try:
        manifest = _load_manifest()
        _verify_manifest(manifest)
        _verify_git_state(manifest)
        _verify_sources()
        _verify_cute_overlay()
        _verify_hosting_surface(manifest)
        _verify_legacy_backward_inventory()
        _verify_route_catalog()
        _verify_v510_snapshot()
        _verify_inventory_and_secrets()
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS: source identities, dependency pins, routes, and credential hygiene verified"
    )
    print(
        "NOTE: GPU build/performance and external-data reproduction remain separate gates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
