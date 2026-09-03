# Third-party license payload map

This directory collects license payloads for source and exact dependency pins
used by the FP4 FlashAttention artifact. Project-specific source is released
under the Apache License 2.0 in the root `LICENSE`, with the approved copyright
notice in the root `NOTICE`. The owner confirmed permission to publish the
source and retained publication assets on 2026-09-03. The pinned TorchTitan
base remains under its original BSD 3-Clause terms, reproduced byte-for-byte
as `TorchTitan-20b3de7-LICENSE.txt`; other third-party and file-level terms are
preserved below.

The copies below were taken from the named upstream files at the full revisions
shown. Their license wording is unmodified. When a Git submodule is initialized,
the license files inside that submodule remain authoritative; these copies make
the terms visible in source archives and before submodule initialization.

| Component or source boundary | Exact revision | Upstream payload | Local payload |
| --- | --- | --- | --- |
| TorchTitan repository base | `20b3de7585696c327bd5aa9f9627f0300abdbf9d` | [`LICENSE`](https://raw.githubusercontent.com/pytorch/torchtitan/20b3de7585696c327bd5aa9f9627f0300abdbf9d/LICENSE) | `TorchTitan-20b3de7-LICENSE.txt` |
| NVIDIA TransformerEngine-derived quantization source and benchmark dependency | `06b44b8eff1f81f33c2a378515cf05fe2fade3cb` | [`LICENSE`](https://raw.githubusercontent.com/NVIDIA/TransformerEngine/06b44b8eff1f81f33c2a378515cf05fe2fade3cb/LICENSE) | `TransformerEngine-06b44b8-LICENSE.txt` |
| ThunderKittens submodule | `9ee85b4afcdea1478b4dda8bb01f8907ab7edb0b` | [`LICENSE`](https://raw.githubusercontent.com/MrHuff/ThunderKittens/9ee85b4afcdea1478b4dda8bb01f8907ab7edb0b/LICENSE) | `ThunderKittens-9ee85b4-LICENSE.txt` |
| SageAttention submodule | `681004015c42b8ac543302235652e618ac66f966` | [`LICENSE`](https://raw.githubusercontent.com/MrHuff/SageAttention/681004015c42b8ac543302235652e618ac66f966/LICENSE) | `SageAttention-6810040-LICENSE.txt` |
| Graphcore FlashAttention fork | `b531f67557b8213db339492cd1629e721776f758` | [`LICENSE`](https://raw.githubusercontent.com/graphcore-research/flash-attention/b531f67557b8213db339492cd1629e721776f758/LICENSE) | `FlashAttention-b531f67-LICENSE.txt` |
| Vendored HAO FlashAttention FP4 comparator | `9b0abefdbbbe4d0da1d4e0c7aa128e3338c4b247` | in-tree `third_party/hao_flash_attention_fp4/LICENSE` | in-tree `third_party/hao_flash_attention_fp4/LICENSE` |
| NVIDIA CuTe DSL helpers vendored in the HAO comparator | `9b0abefdbbbe4d0da1d4e0c7aa128e3338c4b247` | NVIDIA EULA named by each file's `LicenseRef-NvidiaProprietary` header | `NVIDIA-CuTeDSL-EULA.txt` |
| qutlass submodule | `406e86fb2d7df436e94f825bcda8e59b1a7250a6` | [`LICENSE`](https://raw.githubusercontent.com/graphcore-research/qutlass/406e86fb2d7df436e94f825bcda8e59b1a7250a6/LICENSE) | `qutlass-406e86f-LICENSE.txt` |
| Root NVIDIA CUTLASS submodule | `acb45938e9cb3e4db8c1d75155b63d31791e0e5d` | [`LICENSE.txt`](https://raw.githubusercontent.com/NVIDIA/cutlass/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/LICENSE.txt) | `CUTLASS-acb4593-LICENSE.txt` |
| Root NVIDIA CUTLASS Python source | `acb45938e9cb3e4db8c1d75155b63d31791e0e5d` | [`python/LICENSE.txt`](https://raw.githubusercontent.com/NVIDIA/cutlass/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/python/LICENSE.txt) | `CUTLASS-acb4593-python-LICENSE.txt` |
| FlashAttention's nested NVIDIA CUTLASS | `7127592069c2fe01b041e174ba4345ef9b279671` | [`LICENSE.txt`](https://raw.githubusercontent.com/NVIDIA/cutlass/7127592069c2fe01b041e174ba4345ef9b279671/LICENSE.txt) | `CUTLASS-7127592-b2ca083-LICENSE.txt` |
| FlashAttention's nested NVIDIA CUTLASS Python source | `7127592069c2fe01b041e174ba4345ef9b279671` | [`python/LICENSE.txt`](https://raw.githubusercontent.com/NVIDIA/cutlass/7127592069c2fe01b041e174ba4345ef9b279671/python/LICENSE.txt) | `CUTLASS-7127592-b2ca083-python-LICENSE.txt` |
| qutlass's nested NVIDIA CUTLASS | `b2ca083d2bb96c41d9b3c5a930637c641f6669bf` | [`LICENSE.txt`](https://raw.githubusercontent.com/NVIDIA/cutlass/b2ca083d2bb96c41d9b3c5a930637c641f6669bf/LICENSE.txt) | `CUTLASS-7127592-b2ca083-LICENSE.txt` |
| qutlass's nested NVIDIA CUTLASS Python source | `b2ca083d2bb96c41d9b3c5a930637c641f6669bf` | [`python/LICENSE.txt`](https://raw.githubusercontent.com/NVIDIA/cutlass/b2ca083d2bb96c41d9b3c5a930637c641f6669bf/python/LICENSE.txt) | `CUTLASS-7127592-b2ca083-python-LICENSE.txt` |
| NVIDIA CUTLASS/CuTe DSL EULA | all three CUTLASS revisions above | [`python/CuTeDSL/EULA.txt`](https://raw.githubusercontent.com/NVIDIA/cutlass/acb45938e9cb3e4db8c1d75155b63d31791e0e5d/python/CuTeDSL/EULA.txt) | `NVIDIA-CuTeDSL-EULA.txt` |

The two nested CUTLASS revisions have identical root and Python license
payloads. The root `EULA.txt` and `python/CuTeDSL/EULA.txt` payloads are
byte-identical at all three pinned CUTLASS revisions, so one local EULA copy
covers those exact files.

Additional in-tree boundaries are handled as follows:

- `third_party/hao_flash_attention_fp4` carries its own `LICENSE`; its text is
  byte-identical to `FlashAttention-b531f67-LICENSE.txt`, except that the two
  files listed below carry explicit `LicenseRef-NvidiaProprietary` headers and
  are governed by `NVIDIA-CuTeDSL-EULA.txt`:
  - `third_party/hao_flash_attention_fp4/flash_attn/cute/modified_utils/block_scaled_layout_test.py`
  - `third_party/hao_flash_attention_fp4/flash_attn/cute/modified_utils/helpers.py`
  They are outside the project's Apache-2.0 outbound-license boundary. As
  required by Section 1.1(d)(iii) of the NVIDIA EULA, the notice for these
  files is: "This software contains source code provided by NVIDIA Corporation."
- Files under `baseline_kernels/csrc` and `fused_ops/csrc` marked
  `SPDX-License-Identifier: Apache-2.0` are accompanied by the canonical
  [`Apache-2.0.txt`](https://www.apache.org/licenses/LICENSE-2.0.txt) payload
  fetched from the Apache Software Foundation.
- CUTLASS-derived files vendored under
  `fused_ops/csrc/old_ideas/include/cutlass_extensions` retain complete NVIDIA
  BSD 3-Clause headers in each source file and are accompanied by the pinned
  CUTLASS payloads above.

TransformerEngine is not retained as a complete submodule in this export.
Files copied or adapted from its NVFP4 quantization implementation retain
NVIDIA's notices under `TK_quantisation` and the historical source snapshots;
the Apache-2.0 payload above accompanies those files. The remaining historical
benchmarks import TransformerEngine as an external dependency. Revision
`06b44b8eff1f81f33c2a378515cf05fe2fade3cb` is the recovered public pin from
the training repository's TransformerEngine gitlink.
