# Third-party notices

## Export and permission status

This tree is a parentless public-export candidate. The owner confirmed
permission to publish the source and retained publication assets on
2026-09-03. Project-specific source is released under the Apache License 2.0
in the root `LICENSE`, with `Copyright (c) 2026 Graphcore Ltd.` recorded in
the root `NOTICE`. Nothing in that grant expands a third party's terms.

## TorchTitan

This repository starts from a pinned snapshot of
[TorchTitan](https://github.com/pytorch/torchtitan). TorchTitan is copyright
Meta Platforms, Inc. and affiliates and is distributed under the BSD 3-Clause
License reproduced in `LICENSES/TorchTitan-20b3de7-LICENSE.txt`. The parentless
export contains the attributed source snapshot, not TorchTitan's Git history.

## Project FP4 source

The project source exported into `tk_fa4`, `TK_quantisation`, `baseline_kernels`,
`fused_ops`, `qutlass_binding`, and `torchtitan/experiments/fa4` comes from the
pinned Graphcore research revisions recorded in `release/manifest.json` and
`release/SOURCE_PROVENANCE.md`. Publication permission was confirmed on
2026-09-03. Except where an existing file-level or third-party license says
otherwise, this project-specific source is released under Apache-2.0 with the
copyright notice recorded in the root `NOTICE`.

Some imported source files retain their own BSD-3-Clause or Apache-2.0 SPDX
headers. Those file-level notices continue to apply.

## ThunderKittens

`ThunderKittens` is pinned as a Git submodule at
`9ee85b4afcdea1478b4dda8bb01f8907ab7edb0b` and is distributed under the MIT
License included in that submodule.

## SageAttention

`SageAttention` is pinned as a Git submodule at
`681004015c42b8ac543302235652e618ac66f966`. It is retained because historical
forward diagnostic programs reference that source lineage. The submodule
includes its Apache License 2.0 text; its presence does not make a Sage route a
supported training method.

## FlashAttention implementations

`flash-attention` is the pinned Graphcore research fork used by the causal
reference path. `third_party/hao_flash_attention_fp4` is the source snapshot
used for HAO comparisons. Both retain their BSD 3-Clause licenses and upstream
author notices, subject to the explicit file-level exception below. The
vendored HAO copy contains a recorded compatibility patch; its exact base and
patch identities are listed in `release/SOURCE_PROVENANCE.md`.

## NVIDIA CuTe DSL helper files in the HAO snapshot

The following files carry NVIDIA copyright notices and
`SPDX-License-Identifier: LicenseRef-NvidiaProprietary` headers:

- `third_party/hao_flash_attention_fp4/flash_attn/cute/modified_utils/block_scaled_layout_test.py`
- `third_party/hao_flash_attention_fp4/flash_attn/cute/modified_utils/helpers.py`

They are governed by `LICENSES/NVIDIA-CuTeDSL-EULA.txt`, not by the project's
Apache-2.0 project license. Their original headers remain in place. The notice
required by Section 1.1(d)(iii) of that EULA is:

> This software contains source code provided by NVIDIA Corporation.

## qutlass and CUTLASS

`qutlass` is distributed under Apache-2.0. The root `cutlass` submodule and the
nested CUTLASS revisions retain NVIDIA's BSD-3-Clause license and any separate
terms identified inside those repositories. In particular, CUTLASS Python
CuTe DSL files are subject to NVIDIA's separate EULA where their headers or
the pinned CUTLASS license map says so. The exact retained payload is
`LICENSES/NVIDIA-CuTeDSL-EULA.txt`.

## Paper and publication assets

The `results` snapshot includes paper sources, figures, receipts, and the
Graphcore symbol used by the paper. Permission to distribute the retained
paper and Graphcore symbol was confirmed on 2026-09-03. Unused proprietary
fonts and unused HAO assets were removed from this export. This authorization
does not change the terms of any third-party source. The Apache-2.0 grant
applies to project-specific software source, not external assets or marks.

## External assets

Datasets, tokenizers, model weights, GPU drivers, CUDA components, and other
external dependencies are not redistributed by this repository. Users are
responsible for obtaining them under their respective licenses and terms.
