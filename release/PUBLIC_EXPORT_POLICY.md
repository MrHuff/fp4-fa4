# Public export policy

The recovery authority is a private continuation workspace. It must not be
made public by changing the visibility of its existing remote. This directory
is the separate public-export candidate. Its complete tree is staged in an
unborn Git repository so that the first commit can be parentless. That commit,
the required clean-clone audit, and the public push have not yet happened. A
conventional squash commit that still has a private parent would not satisfy
this policy.

This policy protects two independent properties:

1. **Research continuity.** The public tree must retain every source lineage,
   diagnostic route, manifest, test, and scientific handoff needed to continue
   the work.
2. **Public safety and redistribution rights.** Private operational metadata,
   credentials, restricted assets, and source without outbound approval must
   not become reachable from the public repository.

Passing the current credential scanner is necessary but not sufficient for
public release. It does not prove that a file is anonymous, redistributable,
or free of private infrastructure identifiers.

## Verified state and export preparation

The following facts are established by the checked repository:

- An earlier private preparation commit contained removed patch artifacts
  with a private object-store locator and an unused PDF with a personal build
  path. Those bytes remain in private Git history even though they are absent
  from the current tree.
- On 2026-09-03, the authorized owner confirmed consent to publish the project
  source and the paper's fonts and logos, selected Apache-2.0 for
  project-specific source, and approved `Copyright (c) 2026 Graphcore Ltd.`
  for the root `NOTICE`. The inherited TorchTitan source retains its original
  BSD-3-Clause terms.
- Unused proprietary fonts and comparator publication assets were removed.
  Necessary public-surface normalizations are recorded in
  [`PUBLIC_SANITIZATION.md`](PUBLIC_SANITIZATION.md).
- All five root submodule pins and both nested pins were reachable without
  authentication at their exact recorded revisions. Their license payloads
  and notices are included in the export candidate. The final clean recursive
  clone must confirm this closure from the parentless commit.
- Some historical evidence is available only as a normalized receipt. Other
  inputs, including exact historical training data order and several raw
  captures, are absent or external. A public export must preserve those limits
  rather than silently substitute new data.
- The public-surface hygiene pass normalized private host, scheduler, and
  experiment-tracker identifiers without changing scientific values. It also
  removed generated binaries and unused restricted assets. The final
  reachable commit still needs automated and manual reinspection from a clean
  clone.
- Inherited continuous-integration workflows were removed from the public
  tree. Publication must not expose private workflow history or enable jobs as
  a side effect.

The source and evidence boundaries are described in
[`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md),
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md), and
[`../RELEASE_STATUS.md`](../RELEASE_STATUS.md).

## Publication rights

Publication consent and source/font/logo rights were confirmed by the
authorized owner on 2026-09-03. Project-specific source is licensed under
Apache-2.0 and the approved copyright notice is recorded in `NOTICE`.
Third-party and file-level terms remain separate as documented in
`THIRD_PARTY_NOTICES.md` and `LICENSES/`.

Dataset, tokenizer, checkpoint, and model assets that are not redistributable
remain omitted under the documented acquisition and evidence boundary. That
omission is a reproducibility limitation, not an unresolved grant to publish
the source tree.

The private candidate remains the recovery authority. Public-release cleanup
belongs only in this disposable export workspace; it must not delete or
rewrite private provenance.

## What the public tree must contain

Public hygiene is not permission to reduce the release to the advertised
kernel. The exported tree must retain:

- the complete causal forward and backward development tree in `tk_fa4`;
- the quantization and projection sources in `TK_quantisation`;
- the historical non-causal source epoch and its portability patch;
- supported, diagnostic, disabled, and negative-result lineages, including
  the durable direct-CuTe overlay and the v510 snapshot;
- the TorchTitan adapter, route definitions, checkpoint contract, optimizer,
  config renderer, and data-verification code;
- exact source and dependency pins, tests, build tooling, available receipts,
  and the scientific-state handoff; and
- an explicit account of inputs or raw evidence that are not distributed.

Machine-specific shared libraries, credentials, scheduler objects, private
service histories, and restricted datasets are not part of that continuity
contract. Their absence must be explicit, and all supported binaries must be
rebuilt from authenticated source.

## Public-surface audit

Audit the final tracked tree before constructing public history. Inspect text,
binary metadata, archives, generated PDFs, and submodule configuration for:

- private keys, tokens, session credentials, signed URLs, and embedded
  username/password pairs;
- internal object-store paths, service endpoints, repository remotes, cluster
  names, namespaces, scheduler manifests, and container registries;
- personal usernames, email addresses, workstation names, absolute home or
  workspace paths, and document producer metadata;
- W&B entities, project/run URLs, job identifiers, checkpoint buckets, and
  other operational identifiers that are not necessary scientific evidence;
- environment dumps, shell histories, crash dumps, compiled binaries, caches,
  and archives that may carry information not visible in their filenames; and
- fonts, logos, model files, tokenizer files, datasets, or screenshots without
  confirmed redistribution rights.

Not every identifier is secret, but every retained identifier needs a purpose.
Prefer a scientific name such as `bf16_b4_control` over a service-side job ID.
Retain immutable upstream commit IDs and content hashes because they establish
provenance.

Do not edit scientific measurements with broad search-and-replace. When a
receipt needs redaction, generate a public normalized receipt with an explicit
schema. Record the source receipt hash privately, the transformation version,
the fields removed or generalized, and the resulting public hash. Preserve
shape, seed, sample count, timing boundary, numerical result, and source
identity unless the claim is withdrawn. If safe transformation cannot be
demonstrated, omit the receipt and mark the claim as historical or blocked.

The export's public receipts have been normalized under this standard, with
the precise transformations and before/after hashes recorded in
[`PUBLIC_SANITIZATION.md`](PUBLIC_SANITIZATION.md). The private receipts were
not modified. Re-run both automated and manual review on the final reachable
commit rather than treating the worktree pass as final proof.

## Constructing parentless public history

Use a disposable clone at a frozen, reviewed private commit. Never rewrite,
filter, garbage-collect, or force-push the canonical private repository. The
2026-09-03 public export followed the parentless construction and clean-clone
audit below.

1. Record the private commit, submodule revisions, source inventory, and
   verifier output.
2. In the disposable clone, make only the approved public-surface changes:
   replace cleared assets, create normalized receipts, update notices, and
   remove material that cannot be published.
3. Regenerate every affected manifest, hash list, table, figure, and PDF. Run
   the complete CPU/offline verification while the candidate is still private.
4. Write the reviewed index as a Git tree and create a new commit with
   `git commit-tree`, without a parent. This preserves executable modes,
   symlinks, and submodule gitlinks while making the project commit
   parentless. Use an approved public author identity and message.
5. Clone only that export branch with `--no-local --single-branch --no-tags`
   into a second clean directory. This second clone is the candidate that may
   later be pushed; it must have no alternates and no private remote.
6. Verify that exactly one project commit is reachable, that it has no parent,
   and that `git fsck --full --no-reflogs` reports no unexpected unreachable
   private objects. Review all reachable blobs and commit metadata, not only
   the worktree.
7. Recursively initialize dependencies from a credential-free environment.
   Independent reachability checks have passed for all five root and two
   nested pins; repeat that check from the final clean clone.
8. Perform the clean-clone source-inventory, verifier, CPU, and paper checks.
   Run the SM100 build, numerical, liveness, performance, distributed smoke,
   and checkpoint save/resume gates as target-hardware validation. Those GPU
   gates govern support and performance claims, but they do not prevent
   publication of a source snapshot that labels them as pending.
9. Push the parentless branch to a new empty public remote. Do not reuse the
   private remote and do not publish private tags, pull requests, releases,
   actions logs, or branch names.

A safe implementation of this procedure should be scripted and tested before
use. The script must refuse a dirty source checkout, an unapproved license
state, a non-parentless export commit, a nonempty public remote, or a failed
audit. This document is a policy, not evidence that those gates have run.

## Source-publication gates

The source tree may be made public only when all of the following are true:

- **Rights gate:** confirmed publication approval is accompanied by the
  Apache-2.0 project license, the approved Graphcore copyright notice, and
  complete third-party notices and license payloads.
- **History gate:** the public repository has one parentless project commit
  and no reachable private development object or metadata.
- **Identity gate:** manifests, source inventories, submodule pins, and
  generated artifact hashes match the exported tree.
- **Hygiene gate:** both automated secret scanning and manual
  identifier/metadata review pass on all reachable content.
- **Clone gate:** an unauthenticated recursive clone obtains every required
  source dependency at the recorded pin. Independent pin checks already pass;
  the final parentless clone remains to be tested.
- **Offline gate:** source verification, unit tests, and deterministic paper
  regeneration pass from a clean clone.
- **Documentation gate:** unsupported shapes, missing data, absent raw
  captures, diagnostic routes, and unverified long-horizon claims remain
  plainly labelled.

## Scientific validation gates

Two additional gates determine whether a route may be described as validated,
not whether its clearly labelled source may be published:

- **GPU gate:** supported B1/B2/B4 kernels build from source on the stated
  Blackwell target and pass correctness, exact-zero-dO, liveness, and matched
  timing checks.
- **Training gate:** the portable TorchTitan B1/B4 recipes complete a
  distributed smoke test and checkpoint save/fresh-resume test without a
  private service dependency.

The release manifest marks this tree as a public export. That state selects the
verifier's parentless-history checks; it does not turn pending source,
evidence, GPU, or training work into completed validation.

## After publication

Keep the frozen private recovery repository read-only. Development can proceed
from the public root with ordinary reviewable commits. Each result-bearing
change should update the route catalog and scientific handoff, identify the
exact source tree and inputs, and say which gates were run. Security fixes
that reveal private history should be handled through the hosting provider's
incident process, not by assuming a force-push erased disclosed objects.
