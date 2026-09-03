# FP4 FlashAttention releases

The first public release is a source snapshot, not a TorchTitan package or
nightly build. Its history is rooted at the pinned parentless commit described
in [`../release/PUBLIC_EXPORT_POLICY.md`](../release/PUBLIC_EXPORT_POLICY.md).
The release verifier permits ordinary descendant commits while requiring that
this remains the only reachable root, that HEAD descends from it, and that no
unreachable root-repository objects or private project ancestry are present.

This repository intentionally ships no GitHub Actions workflows. Inherited
TorchTitan build, GPU-test, release, Docker, and package-publishing workflows
were removed because their credentials, runner assumptions, package targets,
and ownership rules do not apply to this project. The inherited `CODEOWNERS`
file was removed for the same reason. Repository administrators should also
disable Actions in the hosting settings until project-specific automation has
been reviewed and committed deliberately.

Before creating a source release:

1. Run `make verify-source` from a clean recursive clone.
2. Run the CPU contract suite with `make test`.
3. Regenerate the paper artifacts with `make paper` and verify the recorded
   hashes.
4. Complete the target-GPU and distributed gates listed in
   [`../RELEASE_STATUS.md`](../RELEASE_STATUS.md) for any binary or performance
   claim attached to that release.
5. Review the tracked tree, commit metadata, submodule reachability, licenses,
   and third-party notices before tagging the already-audited commit.

Adding automated tests or package publication later requires a separate
security and cost review. A workflow must use least-privilege permissions,
avoid untrusted code on privileged runners, pin third-party actions, and name
this project's artifacts rather than TorchTitan's packages.
