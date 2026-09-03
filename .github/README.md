# GitHub configuration

This public source snapshot intentionally contains no GitHub Actions workflows
and no inherited `CODEOWNERS` file. The upstream TorchTitan automation targets
different maintainers, secrets, package names, and GPU infrastructure, so it
must not run in this repository.

Repository administrators should keep GitHub Actions disabled in the hosting
settings until a project-specific workflow has been reviewed for triggers,
permissions, secrets, artifact retention, runner labels, and cost. Adding a
workflow is a future reviewed source change, not part of this release.

The YAML files under `ISSUE_TEMPLATE/` configure issue forms only; they are not
executable workflows.
