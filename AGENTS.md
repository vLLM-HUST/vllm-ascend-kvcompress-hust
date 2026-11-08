# Repository working rules

## Scope

- Keep this file concise and in English. State shared rules and link to their
  documentation instead of repeating procedures or implementation details.
- Read [AGENTS.dev.md](AGENTS.dev.md) when present. It is ignored local guidance
  for machine paths, model mounts, persistence, resource limits, Git identity
  and workspace status; keep those details out of shared instructions.
- Preserve pre-existing changes in sibling HUST worktrees.
  Do not modify host repositories merely to make plugin checks pass.

## Development

- Follow the [method architecture](docs/design/methods.md) and
  [contribution guide](CONTRIBUTING.md). Preserve method boundaries,
  default-off activation, compatibility guards and rollback behavior.
- Use the [fixed HUST environment](docs/development/environment-installation.md)
  and [host lock](requirements/ascend-host.json). Keep environment names
  independent of plugin releases; do not change the host stack incidentally.
- Follow [commit conventions](docs/development/git-commit.md). Version edits
  do not authorize commits, pushes or publication.

## Documentation and evidence

- Avoid tying reusable documentation and scripts to specific releases,
  especially environment setup and dataset evaluation workflows that may
  apply across multiple versions. Use version labels only when necessary.
- Keep authored `docs/**/*.md` files in English/Chinese pairs. Update the
  [documentation index](docs/README.md), its Chinese counterpart and affected
  consumers when moving documents.
- Choose the applicable [benchmark protocol](docs/benchmark/README.md) before
  measuring; retain workload boundaries, failures and regressions.
- Follow [original custody](docs/usage/benchmark-originals.md). Measured results,
  logs, profiles and historical measurement narratives belong in
  `vLLM-HUST/vllm-hust-benchmark`; this repository retains implementation,
  protocols, synthetic fixtures and checksum-bound links. Preserve historical
  version, host and model identities rather than relabeling old evidence.
- Keep personal reports and temporary outputs in ignored `dev/`. Store downloaded
  evaluation inputs and plans in `../.benchmarks/` relative to this repository,
  and local run results in its `results/` directory. Local outputs are not
  published evidence. Never publish secrets or private inputs.
