# Packaging and release

English | [简体中文](packaging-and-release.zh.md)

Use this procedure for each release candidate. Read the distribution version
from `pyproject.toml`, inspect the installed package index before uploading,
and review the candidate's [result evidence](../results/README.md). The
[0.9 release recipe and evidence](../archive/development/packaging-and-release-0.9.md)
are historical; do not reuse its fixed artifact filenames.

## Package contract

The distribution is `vllm-ascend-kvcompress-hust`, the import package is
`vllm_ascend_kvcompress`, and the stable Extension Manager ID is
`org.vllm-hust.ascend-kvcompress`. Keep the version in `pyproject.toml`,
`src/vllm_ascend_kvcompress/__init__.py`, and the manifest identical. The
manifest file is `manifests/vllm-hust-extension-v0.3.json`; `0.3` is its schema
line, not the plugin version. Core wheel dependencies remain empty because
the matched accelerator host is provisioned separately. The optional manager
extra targets Extension Manager 0.3.

## Candidate checks

For a release candidate, freeze the exact host, CANN/PyTorch stack,
model revision, calibration SHA-256, workload file hash, configuration, and
source revision. Run the CPU and NPU correctness suites, an isolated wheel
install, manager discovery/configure/enable/check/disable/forget lifecycle,
and each benchmark required by the release's stated claims. Choose protocols
from the [benchmark index](../benchmark/README.md), and record their acceptance
metrics before measuring.
Report all regressions and preserve raw reports and server logs on durable
storage outside temporary build directories.

After validation, build from a clean candidate worktree:

```bash
uv build --no-sources --out-dir dist
python -m twine check dist/*.whl dist/*.tar.gz
python scripts/kvcompress_leaderboard.py --check
```

Inspect the exact generated wheel and sdist, including manifest, entry points,
`LICENSE`, `NOTICE`, and the absence of model weights, `.pt` calibration files,
raw prompts, logs, and `dev/`. Check hashes and install the wheel into the
matched host with `--no-deps`, without the source checkout on `PYTHONPATH`.
Package indices do not allow overwriting an uploaded version; any later code
change needs a new version and full retest.

Only an explicitly authorized release task may commit, push, upload, and
verify on production PyPI. The maintainer's instructions determine the Git
author identity. Follow the [commit guide](git-commit.md) and never place a
PyPI token in a command line argument, log, repository file, or commit.
