# Contributing

English | [简体中文](CONTRIBUTING.zh.md)

Thank you for improving vLLM Ascend KV Cache Compression. Keep changes small,
fail closed on unsupported runtime combinations, and preserve the method-neutral
provider boundary.

## Development Setup

Install matching editable vLLM-HUST and vLLM-Ascend-HUST packages first, then:

```bash
uv pip install -e '.[test]'
```

## Architecture Rule

- Common vLLM/Ascend lifecycle behavior belongs in `provider.py` or `plugin.py`.
- Algorithm-specific configuration, artifacts, compatibility checks, scoring,
  and materialization belong under a dedicated `methods/<method>/` package.
- Keep each method's public factory in `methods/<method>/__init__.py`; split its
  configuration, runtime adapter, scoring, statistics, and cache operations into
  focused modules as needed.
- A new algorithm must implement `KVCompressionMethod`; do not add method-name
  conditionals to the common provider.
- Provider configuration remains a flat JSON-scalar mapping because that is the
  vLLM-HUST schema v1 contract.

See [Compression method architecture](docs/methods.md).

## Required Checks

```bash
python -m pytest -q
ruff check .
ruff format --check .
python -m compileall -q src tests
git diff --check
```

Runtime changes require matched compression-off/on long-context benchmarks on
an idle assigned NPU. Confirm all service processes exit and the NPU allocation
is released. Do not modify sibling HUST repositories to make this plugin pass.

## Documentation

Public, release-ready behavior belongs in the README or public `docs/` files.
Intermediate test reports and machine-specific development logs belong in
git-ignored `docs/dev/`; generated calibration artifacts belong in
git-ignored `artifacts/`. Update each public English document and its `.zh.md`
translation together.
