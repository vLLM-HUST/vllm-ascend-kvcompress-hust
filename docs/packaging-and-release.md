# Packaging and Release Guide

English | [简体中文](packaging-and-release.zh.md)

This project follows the vLLM-HUST BidKV packaging and release process. The
PyPI distribution is `vllm-ascend-kvcompress-hust`, the Python package is
`vllm_ascend_kvcompress`, and the stable Extension Manager ID is
`org.vllm-hust.ascend-kvcompress`.

## Release gates

Do not publish an alpha while the [validation record](validation.md) has a
failed correctness/quality gate or an unvalidated declared dependency stack.
Before tagging a release:

1. keep `[project].version`, package `__version__`, and manifest
   `extension_version` identical;
2. run the CPU suite, Ruff, Ascend numerical smoke, and the complete matched
   service matrix in the acceptance protocol;
3. record exact host commits, model/calibration provenance, raw results, and
   the release commit; and
4. verify the working tree contains only the intended release changes.

PyPI files are immutable. Any source change after upload requires a new
version.

## Build and inspect

Build both wheel and sdist from the repository root with local source
overrides disabled:

```bash
uv build --no-sources --out-dir dist
python -m zipfile -l \
  dist/vllm_ascend_kvcompress_hust-0.3.0-py3-none-any.whl
```

The wheel must contain the package, `LICENSE`, `NOTICE`, and
`manifests/vllm-hust-extension-v0.2.json`. Neither the wheel nor sdist may
contain `artifacts/*.pt`; those calibration files do not have sufficient
redistribution provenance. The wheel's `entry_points.txt` must include:

```ini
[vllm.general_plugins]
ascend_kvcompress = vllm_ascend_kvcompress.plugin:register

[vllm_hust.extension_bundles]
org.vllm-hust.ascend-kvcompress = vllm_ascend_kvcompress.manifests
```

Run a distribution metadata check when the tool is available:

```bash
python -m twine check dist/*
sha256sum dist/*
```

## Isolated lifecycle smoke

Install the wheel and Extension Manager into the same clean environment as the
host. Confirm discovery, then exercise the whole stored-state lifecycle:

```bash
python -m pip install \
  'vllm-hust-ext>=0.2.0.dev0,<0.3' \
  dist/vllm_ascend_kvcompress_hust-0.3.0-py3-none-any.whl

vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /absolute/path/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension status org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension forget org.vllm-hust.ascend-kvcompress
python -m pip uninstall vllm-ascend-kvcompress-hust
```

After uninstall, `extension list` must no longer contain the extension. A
host-less environment may report compatibility as unverified and refuse a
trusted in-process `run --dry-run`; perform the final render and service test
in the exact host environment.

## Publish and verify

Use a project-scoped PyPI token stored in CI secrets. Build, test, and upload
from the same protected tag/commit, explicitly naming only the current wheel
and sdist:

```bash
export UV_PUBLISH_TOKEN='<read from the secret store>'
uv publish --check-url https://pypi.org/simple \
  dist/vllm_ascend_kvcompress_hust-0.3.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.3.0.tar.gz
unset UV_PUBLISH_TOKEN
```

Finally, install the exact version from production PyPI without cache, confirm
Extension Manager discovery/enablement, start a new manager-wrapped host
process, check `/health`, and rerun a representative correctness case. Record
the published file hashes in the validation record and release notes.
