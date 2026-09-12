# Packaging and Release Guide

English | [简体中文](packaging-and-release.zh.md)

This repository follows the BidKV packaging guide. The PyPI distribution is
`vllm-ascend-kvcompress-hust`, the import package is
`vllm_ascend_kvcompress`, and the stable Extension Manager ID is
`org.vllm-hust.ascend-kvcompress`.

## Release gates

Before publishing, keep the project version, `__version__`, and manifest
version identical; run CPU, package, NPU, lifecycle, and declared service
tests; freeze host/model/calibration/data provenance; and review the complete
diff. PyPI files are immutable, so any code change requires a new version.

Version 0.4.0 is an experimental candidate. The current validation supports a
bounded engineering-performance statement but not formal V4.6 or production
claims.

## Build and inspect

Prefer the BidKV `uv` workflow:

```bash
uv build --no-sources --out-dir dist
```

If `uv` is unavailable in the release environment, the equivalent validated
fallback is:

```bash
python -m build --no-isolation --outdir dist
```

Then inspect exactly the candidate files:

```bash
python -m zipfile -l \
  dist/vllm_ascend_kvcompress_hust-0.4.0-py3-none-any.whl
python -m twine check \
  dist/vllm_ascend_kvcompress_hust-0.4.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.4.0.tar.gz
sha256sum dist/vllm_ascend_kvcompress_hust-0.4.0*
```

The wheel must contain code, `LICENSE`, `NOTICE`, and
`manifests/vllm-hust-extension-v0.2.json`; entry-point metadata must contain
both `vllm.general_plugins` and `vllm_hust.extension_bundles`. Neither archive
may contain `artifacts/*.pt`, raw data, service logs, or historical
`docs/dev/results`. The sdist may include only the aggregate evidence summaries
under `docs/evidence`.

## Isolated lifecycle

Install the wheel into the same environment as the supported hosts, with no
source checkout on `PYTHONPATH`:

```bash
python -m pip install --no-deps \
  dist/vllm_ascend_kvcompress_hust-0.4.0-py3-none-any.whl
vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /absolute/path/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension check org.vllm-hust.ascend-kvcompress
vllm-hust-ext run --dry-run -- python -c 'print("manager-run-ok")'
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension forget org.vllm-hust.ascend-kvcompress
python -m pip uninstall -y vllm-ascend-kvcompress-hust
```

Confirm that `extension list` no longer discovers the plugin. Reinstall,
configure, and enable the exact wheel that will be published.

## Publish and verify

Use a project-scoped token owned by the authorized `intellistream` PyPI
organization/project. Do not pass the token on the command line or commit it:

```bash
export UV_PUBLISH_TOKEN='<read from the secret store>'
uv publish --check-url https://pypi.org/simple \
  dist/vllm_ascend_kvcompress_hust-0.4.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.4.0.tar.gz
unset UV_PUBLISH_TOKEN
```

If the protected publisher uses Twine instead, set `TWINE_USERNAME=__token__`
and `TWINE_PASSWORD` from the secret store, then upload the same two explicit
files. Finally install 0.4.0 from production PyPI without cache, verify hashes,
manager discovery, enablement, `/health`, and one correctness case.
