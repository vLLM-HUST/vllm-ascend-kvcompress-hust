# Historical 0.9 Packaging and Release Guide

English | [简体中文](packaging-and-release-0.9.zh.md)

The complete historical measurements, release gates and failures are preserved in the
[benchmark original](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/5eeccc2d69b2c4a634e273a9f926584c8d252cf7/archive/mod-targeted-20261023/compress-release/snapshot/docs/archive/development/packaging-and-release-0.9.md). This page retains historical installation recipes;
use the [current guide](../../development/packaging-and-release.md) for publishing.
Single engineering observations are not current formal results. The 0.8 wheel eager smoke
does not qualify compiled serving; the original compiled-start failure remains unchanged.
Original JSON B0/B1 labels identify their historical pairs, not the fixed 0.18/0.18 B0.

## Original evidence links

These are the saved paired JSON records, not per-request logs. External run logs represented
only by hashes in those JSON files, and the 0.8 wheel smoke logs, have no corresponding
original in this downloaded collection; that availability limitation remains explicit.

- [docs/results/evidence/kvcompress-qwen35-20260927-agentx-postmerge-positive-pair.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20260927-agentx-postmerge-positive-pair.json)
- [docs/results/evidence/kvcompress-qwen35-20260927-swe-postmerge-negative-pair.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20260927-swe-postmerge-negative-pair.json)
- [docs/results/evidence/kvcompress-qwen35-20261008-v090-agentx-pair.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261008-v090-agentx-pair.json)
- [docs/results/evidence/kvcompress-qwen35-20261008-v090-swe-pair.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261008-v090-swe-pair.json)

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
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl
python -m twine check \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.9.0.tar.gz
sha256sum dist/vllm_ascend_kvcompress_hust-0.9.0*
```

The wheel must contain code, `LICENSE`, `NOTICE`, and
`manifests/vllm-hust-extension-v0.3.json`; entry-point metadata must contain
`vllm.general_plugins`, `vllm_hust.extension_bundles`, and the
`vllm-ascend-kvcompress-calibrate` console script. Neither archive
may contain `artifacts/*.pt`, raw data, service logs, or historical
`docs/dev/results`. The old 0.9 recipe bundled leaderboard data in the sdist; this is not a current
packaging requirement. Historical data now lives in benchmark and must not return to
distributions. Follow the [current guide](../../development/packaging-and-release.md).

## Isolated lifecycle

Install the wheel into the same environment as the supported hosts, with no
source checkout on `PYTHONPATH`:

```bash
python -m pip install --no-deps \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl
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

The wheel must have no core `Requires-Dist` entries. In particular, do not add
`vllm`, `vllm-ascend`, Triton-Ascend, NumPy, or OpenCV to the plugin's core
dependencies. The accelerator host is provisioned from a separately validated
lock, and the extension manifest owns the compatible host range. Re-resolving
the host while installing a plugin can combine the NumPy-1-only
Triton-Ascend 3.2.2 line with the NumPy-2-only OpenCV requirement in current
vLLM releases.
The optional `calibration` extra may declare Accelerate; multi-device automatic
calibration must fail with an actionable error when it is absent.

Confirm that `extension list` no longer discovers the plugin. Reinstall,
configure, and enable the exact wheel that will be published.

## Publish and verify

Use a project-scoped token owned by the authorized `intellistream` PyPI
organization/project. Do not pass the token on the command line or commit it:

```bash
export UV_PUBLISH_TOKEN='<read from the secret store>'
uv publish --check-url https://pypi.org/simple \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.9.0.tar.gz
unset UV_PUBLISH_TOKEN
```

If the protected publisher uses Twine instead, set `TWINE_USERNAME=__token__`
and `TWINE_PASSWORD` from the secret store, then upload the same two explicit
files. Finally install 0.9.0 from production PyPI without cache, verify hashes,
manager discovery, enablement, `/health`, and one correctness case.
