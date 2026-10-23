# vLLM Ascend KV Compression

English | [简体中文](README.zh.md)

An independently packaged TriAttention KV-cache compression plugin for the
upstream-aligned vLLM-HUST and vLLM-Ascend-HUST stacks. The 0.7 release added
the paper's V3 position policy and experimental Qwen3.5 support. Version 0.8
adapted to the synchronized hosts without changing their source repositories.
Version 0.9 introduced method API v1 for separately packaged compression
methods, including per-layer physical-state and query-observation contracts.
Version 0.10 batches TP score synchronization and its Qwen3.5 example uses
the measured 4,608-token budget and 2,048-token requested-output guard.

> Engineering validation for 0.10.0: one matched Qwen3.5 BF16/TP=2 SWE
> C4/900-second pair improved output throughput per chip **1.33%** and
> per-request decode P90 **1.62%** versus the compression-off baseline.
> TTFT P95 worsened **8.65%**; the longest observed prompt was 52,505 tokens
> despite 262,144 configured context. This single local pair does not establish
> repeatability, semantic quality, production readiness, or an official
> leaderboard result. MTP2 remains experimental and rejected by default.
> See the [result scope](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.md) and
> [aggregate evidence](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261017-v010-swe-pair.json)
> for the exact host and workload. Earlier 0.9 results remain in the result
> index and [historical campaign](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/frontier-benchmarking-0.9.md).

## Ownership and maintenance

- School: Huazhong University of Science and Technology (HUST)
- Group: CGCL
- Advisor: Prof. Yao Wan (万瑶)
- Project lead: Sichen Liu (刘思辰), [@Seas0](https://github.com/Seas0)
- Maintainers: Jiawan Zhang (张家万), [@Jiawan23](https://github.com/Jiawan23);
  Ruohao Wei (韦若皓), [@kotoriqaq0](https://github.com/kotoriqaq0); and
  [@Seas0](https://github.com/Seas0)

The team agrees to maintain compatibility with vLLM-HUST and
vLLM-Ascend-HUST and to publish this work through the vLLM-HUST Extension
Manager. This is a CGCL-maintained plugin, not code built into either host.

## Source and redistribution scope

The scoring method adapts
[TriAttention](https://github.com/WeianMao/triattention) commit
[`a4bc3c8f`](https://github.com/WeianMao/triattention/tree/a4bc3c8f709db60f016ef42c3feb290fd0c00c1b)
and the paper [*TriAttention: Efficient Long Reasoning with Trigonometric KV
Compression*](https://arxiv.org/abs/2604.04921). The Ascend paged-KV runtime
was rewritten and does not copy the reference CUDA kernels.

Repository code and documentation are Apache-2.0. Model weights, datasets,
raw logs, and calibration statistics are not included in the wheel or sdist.
The development-only committed statistics have incomplete model/data
provenance and must not be redistributed merely under this repository's
license. See [NOTICE](NOTICE), [ownership and licensing](docs/development/ownership-and-licensing.md),
and [calibration artifacts](docs/development/calibration-artifacts.md).

## Compatibility

| Component | Supported line | Validated snapshot |
| --- | --- | --- |
| vLLM-HUST / `vllm` | `>=0.29.1.post1.dev0,<0.30` | `ebfcfba650` (`0.29.1.post1.dev456+gebfcfba65.empty`) |
| vLLM-Ascend-HUST / `vllm-ascend` | `>=0.25.1rc2.dev0,<0.26` | `7c8ec865a1` (`0.25.1rc2.dev774+hust.20260903.4.g7c8ec865a`) |
| Unified Frontier legacy line | Core `0.23.0+empty`, Ascend `0.23.0.post1` | Core `d0f22d2bda562156e4dbf433ce645e1769b4f804`, Ascend `03766ac696fde5ab1980d80ca0b8543d3580c989` |
| Extension Manager | `>=0.3,<0.4` | Manifest 0.3 contract |
| Python | `>=3.10,<3.15` | 3.11.16 |

The standard acceptance topology remains one Ascend NPU, the v1 scheduler and
`NPUModelRunner`, one full-attention KV group, block size 128, and dense
BF16/FP16 K/V. Qwen3.5 text/MoE hybrids additionally support one full-attention
group plus Gated-DeltaNet state groups with `mamba_cache_mode=none` or `align`;
tensor parallelism is accepted only when its size divides the model's KV-head count
(TP=2 for Qwen3.5-35B-A3B). On the validated Ascend host, that hybrid uses a
host-dependent cross-group scheduler alignment, 1,152- or 2,048-token
full-attention pages, and 128-token attention-kernel cache blocks. The plugin
validates the reported scales and expands each logical page into 9 or 16
kernel blocks. Unsupported combinations fail
during startup. The manifest
uses the stable `vllm.general_plugins` discovery entry point; current hosts do
not expose a frozen native KV-lifecycle API, so the narrow host range and
contract tests intentionally guard the internal integration seams.
The disjoint legacy line is admitted only for the exact unified Frontier source
pair above. Its older inline KV-binding ABI is covered by a separate contract
test; unknown 0.24 and 0.25.0 hosts remain excluded by the manifest.

## Install, enable, disable, and uninstall

For a step-by-step deployment, use the [user guide](docs/usage/README.md):
installation, configuration, calibration, serving, and troubleshooting.

Install the matched host stack first, then the released wheel. The plugin wheel
deliberately does not declare `vllm` or `vllm-ascend` as Python package
dependencies: those hardware-specific packages must be provisioned as one
tested runtime lock, while host compatibility is enforced by the Extension
Manager manifest and startup checks. This also prevents plugin installation
from asking pip to re-resolve an already provisioned host environment.

For the current vLLM 0.29 / vLLM-Ascend 0.25 development line, do not mix in
Triton-Ascend 3.2.2. That older wheel pins NumPy 1.26.4, whereas the supported
vLLM line requires `opencv-python-headless>=4.13`, whose available wheels
require NumPy 2. Downgrading OpenCV to 4.9 violates the vLLM requirement. Use
the matched Triton-Ascend 3.6 host stack instead.

After provisioning the matched host stack, choose a compatible published
plugin version and install it without re-resolving hardware-specific
dependencies. Set `KVCOMPRESS_PLUGIN_VERSION` before running:

```bash
python -m pip install --no-deps "vllm-ascend-kvcompress-hust==${KVCOMPRESS_PLUGIN_VERSION}"
```

For a fresh environment, install `vllm-hust-ext` separately from its approved
index before the command above. Source/plugin-test installs should likewise use
`python -m pip install --no-deps .` after provisioning the locked host stack.

Copy [examples/triattention.json](examples/triattention.json) and set
`stats_path` to the artifact location. If it does not exist, the enabled plugin
generates model-matched statistics before vLLM loads its serving weights,
atomically saves them, releases the temporary model, and continues startup.
For production, set `calibration_input_path` to a licensed representative
UTF-8 corpus; the bundled text is a bootstrap default. Then run:

```bash
vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /absolute/path/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension status org.vllm-hust.ascend-kvcompress

export VLLM_PLUGINS=ascend,ascend_kvcompress
vllm-hust-ext run -- vllm serve /path/to/model \
  --block-size 128 --no-enable-prefix-caching --no-async-scheduling
```

If `VLLM_PLUGINS` is unset, vLLM discovers all installed plugins. The package
remains inert unless the manager or direct enable flag activates it. Stop all
host processes before changing state:

```bash
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension forget org.vllm-hust.ascend-kvcompress
python -m pip uninstall vllm-ascend-kvcompress-hust
```

These operations only change the plugin package and manager state. Detailed
source/environment instructions are in the
[installation guide](docs/development/environment-installation.md), and the isolated wheel
workflow is in [packaging and release](docs/development/packaging-and-release.md).

For development without Extension Manager:

```bash
export VLLM_PLUGINS=ascend,ascend_kvcompress
export VLLM_ASCEND_KVCOMPRESS_ENABLED=1
export VLLM_ASCEND_KVCOMPRESS_CONFIG=/absolute/path/triattention.json
vllm serve /path/to/model --block-size 128 --no-enable-prefix-caching
```

Unset both `VLLM_ASCEND_KVCOMPRESS_*` variables and restart to disable it.

Qwen3.5-35B-A3B uses the dedicated
[Qwen3.5 example](examples/qwen3.5-35b-a3b-triattention.json) and TP=2. Its
temporary calibration model is automatically distributed across visible NPUs
before serving weights are loaded. See the
[Qwen3.5 adaptation note](docs/design/qwen3.5-35b-a3b-adaptation.md). The validated
earlier host snapshot required `--enforce-eager` and the explicit
`VLLM_ASCEND_KVCOMPRESS_QWEN_GDN_LIST_COMPAT=1` bridge for a legacy `int[]`
metadata ABI. The synchronized host exposes the Tensor ABI and has completed
FULL_AND_PIECEWISE graph capture without `--enforce-eager`; the opt-in bridge
is schema-gated and does not modify either host repository.

## Runtime and evidence

TriAttention retains calibrated scoring, V3 prefix/recent protection and
proportional middle-segment quotas. It scores only the ten Qwen3.5
full-attention layers; Gated-DeltaNet state remains with the host. Scheduler
and worker commit a compression transaction at a step boundary, preserving
semantic RoPE positions while shortening physical KV length. See the
[method design](docs/design/methods.md) and
[Qwen3.5 adaptation](docs/design/qwen3.5-35b-a3b-adaptation.md).

The 2026-10-17 SWE C4/900-second pair improved per-chip output
throughput 1.33% and per-user decode P90 1.62%, with an 8.65% TTFT P95
regression. The 0.9 SWE pair improved decode P90 1.05% while per-chip
throughput fell 2.71%; AgentX is a separate cohort. See
[result scope](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.md), the
[full 0.9 campaign](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/frontier-benchmarking-0.9.md),
and the [benchmark procedure index](docs/benchmark/README.md).
The historical Qwen2.5 A2/A3 and public long-context results stay in
[the archive](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/public-long-context-benchmarks.md).

## Conflict matrix

| Feature | Implementation status | Behavior |
| --- | --- | --- |
| Prefix cache | Local C4 pairs protocol-valid | Private destination blocks preserve shared source hashes; the selected 0.10 SWE pair improved both target metrics |
| Speculative decoding | Experimental | Opt-in MTP2 served both SWE 900-second arms with real acceptance; default startup guard remains |
| KV transfer / disaggregated P/D | Conflict | Rejected |
| Quantized KV | Conflict | Rejected; dense BF16/FP16 only |
| Qwen3.5 full-attention + Gated-DeltaNet hybrid | Experimental | `mamba_cache_mode=none` or `align`; only full-attention KV uses compacted physical length, while Gated-DeltaNet keeps semantic length |
| Other hybrid / MLA / sliding / local attention | Conflict | Rejected |
| Async scheduling | Official C4 pairs protocol-valid after merge | Commit waits for matching output; original source blocks remain reserved until already queued batches finish |
| TP > 1 | Conditional | Must divide KV heads; TP ranks synchronize scores |
| PP/DP/DCP/PCP > 1 | Conflict | Rejected |
| BidKV or another scheduler | Conflict | Standard v1 scheduler required; active balance scheduling rejected |
| Removed Prefix Router, KV Tiering, KNorm, PyramidKV Ascend, SliceGPT | Not integrated | No imports or assumptions about former host code |
| Other general plugins | Unverified | Use an explicit allowlist and test the combination |

## Configuration and validation

The general example uses an 8192-token budget, 1024-token recompute window,
V3 prefix/recent protection of 128/512 tokens, eight position segments,
8192-token scoring chunks, and every eighth scoring layer. It bypasses
compression below 64 requested output tokens. The 0.10 Qwen3.5 example samples
the ten full-attention layers with stride 2, a 4608-token budget, and a
2048-token requested-output guard. KV-related token counts must be positive multiples
of block size 128. For a Qwen3.5 runtime with a promoted 1,152- or 2,048-token attention
page, `kv_budget` must also be divisible by the active page size; it need not
be divisible by the scheduler's cross-group alignment.

- [Documentation map (current vs. historical)](docs/README.md)
- [Historical 2026-09-20 engineering validation](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/validation.md)
- [Benchmark procedures](docs/benchmark/README.md)
- [Historical V4.6-derived requirements](docs/archive/benchmark/kv-compress-test-requirements.md)
- [Historical A2/A3 benchmark protocol](docs/archive/benchmark/benchmarking.md)
- [Historical public long-context benchmarks](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/public-long-context-benchmarks.md)
- [HTML benchmark leaderboard (Frontier measured points and historical evidence)](docs/results/benchmark-leaderboard.html)
- [Qwen3.5-35B-A3B adaptation](docs/design/qwen3.5-35b-a3b-adaptation.md)
- [Packaging and release](docs/development/packaging-and-release.md)
- [Optimization records](docs/optimization/README.md)
- [Method extension API](docs/design/methods.md)
- [Calibration artifacts](docs/development/calibration-artifacts.md)

```bash
VLLM_PLUGINS='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

An NPU release candidate must also pass numerical kernel smoke and the
declared long-context quality/performance/HBM matrix on the exact host stack.

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
