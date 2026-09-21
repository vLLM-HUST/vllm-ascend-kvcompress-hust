# vLLM Ascend KV Compression

English | [简体中文](README.zh.md)

An independently packaged TriAttention KV-cache compression plugin for the
upstream-aligned vLLM-HUST and vLLM-Ascend-HUST stacks. The current working tree
adds the paper's V3 position policy and an experimental Qwen3.5 hybrid-model
path on top of the 0.6 phase-precompute release, without changing either host
repository.

> Status: experimental release candidate. Package lifecycle, Ascend kernel
> smoke, the complete three-cold-start A2 engineering matrix, and bounded public
> LongBench-v2/LongBench quality checks pass. The completed A3 matrix fails its
> minimum-completion and six-window throughput-CV gates. The official V4.6 B0,
> authorized production data, and a passing stability matrix remain release
> gates. See [Validation](docs/validation.md).

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
license. See [NOTICE](NOTICE), [ownership and licensing](docs/ownership-and-licensing.md),
and [calibration artifacts](docs/calibration-artifacts.md).

## Compatibility

| Component | Supported line | Validated snapshot |
| --- | --- | --- |
| vLLM-HUST / `vllm` | `>=0.28.1.post1.dev0,<0.29` | `6cdc0304a8` (`0.28.1.post1.dev260`) |
| vLLM-Ascend-HUST / `vllm-ascend` | `>=0.25.1rc2.dev0,<0.26` | `5901bedbb7` (`0.25.1rc2.dev232+hust.20260903.4.g5901bedbb`) |
| Extension Manager | `>=0.2.0.dev0,<0.3` | `cf1ea71e3e` |
| Python | `>=3.10,<3.15` | 3.11.16 |

The standard acceptance topology remains one Ascend NPU, the v1 scheduler and
`NPUModelRunner`, one full-attention KV group, block size 128, and dense
BF16/FP16 K/V. Qwen3.5 text/MoE hybrids additionally support one full-attention
group plus Gated-DeltaNet state groups with `mamba_cache_mode=none`; tensor
parallelism is accepted only when its size divides the model's KV-head count
(TP=2 for Qwen3.5-35B-A3B). On the validated Ascend host, that hybrid uses a
32,768-token cross-group scheduler alignment, 2,048-token full-attention pages,
and 128-token attention-kernel cache blocks. The plugin validates all three
scales and expands the attention mapping 16:1. Unsupported combinations fail
during startup. The manifest
uses the stable `vllm.general_plugins` discovery entry point; current hosts do
not expose a frozen native KV-lifecycle API, so the narrow host range and
contract tests intentionally guard the internal integration seams.

## Install, enable, disable, and uninstall

Install the matched host stack first, then the released wheel. The plugin wheel
deliberately does not declare `vllm` or `vllm-ascend` as Python package
dependencies: those hardware-specific packages must be provisioned as one
tested runtime lock, while host compatibility is enforced by the Extension
Manager manifest and startup checks. This also prevents plugin installation
from asking pip to re-resolve an already provisioned host environment.

For the currently supported vLLM 0.28 / vLLM-Ascend 0.25 line, do not mix in
Triton-Ascend 3.2.2. That older wheel pins NumPy 1.26.4, whereas the supported
vLLM line requires `opencv-python-headless>=4.13`, whose available wheels
require NumPy 2. Downgrading OpenCV to 4.9 violates the vLLM requirement. Use
the matched Triton-Ascend 3.6 host stack instead.

After the host and Extension Manager are present, install the plugin without
changing host packages:

```bash
python -m pip install --no-deps vllm-ascend-kvcompress-hust==0.6.0
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
[installation guide](docs/environment-installation.md), and the isolated wheel
workflow is in [packaging and release](docs/packaging-and-release.md).

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
[Qwen3.5 adaptation note](docs/qwen3.5-35b-a3b-adaptation.md). The validated
host stack also requires `--enforce-eager` and the explicit
`VLLM_ASCEND_KVCOMPRESS_QWEN_GDN_LIST_COMPAT=1` bridge because its installed
GDN operator still exposes the legacy non-optional `int[]` metadata ABI. The
bridge is schema-gated and does not modify either host repository.

## Runtime and long-context optimization

After explicit activation, the adapter validates and hooks the current
scheduler, KV-cache manager, concrete Ascend block table, and
`NPUModelRunner`. Compression is committed only after a synchronous model
step: semantic RoPE positions stay unchanged, physical attention/slot indices
use a per-request offset, and old blocks are released at the next scheduler
barrier.

Version 0.6 prepares RoPE phase coefficients once for all sampled layers and
reuses them in direct paged-K scoring. On Ascend 910B2, the median amortized
scoring cost was 0.270 ms per sampled layer versus 0.460 ms for the former
in-kernel phase path, a 41.3% reduction. The validated stride is 8 (six of 48
layers for selection); every K/V layer is still materialized. Persistent copy,
fused aggregation, device-resident offsets, dynamic JIT lengths, and the
short-output bypass remain. A fused Triton K/V copy experiment was removed
after measuring 14.504 ms versus 0.334 ms for the retained workspace path.

The V3 policy hard-protects a configurable prefix and recent window, then
distributes eviction proportionally across equal middle-context segments. For
Qwen3.5, scoring only the 64 rotated dimensions would ignore 75% of each
256-dimensional key, so the adapter also adds a calibrated direct Q·K content
term over the 192 unrotated dimensions. Only the ten full-attention layers are
scored and compacted; Gated-DeltaNet recurrent state remains under the host's
native lifecycle. Tensor-parallel ranks synchronize each layer's selection
score before choosing one identical token set.

The complete Qwen2.5 A2 engineering matrix covers 8K/16K at
0.05/0.1/0.2/0.4 RPS, three cold lifecycles per arm. All eight cells passed;
all 768 measured arm-requests were correct and complete. The 16K cells reduced
physical KV by at least 50%, with median total-throughput changes from +0.71%
to +16.36%; 8K stayed below the compression threshold and was effectively
neutral. The completed A3 matrix improved median total throughput 13.17% and
reduced physical KV at least 73.33%, but failed because B0 completed 23 rather
than 24 requests per lifecycle and six-window throughput CV was 12.86% for B0
and 8.94% for B1, both above 5%. These are same-host commissioning results, not
an official V4.6 B0 claim.

A separate public long-context run used all eligible non-truncated cases from
LongBench-v2 (116), LongBench `passage_retrieval_en` (200), and LongBench
`qasper` (93) with Qwen2.5-14B-Instruct and the frozen sampling contract.
At the 8K physical budget, quality changed by −0.86 pp, 0.00 pp, and −0.24 pp,
respectively, so all three passed the 1 pp gate. Request throughput changed by
+7.78%, +2.41%, and +1.35%; B1 recorded 127/127 scheduler/worker commits and
reduced physical blocks 60.67% over compressed cases. A supplementary
Qwen3.5-35B-A3B TP=2 run also passed all quality and transaction gates, although
its performance did not beat B0. See the full
[public benchmark record](docs/public-long-context-benchmarks.md) and its
single-run limitations.

The supplementary Qwen3.5 transferred A2/A3 matrix is intentionally retained
as a negative result: all 792 measured arm-requests were correct and TP=2
transactions matched exactly, but only two of eight A2 cells met the 1%
throughput-regression budget, while A3 completed 4 rather than 24 requests per
lifecycle and reported 100% six-window throughput CV. This BF16/eager evidence
does not change Qwen2.5-14B-Instruct as the standard release model.

## Conflict matrix

| Feature | 0.6 status | Behavior |
| --- | --- | --- |
| Prefix cache | Conflict | Rejected; must be disabled |
| Speculative decoding | Conflict | Rejected |
| KV transfer / disaggregated P/D | Conflict | Rejected |
| Quantized KV | Conflict | Rejected; dense BF16/FP16 only |
| Qwen3.5 full-attention + Gated-DeltaNet hybrid | Experimental | `mamba_cache_mode=none`; only full-attention KV is compacted |
| Other hybrid / MLA / sliding / local attention | Conflict | Rejected |
| Async scheduling | Conflict | Rejected |
| TP > 1 | Conditional | Must divide KV heads; TP ranks synchronize scores |
| PP/DP/DCP/PCP > 1 | Conflict | Rejected |
| BidKV or another scheduler | Conflict | Standard v1 scheduler required; active balance scheduling rejected |
| Removed Prefix Router, KV Tiering, KNorm, PyramidKV Ascend, SliceGPT | Not integrated | No imports or assumptions about former host code |
| Other general plugins | Unverified | Use an explicit allowlist and test the combination |

## Configuration and validation

The default example uses an 8192-token budget, 1024-token recompute window,
V3 prefix/recent protection of 128/512 tokens, eight position segments,
8192-token scoring chunks, and every eighth scoring layer. It bypasses
compression below 64 requested output tokens. The Qwen3.5 example scores all
ten full-attention layers. KV-related token counts must be positive multiples
of block size 128. For a Qwen3.5 runtime with a promoted 2,048-token attention
page, `kv_budget` must also be divisible by 2,048; it need not be divisible by
the scheduler's 32,768-token cross-group alignment.

- [Current validation record](docs/validation.md)
- [V4.6-derived requirements](docs/kv-compress-test-requirements.md)
- [Benchmark protocol](docs/benchmarking.md)
- [Public long-context benchmarks](docs/public-long-context-benchmarks.md)
- [HTML benchmark leaderboard](docs/benchmark-leaderboard.html)
- [Qwen3.5-35B-A3B adaptation](docs/qwen3.5-35b-a3b-adaptation.md)
- [Packaging and release](docs/packaging-and-release.md)
- [Method extension API](docs/methods.md)
- [Calibration artifacts](docs/calibration-artifacts.md)

```bash
VLLM_PLUGINS='' TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

An NPU release candidate must also pass numerical kernel smoke and the
declared long-context quality/performance/HBM matrix on the exact host stack.
