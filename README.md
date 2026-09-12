# vLLM Ascend KV Compression

English | [简体中文](README.zh.md)

An independently packaged TriAttention KV-cache compression plugin for the
upstream-aligned vLLM-HUST and vLLM-Ascend-HUST stacks. Version 0.4 adapts the
current hosts and Extension Manager without changing either host repository.

> Status: experimental release candidate. Package lifecycle, Ascend kernel
> smoke, three cold 16K engineering-control pairs, and bounded public
> LongBench-v2/LongBench quality checks pass. The full V4.6 official baseline,
> fully traced calibration, repeated public performance runs, and stability
> matrix remain release gates. See [Validation](docs/validation.md).

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

The supported topology is one Ascend NPU, the v1 scheduler and
`NPUModelRunner`, one full-attention KV group, block size 128, and dense
BF16/FP16 K/V. Unsupported combinations fail during startup. The manifest
uses the stable `vllm.general_plugins` discovery entry point; current hosts do
not expose a frozen native KV-lifecycle API, so the narrow host range and
contract tests intentionally guard the internal integration seams.

## Install, enable, disable, and uninstall

Install the host stack first, then the released wheel:

```bash
python -m pip install 'vllm-ascend-kvcompress-hust[manager]==0.4.0'
```

Copy [examples/triattention.json](examples/triattention.json), set
`stats_path` to model- and revision-matched statistics, then run:

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

## Runtime and long-context optimization

After explicit activation, the adapter validates and hooks the current
scheduler, KV-cache manager, concrete Ascend block table, and
`NPUModelRunner`. Compression is committed only after a synchronous model
step: semantic RoPE positions stay unchanged, physical attention/slot indices
use a per-request offset, and old blocks are released at the next scheduler
barrier.

Version 0.4 uses direct paged-K scoring, persistent workspaces, fused NPU
normalization/head/layer aggregation, dynamic JIT lengths, layer-stride
sampling, device-resident offsets, and a configurable requested-output gate
that skips scoring/copy when generation is too short to amortize it. The public quality sweep rejected the
aggressive 4,096-token budget and recommends an 8,192-token physical budget;
4K remains only a workload-specific option requiring separate quality checks.
The final 910B2 kernel benchmark measured 1.82x paged-copy, 2.81x direct-score,
and 1.34x aggregate speedups against generic references; offset update was 0.83x
and is not claimed as an improvement.

In three cold engineering-control pairs on Qwen2.5-14B, each with four fixed
16,384+1,024 requests at 0.4 RPS/concurrency 4, all 24 requests completed with
the expected output length. Compression retained 32/128 blocks per request
(75% physical KV reduction). Median total throughput was 1431.3 versus 1129.5
tok/s (+26.7%); mean TPOT was 39.16 versus 52.43 ms (-25.3%); mean E2E was
44.37 versus 57.81 s (-23.2%). This is engineering evidence against a
same-host compatibility control, not the required official V4.6 B0 claim.

A separate public A3 sweep used all eligible non-truncated cases from
LongBench-v2 (116), LongBench `passage_retrieval_en` (200), and LongBench
`qasper` (93). With an 8K budget, LongBench-v2 accuracy was unchanged and
request throughput improved 2.13%; retrieval stayed at 100% and deliberately
bypassed compression because its output cap is 32 tokens, with request
throughput changing -0.04%; Qasper F1 changed by -0.48 percentage points and
request throughput by -0.11%. B1 recorded 127/127 scheduler/worker commits and
reduced physical blocks 60.67% over compressed cases. See the full
[public benchmark record](docs/public-long-context-benchmarks.md), including
the rejected 4K Qasper result and single-run limitations.

## Conflict matrix

| Feature | 0.4 status | Behavior |
| --- | --- | --- |
| Prefix cache | Conflict | Rejected; must be disabled |
| Speculative decoding | Conflict | Rejected |
| KV transfer / disaggregated P/D | Conflict | Rejected |
| Quantized KV | Conflict | Rejected; dense BF16/FP16 only |
| Hybrid/MLA/sliding/local attention | Conflict | Rejected |
| Async scheduling | Conflict | Rejected |
| TP/PP/DP/DCP/PCP > 1 | Conflict | Rejected |
| BidKV or another scheduler | Conflict | Standard v1 scheduler required; active balance scheduling rejected |
| Removed Prefix Router, KV Tiering, KNorm, PyramidKV Ascend, SliceGPT | Not integrated | No imports or assumptions about former host code |
| Other general plugins | Unverified | Use an explicit allowlist and test the combination |

## Configuration and validation

The example uses an 8192-token budget, 1024-token recompute window, 512-token
protected recent window, 8192-token scoring chunks, and every fourth scoring
layer. It bypasses compression below 64 requested output tokens. KV-related
token counts must be positive multiples of block size 128.

- [Current validation record](docs/validation.md)
- [V4.6-derived requirements](docs/kv-compress-test-requirements.md)
- [Benchmark protocol](docs/benchmarking.md)
- [Public long-context benchmarks](docs/public-long-context-benchmarks.md)
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
