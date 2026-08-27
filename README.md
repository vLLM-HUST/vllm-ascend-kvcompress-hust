# vLLM Ascend KV Cache Compression

English | [简体中文](README.zh.md)

An extensible KV-cache compression plugin for the HUST-maintained vLLM Ascend
stack. It connects compression methods to vLLM-HUST's transactional KV-cache
lifecycle without modifying the vLLM-HUST or vLLM-Ascend-HUST source trees.

The current release includes a correctness-first TriAttention method and a
public method interface for adding other compression algorithms.

## Highlights

- Method-neutral Ascend provider with configuration-based algorithm selection.
- Built-in `triattention` token-selection method.
- Public Python registry and third-party entry-point group for new methods.
- Fail-closed compatibility checks before formal KV-cache allocation.
- Atomic scheduler commit, block reclamation, and semantic/physical position
  separation through the native vLLM-HUST lifecycle.
- English and Simplified Chinese user documentation.

## Compatibility

This release targets these source lines:

| Component | Compatible version |
| --- | --- |
| vLLM-HUST | `>=0.23.1,<0.24` |
| vLLM-Ascend-HUST | `>=0.19.1,<0.20` |
| Python | `>=3.10,<3.15` |

The tested reference snapshots are vLLM-HUST
`1a06c55468966de8ef471ecb7612c199e15a153a` and vLLM-Ascend-HUST
`ac2b94f1536090e2cd0d6c2f8bc8087e336193d5`.

Schema v1 currently supports one Ascend NPU, eager execution, one plain
full-attention KV group, block size 128, and separate contiguous BF16/FP16 K/V
tensors. Multi-device execution, graph mode, hybrid/MLA caches, sliding-window
attention, speculative decoding, KV transfer, sparse layouts, and quantized KV
caches fail closed.

## Installation

Install into the environment that already contains matching editable
vLLM-HUST and vLLM-Ascend-HUST packages:

```bash
uv pip install -e /path/to/vllm-ascend-kvcompress-hust
```

If `VLLM_PLUGINS` is set explicitly, include the canonical plugin name:

```bash
export VLLM_PLUGINS=ascend_kvcompress
```

vLLM-HUST currently enables its independent Knorm compressor by default when
prefix caching is disabled. Disable it so only one component owns KV block
table mutations:

```bash
export VLLM_KNORM_ENABLED=0
```

## Quick Start: TriAttention

The provider name is stable across algorithms. Select the implementation with
the flat JSON-scalar `method` option:

```bash
vllm serve /path/to/model \
  --enforce-eager \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --block-size 128 \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.8 \
  --kv-cache-compression-config '{
    "schema_version": 1,
    "provider": "ascend_kvcompress",
    "provider_config": {
      "method": "triattention",
      "stats_path": "/path/to/triattention_stats.pt",
      "kv_budget": 2048,
      "recompute_window": 128,
      "protected_recent_window": 128,
      "score_aggregation": "mean",
      "layer_aggregation": "mean",
      "score_chunk_size": 512
    }
  }'
```

Compression starts after `kv_budget + recompute_window` tokens. Budgets,
recompute windows, and score chunks must be positive multiples of block size
128. The legacy provider name `triattention_ascend` remains accepted for
existing configurations but cannot select another method.

`score_chunk_size=512` is a conservative default. Larger chunks reduce kernel
launch overhead but consume more temporary device memory. The reference
Qwen2.5-Coder-14B pressure run below used 8192 after validating that it fit on
one 64-GiB Ascend 910B2; tune this value for each model and device.

## Built-in Methods

| Method | Strategy | Required artifact | Status |
| --- | --- | --- | --- |
| `triattention` | Query-aware post-RoPE token selection and KV compaction | Complete per-layer TriAttention statistics | Single-NPU eager validated |

TriAttention accepts flat per-query-head or per-KV-head statistics and
structured `layer_stats` tensors. The loader uses
`torch.load(..., weights_only=True)` and never falls back to unsafe pickle
loading. Scaled RoPE requires exact `inv_freq` and per-layer `freq_scale_sq`
values.

## Adding a Compression Method

New methods implement the `KVCompressionMethod` contract and register a
factory either in process or through the
`vllm_ascend_kvcompress.methods` Python entry-point group. The common provider
continues to own vLLM hooks, cache-layout checks, scheduling transactions,
commit acknowledgement, and physical decode positions.

See [Compression method architecture](docs/methods.md) for the complete API,
configuration contract, extension example, and required tests.

## Reference A/B Results

The post-refactor A/B suite used one Ascend 910B2,
Qwen2.5-Coder-14B-Instruct, BF16 KV, block size 128, eager mode, and a
20.93-GiB preallocated KV pool. The only service-side A/B difference was the
TriAttention configuration: 2048-token budget, 128-token recompute and recent
windows, and `score_chunk_size=8192`.

All four matched pairs completed with identical persisted input/output lengths
and no failures: 30/30 requests in the baseline and 30/30 with TriAttention.
Latency is shown in milliseconds; changes are TriAttention relative to the
baseline.

| Scenario | Input/output x requests | Total tok/s, off to on | P99 TTFT, off to on | Mean TPOT, off to on |
| --- | ---: | ---: | ---: | ---: |
| prefix-repetition-online | 2560/256 x 4 | 131.76 to 129.72 (-1.55%) | 590.92 to 747.60 (+26.52%) | 81.79 to 82.69 (+1.11%) |
| random-online | 2560/32 x 2 | 796.29 to 769.50 (-3.36%) | 365.03 to 455.17 (+24.69%) | 80.72 to 81.55 (+1.03%) |
| knorm-kv-compression-longctx | 8192/256 x 8 | 1246.91 to 1219.72 (-2.18%) | 2628.74 to 2808.23 (+6.83%) | 93.84 to 95.72 (+2.00%) |
| kv-pressure-online | 8192/64 x 16 | 5060.13 to 5113.44 (+1.05%) | 20440.16 to 20057.73 (-1.87%) | 182.85 to 194.95 (+6.61%) |

Every TriAttention request produced a compression commit and acknowledgement.
The 2560-token prompts changed from 20 to 16 physical blocks; the 8192-token
prompts changed from 64 to 16 and returned 48 blocks per request. Observed
active KV-pool peaks changed from 29.6% to 12.9% in the long-context run and
from 94.7% to 26.2% in the pressure run.

The allocator reserves the KV pool at startup, so process-level HBM does not
fall when blocks are reclaimed. Active KV-pool usage and returned blocks are
the meaningful capacity signals. These are single-run engineering results,
not a cross-platform guarantee; lossy quality must be evaluated separately.

See [Benchmarking and result interpretation](docs/benchmarking.md) for exact
baseline/TriAttention service startup commands, all four benchmark client
commands, the reproduction contract, and reporting requirements.

## Validation

```bash
python -m pytest -q
ruff check .
ruff format --check .
python -m compileall -q src tests
```

Runtime changes must also be compared with compression disabled and enabled on
an otherwise identical long-context workload. Record prompt and output lengths,
concurrency, TTFT, TPOT/ITL, throughput, successful requests, KV blocks released,
and NPU memory. Always select an idle assigned NPU and verify that the server
process exits after the run.

## Documentation Policy

- `README.md`, `README.zh.md`, `CONTRIBUTING.md`, `CONTRIBUTING.zh.md`, and
  public files under `docs/` describe stable, release-ready behavior.
- Local experiments, intermediate benchmark logs, machine paths, and
  model-specific validation reports belong under `docs/dev/`.
- Generated calibration files and local artifacts belong under `artifacts/`.
- `docs/dev/` and `artifacts/` are intentionally git-ignored.
- English public documentation is canonical; update the matching `.zh.md`
  document in the same change.

## Related HUST Projects

- [vLLM-HUST](https://github.com/vLLM-HUST/vllm-hust)
- [vLLM-Ascend-HUST](https://github.com/vLLM-HUST/vllm-ascend-hust)
- [vLLM-HUST Benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark)

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
