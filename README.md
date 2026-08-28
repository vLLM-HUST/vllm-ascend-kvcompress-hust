# vLLM Ascend KV Cache Compression

English | [简体中文](README.zh.md)

An extensible KV-cache compression plugin for the HUST-maintained vLLM Ascend
stack. It connects compression methods to vLLM-HUST's transactional KV-cache
lifecycle without modifying the vLLM-HUST or vLLM-Ascend-HUST source trees.

The current release includes a correctness-first TriAttention method and a
public method interface for adding other compression algorithms.

## Ecosystem classification

This repository delivers an Ascend runtime compression component. Its
`vllm.general_plugins` entry point installs compatibility hooks, while the
provider and transactional KV-lifecycle interfaces are existing integration
surfaces rather than versioned Extension Bundle domain contracts. “Plugin”
therefore describes delivery and activation; the component's system role is KV
state transformation across scheduler, worker, and device planes.

It is not a KV store, KV connector, external state system, scheduler policy,
platform profile, or control plane. Compression algorithms are a second-level
extension surface owned by the common provider and must not be promoted to
top-level vLLM plugins. See
[`.vllm-hust/repository-profile.json`](./.vllm-hust/repository-profile.json) for
the machine-readable boundary.

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

Schema v1 currently supports one Ascend NPU, eager or ACL graph execution, one
plain full-attention KV group, block size 128, and separate contiguous BF16/FP16
K/V tensors. Multi-device execution, hybrid/MLA caches, sliding-window attention,
speculative decoding, KV transfer, sparse layouts, and quantized KV caches fail
closed.

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
      "score_chunk_size": 512,
      "score_layer_stride": 4
    }
  }'
```

Compression starts after `kv_budget + recompute_window` tokens. Budgets,
recompute windows, and score chunks must be positive multiples of block size
128. The legacy provider name `triattention_ascend` remains accepted for
existing configurations but cannot select another method.

`score_chunk_size=512` is a conservative default. Larger chunks reduce kernel
launch overhead but consume more temporary device memory. The reference
Qwen2.5-Coder-14B pressure run in the current benchmark report used 8192 after
validating that it fit on one 64-GiB Ascend 910B2; tune this value for each
model and device.
`score_layer_stride=4` uniformly samples calibrated layers for global token
selection by default while still compacting every KV layer. Set it to `1` for
full-layer scoring at the cost of compression-transaction latency.

## Built-in Methods

| Method | Strategy | Required artifact | Status |
| --- | --- | --- | --- |
| `triattention` | Query-aware post-RoPE selection and stateful repeated KV compaction | Complete per-layer TriAttention statistics | Single-NPU eager/ACL graph |

TriAttention accepts flat per-query-head or per-KV-head statistics and
structured `layer_stats` tensors. The loader uses
`torch.load(..., weights_only=True)` and never falls back to unsafe pickle
loading. Scaled RoPE requires exact `inv_freq` and per-layer `freq_scale_sq`
values.

See [TriAttention calibration artifacts](docs/calibration-artifacts.md) for the
purpose of each statistic, the current local artifacts, reproducible smoke and
production generation commands, payload schemas, validation, and provenance
requirements.

## Adding a Compression Method

New methods implement the `KVCompressionMethod` contract and register a
factory either in process or through the
`vllm_ascend_kvcompress.methods` Python entry-point group. The common provider
continues to own vLLM hooks, cache-layout checks, scheduling transactions,
commit acknowledgement, and physical decode positions.

See [Compression method architecture](docs/methods.md) for the complete API,
configuration contract, extension example, and required tests.

## Documentation

- [Current benchmark results](docs/resuts.md)
- [Core Ascend adaptations from the TriAttention vLLM runtime](docs/ascend-adaptation-vs-triattention-vllm.md)
- [TriAttention calibration artifacts](docs/calibration-artifacts.md)
- [Benchmarking and result interpretation](docs/benchmarking.md)
- [Compression method architecture](docs/methods.md)

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
