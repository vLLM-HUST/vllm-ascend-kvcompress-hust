# Compression Method Architecture

English | [简体中文](methods.zh.md)

## Runtime layers

Version 0.6 is a self-contained plugin and does not use the removed
vLLM-HUST compression lifecycle:

1. `plugin.py` is the opt-in `vllm.general_plugins` entry point. It patches
   scheduler-side symbols immediately and installs the Ascend worker hook
   lazily, so manager and API-only processes do not import the NPU runner.
2. `calibration.py` generates a missing model-bound artifact before serving
   weights load, or exposes the same generator through a console command.
3. `stateful.py` owns scheduler/KV-manager state. It commits a completed
   compression at the next synchronous scheduling barrier and frees the old
   block-table tail.
4. `provider.py` validates current host objects, mirrors transactions in the
   worker, translates semantic positions to physical slots, and delegates
   selection/materialization to a method.
5. `methods/<name>/` owns an algorithm's options, compatibility checks,
   scoring, calibration, and KV materialization.

The current adapter deliberately depends on the internal host symbols listed
in the main README. Those interfaces are not frozen, so support must remain on
the tested host version line and fail closed after incompatible changes.

## Configuration contract

The manager stores one JSON object. `schema_version`, `provider`, `method`, and
`method_config` are required. For example:

```json
{
  "schema_version": 1,
  "provider": "ascend_kvcompress",
  "method": "triattention",
  "method_config": {
    "stats_path": "/absolute/path/stats.pt",
    "auto_calibrate": true,
    "calibration_input_path": "/absolute/path/calibration.txt",
    "calibration_max_length": 4096,
    "kv_budget": 8192,
    "recompute_window": 1024,
    "protected_recent_window": 512,
    "score_aggregation": "mean",
    "layer_aggregation": "mean",
    "score_chunk_size": 8192,
    "score_layer_stride": 8,
    "min_output_tokens_for_compression": 64
  }
}
```

Unknown keys, methods, incompatible block alignment, and invalid calibration
files are rejected rather than ignored. A missing file is generated only when
`auto_calibrate` is enabled; see the calibration artifact guide.

`min_output_tokens_for_compression` is a non-negative requested-output gate.
When a request's maximum generation is below it, both scheduler and worker skip
the transaction. `0` preserves the legacy always-eligible behavior; `64` is the
quality-qualified public-benchmark default that removes overhead from the
32-token retrieval workload.

`score_layer_stride=8` uniformly samples six of the 48 calibrated layers for
selection while all 48 K/V layers are still materialized. On the frozen public
Qasper and LongBench-v2 sets this setting preserved baseline quality and
improved the matched end-to-end result compared with the former value `4`.

## Method contract

Implement `KVCompressionMethod` from `methods/base.py`:

- `name`: stable registry/configuration name.
- `runtime_spec`: block-aligned threshold, recompute window, maximum physical
  length, and destination requirements visible to the scheduler adapter.
- `compatibility_reasons(worker)`: side-effect-free method checks.
- `bind_model_runner(runner, layer_caches)`: allocate state after common cache
  validation.
- `compress(request)`: synchronously materialize one transaction and return
  the new physical length, plus optional per-layer lengths.

Methods receive validated cache bindings. They must not mutate scheduler-owned
request objects or block tables. Returned lengths must be positive, no larger
than the declared maximum, and cover every layer when per-layer lengths are
used.

## Register another method

In-process registration:

```python
from vllm_ascend_kvcompress import register_method

register_method("my_method", create_method)
```

External packages may declare:

```toml
[project.entry-points."vllm_ascend_kvcompress.methods"]
my_method = "my_package.method:create_method"
```

Names are lowercase letters, digits, hyphens, or underscores. Duplicate names,
bad factories, and name mismatches fail closed.

## Acceptance checklist

- Parse only method-owned options and reject unknown values.
- Make all scheduler limits deterministic and block aligned.
- Validate model, RoPE, calibration, dtype, and cache layout before serving.
- Materialize every layer before returning from `compress`.
- Add registry, configuration, transaction, and numerical tests.
- Run matched compression-off/on correctness, quality, throughput, latency,
  and HBM acceptance on the exact supported host revisions.
