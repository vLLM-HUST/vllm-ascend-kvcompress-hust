# Compression Method Architecture

English | [简体中文](methods.zh.md)

## Layers

The plugin has three explicit layers:

1. `plugin.py` installs idempotent vLLM-Ascend hooks.
2. `provider.py` owns the common Ascend cache contract and vLLM-HUST
   transaction lifecycle.
3. Each `methods/<name>/` package owns one algorithm's configuration,
   compatibility, state, scoring, and KV materialization.

The common package root must not contain built-in-method scoring, calibration,
or cache-manipulation modules. The method registry depends only on the stable
contracts in `methods/base.py`, so adding or changing an algorithm does not
expand the common provider's dependency surface.

The built-in TriAttention package illustrates the expected organization:

| Path | Responsibility |
| --- | --- |
| `methods/triattention/__init__.py` | Public method exports and factory |
| `methods/triattention/config.py` | Method-owned option parsing and validation |
| `methods/triattention/method.py` | `KVCompressionMethod` implementation |
| `methods/triattention/scoring.py` | Vectorized token scoring |
| `methods/triattention/stats.py` | Calibration artifact loading and validation |
| `methods/triattention/cache.py` | Paged-cache gather and materialization |

The artifact generator, accepted payload schemas, field semantics, validation,
and lifecycle are documented in
[TriAttention calibration artifacts](calibration-artifacts.md).

The outer vLLM provider is always `ascend_kvcompress`. The flat
`provider_config.method` scalar selects a registered method. Remaining scalar
options are passed unchanged to that method's factory.

## Method Contract

Implement `KVCompressionMethod` from `methods/base.py`:

- `name`: stable configuration and registry name.
- `runtime_spec`: scheduler-visible threshold, recompute window, maximum
  physical length, and private-destination requirement.
- `compatibility_reasons(worker)`: side-effect-free method-specific checks.
- `bind_model_runner(runner, layer_caches)`: initialize state after common cache
  layout validation and allocation.
- `compress(request)`: materialize one initial or repeated transaction and return its
  physical length, plus optional per-layer lengths.

The common provider validates method results before creating a scheduler plan.
Methods must not mutate scheduler ownership or model-runner block tables.
If a method returns per-layer physical lengths, it must report every bound
layer exactly once with a positive value no larger than the global physical
length.

## In-Process Registration

```python
from collections.abc import Mapping
from typing import Any

from vllm_ascend_kvcompress import register_method
from vllm_ascend_kvcompress.config import JsonScalar
from vllm_ascend_kvcompress.methods.base import KVCompressionMethod, ModelShape


def create_method(
    options: Mapping[str, JsonScalar],
    vllm_config: Any,
    model_shape: ModelShape,
) -> KVCompressionMethod:
    return MyCompressionMethod(options, vllm_config, model_shape)


register_method("my_method", create_method)
```

## Third-Party Entry Point

External packages can register without importing the plugin eagerly:

```toml
[project.entry-points."vllm_ascend_kvcompress.methods"]
my_method = "my_package.method:create_method"
```

Use it with:

```json
{
  "schema_version": 1,
  "provider": "ascend_kvcompress",
  "provider_config": {
    "method": "my_method",
    "option_owned_by_my_method": 128
  }
}
```

Names are normalized to lowercase and may contain letters, digits, hyphens,
and underscores. Duplicate names, unknown methods, invalid factories, and
factory/name mismatches fail closed.

## Method Checklist

- Parse only owned options and reject unknown keys.
- Return deterministic, positive scheduler limits before KV allocation.
- Validate model/calibration constraints in `compatibility_reasons`.
- Use only the validated `LayerCache` bindings passed by the provider.
- Materialize into the supplied destination blocks before returning.
- Return a valid physical length no larger than the advertised maximum.
- Add registry, configuration, compatibility, materialization, and transaction
  tests.
- Run matched long-context compression-off/on benchmarks and document stable
  public behavior; keep raw machine-specific logs under `docs/dev/`.
