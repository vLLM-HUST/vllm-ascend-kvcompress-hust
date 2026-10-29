# Source reading guide

English | [简体中文](source-reading.zh.md)

## Overall structure

The package root contains only the public `__init__.py`. Implementations are
grouped by responsibility:

```text
vllm_ascend_kvcompress/
├── __init__.py             # Public API and version
├── core/                   # Shared configuration, constants and validators
├── methods/                # Backend/algorithm contracts and registry
│   └── triattention/       # Built-in algorithm, calibration and kernels
├── runtime/                # Worker/scheduler transactions and coordinates
├── host/                   # Model configuration, host integration and ABI compatibility
├── extensions/             # Explicitly selected migration adapters
└── manifests/              # Static Extension Manager declarations
```

`CompressionBackend` describes static limits and chooses runtime factories.
`KVCompressionMethod` owns worker-side algorithm execution. The uniform core
provider and scheduler manage block ownership, acknowledgement and length
mapping. Query observation and per-layer physical lengths belong to downstream
implementations; the temporary `extensions/legacy.py` adapter retains these
features separately from the default TriAttention path.

Compression preserves semantic history. If semantic length S=8192 is compressed
to P=2048 KV tokens, the offset is S-P=6144. The next token at semantic position
8192 writes to physical position 2048, but RoPE still uses 8192. Retained keys
keep their original RoPE encoding. Mamba/GDN recurrent state keeps semantic
lengths.

Also distinguish the scheduler's cross-group LCM alignment, the full-attention
manager's logical pages, and the device cache's 128-token kernel pages. They can
have different sizes in hybrid models.

## File map

Paths below are relative to `src/vllm_ascend_kvcompress/`. Each file links to the
current source location.

| File | Responsibility and reading focus |
| --- | --- |
| [__init__.py](../../src/vllm_ascend_kvcompress/__init__.py) | Public package exports: version, provider name, backend base class and method registration/discovery. |
| [core/__init__.py](../../src/vllm_ascend_kvcompress/core/__init__.py) | Shared configuration, constants and validators; no eager host or algorithm initialization. |
| [core/config.py](../../src/vllm_ascend_kvcompress/core/config.py) | Explicit activation, environment/Extension Manager configuration, provider/method selection; ignore unknown top-level fields and pass method options to the backend. |
| [core/constants.py](../../src/vllm_ascend_kvcompress/core/constants.py) | Shared constants used across plugin modules; each constant documents its meaning and scope. |
| [core/validation.py](../../src/vllm_ascend_kvcompress/core/validation.py) | Shared configuration option validators for integers, booleans and allowed string choices. |
| [methods/__init__.py](../../src/vllm_ascend_kvcompress/methods/__init__.py) | Public interface exports and built-in factory registration; algorithm implementations and external entry points load on demand. |
| [methods/base.py](../../src/vllm_ascend_kvcompress/methods/base.py) | Model/runtime/cache/request/result structures and the worker algorithm contract. Core results describe one uniform cache length. |
| [methods/backend.py](../../src/vllm_ascend_kvcompress/methods/backend.py) | Static runtime specification and overridable preparation, algorithm, provider and scheduler factories. |
| [methods/registry.py](../../src/vllm_ascend_kvcompress/methods/registry.py) | Normalize names, register backend factories, discover third-party entry points and validate backend type/name; reject registration collisions. |
| [methods/builtins.py](../../src/vllm_ascend_kvcompress/methods/builtins.py) | Lazy factories and registration for built-in methods, assembled by methods/__init__.py; registry.py contains no built-in algorithm names. |
| [methods/triattention/__init__.py](../../src/vllm_ascend_kvcompress/methods/triattention/__init__.py) | Built-in configuration and backend exports; worker implementation loads on demand. |
| [methods/triattention/config.py](../../src/vllm_ascend_kvcompress/methods/triattention/config.py) | Budget, recomputed suffix, protected windows, scoring/selection, layer sampling and calibration options; threshold = budget + recompute_window. |
| [methods/triattention/backend.py](../../src/vllm_ascend_kvcompress/methods/triattention/backend.py) | Parse configuration and describe runtime limits without loading calibration or allocating device state; prepare artifacts and create the worker algorithm later. |
| [methods/triattention/method.py](../../src/vllm_ascend_kvcompress/methods/triattention/method.py) | Bind TP-local statistics/cache, allocate workspaces, sample scoring layers, aggregate across TP and materialize one shared token selection for all layers. |
| [methods/triattention/stats.py](../../src/vllm_ascend_kvcompress/methods/triattention/stats.py) | Load calibration formats, validate model/RoPE/numerics, group GQA heads, shard TP statistics and construct device values. |
| [methods/triattention/scoring.py](../../src/vllm_ascend_kvcompress/methods/triattention/scoring.py) | PyTorch scoring formula: geometric future positions, post-RoPE keys, magnitude correction, partial-RoPE content terms and head normalization. |
| [methods/triattention/selection.py](../../src/vllm_ascend_kvcompress/methods/triattention/selection.py) | Global/v3 policies, prefix/recent protection, integer segment quotas, batched top-k and sorted output indices. |
| [methods/triattention/cache.py](../../src/vllm_ascend_kvcompress/methods/triattention/cache.py) | Token-to-slot mapping, paged gathers and selected-token writes through overlap-safe workspaces. |
| [methods/triattention/calibration.py](../../src/vllm_ascend_kvcompress/methods/triattention/calibration.py) | CLI/automatic calibration: temporary HF model, pre-RoPE Q capture, statistics validation, process locking and atomic publication. |
| [methods/triattention/kernels.py](../../src/vllm_ascend_kvcompress/methods/triattention/kernels.py) | Mean-phase preparation, direct paged scoring and fused normalization/query-head maximum/cross-layer accumulation. |
| [runtime/__init__.py](../../src/vllm_ascend_kvcompress/runtime/__init__.py) | Runtime package boundary; importing the package does not eagerly import worker or scheduler implementations. |
| [runtime/transaction.py](../../src/vllm_ascend_kvcompress/runtime/transaction.py) | Serializable CompressionPlan, shared CompressionAnchor and the SchedulerOutput plan attribute; authorization and logical block IDs stay outside algorithm inputs. |
| [runtime/coordinates.py](../../src/vllm_ascend_kvcompress/runtime/coordinates.py) | Block-count rounding, logical-to-kernel page expansion and semantic-to-physical position shifting. |
| [runtime/provider.py](../../src/vllm_ascend_kvcompress/runtime/provider.py) | Worker cache binding, authorized execution, pending/active mirrors, request cleanup and uniform attention length/slot translation; overridable lifecycle methods. |
| [runtime/scheduler.py](../../src/vllm_ascend_kvcompress/runtime/scheduler.py) | Private destination reservation, acknowledgement, block-table commit and source release; async freezing, APC, group lengths and cleanup. |
| [host/__init__.py](../../src/vllm_ascend_kvcompress/host/__init__.py) | Host integration package boundary; importing it does not install hooks or load device dependencies. |
| [host/model_config.py](../../src/vllm_ascend_kvcompress/host/model_config.py) | Extract global model dimensions, original layer indices, full-attention layers and partial-RoPE dimensions. |
| [host/contracts.py](../../src/vllm_ascend_kvcompress/host/contracts.py) | Shared worker/scheduler model and cache constraints, supported layouts, MTP2 opt-in and cache tensor validation. |
| [host/plugin.py](../../src/vllm_ascend_kvcompress/host/plugin.py) | Registration, lazy runner loading and lifecycle dispatch; connect provider/scheduler state to host execution, metadata and slot mapping. |
| [host/compat.py](../../src/vllm_ascend_kvcompress/host/compat.py) | Host ABI bridges for routing, removed symbols/output fields, GDN list metadata and MTP2 cache binding; separate from algorithm preparation. |
| [host/qwen_gdn_ops.py](../../src/vllm_ascend_kvcompress/host/qwen_gdn_ops.py) | Missing GDN chunk-output Triton operator, Meta registration and experimental MTP2 reference convolution. |
| [extensions/__init__.py](../../src/vllm_ascend_kvcompress/extensions/__init__.py) | Optional adapters; the default runtime imports none of their implementations. |
| [extensions/legacy.py](../../src/vllm_ascend_kvcompress/extensions/legacy.py) | Explicitly selected, unmaintained migration adapter owning Query/per-layer request, result, pending, active and metadata state; scheduled for deletion. |
| [extensions/legacy_hooks.py](../../src/vllm_ascend_kvcompress/extensions/legacy_hooks.py) | Query custom-op observation and layer-aware FIA graph replay hooks used by the migration adapter. |
| [manifests/__init__.py](../../src/vllm_ascend_kvcompress/manifests/__init__.py) | Static Extension Manager package; manifests/*.json declares host ranges, activation, execution planes and the exclusive KV-mutation resource. |

Surrounding repository paths:

| Path | Responsibility |
| --- | --- |
| `pyproject.toml`, `MANIFEST.in`, `MOD_METADATA.json` | Packaging, plugin/calibration/bundle entry points and distribution metadata. |
| `examples/` | TriAttention JSON inputs to follow through configuration parsing. |
| `tests/test_*.py` | CPU-capable algorithm, transaction, hook, model/host contract, tool and documentation tests; some require installed host packages. |
| `tests/run_npu_*.py` | Explicit device smoke/benchmarks for kernels, partial RoPE, GDN and Qwen3.5 transactions. |
| `scripts/` | Input preparation, service load generation, scoring, paired comparison, leaderboard and evidence validation; outside the plugin inference call chain. |
| `docs/` | Paired user, design, protocol, development and historical navigation documents. |
| `LICENSE`, `NOTICE`, `LICENSES/` | Licensing and derived-code attribution, particularly the GDN operator boundary. |
| `dev/`, `dist/`, local cache/build directories | Ignored personal/generated artifacts, outside the main source reading path. |

The `scripts/` tools:

| File | Responsibility |
| --- | --- |
| `kvcompress_ascend_env.sh` | Configure the local Ascend environment and launch commands. |
| `benchmark_inputs.py`, `fetch_benchmark_originals.py` | Resolve external inputs and retrieve pinned, verifiable historical originals. |
| `kvcompress_prepare_dataset.py`, `kvcompress_benchmark_data.py` | Prepare acceptance inputs and acquire/freeze public long-context requests. |
| `kvcompress_dataset_applicability.py` | Audit whether frozen tokenized tasks can trigger compression. |
| `kvcompress_benchmark_run.py`, `kvcompress_benchmark_score.py` | Run requests against an OpenAI-compatible service and score public LongBench-family results. |
| `kvcompress_long_context_run.py`, `kvcompress_acceptance_compare.py` | Deterministic streaming long-context runs and three-cold-start B0/B1 acceptance comparison. |
| `kvcompress_m3_quality.py`, `kvcompress_m3_analyze.py` | M3 quality guard and three-lifecycle aggregation. |
| `kvcompress_frontier_preflight.py` | Read-only readiness check for frozen Frontier conditions. |
| `kvcompress_frontier_swe_pair.py`, `kvcompress_frontier_agentx_pair.py` | Validate and compare pinned SWE/AgentX pairs. |
| `kvcompress_leaderboard.py`, `verify_leaderboard_browser.py` | Build/verify the leaderboard, browser behavior, original links and failure presentation. |

## Recommended reading order

Follow algorithm contracts → the built-in algorithm → transactional runtime →
host integration. Package-relative paths below refer to the file map above;
`examples/` and `tests/` paths start at the repository root.

1. `__init__.py` → `examples/triattention.json` → `core/config.py` → `host/model_config.py`: understand public exports, activation, method selection and model dimensions.
2. `methods/base.py`: establish the meaning of ModelShape, MethodRuntimeSpec, LayerCache, CompressionRequest/Result and KVCompressionMethod before following execution.
3. `methods/backend.py` → `methods/registry.py` → `methods/builtins.py`: distinguish static backends from worker algorithm instances; read registration, lazy discovery and downstream factory overrides.
4. `methods/triattention/config.py` → `backend.py` → `method.py` in the same directory: connect configuration, runtime limits, preparation, construction, cache binding and compression. Defer kernel details initially.
5. TriAttention `stats.py` → `scoring.py` → `selection.py` → `cache.py`: follow calibration statistics → token scores → retained indices → K/V movement.
6. `runtime/transaction.py` → `runtime/coordinates.py`: understand plans, anchors, block ownership and semantic/physical coordinates.
7. `runtime/provider.py`: follow `bind_model_runner` → `compress_scheduled_requests` → pending state → `before_update_states` → row-offset/slot/attention-length updates. Inspect overridable request, execution and activation methods.
8. `runtime/scheduler.py`: follow `after_schedule` → `after_output` → `_commit_pending`, then request cleanup, deferred release, async freezing, APC and group-length hooks.
9. `host/contracts.py` → `host/plugin.py`: read supported host/cache layouts, then return to `register` to connect scheduler and provider lifecycles to the actual host.
10. `host/compat.py` → `host/qwen_gdn_ops.py`: inspect host ABI and GDN compatibility after understanding the main transaction path.

On a second pass, read `methods/triattention/calibration.py` for artifact
generation and `methods/triattention/kernels.py` for NPU implementation details.
Read `extensions/legacy.py` and `legacy_hooks.py` last, keeping their optional
protocol separate from the core API. Inspect `manifests/` and `pyproject.toml`
for discovery and activation declarations.

For architecture work, prioritize `methods/base.py` → `methods/backend.py` →
`runtime/transaction.py` → `runtime/provider.py` → `runtime/scheduler.py`.
These files define the boundary between algorithm policy, transaction ownership
and host state.

### Tests to read alongside the source

These files are useful as behavior specifications; reading them does not require
executing them. Some tests require installed host packages or accelerator support.

| Topic | Tests under `tests/` |
| --- | --- |
| Registration and initialization boundaries | `test_method_registry.py`, `test_lifecycle.py` |
| Worker transactions | `runtime_fakes.py`, `test_provider_integration.py` |
| Scheduling, acknowledgement and cache release | `test_stateful.py` |
| Host constraints and integration | `test_host_contract.py`, `test_current_host_contract.py`, `test_plugin.py`, `test_host_compat.py` |
| Host router and GDN ABI | `test_router_compat.py`, `test_qwen_gdn_ops.py` |
| TriAttention statistics and algorithm | `test_stats.py`, `test_scoring.py`, `test_selection.py`, `test_cache.py` |
| Calibration | `test_calibration.py` |
| Optional migration adapter | `test_legacy_extensions.py` |
| Package ownership and entry declarations | `test_package_layout.py`, `test_manifest.py` |

## One transaction in execution order

1. The host schedules and advances scheduler counts. `after_schedule` checks eligibility, reserves private destinations and attaches a plan to SchedulerOutput. Async mode freezes subsequent scheduling of that request.
2. The worker executes forward/sampling using source blocks. `compress_scheduled_requests` validates the plan, invokes the method to write destinations and records worker pending state.
3. After the host accepts the output, `after_output` marks scheduler pending ready. Synchronous mode commits before the next schedule; async mode commits after acknowledgement.
4. Commit switches the scheduler's full-attention block table, records active state, releases or defers releasing sources and restores async scheduling eligibility.
5. When scheduled again, the worker's `before_update_states` switches its local table and promotes pending to active. Each later step synchronizes current batch-row offsets and adjusts attention lengths/write slots.

The barrier is the host's output-acceptance lifecycle boundary, not an explicit
per-transaction `torch.npu.synchronize()`. The plan carries no separate
compression-success response; both sides rely on plan/state validation and the
accepted-output path.

## Current extension and state boundaries

The following describes the implemented interface boundaries, rather than an
unimplemented refactoring proposal.

| Concern | Current owner and contract |
| --- | --- |
| Algorithm policy | `methods/triattention/` owns calibration, scoring, protected/segmented selection and selected-token movement. Other methods own their implementations in their repositories. |
| Static construction | CompressionBackend provides runtime_spec without constructing a worker algorithm. prepare_model_load prepares artifacts; create_method constructs worker state. |
| Runtime customization | Downstream backends override create_provider/create_scheduler; providers can override lifecycle, request construction, algorithm execution and pending activation without adding downstream policy fields to the core. |
| Uniform algorithm contract | CompressionRequest contains forward-complete lengths and device page tables; it does not contain the scheduler plan or CPU block ownership. CompressionResult reports one length equal to the backend target. All core transactions use private destinations. |
| Necessary transaction state | runtime owns destination reservation, accepted-output acknowledgement, pending/active mirrors, coordinate mapping, APC protection and source release. These mechanisms remain necessary without Query observation. |
| Query/per-layer extensions | Only the explicitly selected extensions.legacy adapter retains these protocols and buffers. It is unmaintained and scheduled for deletion; downstream packages should own their adapters. TriAttention does not load it. |
| Shared utilities | runtime.coordinates holds coordinate operations; host.contracts holds common host/cache checks. The core provider does not import TriAttention-private kernels. |
| Host/model ABI | host owns GDN, MTP2 and routing compatibility. These are host-integration concerns rather than sparsification policy. |
| Import paths | Implementations now live under core/runtime/host or methods/triattention. Old top-level module forwarding aliases are not provided; package-level public exports remain available. |


### Entry points after the move

- vLLM plugin: `vllm_ascend_kvcompress.host.plugin:register`.
- Calibration CLI: `vllm_ascend_kvcompress.methods.triattention.calibration:main`; the command remains `vllm-ascend-kvcompress-calibrate`.
- Extension bundle: `vllm_ascend_kvcompress.manifests`.

`pyproject.toml` and the static manifest use these locations. Reinstall an
existing installation after updating the source to refresh entry-point metadata.
Downstream code importing moved implementation modules must update its imports.

See the [method architecture](../design/methods.md) for additional context;
where older descriptions differ, use current source and tests as the reference.
