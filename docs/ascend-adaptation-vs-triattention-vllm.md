# Core Ascend Adaptations from the TriAttention vLLM Runtime

English | [简体中文](ascend-adaptation-vs-triattention-vllm.zh.md)

This document describes the current implementation in this repository relative
to the reference runtime under `triattention/triattention/vllm/`. It covers the
runtime and platform adaptations, not a line-by-line port or a claim of feature
parity.

## Summary

The Ascend implementation keeps the central TriAttention algorithm: score
cached post-RoPE keys with calibrated future-query statistics, protect recent
tokens, select the highest-value tokens, and preserve their causal order.

The integration around that algorithm is different. The reference runtime is a
TriAttention-specific compatibility layer for generic vLLM. It patches
scheduler, worker, allocation, input, engine, and runner paths. This repository
uses vLLM-HUST's transactional KV-compression contract, adds narrow hooks for
the Ascend model runner, and exposes TriAttention through a method-neutral
provider.

## Architecture at a Glance

| Area | Reference `triattention/vllm` runtime | Current Ascend implementation |
| --- | --- | --- |
| Integration | TriAttention-specific scheduler, worker, runner, allocation, and engine patches | Native vLLM-HUST compression plans plus narrow Ascend hooks |
| Configuration | `TRIATTENTION_*` and `TRIATTN_RUNTIME_*` environment variables | Validated `--kv-cache-compression-config` JSON with a named method |
| Execution | CUDA-oriented PyTorch/Triton paths | torch-npu plus an Ascend Triton scoring kernel and PyTorch fallback |
| Cache layout | Adapts several inferred runner and combined-KV layouts | Exact separate K/V paging: `[blocks, 128, kv_heads, head_dim]` |
| Selection layout | Per-head, per-layer, and per-layer-per-head options | One ordered request-wide token set shared by every head and layer |
| Compression cycles | Stateful runtime-managed cycles | Initial final-prefill transaction plus repeated decode-time transactions |
| Block ownership | Custom events and direct block-manager reconciliation | Scheduler-owned reserve, validate, commit, release, and acknowledgement |
| Position handling | Effective-length tracking and multiple input/allocation patches | Semantic RoPE positions retained; only physical slots and lengths are shifted |
| Execution mode | Eager-oriented compatibility surface | Validated in eager and ACL graph execution; async scheduling still rejected |
| Extension unit | TriAttention runtime | Common provider and public `KVCompressionMethod` registry |

## 1. Transactional Scheduling and Block Ownership

The reference runtime attaches TriAttention signals and compression events to
generic vLLM objects. Its compatibility layer reconciles those events with the
block manager and patches asynchronous queue boundaries where necessary.

The Ascend implementation delegates the first compression cycle to the native
vLLM-HUST lifecycle:

1. Before formal KV allocation, the provider advertises the compression
   threshold, recompute requirement, maximum physical length, and destination
   policy.
2. At final prefill, the scheduler starts a transaction and supplies the source
   block table and private destination blocks.
3. `NPUModelRunner` materializes compacted K/V and returns a
   `KVCacheCompressionPlan`; it never releases scheduler-owned blocks itself.
4. The scheduler validates the expected table, commits atomically, releases
   blocks, and later sends an acknowledgement.
5. Only after that acknowledgement does the runner replace its local request
   and input-batch block tables.

For subsequent cycles, the plugin-local compatibility layer in `stateful.py`
extends the same validation and commit rules. It arms a new transaction after
the physical cache again reaches `kv_budget + recompute_window`, reserves a
fresh private destination, and verifies the current block table before
replacement. This adds repeated compression without changing either the
vLLM-HUST or vLLM-Ascend-HUST source tree.

Async scheduling remains fail-closed because the current schema does not add a
separate asynchronous plan-delivery contract.

## 2. Ascend Paged-Cache Materialization

Ascend attention binds separate, contiguous K and V tensors with shape
`[num_blocks, 128, num_kv_heads, head_dim]`. The implementation therefore:

- maps logical token indices to physical slots through request block IDs;
- computes source and destination slot mappings once per transaction;
- gathers K and V into persistent device workspaces before any destination is
  overwritten;
- reuses those mappings and workspaces for every layer; and
- writes selected tokens with device-side `index_copy_` operations.

The temporary workspaces make overlapping source and destination blocks safe
without allocating a new dense K/V buffer for every layer and transaction.
Block size 128 and the exact separate-K/V layout are checked before cache
binding; unsupported layouts fail closed.

## 3. Ascend Scoring Hot Path

The initial Ascend port used only vectorized PyTorch operations. The current
mean-aggregation path adds an Ascend Triton kernel that reads post-RoPE keys
directly from paged cache and fuses:

- logical-token to paged-cache addressing;
- calibrated complex query means;
- RoPE phase and precomputed future-offset trigonometric means;
- frequency scaling; and
- the magnitude-regression correction term.

This removes the dense key gather from the scoring hot path. Calibration-derived
frequency scales, correction coefficients, and offset trigonometric means are
precomputed when the cache is bound. A device-resident score workspace is also
reused across transactions.

If the specialized kernel is not applicable, including unsupported device or
aggregation combinations, the method falls back to chunked vectorized PyTorch.
`score_chunk_size` bounds that fallback's working set.

## 4. One Shared Token Layout

The reference runtime can retain different token sets by layer or head. The
current vLLM-HUST request block table represents one physical order shared by
all attention layers, so this implementation produces one request-wide
`keep_indices` tensor:

1. Normalize scores independently for each calibrated query head.
2. Reduce query heads with `max` for each scoring layer.
3. Combine layers with configured `mean` or `max` aggregation.
4. Force the protected recent window to remain selected.
5. Run one global Top-K and sort the selected indices.
6. Materialize the same ordered indices for every K/V layer.

By default, `score_layer_stride=4` uniformly samples calibrated layers for
selection while still compacting every cache layer. Setting it to `1` restores
all-layer scoring at higher transaction cost. This is an Ascend performance
policy, not a change to the cache layout or block ownership contract.

## 5. Semantic and Physical State

After compression, each request has a semantic length used for model progress
and RoPE and a shorter physical length used for KV allocation and slot mapping.
The provider stores an anchor pair for each committed request:

```text
removed tokens = semantic anchor - physical anchor
physical position = semantic position - removed tokens
```

Model-visible positions remain semantic. The provider applies the offset only
to attention sequence lengths, optimistic physical lengths, and the existing
block-table slot-mapping call. Per-request offsets live on the NPU and are
updated only when request rows or compression state change; stable decode steps
avoid a CPU-to-NPU offset copy and avoid recomputing a second full slot mapping.

The offset is updated after every commit acknowledgement, so repeated
compression preserves monotonic semantic positions while keeping physical
decode writes dense.

## 6. ACL Graph Compatibility

The provider no longer requires `--enforce-eager`. Its physical-position
transformation uses fixed device buffers and graph-compatible NPU tensor
operations, and the normal torch-npu startup preflight remains enabled. Both
eager and ACL graph execution are supported within the schema-v1 compatibility
scope.

This does not enable async scheduling, multiple devices, alternate model
runners, or arbitrary cache layouts. Those remain explicit compatibility
failures.

## 7. Strict Calibration and RoPE Validation

The reference loader offers several model and RoPE reconstruction fallbacks.
The Ascend loader deliberately validates more before KV allocation:

- `torch.load(..., weights_only=True)` is used on CPU;
- flat per-head and structured per-layer payloads are accepted;
- layer coverage, head counts, head dimension, model type, RoPE style, and
  `rope_theta` are checked;
- query-head statistics are grouped for GQA without duplicating KV heads;
- scaled or non-default RoPE requires exact per-layer `inv_freq`; and
- scaled RoPE also requires exact per-layer `freq_scale_sq`.

Ambiguous metadata fails closed instead of silently choosing a frequency model
that could change selection semantics.

Generation, schemas, current local files, and provenance requirements are
documented in [TriAttention calibration artifacts](calibration-artifacts.md).

## 8. Method-Neutral Provider Boundary

Runtime ownership is split deliberately:

- `provider.py` owns compatibility, transactions, acknowledgements, and
  semantic/physical request state;
- `stateful.py` adds repeated-transaction compatibility to the native manager
  and scheduler contract;
- `methods/base.py` defines the framework-facing method interface;
- `methods/registry.py` resolves built-in and third-party factories; and
- `methods/triattention/` owns calibration, scoring, selection, and K/V
  materialization.

Another compression algorithm can reuse the Ascend lifecycle without copying
the TriAttention integration.

## Preserved Algorithmic Behavior

The adaptation retains these central TriAttention properties:

- calibrated complex future-query statistics;
- post-RoPE key scoring at geometrically spaced future offsets;
- the magnitude-regression correction;
- head-wise score normalization;
- protected recent tokens;
- Top-K selection followed by causal-order sorting; and
- identical selection indices for K and V materialization.

## Current Scope and Evidence

The validated schema covers one Ascend NPU, the standard v1
`NPUModelRunner`/`AscendAttentionBackend`, eager or ACL graph execution, one
plain full-attention KV group, separate BF16/FP16 K/V, and block size 128.

It rejects multi-device parallelism, async scheduling, hybrid/MLA caches,
sliding-window or chunked-attention cache layouts, speculative decoding, KV
transfer, sparse/model-native compression, and quantized KV. Prefix caching and
the independent Knorm compressor must be disabled for isolated validation.

See [Current benchmark results](resuts.md) for performance, capacity, quality,
and validation status. See [Benchmarking and result interpretation](benchmarking.md)
for the reproducibility contract.
