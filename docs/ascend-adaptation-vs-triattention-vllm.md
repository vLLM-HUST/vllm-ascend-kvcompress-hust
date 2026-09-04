# Ascend Adaptation Compared with TriAttention

English | [简体中文](ascend-adaptation-vs-triattention-vllm.zh.md)

## Source boundary

The algorithmic source is the TriAttention paper and the Apache-2.0 reference
repository at commit `a4bc3c8f709db60f016ef42c3feb290fd0c00c1b`. The reference
scores unrotated-query statistics against post-RoPE keys and preserves a
recent window. This repository keeps that scoring model but implements a new
runtime for vLLM Ascend paged KV caches. No upstream CUDA runtime kernel is
copied here.

## What changes on Ascend

| Area | Reference implementation | This plugin |
| --- | --- | --- |
| Packaging | Standalone research scripts/patches | Python package with `vllm.general_plugins` and Extension Manager manifest |
| Cache layout | Reference contiguous tensors | Current vLLM-Ascend paged K/V cache unpacked by the attention implementation |
| Scoring | PyTorch/CUDA-oriented path | Direct paged-cache Triton-Ascend scoring |
| Aggregation | Intermediate normalized score tensors | Fused normalize, query-head max, and layer accumulation kernel |
| Memory | Per-operation temporaries | Persistent score, aggregate, dense-index, and K/V copy workspaces |
| Lifecycle | Research integration | Scheduler and worker mirrors with commit at a synchronous scheduling barrier |
| Positions | Reference sequence manipulation | Semantic RoPE positions remain monotonic; only physical slots and attention lengths shift |
| Activation | Environment/research integration | Manager-owned JSON or explicit development environment variables |

## Current host hooks

The public discovery contract is `vllm.general_plugins`. Once enabled, version
0.3 adapts the current internal host symbols:

- `Scheduler` and `KVCacheManager.allocate_slots` for physical block ownership;
- `BlockTable.compute_slot_mapping` for semantic-to-physical slot translation;
- `NPUModelRunner.initialize_kv_cache`, `_update_states`,
  `_build_attention_metadata`, and `sample_tokens` for cache binding,
  transaction mirroring, attention lengths, and post-step materialization.

The NPU runner is hooked lazily. Exact class and method signatures are checked
by tests and runtime validation, but these are not frozen upstream APIs. A host
upgrade requires a new compatibility review and acceptance run.

## Transaction invariant

Compression is synchronous with one model step:

1. Scheduler and worker independently arm the same request at a deterministic
   token threshold.
2. After sampling, the worker selects tokens and copies every layer into a
   private, block-aligned destination.
3. At the next scheduler barrier, the scheduler records the removed-token
   offset, trims the physical token count, and frees the unused tail blocks.
4. Future RoPE positions use the original semantic token number, while cache
   slot mapping and attention metadata use the shortened physical length.

This design intentionally rejects async scheduling, speculative decoding,
prefix caching, KV transfer, quantized KV, hybrid/MLA/local attention,
distributed parallel modes, and nonstandard schedulers such as BidKV until
each has a dedicated protocol and acceptance suite.

## Optimization boundary

The fused aggregation and persistent workspaces reduce allocations and kernel
launches in the compression hot path. `score_layer_stride` can sample scoring
layers while materialization still copies all layers. These are implementation
optimizations, not a performance guarantee. Only matched NPU measurements on
the supported host snapshots may be reported as version 0.3 evidence.
