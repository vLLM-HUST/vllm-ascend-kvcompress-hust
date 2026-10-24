# TriAttention TP score batching and SWE configuration search

English | [简体中文](triattention-tp-score-batching-20261017.zh.md) |
[Optimization index](README.md)

Date: 2026-10-17. Scope: the 0.10.0 source implementation on the pinned
Qwen3.5-35B-A3B BF16, TP=2 Ascend 910B2 host. This is an engineering record,
not a claim about every model, host, or request mix.

## Problem and implementation

In the TP=2 scoring path, each sampled attention layer previously
performed its own HCCL MAX reduction. With stride 2 on Qwen3.5, five sampled
layers meant five collectives per compression selection.

The change allocates a persistent contiguous score workspace, normalizes
each sampled layer into its own workspace row, performs one HCCL MAX over the
whole workspace, then reduces the layer axis. MAX acts independently on each
layer/token entry, so batching preserves the per-layer cross-rank result.
The Ascend normalization kernel uses the same minimum variance as the Torch
fallback. TriAttention's calibrated scoring, V3 protected windows, segment
quotas, and KV materialization for all ten full-attention layers remain in
place. See [method.py](../../src/vllm_ascend_kvcompress/methods/triattention/method.py)
and [kernels.py](../../src/vllm_ascend_kvcompress/methods/triattention/kernels.py).

This code change reduces synchronization work during a compression event.
The service result also depends on **when** that event occurs. The candidate
search therefore kept the same optimized source and changed the physical KV
budget and requested-output guard. The guard checks the request's configured
`max_tokens`; a 2,048-token guard bypasses requests with smaller limits.

## Historical testing and evidence

The [complete frozen record](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b3d2ed2692b5575b5ae77b93fc82e32c9119be8e/archive/mod-targeted-20261023/compress-mixed/snapshot/docs/optimization/triattention-tp-score-batching-20261017.md) preserves the workload, measurements, negative results and limitations.
[Aggregate original data](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261017-v010-swe-pair.json).
B0/B1 in the immutable original are historical run labels, not a redefinition of the current project baseline.
