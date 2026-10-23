# TriAttention adaptation for vLLM Ascend

English | [简体中文](ascend-adaptation-vs-triattention-vllm.zh.md)

The scoring idea and V3 position policy derive from the
[TriAttention paper](https://arxiv.org/abs/2604.04921) and its
[Apache-2.0 reference](https://github.com/WeianMao/triattention/tree/a4bc3c8f709db60f016ef42c3feb290fd0c00c1b).
The plugin owns the Ascend paged-cache runtime, transaction protocol, and
calibration format. It does not copy the reference CUDA runtime kernels. The
[0.4 design](../archive/design/ascend-adaptation-vs-triattention-vllm-0.4.md)
is retained for historical comparison.

## Cache and scoring

The model runner binds dense BF16/FP16 K/V tensors in 128-token kernel blocks.
Qwen3.5 hybrid cache managers may expose 1,152- or 2,048-token attention pages;
the plugin checks the actual layout and expands each logical page into 9 or 16
kernel blocks. It scores only full-attention layers. Gated-DeltaNet state stays
with the host. Partial RoPE uses the trigonometric score on rotated dimensions
and a calibrated content dot product on pass-through dimensions. The fused
paged-key kernel avoids gathering a full contiguous K tensor for the supported
mean-score path; the Torch fallback covers unsupported kernel shapes. See the
[method design](methods.md) and [calibration guide](../development/calibration-artifacts.md).

V3 selection protects the configured prefix and recent suffix, then assigns
exact proportional eviction quotas across the middle segments. Every TP rank
must select the same tokens: ranks synchronize per-layer scores before
selection. A scoring-layer stride can reduce scoring work, while selected K/V
tokens are still materialized for every target layer.

## Transaction boundary

The scheduler arms an eligible request and reserves private destination
blocks. The worker scores current keys and materializes selected K/V after a
model step. The scheduler commits the corresponding physical-length offset at
the next barrier and releases old blocks only after already queued batches
finish. Semantic RoPE positions continue monotonically; only the physical
attention length and slot mapping shorten. Per-layer state and graph-capture
write slots are checked so MTP attention caches and recurrent GDN state do not
receive an unrelated full-attention offset.

The implementation has bounded support for APC, async scheduling, Qwen3.5
`mamba_cache_mode=none|align`, TP=2 where KV heads divide, and opt-in real
MTP2. KV transfer, quantized KV, MLA/local/sliding attention, and PP/DP/DCP/PCP
remain rejected by startup checks. These are internal host contracts guarded
by version ranges and tests, not a frozen upstream KV lifecycle API.

## Performance evidence boundary

Kernel improvements do not imply a service speedup. The published 0.9 SWE pair
improved per-user decode P90 but reduced output tokens/s/chip. For any
optimization claim, select the [protocol](../benchmark/README.md) that covers
its workload and reported metrics. See [result scope](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.md)
for the exact 0.9 values.
