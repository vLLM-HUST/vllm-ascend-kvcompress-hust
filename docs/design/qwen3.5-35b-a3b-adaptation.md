# Qwen3.5-35B-A3B adaptation

English | [简体中文](qwen3.5-35b-a3b-adaptation.zh.md)

Updated: 2026-10-17. Qwen3.5-35B-A3B BF16/TP=2 is a model cohort in the
[Frontier protocol](../benchmark/frontier-benchmarking.md).
The Qwen2.5 V4.6 matrix is [historical](../archive/benchmark/kv-compress-test-requirements.md).

## Frozen model identity

- Upstream: [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), Apache-2.0.
- Hugging Face revision: `712cf74392b05026a6db2bf213d343747d1f6d45`.
- Obtain a local snapshot of the pinned revision in a model directory chosen
  for the serving host.
- Weight index total: 71,903,655,008 bytes in 14 safetensors shards.
- Native context: 262,144 tokens.

The [historical model manifest](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/qwen3.5-35b-a3b-model-manifest.json)
records the earlier 59d61f3c revision; do not use it as the hash receipt for
the current 712cf743 revision.

If the Hugging Face endpoint is unavailable, an official Qwen mirror can
provide the snapshot, provided its revision and file hashes match. Model
configuration and the complete weight index must be present before testing;
partial `.safetensors` files are not accepted.

## Architecture facts that affect compression

| Field | Value | Adaptation consequence |
| --- | ---: | --- |
| Text model type | `qwen3_5_moe_text` | Explicitly allowlisted |
| Total / active parameters | 35B / 3B | BF16 serving requires more than one 910B2 |
| Decoder layers | 40 | Hybrid layout validation is required |
| Layer pattern | 10 × (3 Gated-DeltaNet + 1 full attention) | Only layers 3, 7, …, 39 own compressible KV |
| Query / KV heads | 16 / 2 | TP=2 produces one local KV head per rank |
| Head / rotary dimensions | 256 / 64 | Add content scoring for 192 non-rotated dimensions |
| Experts | 256, top-8 + shared expert | Calibration must load the MoE model, not a dense proxy |
| RoPE theta | 10,000,000 | Exact per-layer frequencies are stored in calibration |

The model thinks by default. Quality runners should either apply the tokenizer's
non-thinking template (`enable_thinking=false`) or freeze an explicit reasoning
budget; mixing the two modes invalidates a B0/B1 comparison.

## Runtime adaptation

The plugin accepts exactly one full-attention KV group plus optional Mamba/GDN
groups for Qwen3.5, with `mamba_cache_mode=none` or `align`. Scheduler and worker
commits change only the full-attention block table. Slot positions and attention
lengths use the compacted physical view, while semantic RoPE positions and GDN
state progression remain unchanged. The `align` path and prefix-cache copy-on-
write commit passed a bounded TP=2/eager NPU smoke, not a full workload.

Ascend retains 128-token dense K/V kernel blocks. For this hybrid family,
the host can promote the attention manager page to 1,152 or 2,048 tokens;
the 0.9.0 release host reported 1,152. The plugin verifies the reported
scheduler alignment against the cache-manager LCM and expands each
attention page into 9 or 16 kernel blocks. `kv_budget` must be divisible
by the active attention page, but not by the cross-group alignment.

The old `int[]` GDN ABI bridge is opt-in and schema-gated. Historical
host ABI failures and the 2026-09-27 performance pairs are retained in the
[0.9 campaign](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/frontier-benchmarking-0.9.md).

Partial-RoPE calibration captures normalized query heads before rotation. The
64 rotated dimensions use TriAttention's trigonometric future-position score;
the 192 pass-through dimensions add their calibrated direct Q·K contribution.
The plugin's fused paged-key scorer now handles both terms for this model;
its historical 32,768-token single-layer NPU microbenchmark improved 10.08×.
That isolated speedup did not imply an end-to-end SWE speedup. With TP=2, each
rank scores its local KV head. The implementation batches all sampled
layer scores into one all-reduce maximum, producing the same token selection
on both ranks while reducing synchronization overhead. The dated
[optimization record](../optimization/triattention-tp-score-batching-20261017.md)
contains the implementation and service measurements.

V3 selection protects the first 128 and most recent 512 tokens, splits the
middle into eight segments, and assigns an exact proportional eviction quota
to every segment. The checked-in Qwen3.5 example samples five of the ten
full-attention layers (`score_layer_stride=2`) while materializing all ten
target layers. Its 2,048-token requested-output guard and 4,608-token
physical budget are workload-specific choices.

## Calibration and validation

Use the [calibration guide](../development/calibration-artifacts.md) and choose
the relevant protocol from the [benchmark index](../benchmark/README.md). The
older eager launch commands, 2026-09-20 quality matrix, and 2026-09-27
diagnostics are preserved in the [historical adaptation record](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/qwen3.5-20260920-adaptation-results.md).
