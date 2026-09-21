# Qwen3.5-35B-A3B adaptation

English | [简体中文](qwen3.5-35b-a3b-adaptation.zh.md)

Updated: 2026-09-20. This is an engineering adaptation target, not a change to
the standard acceptance model. Routine release testing continues to use
`Qwen/Qwen2.5-14B-Instruct` as specified by
[the acceptance requirements](kv-compress-test-requirements.md). Qwen3.5 is
covered because its full-attention/Gated-DeltaNet hybrid is a useful precursor
to the related Intern-S2 architecture.

## Frozen model identity

- Upstream: [`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), Apache-2.0.
- Hugging Face revision: `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- Local snapshot: `/workspace/models/Qwen--Qwen3.5-35B-A3B`.
- Weight index total: 71,903,655,008 bytes in 14 safetensors shards.
- Native context: 262,144 tokens.

The complete shard, index-consistency, and SHA-256 record is in
[`qwen3.5-35b-a3b-model-manifest.json`](evidence/qwen3.5-35b-a3b-model-manifest.json).

The snapshot is downloaded through the official ModelScope Qwen mirror when
the Hugging Face endpoint is unavailable. Model configuration and the complete
weight index must be present before testing; partial `.safetensors` files are
not accepted.

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
groups for Qwen3.5, with `mamba_cache_mode=none`. Scheduler and worker commits
truncate only the full-attention block table. Slot positions and attention
lengths use the compacted physical view, while semantic RoPE positions and GDN
state progression remain unchanged.

The validated Ascend host exposes three distinct granularities: a 32,768-token
cross-group scheduler alignment (the LCM of all cache managers), a 2,048-token
full-attention manager page, and 128-token dense K/V kernel blocks. The plugin
validates the LCM and uses the attention manager's block table for compression,
then expands every 2,048-token attention ID into 16 consecutive kernel IDs for
scoring/materialization. `kv_budget` must be divisible by 2,048, but does not
need to be divisible by the 32,768-token scheduler alignment. Other layouts
fail closed.

The validated host combination currently requires `--enforce-eager`. It also
contains a binary/source ABI mismatch: the installed GDN causal-convolution
operator declares four non-optional `int[]` metadata inputs, while the current
Python call site supplies device tensors and uses `None` for absent metadata.
Set `VLLM_ASCEND_KVCOMPRESS_QWEN_GDN_LIST_COMPAT=1` to enable the plugin-owned,
opt-in bridge for exactly that legacy schema. The bridge converts tensors to
integer lists, flattens the two-dimensional cache-index table in row-major
order, and maps absent metadata to an empty list. A native `Tensor?` schema is
left untouched and every unknown schema fails closed. No host repository patch
is required. Remove the environment variable and eager restriction only after
validating an aligned host source/binary release.

Partial-RoPE calibration captures normalized query heads before rotation. The
64 rotated dimensions use TriAttention's trigonometric future-position score;
the 192 pass-through dimensions add their calibrated direct Q·K contribution.
The full-RoPE fused scoring kernel is not used for this model. With TP=2, each
rank scores its local KV head and an all-reduce maximum produces the same token
selection on both ranks.

V3 selection protects the first 128 and most recent 512 tokens, splits the
middle into eight segments, and assigns an exact proportional eviction quota
to every segment. Qwen3.5 scores all ten full-attention layers
(`score_layer_stride=1`).

## Calibration and launch

Generate a model-bound artifact across two free devices:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m vllm_ascend_kvcompress.calibration \
  --model /workspace/models/Qwen--Qwen3.5-35B-A3B \
  --output artifacts/qwen3.5-35b-a3b-stats-v3.pt \
  --max-length 4096 --device npu:0 --device-map auto \
  --dtype bfloat16 --attn-implementation eager --local-files-only
```

Configure the plugin with
[`examples/qwen3.5-35b-a3b-triattention.json`](../examples/qwen3.5-35b-a3b-triattention.json),
then launch the B1 service:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_PLUGINS=ascend,ascend_kvcompress
export VLLM_ASCEND_KVCOMPRESS_ENABLED=1
export VLLM_ASCEND_KVCOMPRESS_CONFIG=/absolute/path/qwen3.5-35b-a3b-triattention.json
# Required only for the validated host's legacy non-optional int[] GDN ABI.
export VLLM_ASCEND_KVCOMPRESS_QWEN_GDN_LIST_COMPAT=1
# Needed only when the container has /dev/shm=64 MiB; C4 uses a 40 MiB ring.
export VLLM_MQ_MAX_CHUNK_BYTES_MB=4
# Prepend the plugin source; do not replace the CANN Python path containing acl.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
vllm serve /workspace/models/Qwen--Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b --tensor-parallel-size 2 \
  --language-model-only --dtype bfloat16 --block-size 128 --mamba-cache-mode none \
  --max-model-len 32768 --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching --enable-chunked-prefill --no-async-scheduling \
  --generation-config vllm --enforce-eager
```

Although the command requests `--block-size 128`, the host reports the
32,768-token scheduler alignment and 2,048-token full-attention page separately.
The scheduler bind log records both, while the worker bind log also records the
128-token kernel block.

For B0, restart independently while retaining the same plugin and model-bound
statistics. Set `kv_budget` to 32,768 and `recompute_window` to 128, so the
32,896-token threshold is strictly above the service limit and no request is
compressed. B0 and B1 use the same command apart from this compression-budget
configuration. This keeps the compatibility hooks and runtime stack identical
without modifying either host repository. Never mix the two arms in one
service lifecycle.

The plugin applies an explicitly set `VLLM_MQ_MAX_CHUNK_BYTES_MB` to vLLM's
worker response queue as well as its scheduler broadcast queue. Messages above
the chunk size still use the socket path, so this reduces only the shared-memory
fast-path capacity and does not cap message size. Deployments without the
64 MiB constraint should omit the variable and retain the upstream default.

## Test status and scope

The current worktree passes 114 unit/contract tests with one environment skip.
The 16K TP=2 service smoke completed in both arms: B0 retained all 16,384 input
tokens, while B1 committed 16,384 semantic tokens to 8,192 physical tokens;
both produced all 1,024 forced output tokens, matched the oracle, and reported
no silent truncation. The public-long-context and standard A2/A3 suites remain
the governing model-level tests. Because this target uses TP=2, eager execution,
the explicit legacy-ABI bridge, and hybrid attention, its results are
engineering evidence and cannot be labeled formal V4.6 acceptance. Results are
published in the
[HTML leaderboard](benchmark-leaderboard.html) with links to machine-readable
evidence and explicit limitations.

The public long-context run used a frozen `enable_thinking=false` chat template,
`--generation-config vllm`, and explicit neutral sampling controls. Each arm
completed 403 requests; all 806 requests succeeded without silent truncation:

| Dataset | B0 → B1 quality | Quality delta | Request throughput delta | Physical KV reduction | Commits / TP-rank acks |
| --- | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2 (110) | 47.27 → 47.27 accuracy | 0.00 pp | −5.83% | 63.76% | 110 / 220 |
| passage_retrieval_en (200) | 100.00 → 100.00 | 0.00 pp | −2.39% | not applicable (short-output bypass) | 0 / 0 |
| Qasper (93) | 51.07 → 50.64 F1 | −0.43 pp | −1.28% | 42.11% | 11 / 22 |

All three quality-retention gates passed. See the
[machine-readable summary](evidence/kvcompress-working-tree-20260920-qwen35-public-summary.json)
for bootstrap intervals, latency changes, and raw hashes. Other NPUs on the
same host were executing the standard matrix, so throughput and latency deltas
are exploratory; quality, request integrity, and compression transactions are
the primary adaptation evidence.

## Transferred A2/A3 engineering matrix

The complete commissioning matrix was also transferred to this model. The
profile identifiers retain `FP16` to preserve workload identity, but Qwen3.5
actually ran in BF16, eager mode, TP=2, with the opt-in legacy GDN ABI bridge.
These runs are supplementary engineering evidence, not formal execution of the
standard graph-mode/TP=1 topology.

Each A2 cell used 16 measured requests per lifecycle, concurrency four, and
three independent cold lifecycles per arm:

| Shape | RPS | B0 → B1 median total tok/s | Change | Minimum physical KV reduction | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 8K + 512 | 0.05 | 249.72 → 250.80 | +0.43% | N/A: below threshold | PASS |
| 8K + 512 | 0.10 | 266.08 → 262.28 | −1.43% | N/A: below threshold | **FAIL** |
| 8K + 512 | 0.20 | 272.09 → 266.65 | −2.00% | N/A: below threshold | **FAIL** |
| 8K + 512 | 0.40 | 278.59 → 275.32 | −1.17% | N/A: below threshold | **FAIL** |
| 16K + 1,024 | 0.05 | 266.78 → 262.17 | −1.73% | 50.00% | **FAIL** |
| 16K + 1,024 | 0.10 | 274.73 → 269.07 | −2.06% | 50.00% | **FAIL** |
| 16K + 1,024 | 0.20 | 278.00 → 274.01 | −1.43% | 50.00% | **FAIL** |
| 16K + 1,024 | 0.40 | 275.39 → 273.54 | −0.67% | 50.00% | PASS |

All 768 measured arm-requests and six warm-ups completed correctly, with zero
failures, short forced outputs, or silent truncations. The 16K B1 cells recorded
192 scheduler commits and exactly 384 per-rank acknowledgements. Two cells
passed; six failed only the frozen 1% total-throughput regression budget.

A3 used 30,720 input tokens plus 2,048 forced output tokens, one request at a
time, a five-minute warm-up, and a 30-minute measurement split into six windows:

| Median of three lifecycles | B0 | B1 | Change |
| --- | ---: | ---: | ---: |
| Total token throughput | 69.34 tok/s | 70.19 tok/s | +1.23% |
| Mean TTFT | 1949.71 ms | 2009.13 ms | +3.05% |
| Mean TPOT | 229.91 ms | 227.05 ms | −1.24% |
| Mean E2E | 472.57 s | 466.77 s | −1.23% |

Every B0 and B1 lifecycle completed four correct requests, below the required
24. Every six-window throughput CV was 100%, above the 5% gate, and empty
windows made latency-drift statistics unavailable. B1 nevertheless preserved
73.33% minimum physical-KV reduction with 24 commits and 48 TP-rank
acknowledgements. A3 is therefore **FAIL**, despite passing quality,
transaction, reduction, and throughput-regression checks.

Three initial sandbox launches could not enumerate Ascend devices and failed
before model loading or requests. Their log hashes are retained but excluded
from the three valid lifecycles. Full hashes and per-cell values are in the
[machine-readable standard-matrix summary](evidence/kvcompress-working-tree-20260920-qwen35-standard-summary.json).

This does not negate the V3 paper's hybrid-model warning: its Qwen3.5-27B and
35B-A3B V3-only runs failed strict middle/end NIAH, while the later long-context
rescue was not validated on those larger models. This run's
`passage_retrieval_en` workload requests only 32 output tokens and therefore
bypasses compression by design; it proves the bypass, not post-compression
needle retention. Deployment still needs an authorized multi-position NIAH or
production retrieval gate under active compression. The present conclusion is
strictly limited to the public tasks reported above.
