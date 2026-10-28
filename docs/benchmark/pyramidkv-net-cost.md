# PyramidKV metadata and service net-cost protocol

English | [简体中文](pyramidkv-net-cost.zh.md)

This separate experimental protocol addresses
[PyramidKV issue #1](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/1).
It does not replace the Frontier SWE/AgentX contract or qualify a public wheel.
The previous selected-page operator result does not establish service acceleration.

Freeze the model, tokenizer, host and source hashes before each series. Use the
qualified Qwen3.5-35B-A3B BF16/TP2 experimental host, actual 910B2 device IDs,
FULL_AND_PIECEWISE, MTP2, APC, async scheduling, 32768 context, four sequences,
4096 batched tokens and 8 GiB KV per worker. Retain the 1368/beta-2 method and
its admission threshold. Do not substitute the official v0.18 baseline.

## Diagnostic stage

Instrument host-side metadata construction and scheduled compression separately,
without inserting device synchronization. Preserve every call and its request IDs.
These durations include host dispatch and incidental synchronization, not isolated
NPU kernel time. Instrumented service latency is diagnostic, not the headline.
Use two 8192-token warmups, then two unique prompts at each of 1024/8192/24576
tokens, 128 forced output tokens, concurrency one. A candidate instrumented run
may use the same inputs, but does not replace the uninstrumented series below.

## Uninstrumented service series

Use four fresh process lifecycles in OFF/ON/ON/OFF order. Both arms use the same
candidate source; only compression activation differs. OFF is the native service
baseline, not an artificially expensive no-compression path. Keep old ON evidence
as a different comparison. No profiling decorators run in this series.

Reuse the frozen synthetic input cohort from the prior selected-page series:
1024/8192/24576 input tokens, 128 forced output tokens, concurrency 1 and 4,
three cohorts per cell. Concurrency-one cohorts contain three requests;
concurrency-four cohorts contain eight. Use a unique API `cache_salt` for each synthetic case; reuse exact saved token IDs
and salts across fresh processes, and reject nonzero cached prompt tokens. Preserve 33 designated warmups and five repeatability probes, excluded
before measurement: 137 synthetic requests per process, 99 measured.

Report pooled output tokens divided by cohort wall time, each process/cohort,
TTFT, end-to-end latency, TPOT, failures, preemptions, observed HBM and KV usage.
Verify raw SSE against usage/results, source/input hashes and scheduler plus both
TP acknowledgements for ON transactions; OFF must have no compression commits.
Do not treat within-process cohorts as independent process replications, infer
maximum capacity from C4 success, or infer HBM savings from released KV ownership.
Do not run tests, lint, tokenization or data audits during measured requests.

## Correctness and limits

Retain the first four original development examples per LongBench task as a
12-example smoke, with unchanged prompts, generation limits and original QA-F1.
The stock ON diagnostic and candidate ON outputs can check a narrow metadata
regression boundary. This does not replace the existing 300-example quality
evaluation or qualify new task-quality claims. Do not tune selection or discard
failures based on these outputs.

Changing metadata transport must leave correction values, layer lengths, semantic
positions, private write slots, MTP rejected-draft handling and graph padding
unchanged. CPU regressions and live output/transaction checks are necessary;
the service series determines whether the resulting compression has a net benefit.
Negative and mixed outcomes remain publishable outcomes. Canonical originals,
analysis and exact experiment contracts belong in `vllm-hust-benchmark`.

## Opt-in batched metadata candidate

`VLLM_ASCEND_KVCOMPRESS_BATCHED_METADATA=1` groups the per-layer length and slot
operations into two-dimensional tensors. It remains disabled by default. The
implementation qualifies only single-CP, contiguous, expanded logical block
tables with consistent physical/logical block geometry and shared base sequence
metadata. Other shapes use the existing native per-layer path. Persistent slot
rows preserve capture addresses; unused graph-bucket slots are reset to `-1`.

Before service trials, compare every candidate slot against the pinned host's
actual native NPU block-table kernel, including mixed active/uncompressed rows,
row reordering, unequal query counts, multiple expanded physical pages, padding
and speculative rollback. Synthetic CPU/NPU comparisons establish integer
metadata equivalence only. A live FULL_AND_PIECEWISE/MTP2 smoke is still required.
Freeze the flag and a separate source snapshot for both arms; preserve earlier
candidate series even if negative. Do not combine different implementations'
OFF/ON measurements into a paired comparison.

The qualified builder keeps sequence lengths on CPU and slots on NPU. Batched
length arithmetic follows the actual metadata device and creates independent
storage per call, preserving older in-flight length views. Only slot addresses
are persistent across calls.
