# TriAttention private KV copy and uneven V3 selection

English | [简体中文](triattention-private-copy-and-v3-selection-20261023.zh.md) |
[Optimization index](README.md)

Date: 2026-10-23. Scope: the 0.11.0 source implementation on the pinned
Qwen3.5-35B-A3B BF16, TP=2, two Ascend 910B2 host. This is a dated
engineering observation, not a general claim for other models or hosts.

## Changes

TriAttention previously mapped logical token positions to physical cache slots
with separate division, remainder, gather, and multiply operations. A Triton
kernel now performs that mapping in one NPU launch. For private destination
blocks, a second kernel copies K and V directly between physical slots in one
launch. It handles contiguous and padded page strides for the one-local-KV-head
Qwen3.5 layout. Other layouts retain the persistent workspace path. The
scheduler/provider still requires source and destination block sets to be
disjoint before a compression transaction commits.

V3 token selection now batches TopK across uneven middle-context segments.
It pads score rows without admitting padding as a selected token, preserves
each segment's exact proportional quota, and returns sorted token positions.
The prior per-segment path remains for zero-width or zero-quota cases. The
batched plan caches shape-dependent indices across transactions. This matters
for actual SWE compression lengths such as 9,605, 17,090, and 46,473 tokens;
they did not enter the earlier equal-width fast path. The SWE pair comparator
also recognizes arm-specific `VLLM_HUST_EXT_CONFIG` paths while continuing to
reject changes to other launch arguments.

The full source suite passed 216 tests, the Ascend kernel smoke passed, and
the uneven selection returned exactly the same indices as the per-segment
reference on the three lengths above. One-NPU measurements showed V3 selection
at about 3.25 ms before and 0.49 ms after batching (6.6–6.7× for those
lengths). For 4,608 selected tokens, the production direct copy took 0.117 ms
on contiguous K/V and 0.112 ms on padded K/V, versus 0.299 ms through the
workspace for one layer. These are **isolated operator measurements**; they
do not establish a service throughput change.

## Matched SWE service measurements

The [SWE protocol](../benchmark/frontier-benchmarking.md) used the same model
revision, calibrated statistics, prepared workload, host commits, C4 client,
APC, real MTP2, async scheduling, `mamba_cache_mode=align`, and
`FULL_AND_PIECEWISE` graph mode. Each arm had a separate cold server, a valid
60-second check, and a valid 900-second window with zero failed requests.
The configured context was 262,144 tokens; the longest observed prompt was
52,505 tokens. B0 loaded the plugin with compression disabled. All B1 arms
used a 1,024-token recompute window, stride 2, and a 2,048-token requested
output guard. Two source cohorts are shown separately because uneven V3
selection was added between them.

| Source cohort and arm | KV budget | Recent protection | Output/chip token/s | Decode P90 token/s | TTFT P95 ms | Completed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First, B0 | 262656 | — | 73.916 | 47.546 | 1641 | 217 |
| First, B1 | 4608 | 512 | 72.137 (−2.41%) | 47.669 (+0.26%) | 1660 (+1.18%) | 218 |
| First, B1 | 4608 | 1024 | 72.888 (−1.39%) | 47.326 (−0.46%) | 1753 (+6.84%) | 215 |
| Uneven V3, B0 | 262656 | — | 73.576 | 47.628 | 1663 | 216 |
| Uneven V3, B1 | 8064 | 1024 | 73.006 (−0.77%) | 47.874 (+0.52%) | 1695 (+1.90%) | 216 |
| Uneven V3, B1 | 16128 | 1024 | **73.697 (+0.17%)** | **47.692 (+0.13%)** | 1901 (+14.27%) | 216 |

Percentages compare with the B0 in the same source cohort. The 16,128-token
candidate made 13 compression commits during its formal window; each received
one acknowledgement from each TP rank. Four additional deterministic 32K
retrieval requests completed with correct codes and four compression commits.
That limited check does not measure SWE answer quality.

The wide-budget arm was the only measured 0.11 candidate with both target
metrics above its B0. Both gains are below 0.2% in a single 900-second pair,
while TTFT P95 regressed 14.27%. Cross-run variation is larger than these
gains, so this pair is **not evidence of a repeatable service speedup**.
Changing the budget also changes when compression triggers and how much KV is
retained; the service delta cannot be attributed solely to the kernels.
The local runs are not an official leaderboard submission.

[Machine-readable campaign evidence](../results/evidence/kvcompress-qwen35-20261023-v011-swe-campaign.json)
contains both source hashes, run IDs, raw-report hashes, exact metrics, and
the negative candidate results.

## Comparison with 0.10.0 and end-to-end analysis

In the [0.10.0 paired evidence](../results/evidence/kvcompress-qwen35-20261017-v010-swe-pair.json),
the selected 4,608-token candidate improved output per chip and decode P90
over B0 by 1.33% and 1.62%, while TTFT P95 rose 8.65%. The selected
16,128-token candidate here measured +0.17%, +0.13%, and +14.27%,
respectively. Comparing the selected B1 arms directly, this run had 2.28%
lower output per chip, 1.90% lower decode P90, and 12.08% higher TTFT P95.
The runs used physical NPUs 4/5 and 0/2, respectively; this run's B0 output
per chip was also 1.14% lower. The absolute cross-version difference therefore
cannot be assigned solely to the source changes.

The optimized operators covered little of the service workload: the selected
B1 made 13 compression commits in its 900-second formal window, versus 20 for
the selected 0.10.0 B1. The isolated V3 selection measurement saved about
2.76 ms per call; even multiplying that by all 13 commits gives only about
36 ms. The 0.117 ms direct-copy measurement covers one layer and 4,608 tokens;
it does not measure a complete compression transaction at the selected
16,128-token budget or predict throughput over 900 seconds. That wider budget
also changes the trigger point and retains more KV for subsequent attention
than the 4,608-token configuration. This continuing work is absent from the
selection and copy microbenchmarks.

Pairing the uneven-V3 cohort's raw B0 and selected-B1 requests by play,
trajectory ID, and turn matches all 220 requests. Thirteen had both a prompt
of at least 17,152 tokens and a requested output limit of at least 2,048
tokens, making them compression-eligible. TTFT was slower on 11 of those 13
in B1, with a mean paired difference of +174 ms and median of +34 ms. The
other 207 requests had a mean paired difference of −5 ms. The eligible-request
slowdown aligns with the tail-latency regression, but this small group contains
a large outlier; the mean is not a repeatability estimate. The implementation
caps prefix-cache hits for these requests to leave a 1,024-token query suffix
for recomputation, then performs selection, K/V movement, and a scheduler
transaction. Those steps can offset operator gains, but the reports do not
break out their costs, so their individual contributions are unknown.

This comparison supports faster isolated operators without an established
service speedup. Identifying the dominant cause requires the same physical
NPUs, fixed KV budget and request sequence, per-stage measurements of prefix
recomputation, the full compression transaction, scheduler wait, and decode,
and repeated paired runs.
