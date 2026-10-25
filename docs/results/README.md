# Results and evidence

English | [简体中文](README.zh.md)

The [HTML leaderboard](benchmark-leaderboard.html) reads aggregate, versioned
records from [evidence](evidence). The page separates measured Frontier points
from historical experiments; it is not the official vLLM-HUST leaderboard.
Regenerate its JavaScript bundles with `python scripts/kvcompress_leaderboard.py`
and check them with `python scripts/kvcompress_leaderboard.py --check`.

## Published 0.9.0 local validation

The 2026-10-08 Qwen3.5-35B-A3B BF16/TP=2 SWE C4/900-second pair was valid with
zero failed requests and 60 compression commits, each acknowledged by both TP
ranks. B0 to B1 output throughput per chip changed **76.648 → 74.572 tokens/s
(−2.71%)**; per-request P90 decode speed changed **47.920 → 48.424 tokens/s
(+1.05%)**. Thus only one of the two current SWE target metrics improved.
The longest observed prompt was 52,505 tokens. See the
[paired evidence](evidence/kvcompress-qwen35-20261008-v090-swe-pair.json) and
[full archived campaign](../archive/results/frontier-benchmarking-0.9.md).

The corresponding AgentX pair had a positive decode P90 but effectively flat
output throughput; it is a separate cohort. Its
[paired evidence](evidence/kvcompress-qwen35-20261008-v090-agentx-pair.json)
records the original report hashes and limitations. Neither pair is a formal
website submission or proof of repeatability.

## 0.10.0 local validation

On 2026-10-17, a Qwen3.5-35B-A3B BF16/TP=2 SWE C4/900-second pair on the
pinned host was valid, with zero failed requests and 218 in-window completions
in each arm. B0 to the selected B1 configuration changed output throughput
per chip **74.426 → 75.418 tokens/s (+1.33%)** and per-request P90 decode
speed **47.843 → 48.616 tokens/s (+1.62%)**. TTFT P95 worsened
**1561 → 1696 ms (+8.65%)**. The candidate used a 4,608-token KV budget,
1,024-token recompute window, stride 2, and a 2,048-token requested-output
guard. Its service log contains 20 lifetime compression commits, each
acknowledged by both TP ranks. The longest observed prompt was 52,505 tokens.
See the [aggregate evidence](evidence/kvcompress-qwen35-20261017-v010-swe-pair.json)
and the [optimization record](../optimization/triattention-tp-score-batching-20261017.md).

Two earlier candidate settings improved decode P90 but reduced per-chip
throughput (−1.71% and −0.44%). Their metrics and report hashes are kept in
the same evidence record. The selected candidate passed four deterministic
32K retrieval checks with compression; those checks do not measure SWE task
success. This is one local pair, not a repeatability result or official
leaderboard submission. The HTML leaderboard remains a
separate set of measured historical cohorts; it does not pool this pair with
the different 2026-09-27 model revision.

## 0.11.0 source-tree local validation

On 2026-10-23, the Qwen3.5 BF16/TP=2 SWE C4/900-second campaign screened four
compression candidates across two source cohorts. Every arm passed its own
60-second check and formal window with zero failed requests. The first three
candidates reduced per-chip output throughput against their matching B0.
With the uneven-V3 source and a 16,128-token KV budget, the last pair changed
per-chip output throughput **73.576 → 73.697 token/s (+0.17%)** and
per-request P90 decode speed **47.628 → 47.692 token/s (+0.13%)**. TTFT P95
worsened **1663 → 1901 ms (+14.27%)**. Both arms completed 216 in-window
requests; the longest observed prompt was 52,505 tokens. The candidate made
13 formal-window compression commits, each acknowledged by both TP ranks, and
passed four limited 32K deterministic retrieval checks with compression.

The two target gains are smaller than cross-run variation and come from one
pair. They do not establish repeatability or SWE semantic answer quality.
See the [campaign evidence](evidence/kvcompress-qwen35-20261023-v011-swe-campaign.json)
and [optimization record](../optimization/triattention-private-copy-and-v3-selection-20261023.md)
for both source identities, negative candidates, exact metrics, raw-report
hashes, and the separate NPU operator measurements. This local campaign is
not an official leaderboard submission.
