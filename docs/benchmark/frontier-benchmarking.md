# Frontier benchmark protocol

English | [简体中文](frontier-benchmarking.zh.md)

This is the active comparison contract for the Qwen3.5-35B-A3B BF16, TP=2
Ascend 910B2 cohort. Measured releases and source candidates are indexed in
[results](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.md); earlier campaigns are
[archived](../archive/README.md). Qualify each change with a fresh pair on its
exact source and environment.

## Freeze before measuring

Record model and tokenizer revisions, calibration SHA-256, host and plugin
commits or wheel hash, CANN/PyTorch/TorchNPU versions, device IDs, prepared
workload SHA-256, client commit, server flags, and all configuration files.
Use the same model, host, hardware, request file, and concurrency in both arms.
Cold-start each arm separately. B0 loads the same plugin but sets its
compression threshold strictly above the 262,144-token service limit; B1 uses
the candidate configuration. Do not change the host source between arms.

| Workload | Original client | Qualification | Measurement | Headline metrics |
| --- | --- | --- | --- | --- |
| SWE prefix reuse | [swe-prefix-reuse](https://github.com/vLLM-HUST/swe-prefix-reuse) | Separate C4/60-second protocol run per arm | C4/900-second window per arm | `metrics.output_tps / 2` output tokens/s/chip; P90 per-request decode tokens/s |
| AgentX | [agentx-bench](https://github.com/vLLM-HUST/agentx-bench) | Original primer and warmup | Original 900-second smoke per arm | Output tokens/s/chip and per-user inverse-ITL P90 |

For SWE, P90 decode speed is the 90th percentile of each completed request's
`(output_tokens - 1) / (last_token_time - first_token_time)`. It is the
per-user axis for this workload. It must not be replaced by the reciprocal of
aggregate P90 TPOT. Keep original `summary.json`, `config.json`, and raw
requests. Run the repository's read-only comparator:

```bash
python scripts/kvcompress_frontier_swe_pair.py \
  --baseline /path/to/b0-c4-900s \
  --candidate /path/to/b1-c4-900s
```

SWE must retain real MTP2 generation and observed acceptance, APC, async
scheduling, `mamba_cache_mode=align`, and FULL_AND_PIECEWISE graphs in both
arms if these are in the frozen cohort. AgentX must disable MTP in both arms
until its forced-acceptance evidence requirement is met. Do not pool the two
cohorts or alter their original traces, prompts, output budgets, or warmup.

## Acceptance and interpretation

Require valid original client reports, zero hidden request failures, matching
prepared-file hashes, a successful separate C4/60-second SWE check, server
prefix hits where expected, and one acknowledgement from each TP rank for
every B1 scheduler compression commit. Preserve maximum *observed* prompt
length, preemptions, NPU/HBM samples, logs, and exit status. Configuring 262K
capacity alone does not demonstrate a 256K request.

Report absolute B0 and B1 values and deltas for **both** output tokens/s/chip
and per-user decode P90. Also report TTFT P95, completion counts, and any
regressions. A claim that SWE improved on both headline axes requires both to
rise in a matched SWE pair. A faster isolated kernel or a positive AgentX
result alone does not establish that claim. One 900-second pair is an engineering observation,
not a one-hour formal result or proof of statistical significance. The
[official submission guide](https://github.com/vLLM-HUST/vllm-hust-website/blob/codex/frontier-submission-example/data/examples/frontier-submission/README.md)
governs any later website submission.
