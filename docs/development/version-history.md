# Version history

English | [简体中文](version-history.zh.md) | [Project overview](../../README.md)

This page records implementation changes by plugin version.

## v0.11.0

- Separate compression backends, shared runtime state, Ascend host integration
  and legacy extension adapters.
- Use the pinned HUST source base with the [one-command environment installer](environment-installation.md)
  for the persistent `kvcompress-host` overlay shared across plugin releases.
- Adopt Pujiang Laboratory's designated MMLU-Pro, HLE-Verified, SWE-bench-Pro,
  FrontierScience and TerminalBench 2.1 evaluations. MMLU-Pro has a pinned
  client, dataset download, frozen plan and evaluation scripts; see the
  [benchmark protocols](../benchmark/README.md).

To be released

## v0.10.0

- Merge TP score synchronization.
- Use a 4,608-token KV budget, 2,048-token requested-output guard and stride 2
  scoring in the Qwen3.5 example.
- Keep MTP2 experimental and rejected by default.

The historical SWE C4/900-second pair's [result scope](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.md) and
[aggregate evidence](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261017-v010-swe-pair.json) preserve the exact host, BF16/TP=2 topology,
workload, improvements and regressions.

## v0.9.0

- Introduce method API v1 for independent compression methods, including
  per-layer physical state and the Query observation contract.

SWE and AgentX are separate cohorts. See the [result index](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.md) and
[full historical campaign](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/frontier-benchmarking-0.9.md).

## v0.8.0

- Adapt to the synchronized vLLM-HUST and vLLM-Ascend-HUST hosts without
  modifying their source repositories.

## v0.7.0

- Add the paper's V3 position policy and experimental Qwen3.5 support.
