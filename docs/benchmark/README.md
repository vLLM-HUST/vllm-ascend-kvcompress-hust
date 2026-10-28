# Benchmark protocols

English | [简体中文](README.zh.md) | [Documentation map](../README.md)

Choose a protocol for the **specific claim and workload** before measuring.
A service throughput claim, a model quality claim, and a kernel latency claim
may need different tests. Multiple protocols may apply to one change.
Record the protocol revision, model and host identities, workload, configuration,
measurement window, primary metrics, and limitations with the result. Compare
baseline and candidate under the chosen protocol's rules; do not combine
results from different cohorts as if they were one pair.

| Protocol                                                 | Scope                                                                      |
| -------------------------------------------------------- | -------------------------------------------------------------------------- |
| [PyramidKV metadata and net cost](pyramidkv-net-cost.md) | Experimental 32K BF16/TP2 compression OFF/ON and host metadata diagnostics |
| [Frontier SWE and AgentX](frontier-benchmarking.md)      | Qwen3.5-35B-A3B BF16, TP=2, Ascend 910B2 service comparisons               |

Superseded procedures belong in the [benchmark archive](../archive/benchmark/).
Dated measurements and their exact protocol are linked from
[results](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.md) and [optimization records](../optimization/README.md).
Add a new protocol here when a new workload or acceptance requirement needs one.
