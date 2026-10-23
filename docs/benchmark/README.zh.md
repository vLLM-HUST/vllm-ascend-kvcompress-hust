# 测试规程索引

[English](README.md) | 简体中文 | [文档导航](../README.zh.md)

测量前根据**具体结论和负载**选择适用规程。服务吞吐、模型质量和算子延迟
可能需要不同测试；同一项改动也可能适用多个规程。结果须记录规程版本、
模型与宿主身份、负载、配置、测量窗口、主要指标和结论边界。按所选规程的
规则比较基线与候选，不把不同 cohort 的结果合并成一组配对。

| 规程 | 适用范围 |
| --- | --- |
| [Frontier SWE 与 AgentX](frontier-benchmarking.zh.md) | Qwen3.5-35B-A3B BF16、TP=2、Ascend 910B2 服务对照 |

退役流程放在[测试规程归档](../archive/benchmark/)。
[结果](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.zh.md)与[优化记录](../optimization/README.zh.md)
标明各次测量使用的精确规程。出现新的负载或验收要求时，在此增加对应规程。
