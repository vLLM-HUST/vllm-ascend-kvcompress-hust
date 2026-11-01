# 版本记录

[English](version-history.md) | 简体中文 | [项目简介](../../README.zh.md)

本页记录插件各版本的实现变化。

## v0.11.0

- 将压缩后端、共享运行时状态、Ascend 宿主集成和旧扩展适配分层。
- 使用固定 HUST 源码基座，通过[一键安装脚本](environment-installation.zh.md)
  配置 `kvcompress-host` 长期环境，后续插件版本共用该环境。
- 评估采用浦江实验室指定的 MMLU-Pro、HLE-Verified、SWE-bench-Pro、
  FrontierScience 和 TerminalBench 2.1；MMLU-Pro 提供固定客户端、数据下载、
  冻结计划及评估脚本，见[测试规程](../benchmark/README.zh.md)。

待发布

## v0.10.0

- 合并 TP 评分同步。
- Qwen3.5 示例采用 4,608-token KV 预算、2,048-token 请求输出门槛和 stride 2 评分。
- MTP2 仍为默认拒绝启动的实验功能。

历史 SWE C4／900 秒配对的[结果范围](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.zh.md)与
[聚合证据](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261017-v010-swe-pair.json)保留精确的宿主、BF16/TP=2 拓扑、负载与正负结果。

## v0.9.0

- 加入供独立压缩方法使用的 method API v1，包括逐层物理状态与 Query 观测契约。

SWE 与 AgentX 属于不同 cohort；历史测量见[结果索引](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.zh.md)及
[完整历史过程](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/frontier-benchmarking-0.9.zh.md)。

## v0.8.0

- 适配同步后的 vLLM-HUST、vLLM-Ascend-HUST 宿主，不修改宿主源码仓库。

## v0.7.0

- 加入论文 V3 位置策略与实验性 Qwen3.5 支持。
