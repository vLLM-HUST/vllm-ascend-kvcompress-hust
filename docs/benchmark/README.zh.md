# Benchmark 规程

[English](README.md) | 简体中文 | [文档导航](../README.zh.md)

当前采用浦江实验室指定的数据集测试，见[项目指定数据集页面](https://vllm-hust.sage.org.ai/dataset-introductions.html)，沿用固定客户端及各自评分约定。

| 数据集 | 评估方式 | 本次准备范围 |
| --- | --- | --- |
| MMLU-Pro | token-ID completions，本地答案提取 | [客户端、下载、冻结计划与运行入口](mmlu-pro.zh.md) |
| HLE-Verified | Gold / Revision，Solver 与独立结构化 Judge | [统一评测入口](pujiang.zh.md) |
| SWE-bench-Pro | Agent 沙箱与任务测试 | 暂未实现，入口留空 |
| FrontierScience | Olympiad / Research，独立 Judge 与 rubric | [统一评测入口](pujiang.zh.md) |
| TerminalBench 2.1 | ARM 兼容任务沙箱与任务评分器 | 暂未实现，入口留空 |


[MMLU-Pro 操作说明](mmlu-pro.zh.md)。

[统一安装、固定 smoke 与 B0/B1 对比指南](pujiang.zh.md)。
