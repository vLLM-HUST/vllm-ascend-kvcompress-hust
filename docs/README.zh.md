# 文档导航

[English](README.md) | 简体中文

本导航汇集用户说明、设计、测试方法、优化记录与按日期保存的结果。
文档保持中英成对。包版本以 `pyproject.toml` 为准；发布与测量状态见
[结果](results/README.zh.md)。归档后的历史测量仍保留当时的版本、模型与宿主身份。

## 当前工作

| 主题 | 文档 |
| --- | --- |
| 用户指南 | [安装、配置、校准、服务与排障](usage/README.zh.md) |
| 测试规程 | [按负载与结论选择规程](benchmark/README.zh.md) |
| 实现设计 | [方法 API 与 TriAttention](design/methods.zh.md)、[Ascend 适配](design/ascend-adaptation-vs-triattention-vllm.zh.md)、[Qwen3.5 混合模型](design/qwen3.5-35b-a3b-adaptation.zh.md) |
| 优化记录 | [按日期整理的实现与调参记录](optimization/README.zh.md) |
| 开发 | [环境](development/environment-installation.zh.md)、[校准](development/calibration-artifacts.zh.md)、[归属与许可](development/ownership-and-licensing.zh.md)、[提交规范](development/git-commit.zh.md)、[打包发布](development/packaging-and-release.zh.md) |
| 结果 | [结果范围与证据](results/README.zh.md)、[HTML 榜单](results/benchmark-leaderboard.html) |
| 项目议题 | [五数据集适用性，issue #17](issue/dataset-program-issue-17.zh.md) |
| 报告 | [周报](template/weekly-report-template.zh.md)和[月报](template/monthly-report-template.zh.md)模板 |

## 历史归档

按旧测试规程、旧结果、旧设计和旧开发记录浏览
[分类归档](archive/README.zh.md)。

[0.9 Frontier 测试过程](archive/results/frontier-benchmarking-0.9.zh.md)
保留所有本地正、负配对及 2026-10-08 发布验证。
[V4.6 要求](archive/benchmark/kv-compress-test-requirements.zh.md)、
[A2/A3 规程](archive/benchmark/benchmarking.zh.md)、
[公开长上下文实验](archive/results/public-long-context-benchmarks.zh.md)、
[2026-09-20 验证](archive/results/validation.zh.md)和
[0.2 结果](archive/results/results-0.2.zh.md)属于旧工作负载与宿主栈，
不得与不同 cohort 的结果混合计算。
