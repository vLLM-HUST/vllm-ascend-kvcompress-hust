# 文档导航

[English](README.md) | 简体中文

## 当前 0.8 候选版

- [Frontier 刷榜规程](frontier-benchmarking.zh.md)：官方 SWE prefix reuse 与 AgentX 256K、900 秒 B0/B1 配对测量、证据及当前阻碍。
- [HTML 测试榜单](benchmark-leaderboard.html)：Frontier 实测点与历史工程结果分开；目前没有合格的 0.8 Frontier 点。
- [Qwen3.5 适配](qwen3.5-35b-a3b-adaptation.zh.md)：混合缓存、TP、APC、MTP2、async 和 GDN 的实现状态。
- [环境安装](environment-installation.zh.md)：同步后的宿主栈，以及已解决旧版不匹配问题的隔离式 PyTorch 2.13/TorchNPU 2.13/CANN 9.1 安装。
- [方法架构](methods.zh.md)、[校准产物](calibration-artifacts.zh.md)与[打包发布](packaging-and-release.zh.md)。

## 历史归档（不是 0.8 验收门槛）

- [V4.6 派生要求](kv-compress-test-requirements.zh.md)与[A2/A3 规程](benchmarking.zh.md)。
- [公开长上下文实验](public-long-context-benchmarks.zh.md)及[2026-09-20 验证记录](validation.zh.md)，包括负结果。
- [早期结果](resuts.zh.md)保留原文件名，以免破坏已有链接。

历史文件保留稳定路径并在文首标记归档；直接移动会破坏既有证据链接。上面的当前规程取代旧文件中的发布门槛说法。不要把历史百分比或短工程冒烟与官方 Frontier 900 秒 cohort 混合。
