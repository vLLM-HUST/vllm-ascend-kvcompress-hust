# 结果与证据

[English](README.md) | 简体中文

[HTML 榜单](benchmark-leaderboard.html)读取[evidence](evidence)中的聚合、
带版本记录。页面区分 Frontier 实测点与历史实验，不是 vLLM-HUST 官网榜单。
运行 `python scripts/kvcompress_leaderboard.py` 生成 JavaScript 数据包，
再用 `python scripts/kvcompress_leaderboard.py --check` 核对。

## 已发布 0.9.0 的本地验证

2026-10-08 的 Qwen3.5-35B-A3B BF16/TP=2 SWE C4／900 秒配对有效，
0 个失败请求；60 次压缩提交均由两个 TP rank 确认。B0 到 B1 的每卡输出吞吐
**76.648 → 74.572 token/s（−2.71%）**，逐请求 P90 解码速度
**47.920 → 48.424 token/s（+1.05%）**。当前 SWE 目标的两项指标仅一项
提升。实际最长 prompt 为 52,505 token。详见[配对证据](evidence/kvcompress-qwen35-20261008-v090-swe-pair.json)
及[完整历史过程](../archive/results/frontier-benchmarking-0.9.zh.md)。

同次 AgentX 配对的解码 P90 提升，但输出吞吐基本持平，属于另一 cohort。
其[配对证据](evidence/kvcompress-qwen35-20261008-v090-agentx-pair.json)
保留原报告哈希及限制。两组都不是官网正式提交，也不能证明可重复性。

## 0.10.0 本地验证

2026-10-17 在固定宿主上完成 Qwen3.5-35B-A3B BF16/TP=2 SWE
C4／900 秒匹配配对：两侧均有效、0 个失败请求、窗口内各完成 218 个请求。
B0 到所选 B1 的每卡输出吞吐 **74.426 → 75.418 token/s（+1.33%）**，
逐请求 P90 解码速度 **47.843 → 48.616 token/s（+1.62%）**；
TTFT P95 则从 **1561 增至 1696 ms（+8.65%）**。候选采用
4608-token KV 预算、1024-token 重算窗口、stride 2 和 2048-token 请求
输出门槛。服务整个生命周期有 20 次压缩提交，均由两个 TP rank 确认。
实际最长 prompt 为 52,505 token。详见[聚合证据](evidence/kvcompress-qwen35-20261017-v010-swe-pair.json)
和[优化记录](../optimization/triattention-tp-score-batching-20261017.zh.md)。

另两种候选设置的解码 P90 提升，但每卡吞吐分别下降 1.71% 和 0.44%；
同一证据记录保留了它们的指标与报告哈希。所选候选通过四项带真实压缩的
确定性 32K 检索检查，但这不能代表 SWE 任务成功率。这只是一组本地配对，
不是可重复性证明或官网榜单提交。HTML 榜单保留不同模型
revision 的历史实测 cohort，不与本组结果混合。

## 0.11.0 源码本地验证

2026-10-23 的 Qwen3.5 BF16/TP=2 SWE C4／900 秒测试，在两份源码 cohort
中筛选四组压缩候选。各组都通过独立的 60 秒检查和正式窗口，失败请求均为 0。
前三组候选相对各自 B0 的每卡输出吞吐下降。加入非整齐 V3 选择后的源码采用
16,128-token KV 预算时，每卡输出吞吐 **73.576 → 73.697 token/s
（+0.17%）**，逐请求解码 P90 **47.628 → 47.692 token/s（+0.13%）**；
TTFT P95 **1663 → 1901 ms（变差 14.27%）**。两侧窗口内各完成 216 个
请求，实际最长 prompt 为 52,505 token。该候选正式窗口提交 13 次压缩，
均收到两个 TP rank 回执；另通过四项带压缩的有限 32K 确定性检索。

两项微小正差值低于跨次运行波动，仅来自一组配对，不能证明可重复性或 SWE
任务语义答案质量。[多轮证据](evidence/kvcompress-qwen35-20261023-v011-swe-campaign.json)
和[优化记录](../optimization/triattention-private-copy-and-v3-selection-20261023.zh.md)
保留两份源码身份、负候选、精确指标、原始报告哈希与独立 NPU 算子测量。
本地测试不是官网榜单提交。
