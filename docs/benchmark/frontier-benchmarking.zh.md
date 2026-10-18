# Frontier 测试规程

[English](frontier-benchmarking.md) | 简体中文

这是 Qwen3.5-35B-A3B BF16、TP=2、Ascend 910B2 cohort 的现行对照规程。
发行版及源码候选的测量见[结果索引](../results/README.zh.md)，旧过程见
[归档](../archive/README.zh.md)。每项改动必须在精确的源码与环境上
重新配对测量。

## 测量前冻结

记录模型与 tokenizer revision、校准 SHA-256、宿主与插件提交或 wheel 哈希、
CANN/PyTorch/TorchNPU 版本、设备编号、准备文件 SHA-256、客户端提交、服务参数
及所有配置。B0/B1 使用相同模型、宿主、硬件、请求文件和并发数，分别冷启动。
B0 加载同一插件，但压缩门槛必须严格高于 262,144-token 服务上限；B1 使用候选
配置。两侧之间不得修改宿主源码。

| 负载 | 原版客户端 | 资格检查 | 测量 | 核心指标 |
| --- | --- | --- | --- | --- |
| SWE prefix reuse | [swe-prefix-reuse](https://github.com/vLLM-HUST/swe-prefix-reuse) | 两侧分别运行 C4／60 秒协议检查 | 两侧各运行 C4／900 秒窗口 | `metrics.output_tps / 2` 每卡输出 token/s；逐请求 P90 解码 token/s |
| AgentX | [agentx-bench](https://github.com/vLLM-HUST/agentx-bench) | 原版 primer 与预热 | 两侧各运行原版 900 秒 smoke | 每卡输出 token/s 与逐用户反向 ITL P90 |

SWE 的 P90 解码速度是每个完成请求的
`(output_tokens - 1) / (last_token_time - first_token_time)` 的第 90 百分位，
即本负载的逐用户指标，不能用总体 P90 TPOT 的倒数代替。保留原始
`summary.json`、`config.json` 和请求记录，再运行仓库的只读比较器：

```bash
python scripts/kvcompress_frontier_swe_pair.py \
  --baseline /path/to/b0-c4-900s \
  --candidate /path/to/b1-c4-900s
```

若冻结 cohort 含真实 MTP2 生成与接受率、APC、异步调度、
`mamba_cache_mode=align` 及 FULL_AND_PIECEWISE 图，SWE 两侧均须保留。
AgentX 在具备其强制接受率证据前，两侧均关闭 MTP。不得混合两类 cohort，也不得
改写原始 trace、prompt、输出预算或预热流程。

## 验收与判读

要求原版客户端报告有效、无隐藏失败、准备文件哈希一致、SWE 两侧分别通过
C4／60 秒检查；按预期观察服务端 prefix hit，每次 B1 调度压缩提交均有两个 TP
rank 各一次回执。保留*实际*最长 prompt、抢占、NPU/HBM 采样、日志与退出状态。
只设置 262K 容量不证明实际跑到 256K。

同时报告**每卡输出 token/s**和**逐用户解码 P90** 的 B0/B1 绝对值及差值，
以及 TTFT P95、完成请求数和所有退化。若宣称 SWE 的两项核心指标均提升，
须在同源 SWE 配对中同时观察到提升；单算子加速或 AgentX 正收益不足以证明
这一结论。单次 900 秒配对只是
工程观测，不是一小时正式结果或统计显著性证明。未来官网提交另按
[官方提交指南](https://github.com/vLLM-HUST/vllm-hust-website/blob/codex/frontier-submission-example/data/examples/frontier-submission/README.md)
执行。
