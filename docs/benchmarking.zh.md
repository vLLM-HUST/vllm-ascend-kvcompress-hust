# 0.4 版本验收规程

[English](benchmarking.md) | 简体中文

从 HUST V4.6 交付方案提取的项目专项规范见
[KV 压缩测试要求](kv-compress-test-requirements.zh.md)，其优先级高于下述简要流程。

1. 冻结全部宿主/插件提交、Python/NPU 栈、模型/tokenizer revision、校准来源与
   SHA-256、设备状态、配置、命令、数据集哈希和原始输出路径。
2. 完成干净 wheel 安装、Manager 发现/配置/启用/check、禁用惰性、disable/
   forget/卸载、消失及重装的完整生命周期。
3. 运行 `tests/run_npu_kernel_smoke.py`，并针对数值参考测试直接评分、K/V copy、
   聚合、offset、重复压缩、边界长度和支持的数据类型。
4. 使用确定性且有授权的数据，断言无失败、OOM、死锁、串扰、错误答案、短输出或
   静默截断；每次 scheduler 压缩提交必须有对应 worker 确认。
5. A2 固定运行 8,192+512、16,384+1,024，速率 0.05/0.1/0.2/0.4 RPS，
   并发 4，B0/B1 各三轮独立冷生命周期。A3 使用精确 30,720+2,048、并发 1、
   预热五分钟、测量 30 分钟并分为六个窗口。
6. 报告请求/输入/输出/总吞吐、TTFT/TPOT/E2E 分布、失败、HBM 峰值、动态 KV
   block、压缩次数/耗时、质量、逐轮值、中位数、范围、CV 和各类哈希。

仓库工具可生成确定性 commissioning 数据、运行单组服务并比较成对结果：

```bash
python scripts/kvcompress_prepare_dataset.py --output .benchmarks/data/a2.jsonl
python scripts/kvcompress_long_context_run.py --help
python scripts/kvcompress_acceptance_compare.py --help
```

随机数据、截断、复用 prompt 或仅重启热服务都不能替代正式流程。
`kv-pressure-online` 仅是快速压力 smoke。正式 B0 必须使用规定的官方 vLLM 0.18
及匹配官方 Ascend 基线；当前宿主中禁用插件/no-compression 的结果只能标记为工程
兼容性对照。

Prefix cache、speculative decoding、KV transfer、量化 KV、BidKV 和已移除的
宿主优化均是不支持组合，不是调参变量。发布结论必须受已完成矩阵和
[验收记录](validation.zh.md)的限制约束。
