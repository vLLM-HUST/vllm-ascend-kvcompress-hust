# 0.4 版本验收记录

[English](validation.md) | 简体中文

日期：2026-09-11。结论：**工程验收通过，但不是 V4.6 正式验收**。插件包及当前
宿主适配可继续评估；下述缺口使其暂不能宣称生产可用或官方基线收益。

## 冻结环境

| 组件 | 已验证值 |
| --- | --- |
| 插件 | 0.4.0，工作树基于 `db18568aa8` |
| vLLM-HUST | `6cdc0304a8`，`0.28.1.post1.dev260` |
| vLLM-Ascend-HUST | `5901bedbb7`，`0.25.1rc2.dev232+hust.20260903.4.g5901bedbb` |
| Extension Manager | `cf1ea71e3e`，`0.2.0.dev0` |
| Triton Ascend | `8f0a4de84`，wheel `3.6.0+git8f0a4de8` |
| Python / torch / torch-npu | 3.11.16 / 2.10 / 2.10.post2 |
| 设备 | 单卡 Ascend 910B2 |
| 模型 | 本地 Qwen2.5-14B-Instruct，FP16 |

现有 quickstart 使用通用 Triton `setup.py`，不能构建 Ascend 后端。本次在临时
目录通过 `setup_ascend.py` 构建并重装同一份、未经修改的 Triton-Ascend 源码。
来源与恢复流程见[环境安装](environment-installation.zh.md)。

## 插件包与 Extension Manager 生命周期

0.4.0 wheel 在不把源码目录加入 `PYTHONPATH` 的条件下安装。插件发现、manifest
解析、兼容性检查、配置、校验、启用、status/check/plan/env 渲染和
`run --dry-run` 均通过，渲染环境包含：

```text
VLLM_ASCEND_KVCOMPRESS_ENABLED=1
VLLMHUST_EXT_ENABLED_BUNDLES=org.vllm-hust.ascend-kvcompress
```

发布生命周期还覆盖禁用、forget、卸载、从 `extension list` 消失、重装和重新启用；
禁用状态下导入保持惰性。当前 Manager 判定宿主未提供原生扩展 API，因此 manifest
不声明虚假的 `api_range`，而以精确包版本范围和运行时契约测试约束适配。

## 自动化与 NPU 检查

最终候选应复现：

- `pytest`：58 passed，1 skipped；
- Ruff 检查和格式检查：通过；
- 包 metadata 与 wheel/sdist 内容检查：通过；
- Ascend 算子数值 smoke：通过；
- NPU 算子相对通用参考测试：

| 算子路径 | 优化实现 | 通用参考 | 比值 |
| --- | ---: | ---: | ---: |
| 分页 K/V copy | 0.592 ms | 0.325 ms | 1.82x |
| 分页直接评分 | 1.306 ms | 0.464 ms | 2.81x |
| 融合聚合 | 0.158 ms | 0.118 ms | 1.34x |
| Offset 更新 | 0.048 ms | 0.058 ms | 0.83x |

比值为通用参考耗时除以优化实现耗时。Offset 更新存在负收益，不宣称该项优化有效。

## 长上下文工程对照

已完成的主测试单元为 A2-LONG-FP16-16K：每个冷服务生命周期 4 个确定性请求，
16,384 输入 token、强制输出 1,024 token、忽略 EOS、0.4 RPS、并发 4、block
size 128，关闭 prefix cache 和 speculative decoding。两组各运行三个独立冷生命周期。

| 三轮冷启动中位数 | 同宿主对照 | 压缩 | 变化 |
| --- | ---: | ---: | ---: |
| 请求吞吐 | 0.06489 req/s | 0.08222 req/s | +26.7% |
| 输入吞吐 | 1063.1 tok/s | 1347.1 tok/s | +26.7% |
| 输出吞吐 | 66.44 tok/s | 84.19 tok/s | +26.7% |
| 总吞吐 | 1129.5 tok/s | 1431.3 tok/s | +26.7% |
| 平均 TTFT | 4167.7 ms | 4326.3 ms | +3.8%（变差） |
| 平均 TPOT | 52.43 ms | 39.16 ms | -25.3% |
| 平均端到端时延 | 57.81 s | 44.37 s | -23.2% |
| p99 端到端时延 | 61.45 s | 48.03 s | -21.8% |

24 个请求（每组 12 个）均达到指定输出长度并返回正确检索答案。压缩组记录到 12 次
调度提交及 12 次 worker 确认；每个请求均从 128 个物理 block 压至 32 个，减少
75%。由于 vLLM 启动时预留 KV 池，设备级 HBM 峰值均为 87%；服务日志中的动态
KV block 使用峰值约为压缩组 16.5%、对照组 56%。因此这里只宣称物理 KV 压力
降低，不宣称预留 HBM 池变小。

机器可读摘要见
[kvcompress-v0.4.0-a2-16k-c4-summary.json](evidence/kvcompress-v0.4.0-a2-16k-c4-summary.json)。
确定性数据 SHA-256 为
`3f68a54ee4028aa63418534466e3763d8e7c1da308af58cec2506cc16b067ab0`。

8,192+512 和 16,384+1,024 的单请求 commissioning smoke 也已通过，仅用于确认
边界行为，不能替代三轮测试矩阵。

## V4.6 缺口与可发布结论

本记录不是 V4.6 正式 PASS，原因如下：

1. B0 是当前宿主下禁用压缩/no-op 的兼容性对照，而不是规定的官方 vLLM 0.18 与
   匹配官方 Ascend 栈；
2. 测试数据是确定性合成检索样例，不是授权冻结的正式业务/质量数据集；
3. 现有 Qwen2.5-Coder-14B 聚合统计与被测 Qwen2.5-14B-Instruct 不匹配，且生成
   provenance 不完整；
4. 只有 A2 的 16K/0.4-RPS/并发 4 单元完成三轮冷测，完整 A2 速率矩阵和 A3
   30 分钟/六窗口稳定性测试尚未完成；
5. 宿主预分配 KV 池使 HBM 百分比未下降，系统 cost/token 也未单独测量。

因此当前可以声明：插件包/Manager 兼容、声明快照上的 NPU 正确性、物理 KV block
减少 75%，以及上述有边界的工程对照收益；不能声明 V4.6 正式通过、模型质量保持或
生产就绪。
