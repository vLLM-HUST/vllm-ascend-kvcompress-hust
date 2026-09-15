# 0.5 版本验收记录

[English](validation.md) | 简体中文

日期：2026-09-15。结论：**工程验收通过，但不是 V4.6 正式验收**。插件包及当前
宿主适配可继续评估；下述缺口使其暂不能宣称生产可用或官方基线收益。

## 0.5 校准能力验收

0.5 新增插件内校准生成。在物理 NPU 6 上，最终生成器加载本地
Qwen2.5-Coder-14B-Instruct，以默认 4,096-token 长度生成全部 48 x 40 个查询 head
统计。schema-2 产物为 1,565,471 字节，通过模型形状、RoPE、有限值、模型标识、
revision 和 checkpoint manifest 指纹校验，SHA-256 为
`f53898b153fe8b3177b7e10e3c0f979eb6b470a9e5fc335a3f0ed3258f42846b`。

一次 `stats_path` 缺失的干净服务启动确认：校准在原始服务权重加载之前完成，临时
模型随后释放，48 层 KV 成功绑定，`/health` 返回 200，短 completion 成功。第二次
服务使用新生成的 4,096 产物完成一个精确 3,072-token 检索用例：答案正确，记录一次
scheduler commit 和一次 worker 回执，并将 24 个 source block 压至 16 个
destination block。Ascend 算子数值 smoke 也通过。

内置语料只是 bootstrap 默认值。单个使用用例不能替代下文已有的重复性能与公开质量
证据；生产环境必须使用具有代表性、许可清晰的语料生成，并重新执行质量、性能和 HBM
矩阵。机器可读记录见
[kvcompress-v0.5.0-auto-calibration-summary.json](evidence/kvcompress-v0.5.0-auto-calibration-summary.json)。
最终自动化套件为 77 passed、1 skipped；Ruff 检查与格式检查通过。

## 冻结环境

| 组件 | 已验证值 |
| --- | --- |
| 插件 | 0.5.0，工作树基于 `7c0d21144d` |
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

0.5.0 wheel 在不把源码目录加入 `PYTHONPATH` 的条件下安装。插件发现、manifest
解析、兼容性检查、配置、校验、启用、status/check/plan/env 渲染和
`run --dry-run` 均通过，渲染环境包含：

```text
VLLM_ASCEND_KVCOMPRESS_ENABLED=1
VLLMHUST_EXT_ENABLED_BUNDLES=org.vllm-hust.ascend-kvcompress
```

0.5 生命周期覆盖禁用、卸载、从 `extension list` 消失、重装、重新发现、check 和
重新启用，并保留此前配置；更早的 0.4 检查还覆盖了 `forget`。
禁用状态下导入保持惰性。当前 Manager 判定宿主未提供原生扩展 API，因此 manifest
不声明虚假的 `api_range`，而以精确包版本范围和运行时契约测试约束适配。

## 自动化与 NPU 检查

最终候选应复现：

- `pytest`：77 passed、1 skipped；
- Ruff 检查和格式检查：通过；
- 包 metadata 与 wheel/sdist 内容检查：通过；
- Ascend 算子数值 smoke：通过；
- 下列 NPU 算子比值继承自运行时 kernel 未变化的 0.4 版本，本次 0.5 未重新测量：

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

## 公开 A3 长上下文质量测试

为补足合成数据，本次在本地 Qwen2.5-Coder-14B-Instruct FP16 上，让三个公开场景
依次运行于一个冷启动 B0 和一个冷启动 B1 服务生命周期。B0 在相同宿主与插件下
使用 32,768-token 无压缩预算；B1 使用推荐的 8,192-token 预算。所有接纳的 prompt
均精确 token 化且不截断，超限输入单独登记为 unsupported。

| 全部可接纳样本 | 质量 B0 → B1 | 请求吞吐 | 平均 TTFT | 平均 TPOT | 物理 block |
| --- | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2，116 条（10K–32K） | 34.48 → 34.48 accuracy | +2.13% | -3.84% | -7.80% | -61.96% |
| LongBench 段落检索，200 条（10K–16K） | 100.00 → 100.00 | -0.04% | -0.28% | +0.27% | 绕过（0 次提交） |
| LongBench Qasper，93 条（5K–22K） | 42.51 → 42.03 F1（-0.48 pp） | -0.11% | -2.89% | +11.31% | 11 条触发样本 -38.73% |

两组共 818 个请求全部完成，静默截断为 0。B1 记录 127 次 scheduler 提交和 127
次 worker 回执；触发压缩样本的物理 block 从 20,666 降至 8,128（-60.67%）。
LongBench-v2 动态 KV 峰值从 B0 的 90.4% 降至 39.8%；因启动时预留池，两组设备
HBM 仍均为 87%。

4,096-token 调参候选因 Qasper F1 下降 4.71 个百分点而被淘汰。8K 结果满足 1 pp
质量容差，但性能与负载相关：LongBench-v2 有收益；优化后的 64-token 请求输出门槛
会为 32-token 检索负载绕过压缩，将原有额外开销降至近似中性。
由于每组只有一次运行，这些性能数值是方向性工程证据，不是重复测量统计。数据固定
版本、许可、命令、评分器、完整指标、负结果和哈希见
[公开长上下文 Benchmark](public-long-context-benchmarks.zh.md)及其
[机器可读证据](evidence/kvcompress-v0.4.0-public-long-context-summary.json)。

## V4.6 缺口与可发布结论

本记录不是 V4.6 正式 PASS，原因如下：

1. B0 是当前宿主下禁用压缩/no-op 的兼容性对照，而不是规定的官方 vLLM 0.18 与
   匹配官方 Ascend 栈；
2. 公开质量任务已补充确定性合成数据，但公开性能组每组仅一次冷测，也不是经授权的
   生产数据集；
3. 公开测试模型与统计文件属于同一 Qwen2.5-Coder-14B 系列，但缺少准确上游
   revision 与生成 provenance；此前合成测试使用了另一 Qwen2.5-14B 变体；
4. 只有 A2 的 16K/0.4-RPS/并发 4 单元完成三轮冷测，完整 A2 速率矩阵和 A3
   30 分钟/六窗口稳定性测试尚未完成；
5. 宿主预分配 KV 池使 HBM 百分比未下降，系统 cost/token 也未单独测量。

因此当前可以声明：插件包/Manager 兼容、声明快照上的 NPU 正确性、推荐 8K 预算
下有边界的公开质量保持，以及上述负载特定工程测量；不能声明 V4.6 正式通过、普遍
质量/吞吐保持或生产就绪。
