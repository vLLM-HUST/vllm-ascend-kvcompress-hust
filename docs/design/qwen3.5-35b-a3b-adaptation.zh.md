# Qwen3.5-35B-A3B 适配说明

[English](qwen3.5-35b-a3b-adaptation.md) | 简体中文

更新日期：2026-10-17。Qwen3.5-35B-A3B BF16/TP=2 是
[Frontier 规程](../benchmark/frontier-benchmarking.zh.md)中的一个模型 cohort。
旧 Qwen2.5 V4.6 矩阵已[归档](../archive/benchmark/kv-compress-test-requirements.zh.md)。

## 冻结模型身份

- 上游：[`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)，Apache-2.0；
- Hugging Face revision：`712cf74392b05026a6db2bf213d343747d1f6d45`；
- 在服务宿主选定的模型目录获取上述固定 revision 的本地快照；
- 权重索引总大小：71,903,655,008 bytes，共 14 个 safetensors 分片；
- 原生上下文：262,144 token。

[历史模型清单](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/qwen3.5-35b-a3b-model-manifest.json)
记录的是旧版 59d61f3c revision，不能充当当前 712cf743 revision 的哈希凭据。

Hugging Face 端点不可用时，可使用 Qwen 官方镜像，但须核对 revision 与文件哈希。
测试前必须同时存在完整模型配置和权重索引，任何未完成的 `.safetensors` 分片
都不能进入测试。

## 影响压缩的架构事实

| 字段 | 数值 | 对适配的影响 |
| --- | ---: | --- |
| 文本模型类型 | `qwen3_5_moe_text` | 加入显式 allowlist |
| 总参数 / 激活参数 | 35B / 3B | BF16 服务需多张 910B2 |
| 解码层 | 40 | 必须校验混合层布局 |
| 层模式 | 10 ×（3 Gated-DeltaNet + 1 全注意力） | 只有第 3、7、…、39 层有可压缩 KV |
| Query / KV 头 | 16 / 2 | TP=2 时每个 rank 有 1 个本地 KV 头 |
| Head / RoPE 维度 | 256 / 64 | 需为 192 个非旋转维度增加内容评分 |
| 专家 | 256，top-8 + 共享专家 | 校准必须加载 MoE 本体，不能用稠密代理 |
| RoPE theta | 10,000,000 | 校准产物保存精确逐层频率 |

模型默认进入思考模式。质量测试必须统一使用 tokenizer 的非思考模板
（`enable_thinking=false`），或冻结明确的推理预算；B0/B1 混用两种模式会使比较无效。

## 运行时适配

插件只为 Qwen3.5 接受“一个全注意力 KV 组 + 可选 Mamba/GDN 组”，支持
`mamba_cache_mode=none` 或 `align`。scheduler 与 worker 提交只更改全注意力 block table。
slot 位置和 attention length 使用压缩后的物理视图，语义 RoPE 位置与 GDN 状态推进
保持不变。`align` 路径与前缀缓存写时复制提交已通过 TP=2/eager 有界 NPU 冒烟测试，
但尚未完成正式负载验证。

Ascend 的稠密 K/V 内核块为 128 token。此混合模型的宿主可将全注意力
manager 页提升至 1,152 或 2,048 token；0.9.0 发布宿主报告为 1,152。
插件根据各 cache manager 的最小公倍数校验 scheduler 对齐，并把每个注意力页
展开为 9 或 16 个内核块。`kv_budget` 必须能被当前注意力页大小整除，
但不必能被跨组对齐粒度整除。

旧 `int[]` GDN ABI 兼容桥需要显式开启，并受 schema 约束。
旧宿主 ABI 故障及 2026-09-27 性能配对见
[0.9 历史过程](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/frontier-benchmarking-0.9.zh.md)。

部分 RoPE 校准会在旋转前采集归一化 query head。64 个旋转维度使用 TriAttention
面向未来位置的三角评分；192 个直通维度加入校准后的直接 Q·K 内容项。插件自有的
分页键融合评分算子现可同时处理两部分；历史 32,768-token 单层 NPU 微基准提升 10.08 倍；
该单算子收益不能推导 SWE 端到端收益。
TP=2 时，各 rank 评分自己的本地 KV 头。实现将采样层的评分合并，
只执行一次最大值 all-reduce，使两个 rank 的 token 选择保持一致并减少同步开销。
具体改动与服务测量见按日期编写的
[优化记录](../optimization/triattention-tp-score-batching-20261017.zh.md)。

V3 会保护开头 128 个和最近 512 个 token，把中间上下文切成 8 段，并为每段分配
精确的比例驱逐配额。仓库中的 Qwen3.5 示例以 stride 2 采样十个全注意力层中的
五层评分，同时仍对十个目标层实施物理压缩。2048-token 请求输出门槛与
4608-token 物理预算是特定负载的选择。

## 校准与验证

使用[校准指南](../development/calibration-artifacts.zh.md)，并从
[测试规程索引](../benchmark/README.zh.md)选择适用规程。旧版 eager 启动命令、
2026-09-20 质量矩阵和 2026-09-27 诊断保存在
[历史适配记录](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/qwen3.5-20260920-adaptation-results.zh.md)。
