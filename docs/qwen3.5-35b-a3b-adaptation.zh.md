# Qwen3.5-35B-A3B 适配说明

[English](qwen3.5-35b-a3b-adaptation.md) | 简体中文

更新日期：2026-09-20。该模型是工程适配目标，不改变标准验收模型。日常发布测试仍按
[验收要求](kv-compress-test-requirements.zh.md)使用
`Qwen/Qwen2.5-14B-Instruct`。之所以额外覆盖 Qwen3.5，是因为其“全注意力 +
Gated-DeltaNet”混合架构可作为后续相关 Intern-S2 适配的前置验证。

## 冻结模型身份

- 上游：[`Qwen/Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)，Apache-2.0；
- Hugging Face revision：`59d61f3ce65a6d9863b86d2e96597125219dc754`；
- 本地快照：`/workspace/models/Qwen--Qwen3.5-35B-A3B`；
- 权重索引总大小：71,903,655,008 bytes，共 14 个 safetensors 分片；
- 原生上下文：262,144 token。

完整分片、索引一致性和 SHA-256 记录见
[`qwen3.5-35b-a3b-model-manifest.json`](evidence/qwen3.5-35b-a3b-model-manifest.json)。

Hugging Face 端点不可用时，通过 ModelScope 的 Qwen 官方镜像下载。测试前必须同时
存在完整模型配置和权重索引，任何未完成的 `.safetensors` 分片都不能进入测试。

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

插件只为 Qwen3.5 接受“一个全注意力 KV 组 + 可选 Mamba/GDN 组”，并要求
`mamba_cache_mode=none`。scheduler 与 worker 提交只截断全注意力 block table。
slot 位置和 attention length 使用压缩后的物理视图，语义 RoPE 位置与 GDN 状态推进
保持不变。

已验证的 Ascend 宿主暴露三层不同粒度：32,768-token 跨组 scheduler 对齐（所有
cache manager block size 的最小公倍数）、2,048-token 全注意力 manager 页，以及
128-token 稠密 K/V 内核块。插件校验该最小公倍数，并使用全注意力 manager 的 block
table 做压缩；评分和物化时把每个 2,048-token attention ID 展开为连续 16 个内核
ID。`kv_budget` 必须能被 2,048 整除，但无需被 32,768-token scheduler 对齐整除。
其他布局会在启动时拒绝。

当前已验证的宿主组合运行该模型时必须带 `--enforce-eager`，并且存在二进制/源码
ABI 不一致：已安装的 GDN 因果卷积算子把四个元数据参数声明为不可选 `int[]`，当前
Python 调用点却传入设备 tensor，并用 `None` 表示缺省值。设置
`VLLM_ASCEND_KVCOMPRESS_QWEN_GDN_LIST_COMPAT=1` 后，插件只为这一精确旧 schema
启用显式兼容桥：把 tensor 转成整数列表，把二维 cache-index 表按行优先展平，并把
缺省元数据映射为空列表。原生 `Tensor?` schema 保持不变，任何未知 schema 都会
fail closed。全程无需修改宿主仓库。只有在新宿主版本已验证源码与二进制对齐后，
才应移除该环境变量和 eager 限制。

部分 RoPE 校准会在旋转前采集归一化 query head。64 个旋转维度使用 TriAttention
面向未来位置的三角评分；192 个直通维度加入校准后的直接 Q·K 内容项。该模型不会
使用只适用于完整 RoPE 的融合评分算子。TP=2 时，各 rank 评分自己的本地 KV 头，
再以最大值 all-reduce 得到完全相同的 token 选择。

V3 会保护开头 128 个和最近 512 个 token，把中间上下文切成 8 段，并为每段分配
精确的比例驱逐配额。Qwen3.5 会评分全部 10 个全注意力层
（`score_layer_stride=1`）。

## 校准与启动

在两张空闲设备上生成模型绑定的校准产物：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m vllm_ascend_kvcompress.calibration \
  --model /workspace/models/Qwen--Qwen3.5-35B-A3B \
  --output artifacts/qwen3.5-35b-a3b-stats-v3.pt \
  --max-length 4096 --device npu:0 --device-map auto \
  --dtype bfloat16 --attn-implementation eager --local-files-only
```

使用
[`examples/qwen3.5-35b-a3b-triattention.json`](../examples/qwen3.5-35b-a3b-triattention.json)
配置插件，再启动 B1：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_PLUGINS=ascend,ascend_kvcompress
export VLLM_ASCEND_KVCOMPRESS_ENABLED=1
export VLLM_ASCEND_KVCOMPRESS_CONFIG=/absolute/path/qwen3.5-35b-a3b-triattention.json
# 仅当前已验证宿主使用旧式不可选 int[] GDN ABI 时需要。
export VLLM_ASCEND_KVCOMPRESS_QWEN_GDN_LIST_COMPAT=1
# 仅当容器 /dev/shm=64 MiB 时需要；并发 4 使用 40 MiB 环形区。
export VLLM_MQ_MAX_CHUNK_BYTES_MB=4
# 只把插件源码路径前置；不能覆盖包含 acl 的 CANN Python 路径。
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
vllm serve /workspace/models/Qwen--Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b --tensor-parallel-size 2 \
  --language-model-only --dtype bfloat16 --block-size 128 --mamba-cache-mode none \
  --max-model-len 32768 --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching --enable-chunked-prefill --no-async-scheduling \
  --generation-config vllm --enforce-eager
```

尽管命令请求 `--block-size 128`，宿主会分别报告 32,768-token scheduler 对齐和
2,048-token 全注意力页。scheduler bind 日志记录这两者，worker bind 日志还会记录
128-token 内核块。

B0 必须独立重启，但仍加载同一插件及模型绑定统计；把 `kv_budget` 设为 32,768、
`recompute_window` 设为 128，使 32,896-token 阈值严格大于服务上限，从而得到
零压缩工程对照。B0/B1 除配置文件中的压缩预算外保持同一命令。这样兼容挂钩和
运行栈完全一致，也无需修改任一宿主仓库。禁止用同一服务生命周期混跑两组结果。

插件会把显式设置的 `VLLM_MQ_MAX_CHUNK_BYTES_MB` 同步应用到 vLLM worker 回传
队列；上游只把它应用于 scheduler 广播队列。超出 chunk 的消息仍使用 socket，
因此该设置只缩小共享内存快速路径，不限制消息大小。不受 64 MiB 限制的部署应省略
此变量并保留上游默认值。

## 测试状态与范围

当前工作树已通过 114 项单元/契约测试，另有 1 项环境跳过。TP=2 的 16K 服务冒烟
已在两组完成：B0 保留全部 16,384 个输入 token；B1 把 16,384 个语义 token 提交为
8,192 个物理 token；两组均生成完整 1,024 个强制输出 token、命中 oracle，且无
静默截断。模型级结论仍以公开长上下文和标准 A2/A3 套件为准。由于本目标使用 TP=2、
eager、显式旧 ABI 兼容桥和混合注意力，其结果属于工程证据，不能标为 V4.6 正式
验收。结果统一发布到 [HTML 榜单](benchmark-leaderboard.html)，并链接机器可读证据
和明确限制。

公开长上下文测试使用 `enable_thinking=false` 的冻结 chat template、
`--generation-config vllm` 和显式中性采样参数。B0/B1 各完成 403 个请求，合计
806 个请求全部成功且无静默截断：

| 数据集 | B0 → B1 质量 | 质量差 | 请求吞吐差 | 物理 KV 缩减 | 提交 / TP rank 回执 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2（110） | 47.27 → 47.27 accuracy | 0.00 pp | −5.83% | 63.76% | 110 / 220 |
| passage_retrieval_en（200） | 100.00 → 100.00 | 0.00 pp | −2.39% | 不适用（短输出绕过） | 0 / 0 |
| Qasper（93） | 51.07 → 50.64 F1 | −0.43 pp | −1.28% | 42.11% | 11 / 22 |

三项质量保持门槛均通过；详细 bootstrap 区间、时延与原始哈希见
[机器可读摘要](evidence/kvcompress-working-tree-20260920-qwen35-public-summary.json)。
服务运行时同一宿主的其他 NPU 还在执行标准矩阵，因此吞吐和时延差只作为探索性
数据；质量、请求完整性和压缩事务才是本次适配的主要结论。

## 移植的 A2/A3 工程矩阵

完整 commissioning 矩阵也已移植到该模型。profile 名保留 `FP16` 以维持工作负载
身份，但 Qwen3.5 实际使用 BF16、eager、TP=2 和显式旧 GDN ABI 兼容桥。因此这些
是补充工程证据，不是标准 graph-mode/TP=1 拓扑的正式执行。

A2 每个 cell、每轮测量 16 个请求，并发 4；B0/B1 各使用三轮独立冷生命周期：

| 形状 | RPS | B0 → B1 总吞吐中位数 tok/s | 变化 | 最小物理 KV 缩减 | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| 8K + 512 | 0.05 | 249.72 → 250.80 | +0.43% | 不适用：未跨阈值 | 通过 |
| 8K + 512 | 0.10 | 266.08 → 262.28 | −1.43% | 不适用：未跨阈值 | **失败** |
| 8K + 512 | 0.20 | 272.09 → 266.65 | −2.00% | 不适用：未跨阈值 | **失败** |
| 8K + 512 | 0.40 | 278.59 → 275.32 | −1.17% | 不适用：未跨阈值 | **失败** |
| 16K + 1,024 | 0.05 | 266.78 → 262.17 | −1.73% | 50.00% | **失败** |
| 16K + 1,024 | 0.10 | 274.73 → 269.07 | −2.06% | 50.00% | **失败** |
| 16K + 1,024 | 0.20 | 278.00 → 274.01 | −1.43% | 50.00% | **失败** |
| 16K + 1,024 | 0.40 | 275.39 → 273.54 | −0.67% | 50.00% | 通过 |

768 个实测组别请求和六个预热请求全部正确完成，失败、强制输出不足和静默截断均为
0。16K B1 共记录 192 次 scheduler 提交和精确的 384 次逐 rank 回执。两个 cell
通过；六个 cell 仅因超过冻结的 1% 总吞吐回退预算而失败。

A3 使用 30,720 输入 token 加 2,048 强制输出 token，并发 1，预热五分钟，测量
30 分钟并切分为六个窗口：

| 三轮中位数 | B0 | B1 | 变化 |
| --- | ---: | ---: | ---: |
| 总 token 吞吐 | 69.34 tok/s | 70.19 tok/s | +1.23% |
| 平均 TTFT | 1949.71 ms | 2009.13 ms | +3.05% |
| 平均 TPOT | 229.91 ms | 227.05 ms | −1.24% |
| 平均 E2E | 472.57 s | 466.77 s | −1.23% |

B0/B1 每轮都只完成 4 个正确请求，低于至少 24 个的要求；所有六窗口吞吐 CV 都是
100%，高于 5% 门槛，空窗口还使时延漂移无法计算。B1 仍实现最小 73.33% 物理 KV
缩减，24 次提交对应 48 次 TP rank 回执。因此 A3 结论为**失败**，尽管质量、事务、
缩减和吞吐回退检查均通过。

最初三个沙箱启动因无法枚举 Ascend 设备，在模型加载和请求执行前失败；其日志哈希
已保留，但不计入三轮有效生命周期。完整哈希和逐 cell 数值见
[标准矩阵机器可读摘要](evidence/kvcompress-working-tree-20260920-qwen35-standard-summary.json)。

这不能推翻 V3 论文对混合架构的风险提示：论文中的 Qwen3.5-27B/35B-A3B 在
V3-only 严格 NIAH 的中部和尾部位置失败，而后续 longctx rescue 尚未在这两个大
模型上验证。本次 `passage_retrieval_en` 因最大输出仅 32 token 按规则绕过压缩，
所以它只验证 bypass，不构成“压缩后 needle 检索”证据。部署前仍应增加经授权的
压缩态多位置 NIAH/生产检索门槛；当前结论严格限于已列出的公开任务。
