# 浦江五数据集适配——issue #17

[English](dataset-program-issue-17.md) | 简体中文

[候选适用性矩阵](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/6c49429c3211de0363c534a83fcffef5267c5d9d/reports/kvcompress-five-dataset-applicability-20261011)已提交至[基准仓库 PR #276](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/276)，由[本 MOD 的 issue #17](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/issues/17)和[项目执行总账 #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254)跟踪。矩阵仍待维护者审查，目前没有五数据集任务质量成绩或优化结论。

| 数据集 | 状态 | 当前门禁 |
| --- | --- | --- |
| MMLU-Pro | `not-exercised` | 已审计的 Qwen 参考五样例提示词覆盖 12,032 道测试题，最长 2,910 token；输出上限 2,048 token，因此总长度最多 4,958 token，低于当前 Qwen3.5 TriAttention V3 的 9,216 token 压缩门槛。这只是长度上界判断，不是本 MOD 的运行结果。 |
| HLE-Verified | `blocked` | 权利核验、全量或 Gold 子集选择、图像处理、judge 与运行合同尚待冻结。 |
| SWE-bench-Pro | `blocked` | 权利核验、任务与代码仓/镜像映射、新鲜隔离沙箱和复判器尚待落实。既有 SWE prefix-reuse 吞吐遥测不是任务解决率。 |
| FrontierScience | `blocked` | Olympiad 和 Research 需分别冻结评分器，最终 solver 提示词与输出预算尚未确定。仅凭候选题面长度不能判定最终适用性。 |
| Terminal-Bench 2.1 | `blocked` | Harbor 镜像、agent/工具/预算/评分合同及逐题权利核验尚待落实。 |

## 存储容量规划（估算）

下表按本次提供的数字整理，用于准备数据集和评测环境；尚未在本仓库独立核验。具体占用取决于下载范围、镜像缓存策略与运行环境，不能将数据集本体大小当作完整评测环境的大小。

| 数据集 | 数据集本体下载 | 完整评测环境可能占用 | 建议预留 |
| --- | ---: | ---: | ---: |
| **MMLU-Pro** | **≈ 4.2 MB** | 基本无额外大资源 | **< 100 MB** |
| **HLE-Verified** | **≈ 335 MB**（含图片完整版） | 很小 | **0.5–1 GB** |
| **SWE-bench-Pro** | **≈ 15 MB** | 若缓存全部 task 镜像，约 **1–2+ TB** | **2–3 TB** |
| **FrontierScience** | **≈ 518 KB** | 基本无额外本地资源 | **< 100 MB** |
| **Terminal-Bench 2.1** | **≈ 58.4 MB** | Docker 环境约 **几十 GB** | **≈ 80–150 GB** |

## Token 规模与 9,216-token 门槛初筛

以下输入、输出和 token 数字按本次提供的表格整理，尚未绑定可核查的原始产物，不能作为本 MOD 在这五项上的配对评测结果。`total` 对单次请求和多轮 agent 任务的含义不同；多轮任务累计 token 超过门槛，**不能证明任一单次请求的物理 KV 长度达到 9,216 token**。实际适用性仍须按冻结后的每次请求审计，并核对 B1 的压缩提交与 worker 确认。

| Benchmark | 样本输入规模 | 输出规模 | 代表性 total | `total ≥ 9,216` | 判断可信度 |
| --- | ---: | ---: | ---: | --- | --- |
| **MMLU-Pro** | 当前 5-shot 最大 **2,910** | 当前上限 **2,048**；Qwen3.5 参考统计均值 **1,179** | 当前最多 **4,958** | **0%（当前配置）** | **高** |
| **HLE-Verified** | question 最大 16.9k chars，约数千 token | 类 HLE 的 Qwen3.5 参考统计均值 **6,625** | 可能接近或超过 9k | **未知，但预计不可忽略** | 中 |
| **SWE-bench-Pro** | 静态 issue prompt 通常数百～数千 token | agent 多轮；参考值约 **6.1k–8.9k output/task** | fresh input+output 约 **9.4k–14.4k/task** | **任务累计很可能大量超过；单请求未知** | 中高 |
| **FrontierScience** | problem 408–6,590 chars | 强烈取决于 reasoning model：参考值约 **3.3k–29k** | 可低于也可远高于 9,216 | **高度模型相关，不能从数据集本身确定** | 高 |
| **Terminal-Bench 2.1** | 初始 instruction 通常 <~1–1.5k token | agent 多轮；参考值约 **12k–18k output/task** | fresh input+output 约 **12.5k–47k/task** | **任务累计预计多数超过；单请求未知** | 中高 |

当前 B1 配置为 [`.benchmarks/config/qwen3.5-35b-a3b-v3-batched-stride2-min512.json`](../../.benchmarks/config/qwen3.5-35b-a3b-v3-batched-stride2-min512.json)。压缩门槛由 `kv_budget + recompute_window` 决定，`min_output_tokens_for_compression` 是另一项门禁。评测方若冻结新的任务提示词，应对准确的分词后任务清单重新审计：

```sh
python scripts/kvcompress_dataset_applicability.py \
  --manifest /path/to/frozen-task-prompts.jsonl \
  --method-config .benchmarks/config/qwen3.5-35b-a3b-v3-batched-stride2-min512.json \
  --output /path/to/applicability-audit.json
```

每条 JSONL 记录必须包含唯一的 `case_id`、准确的 `prompt_token_ids` 和 `max_tokens`，也可包含 `input_len`。输出会记录输入和配置的 SHA-256，并列出在预算上**可能**达到压缩门槛的请求。即使长度允许，也必须看到真实 B1 服务端的 scheduler commit 与 worker acknowledgement，才能确认压缩已执行。不得通过填充提示词或降低门槛制造适用性。

数据集达到可执行条件后，须先冻结评分器与完整 B0/B1 合同，再测量。任务质量是主指标；吞吐与时延只是辅助遥测。
