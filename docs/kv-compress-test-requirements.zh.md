# KV 压缩测试与验收要求

[English](kv-compress-test-requirements.md) | 简体中文

本文把受控文档《vLLM-HUST 标准交付测试方案 V4.6》中与本插件有关的要求
落实为可执行规程。源文件位于
`vllm-hust-benchmark/docs/assets/vLLM-HUST标准交付测试方案_V4.6.pdf`，
共 63 页、980,035 bytes，SHA-256 为
`ab2ceacf25f9ba82da39dfd7bd457d6b620b6ad33d121aee46fa5c044f58c370`。
如本文与受控 PDF 或独立评测方冻结的运行声明冲突，以后两者为准。

## 结论等级

| 等级 | 用途 | 可宣称 V4.6 正式验收吗 |
| --- | --- | --- |
| 单元/契约 | PR 中检查 schema、包、宿主 seam 和数值 reference | 否 |
| `kv-pressure-online` | 当前宿主上的随机 token KV 压力快速冒烟 | 否；registry 标为 provisional，且基线为 vLLM 0.18 |
| 仓库自验 | 使用仓库生成 fixture 做 A2/A3 形状的确定性服务测试 | 否；fixture 明确标记 `commissioning_only` |
| 正式交付 | 独立评测方以批准的 LONG-PUBLIC 数据执行、签名并固化 B0/B1 | 是 |

不得把自验结果发布为正式吞吐、质量、HBM 或 V4.6 验收结论。

## 冻结目标与生命周期

- 模型：`Qwen/Qwen2.5-14B-Instruct`，冻结模型和 tokenizer revision。
- 硬件：单节点单张 Ascend 910B2；模型 FP16，实际 KV FP16。
- 拓扑：TP=PP=DP=EP=DCP=1，统一非 P-D 服务，FCFS、multiprocessing，
  不使用 CPU offload 或 swap。
- 服务公共参数：block size 128、显存利用率 0.85、关闭 prefix cache、
  speculative decoding、KV transfer、async scheduling，开启 chunked prefill，
  禁止静默截断。
- A2/A3 使用 `FULL_DECODE_ONLY`，capture size 按场景冻结；
  `--enforce-eager` 结果不能作为正式 A2/A3 证据。
- 服务必须使用 `--generation-config vllm`，不得继承模型目录中的采样默认值。
- 请求采样：temperature 0、top-p 1、top-k -1、min-p 0、presence/frequency
  penalty 0、repetition penalty 1（乘法惩罚的中性值）、n=1、不用 beam、
  stop 为空、seed 0、流式并回传 usage、`add_special_tokens=true`。
- B0、B1 各至少三次独立冷启动生命周期：start、warm、measure、stop。
  失败、超时和重试都要保留；预声明主指标取中位数。可以增加至多两轮，
  但不能用新增轮次替换失败轮次。

V4.6 正式 B0 是受控方案冻结的官方 vLLM 0.18 和对应官方 vLLM-Ascend。
当前 HUST 宿主上的配对测试让 B0/B1 都加载同一插件：B0 以高于服务上限的阈值
禁止压缩，B1 使用候选预算。它只属于工程对照，不能替代正式 B0。整个流程不得
修改 `vllm-hust` 或 `vllm-ascend-hust` 宿主仓库。

Qwen3.5-35B-A3B 是补充的工程目标。它使用 TP=2 和“全注意力 +
Gated-DeltaNet”混合缓存，因此结果必须单独保存，不能替代、放宽或并入冻结的
Qwen2.5-14B-Instruct 验收矩阵。

## 本项目选定的交付场景

### A2-LONG-FP16：长上下文质量与服务性能

使用至少 64 篇获授权的 LONG-PUBLIC 文档，冻结来源、许可证、revision 和
校验和。prompt 必须包含可验证事实与 oracle，不能使用随机 token，不能截断。

| cell | 请求数 | 渲染后输入 | 最大输出 | 到达率 | 最大并发 |
| --- | ---: | ---: | ---: | --- | ---: |
| LONG-8K | 16 | 8,192 | 512 | 0.05、0.1、0.2、0.4 RPS | 4 |
| LONG-16K | 16 | 16,384 | 1,024 | 0.05、0.1、0.2、0.4 RPS | 4 |

服务参数为 `max_num_seqs=4`、`max_num_batched_tokens=16384`、显存利用率
0.85、`FULL_DECODE_ONLY` capture size `[1,2,4]`。单请求超时 1,800 秒，
容量 cell 超时 7,200 秒。

必须报告请求/输入/输出/总吞吐，TTFT/TPOT/E2E mean、p95、p99，成功率、
oracle 正确率、OOM、静默截断、B1/B0 比值与增量、HBM 峰值、KV block
缩减、NPU 利用率和三轮波动。B1 相对 B0 的质量下降不得超过 1 个百分点。

### A3-32K-FP16：32K 稳定性

- 精确总上下文为 30,720 个渲染后输入 token 加 2,048 个强制输出 token，
  合计 32,768；设置 `ignore_eos=true`。
- 闭环、并发 1、seed 0、`max_num_seqs=1`、
  `max_num_batched_tokens=16384`、capture size `[1]`。
- 预热 5 分钟、测量 30 分钟，报告六个 5 分钟窗口；最多 64 个请求，
  至少完成 24 个。
- 单请求超时 1,800 秒，单生命周期超时 3,600 秒。
- 请求失败、OOM、死锁、进程退出、静默截断均为 0。
- 窗口吞吐 CV 不超过 5%；TTFT/TPOT 中位数漂移不超过 10%，p99 漂移
  不超过 20%。

### M3 压缩专项门槛

质量必须通过，并且“物理每 token KV 使用量降低至少 20%”或“全系统每 token
成本降低至少 15%”至少满足一项。调度器 commit 与 worker ack 数量必须一致。
宿主静态预留 KV pool 不等于 KV 已减少，必须保留 block 释放事务和 1 秒粒度
HBM 采样。

## 适用性与冲突

Prefix cache、推测解码、KV transfer/P-D 分离、量化 KV、MLA/local attention、
异步调度、PP/DP/DCP/PCP 大于 1，以及未识别的混合布局都会在启动时拒绝。唯一的
混合架构例外是 `mamba_cache_mode=none` 的 Qwen3.5 全注意力 + Gated-DeltaNet；
TP 只有在大小可整除模型 KV 头数时才允许。这些例外只属于工程范围，不改变正式
验收的单卡拓扑。当前实现仍不适用 A1 的非 chunked 配置和 A4 的 prefix-enabled
配置。当前宿主已不存在 Prefix Router、KV Tiering、KNorm、PyramidKV Ascend、
SliceGPT 代码，本插件不导入、不假设这些旧实现。

## 数据准备

正式评测数据必须由评测方批准。下载时不得在命令或日志中写入凭据：

```bash
python scripts/kvcompress_prepare_dataset.py download \
  --url https://approved.example/LONG-PUBLIC-v1.jsonl \
  --sha256 <approved-sha256> \
  --source-revision <immutable-revision> \
  --license-id <SPDX-or-license-name> \
  --output .benchmarks/data/long-public-v1.jsonl
```

脚本只接受 HTTPS，在替换目标文件前核验 SHA-256，检查 JSONL 字段并生成来源
manifest。本地联调可生成确定性、非随机 token 的 Apache-2.0 fixture：

```bash
python scripts/kvcompress_prepare_dataset.py generate-commissioning \
  --tokenizer /data/shared_models/Qwen--Qwen2.5-14B-Instruct \
  --profile all \
  --output .benchmarks/data/kvcompress-commissioning.jsonl
```

该数据始终标为 `commissioning_only`，不能用于正式结论。

## 执行与配对比较

先运行一次 `kv-pressure-online` 快速压力冒烟，再对禁止压缩的 B0 工程对照和
启用候选压缩的 B1 分别独立重启服务，执行所有冻结 A2 cell 与 A3。两组均加载
插件；B0 的阈值必须严格高于 `max_model_len`。A2 示例：

```bash
python scripts/kvcompress_long_context_run.py \
  --dataset .benchmarks/data/long-public-v1.jsonl \
  --profile A2-LONG-FP16-16K \
  --model /data/shared_models/Qwen--Qwen2.5-14B-Instruct \
  --request-rate 0.1 --concurrency 4 --timeout 1800 \
  --npu-id 0 --run-label B1 --server-log .benchmarks/b1-r1/server.log \
  --result .benchmarks/b1-r1/a2-16k.json
```

若某个冻结 cell 的最大物理长度不超过压缩阈值（例如默认 9,216-token
阈值下的 8K cell），B1 命令必须显式追加
`--compression-evidence optional`。此模式允许零事务，但一旦出现 scheduler
commit，仍要求对应 worker acknowledgement。跨过阈值的 cell 保持默认
`auto`（B1 必须观察到事务）；B0 可用 `--compression-evidence forbidden`
明确断言无压缩。

B0/B1 各收集三轮后比较：

```bash
python scripts/kvcompress_acceptance_compare.py \
  --baseline .benchmarks/b0-r{1,2,3}/a2-16k.json \
  --plugin .benchmarks/b1-r{1,2,3}/a2-16k.json \
  --output .benchmarks/a2-16k-comparison.json
```

冻结的 Qwen2.5 TP=1 矩阵保持默认每次提交一个 worker 回执。补充的 Qwen3.5
TP=2 工程矩阵必须显式追加 `--worker-acks-per-commit 2`，比较器随后严格要求每次
scheduler 提交恰好收到两个 rank 回执；该参数不能用于放宽 TP=1 标准矩阵。

比较器从 B1 原始结果读取冻结的压缩证据声明。`required`（以及旧证据缺少该字段时
采用的 fail-closed `auto`）必须满足 M3 的 20% 物理 KV 缩减门槛；若某个
`optional` cell 三轮都没有压缩事务，则该 cell 的 M3 缩减项标记为“不适用”，
但质量、吞吐、请求完整性及 commit/ack 配对门槛仍全部执行。`optional` cell
一旦出现任何压缩事务，M3 缩减门槛立即恢复为必需。
如果证据由早于该字段的 runner 生成，比较命令必须显式追加
`--compression-evidence optional`；默认不猜测历史意图，仍按 `auto` 失败关闭。

## 证据包与放行

保留精确 Git commit、包版本、镜像 digest、模型/tokenizer/数据 hash、命令与
环境、硬件拓扑、启动参数、原始请求/结果/日志、1 秒设备采样、失败记录、
manifest 和产物 hash。计时使用 monotonic clock，百分位使用 nearest rank，
排除预热和冷却。样本少于 1,000 时不允许失败，否则错误率不超过 0.1%。如涉及
结构化输出，schema 必须 100% 通过；工具/结构化成功率至少 95%。

只有精确 release candidate 的所有预声明 gate 都为真才可放行。缺失、字段不匹配、
测试后被修改或正式证据未签名时一律 fail closed。普通 PR CI 只执行静态、单元、
构建、契约、schema 和 hash 检查；性能验收在隔离的独立评测环境执行。
