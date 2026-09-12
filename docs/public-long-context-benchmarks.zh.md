# 公开长上下文 Benchmark 记录

[English](public-long-context-benchmarks.md) | 简体中文

日期：2026-09-12。本记录在原合成 commissioning 测试之外，新增三个公开 A3
长上下文/KV 场景。它是同宿主工程对照，不是 V4.6 官方基线正式验收。

## 场景与冻结输入

| 场景 | 用途 | 实测范围 | 输入 token | 预处理 SHA-256 |
| --- | --- | ---: | ---: | --- |
| LongBench-v2 | 长上下文多选推理 | short 档中 116 条可容纳样本 | 10,171–31,869 | `06255143...b538be6c` |
| LongBench `passage_retrieval_en` | 段落检索 | 200/200 条 | 10,412–15,670 | `cc015530...98a2cb6` |
| LongBench `qasper` | 科研论文问答 | 200 条中达到 5,120-token 压力下限的 93 条 | 5,168–22,074 | `20b04612...b8a6da94` |

LongBench-v2 固定 Hugging Face revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`，下载文件 `data.json` 的
SHA-256 为 `15d61c22...604c7fe2`。LongBench 固定 revision
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，`data.zip` 的 SHA-256 为
`cb45b11a...857f7f64`。

所有接纳的 prompt 均未截断。服务上限为 32,768 token，因此超出该范围的 387 条
LongBench-v2 medium/long 档样本会登记到 unsupported 文件，不会缩短后混入测试。
Qasper 中低于 5,120 token 的样本无法触发原 4K 压缩阈值，因此不纳入本次压力集。
`--limit 0` 表示全部可接纳样本，而非抽样。

下载脚本在解压前校验固定 revision 与哈希。LongBench-v2 数据卡声明 Apache-2.0；
LongBench 汇集的源数据各有自身条款，因此脚本只从上游下载，本仓库和发行包均不
再分发 benchmark 正文。使用者须遵守各源数据集的许可条件。

## 复现步骤

用被测模型的准确 tokenizer 下载并预处理：

```bash
MODEL=/path/to/Qwen2.5-Coder-14B-Instruct
python scripts/kvcompress_benchmark_data.py download \
  --source all --root .benchmarks/datasets

python scripts/kvcompress_benchmark_data.py prepare \
  --benchmark longbench-v2 --root .benchmarks/datasets \
  --tokenizer "$MODEL" --local-files-only \
  --max-model-len 32768 --min-input-tokens 5120 --limit 0 \
  --output .benchmarks/public/longbench-v2.jsonl
python scripts/kvcompress_benchmark_data.py prepare \
  --benchmark longbench-passage-retrieval-en --root .benchmarks/datasets \
  --tokenizer "$MODEL" --local-files-only \
  --max-model-len 32768 --min-input-tokens 5120 --limit 0 \
  --output .benchmarks/public/longbench-passage-retrieval-en.jsonl
python scripts/kvcompress_benchmark_data.py prepare \
  --benchmark longbench-qasper --root .benchmarks/datasets \
  --tokenizer "$MODEL" --local-files-only \
  --max-model-len 32768 --min-input-tokens 5120 --limit 0 \
  --output .benchmarks/public/longbench-qasper.jsonl
```

两组各使用一个冷服务生命周期。B0 在相同宿主与插件下设 32,768-token 预算和
32,896-token 阈值，所有请求均不发生压缩；B1 使用[示例配置](../examples/triattention.json)：
8,192-token 预算、1,024-token 重算窗口、512-token 最近保护、8,192-token 评分
分块、每四层评分一次，并要求请求至少生成 64 token 才值得进入压缩路径。

```bash
export ASCEND_RT_VISIBLE_DEVICES=6
export VLLM_PLUGINS=ascend,ascend_kvcompress
vllm-hust-ext run -- vllm serve "$MODEL" \
  --served-model-name qwen2.5-coder-14b --dtype float16 \
  --block-size 128 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --max-num-batched-tokens 16384 --max-num-seqs 4 \
  --no-enable-prefix-caching --enable-chunked-prefill --no-async-scheduling \
  --generation-config vllm
```

针对两组服务分别运行每个预处理文件，按实际任务修改路径与标签：

```bash
python scripts/kvcompress_benchmark_run.py \
  --dataset .benchmarks/public/longbench-v2.jsonl \
  --base-url http://127.0.0.1:8000 --model qwen2.5-coder-14b \
  --result .benchmarks/results/b1-longbench-v2.json --run-label B1 \
  --request-rate 0 --concurrency 4 --timeout 1800 --npu-id 6 \
  --server-log .benchmarks/logs/b1.log

# passage_retrieval_en 的 max_output_tokens=32，必须验证绕过路径：
python scripts/kvcompress_benchmark_run.py \
  --dataset .benchmarks/public/longbench-passage-retrieval-en.jsonl \
  --base-url http://127.0.0.1:8000 --model qwen2.5-coder-14b \
  --result .benchmarks/results/b1-passage-retrieval-en.json --run-label B1 \
  --request-rate 0 --concurrency 4 --timeout 1800 --npu-id 6 \
  --server-log .benchmarks/logs/b1.log \
  --compression-expectation forbidden

python scripts/kvcompress_benchmark_score.py \
  --baseline .benchmarks/results/b0-longbench-v2.json \
  --plugin .benchmarks/results/b1-longbench-v2.json \
  --quality-tolerance-pp 1 \
  --output .benchmarks/results/paired-longbench-v2.json
```

运行器固定 `temperature=0`、`top_p=1`、`top_k=-1`、`min_p=0`、`seed=0`，
保存逐请求输出、TTFT/TPOT/E2E、包版本、设备采样、服务端 token 计数和 scheduler/
worker 压缩证据。评分器实现 LongBench-v2 官方答案抽取、LongBench 检索得分及 QA
F1 归一化。如果请求失败、服务端 token 数显示截断、实际行为不符合声明的压缩预期、
worker 回执缺失或质量下降超过 1 个百分点，B1 会被判失败。压缩预期默认是
`required`；短输出检索显式指定 `forbidden`，用于证明绕过路径，而非默许一次没有压缩
的运行。

## 测试结果

模型：本地 Qwen2.5-Coder-14B-Instruct，FP16。设备：单卡 Ascend 910B2。
每组运行一个冷服务生命周期，不限请求到达速率，并发 4。B0 定义如上。

| 场景 | B0 → B1 质量 | 请求数 / 压缩数 | 请求吞吐 | 平均 TTFT | 平均 TPOT | 平均 E2E | 物理 block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2 | 34.48 → 34.48 accuracy（0.00 pp） | 116 / 116 | **+2.13%** | **−3.84%** | **−7.80%** | **−2.17%** | −61.96% |
| 段落检索 | 100.00 → 100.00（0.00 pp） | 200 / 0（绕过） | −0.04% | **−0.28%** | +0.27% | +0.01% | 不适用 |
| Qasper | 42.51 → 42.03 F1（−0.48 pp） | 93 / 11 | −0.11% | **−2.89%** | +11.31% | +0.41% | 触发样本 −38.73% |

两组共 818 个请求全部完成，静默截断为 0。B1 有 127 次 scheduler 提交和 127 次
worker 回执。实际触发压缩的样本合计从 20,666 个源 block 降至 8,128 个目标
block，减少 60.67%。LongBench-v2 动态 KV 使用峰值为 39.8%，B0 为 90.4%。
由于 vLLM 启动时预留 KV 池，两组设备 HBM 峰值仍均为 87%。

工作负载边界很明显：插件在输入 10K–32K、输出较丰富的 LongBench-v2 上提升
吞吐。10K–16K 输入的检索任务最多只请求 32 个输出 token，因此优化后的 64-token
门槛会跳过评分和 copy；200 条均无 scheduler 提交/worker 回执，性能近似持平。
Qasper 大部分请求低于 B1 的 9,216-token 阈值，仅 11 条压缩，因此总体性能也近似
持平。每组只有一次运行，性能数值仅支持有方向性的工程结论，不代表吞吐置信区间。

调参过程中的 4,096-token 候选作为负结果保留：Qasper F1 下降 4.71 个百分点，未
通过 1 pp 质量门槛。因此公开 benchmark 推荐 8,192-token 预算；此前 4K 合成结果
只在其固定工作负载边界内有效。

机器可读证据见
[kvcompress-v0.4.0-public-long-context-summary.json](evidence/kvcompress-v0.4.0-public-long-context-summary.json)。

## 尚存限制

- 性能对比每组只有一次运行，不是三轮冷启动统计。
- B0 不是 V4.6 指定的官方宿主基线。
- 本地模型和校准文件属于同一 Qwen2.5-Coder-14B 系列，但缺少完整上游 revision
  与生成 provenance。
- 超过 32K 的 LongBench-v2 medium/long 档需换用更长上下文服务后测试。
- 本记录不是 30 分钟、六窗口 A3 稳定性测试。
