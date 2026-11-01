# MMLU-Pro 准备与评估

[English](mmlu-pro.md) | 简体中文

客户端安装、数据下载和冻结分词计划无需 NPU，也不需要宿主包。模型评估连接独立启动的服务；
脚本不会启动模型服务。

## 固定输入

- 客户端 `vLLM-HUST/hust-eval-client-mmlu-pro`：`b6215bf76808b1e9d351295e2ecb401542317372`。
- 数据 `TIGER-Lab/MMLU-Pro`：`b189ec765aa7ed75c8acfea42df31fdae71f97be`，
  test 12,032 条；validation 70 条用于各学科 five-shot 示例。
- 上游参考 `TIGER-AI-Lab/MMLU-Pro`：`f418b116db00b065c2aea046518d8fcf74d39872`。
- 固定 transformers 5.14.1、pyarrow 19.0.1、Jinja2 3.1.6。
  tokenizer/chat template 必须与服务完全一致并记录哈希。准备不读取权重，测试集不用于压缩校准。

## CPU 准备

```bash
python3.11 -m venv .benchmarks/mmlu-pro-env
export PYTHONNOUSERSITE=1
unset PYTHONPATH
.benchmarks/mmlu-pro-env/bin/python -m pip install -r requirements/mmlu-pro.txt
export EVAL_PYTHON="$PWD/.benchmarks/mmlu-pro-env/bin/python"
"$EVAL_PYTHON" scripts/kvcompress_mmlu_pro.py download
"$EVAL_PYTHON" scripts/kvcompress_mmlu_pro.py prepare \
  --model-path /path/to/served-model-tokenizer \
  --output .benchmarks/pujiang/plan
```

默认数据目录 `.benchmarks/pujiang/`，其中 `pujiang` 表示浦江实验室指定评测，
不绑定插件版本。两份 parquet 校验固定上游 SHA256，随后由客户端
检查 12,032/70 行及上游 prompt/评分参考。`data-manifest.json` 记录哈希、大小、提交和下载来源。
HF 连接失败时尝试 ModelScope 镜像，但必须与同一固定上游哈希完全一致；不将镜像 master 当作固定版本。

冻结全量 12,032 题、每科 five-shot、token IDs、答案、32,768-token 上下文与
2,048-token 输出上限，关闭 thinking。计划绑定 scorer 哈希并拒绝覆盖已有计划。
分配 NPU 前冻结的计划中服务来源未知；之后可在新目录使用
`prepare --runtime-contract /path/to/runtime-contract.json` 附上服务操作者提供的真实环境。
不得用客户端包身份伪造服务来源。

## 分配 NPU 后评估

先满足[宿主环境要求](../development/environment-installation.zh.md)。`--base-url` 不包含 `/v1`；
服务必须支持 `/v1/completions` 的 token-ID prompt。凭证通过 `--api-key-env MODEL_API_KEY` 指定环境变量。
每组 B0/B1 必须共用同一冻结计划。

```bash
# Run once against B0, restart the service with B1, then repeat with a new output.
"$EVAL_PYTHON" scripts/kvcompress_mmlu_pro.py run \
  --plan .benchmarks/pujiang/plan \
  --base-url http://127.0.0.1:8000 --served-model YOUR_SERVED_MODEL \
  --output dev/results/mmlu-pro/b0-smoke --limit 1

# Full B0 arm; use b1-full for the corresponding B1 service.
"$EVAL_PYTHON" scripts/kvcompress_mmlu_pro.py run \
  --plan .benchmarks/pujiang/plan \
  --base-url http://127.0.0.1:8000 --served-model YOUR_SERVED_MODEL \
  --output dev/results/mmlu-pro/b0-full --concurrency 1
```

每次保留逐请求回复并生成 `summary.json`。传输失败保留在分母中记错，并以非零状态退出；
无法提取答案记错，不随机猜测。中断后可 `--resume`，新汇总另存而不覆盖旧文件。
未完成汇总明确标记为不完整；保留错误及截断输出。一题 smoke 不代表 benchmark 成绩。

报告整体/分类准确率、失败、无效答案、截断及客户端 TTFT/E2E/token；吞吐、HBM 峰值和实际压缩事务
需另外采集服务遥测。2,048-token 输出上限可达到示例配置的压缩门槛，但只有遥测能证明压缩实际发生。

## 成对服务生命周期运行器

对于固定 HUST 栈，通过 `scripts/kvcompress_ascend_env.sh` 调用
`scripts/kvcompress_benchmark_host.py -m vllm.entrypoints.cli.main serve ...`，
可为**两组**及其派生 worker 显式应用相同的 ABI 兼容处理。该入口不会开启压缩；
B0 设置 `VLLM_ASCEND_KVCOMPRESS_ENABLED=0`，B1 使用 `1` 并指定配置路径。
其中 Mamba 兼容层会转发新版核心的 `prefill_end` 边界，保留 Ascend 的外部 KV
预留规则。两组必须使用同一启动器及宿主兼容层，不能拿已适配的 B1 对比无法运行的 B0。

[短上下文 Qwen3.5 候选配置](../../examples/mmlu-pro-qwen35-triattention.json)
用于验证实际压缩是否触发，不作为部署推荐，必须单独验证质量。其使用
1,152-token KV 目标和 128-token 重算窗口，保留原有 2,048-token 请求输出门槛
及保护窗口。这是独立的工作负载配置，不改变通用 Qwen3.5 示例。必须检查实际的
full-attention 页大小，该候选要求 1,152-token 页。在查看测试准确率之前选定预算。
通用示例的 5,632-token 触发门槛可能超过全部提示词与输出预算之和，此时插件虽已启用，
也不能提供实际压缩的质量证据。应检查 B1 日志中的调度端提交及 worker 确认。
内置校准文本是独立于测试集的集成材料，其部署代表性尚未建立。

创建接受 `b0` 或 `b1` 参数的可执行启动器，然后运行：

```bash
python scripts/kvcompress_mmlu_pair.py \
  --launcher /path/to/launch-server.sh \
  --client-python "$EVAL_PYTHON" \
  --plan .benchmarks/pujiang/plan \
  --output /path/to/external/benchmark-submission \
  --served-model YOUR_SERVED_MODEL --concurrency 16 --pairs 3
```

运行器在每组启动新服务，执行相同的 16 题预热，再评估全部任务。并发 16 将被记录为
相对冻结客户端契约中并发 1 的偏离。每组保留生命周期元数据、服务与客户端日志、
逐题原始响应、准确率汇总、Prometheus 快照及采样的 NPU/HBM 状态。HBM 采样峰值
只是观测峰值，不是分配器的精确高水位。启动、预热或客户端失败时，运行器停止当前
配对并保留失败记录；修复后使用新目录重跑。不要跨新服务生命周期恢复失败任务后，
将其称为独立且未中断的运行。失败或中断时会停止所属服务。测量期间保持其他 NPU
工作负载空闲。
