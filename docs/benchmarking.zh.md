# 基准测试与结果解读

[English](benchmarking.md) | 简体中文

KV 缓存压缩必须作为容量、延迟、吞吐和质量之间的权衡进行评估。仅有更低的
物理 KV 占用，不能证明端到端效果得到改善。

## 匹配 A/B 协议

两组测试必须使用相同的模型、revision、tokenizer、NPU、dtype、block size、
KV 池大小、执行模式、提示集、到达模式、并发度、采样参数和输出长度。预期的
唯一差异是是否设置 `--kv-cache-compression-config`。

测试内置方法时必须报告全部方法选项，尤其是 `kv_budget`、
`recompute_window`、`protected_recent_window` 和 `score_chunk_size`。
score chunk 是性能/内存调优参数：增大该值可以减少 kernel 启动开销，但会
增加临时设备内存。

推荐使用以下 vLLM-HUST Benchmark 场景：

- `prefix-repetition-online`：重复前缀缓存行为；
- `random-online`：小规模合成在线 smoke 负载；
- `knorm-kv-compression-longctx`：持续长上下文服务；
- `kv-pressure-online`：接近 KV 容量边界的同时到达负载。

单独验证本插件时，应禁用 prefix caching 和独立的 Knorm owner。两组测试都
使用 eager 模式、block size 128，以及相同的 `max_model_len` 和
`gpu_memory_utilization`。

## 复现环境

激活目标 conda 环境后，在 `vllm-hust-benchmark` checkout 中执行命令。根据
本机路径设置：

```bash
export MODEL=/path/to/Qwen2.5-Coder-14B-Instruct
export STATS=/path/to/triattention-stats.pt
export RESULT_ROOT=/path/to/ab-results
export ASCEND_RT_VISIBLE_DEVICES=5
export VLLM_KNORM_ENABLED=0
export VLLM_ASCEND_TORCH_PREFLIGHT=0
```

启动任一服务前先检查 `npu-smi info`。选中的物理设备必须没有进程，并处于
空闲 HBM 基线。

## 启动服务

不传入压缩配置，启动 baseline 服务：

```bash
vllm serve "$MODEL" \
  --enforce-eager \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --block-size 128 \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.8
```

启用压缩的 A/B 组需要先停止 baseline 服务并确认 NPU 已释放，再以完全相同
的其他参数启动：

```bash
vllm serve "$MODEL" \
  --enforce-eager \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --block-size 128 \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.8 \
  --kv-cache-compression-config "{\
\"schema_version\":1,\
\"provider\":\"ascend_kvcompress\",\
\"provider_config\":{\
\"method\":\"triattention\",\
\"stats_path\":\"$STATS\",\
\"kv_budget\":2048,\
\"recompute_window\":128,\
\"protected_recent_window\":128,\
\"score_aggregation\":\"mean\",\
\"layer_aggregation\":\"mean\",\
\"score_chunk_size\":8192}}"
```

等待日志出现 `Application startup complete`，并在启动客户端前确认
`curl -f http://127.0.0.1:8000/health` 成功。

## 运行四个 Benchmark 客户端

压缩关闭服务设置 `MODE=baseline`，启用服务设置 `MODE=triattention`。对应
服务保持运行时，两组分别执行完全相同的以下四条命令。

### prefix-repetition-online

```bash
python -m vllm_hust_benchmark.cli run prefix-repetition-online \
  --model "$MODEL" \
  --set num_prompts=4 \
  --set prefix_repetition_num_prefixes=2 \
  --set prefix_repetition_prefix_len=2304 \
  --set prefix_repetition_suffix_len=256 \
  --set prefix_repetition_output_len=64 \
  --set custom_output_len=64 \
  --set request_rate=1 \
  --set max_concurrency=1 \
  --set temperature=0 \
  --set ignore_eos=true \
  --set save_result=true \
  --set save_detailed=true \
  --set result_dir="$RESULT_ROOT/prefix-repetition-online/$MODE" \
  --set result_filename=raw.json \
  --execute
```

### random-online

```bash
python -m vllm_hust_benchmark.cli run random-online \
  --model "$MODEL" \
  --set num_prompts=2 \
  --set input_len=2560 \
  --set output_len=32 \
  --set request_rate=1 \
  --set max_concurrency=1 \
  --set temperature=0 \
  --set ignore_eos=true \
  --set save_result=true \
  --set save_detailed=true \
  --set result_dir="$RESULT_ROOT/random-online/$MODE" \
  --set result_filename=raw.json \
  --execute
```

### knorm-kv-compression-longctx

```bash
python -m vllm_hust_benchmark.cli run knorm-kv-compression-longctx \
  --model "$MODEL" \
  --set num_prompts=8 \
  --set prefix_repetition_num_prefixes=2 \
  --set prefix_repetition_prefix_len=7168 \
  --set prefix_repetition_suffix_len=1024 \
  --set prefix_repetition_output_len=128 \
  --set custom_output_len=128 \
  --set request_rate=2 \
  --set max_concurrency=4 \
  --set temperature=0 \
  --set ignore_eos=true \
  --set save_result=true \
  --set save_detailed=true \
  --set result_dir="$RESULT_ROOT/knorm-kv-compression-longctx/$MODE" \
  --set result_filename=raw.json \
  --execute
```

### kv-pressure-online

```bash
python -m vllm_hust_benchmark.cli run kv-pressure-online \
  --model "$MODEL" \
  --set num_prompts=16 \
  --set input_len=8192 \
  --set output_len=64 \
  --set request_rate=inf \
  --set max_concurrency=16 \
  --set temperature=0 \
  --set ignore_eos=true \
  --set save_result=true \
  --set save_detailed=true \
  --set result_dir="$RESULT_ROOT/kv-pressure-online/$MODE" \
  --set result_filename=raw.json \
  --execute
```

已验证的 benchmark 快照中，prefix-repetition 客户端即使收到更小的专用输出
覆盖值，仍可能采用通用的 256-token 默认值。必须比较两份原始 JSON 中实际
持久化的 `input_lens` 和 `output_lens`；不一致的 A/B pair 必须作废。

第四个客户端结束后停止服务，并检查 8000 端口无监听、没有遗留 vLLM 进程，
且 `npu-smi info` 显示所选 NPU 回到运行前空闲基线。

## 必报指标

至少记录：

- 成功和失败请求数；
- 精确输入/输出 token 数和并发度；
- 请求、输出 token 和总 token 吞吐；
- TTFT、TPOT 和 ITL 的平均值与 P99；
- 每次压缩提交的源、目标和释放 KV block 数；
- 活跃 KV 池峰值占用、运行请求数和等待请求数；
- 有损压缩在目标模型上的质量或任务准确率；
- NPU 选择、进程退出和测试后资源释放情况。

## 显存解读

vLLM 会在启动时预分配 KV 池。将物理 block 归还 scheduler 会增加可复用容量，
但通常不会降低 `npu-smi` 显示的进程级 HBM 分配。应将活跃 KV 池占用以及
提交后的物理 token/block 作为压缩容量信号。预分配池不变时，不应宣称
allocator 级 HBM 分配下降。

block size 为 128 时，将 8192-token 提示从 64 个 block 压缩到 16 个 block，
会保留 2048 个物理 token、释放 48 个 block，即提示词 KV 减少 75%。语义
位置仍为 8192；降低的只是物理缓存占用。

## 发布规范

稳定的方法和范围明确的汇总结果放在公开文档中。原始 JSON、带机器路径的完整
命令、失败尝试、profiler 日志和调优笔记放在被 git ignore 的 `docs/dev/`。
单次运行只能作为工程证据，不应视为普遍性能保证。
