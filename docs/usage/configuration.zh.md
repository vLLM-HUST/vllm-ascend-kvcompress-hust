# 配置与启停

[English](configuration.md) | 简体中文 | [用户指南](README.zh.md)

稠密注意力模型可复制[通用示例](../../examples/triattention.json)；
Qwen3.5-35B-A3B 使用[混合模型示例](../../examples/qwen3.5-35b-a3b-triattention.json)。
将 `stats_path` 改成可写的绝对产物路径。需要代表性校准时，将
`calibration_input_path` 指向许可明确的 UTF-8 文本。配置文件应保存在已安装
软件包以外，避免升级时被覆盖。

## 预算与页对齐

`kv_budget` 是压缩后物理 KV 的目标长度，`recompute_window` 是供重算保留
的最近窗口。上下文长度超过两者之和，且请求设置的 `max_tokens` 不小于
`min_output_tokens_for_compression` 时才可能触发压缩。此门槛检查请求的
输出上限，不检查已经生成的 token 数。
这两个值均须为 128 的正整数倍；`protected_prefix_window` 与
`protected_recent_window` 之和不得超过 `kv_budget`。

预算还必须被宿主实际报告的**全注意力页大小**整除。Qwen3.5 示例采用
`kv_budget=4608`，对应 1,152-token 页（`4608 = 4 × 1152`）。
如果宿主报告 2,048-token 页，`8192` 可满足对齐（`8192 = 4 × 2048`），
但此配置尚未通过[已记录的配对性能验证](../optimization/triattention-tp-score-batching-20261017.zh.md)。
请在服务启动时核对布局报告；页大小不同则先调整预算。
注意力内核仍使用 `--block-size 128`，它与逻辑注意力页大小不同。

`score_layer_stride` 控制参与评分的注意力层采样间隔。
仓库中的 Qwen3.5 示例采用 stride 2 和
`min_output_tokens_for_compression=2048`；测量结果及早期候选设置见
[优化记录](../optimization/triattention-tp-score-batching-20261017.zh.md)。
这些是负载选择，并非通用质量保证。除非明确评估其他方法，
应保持 `position_policy=v3`。

## 启用与检查

切换 Manager 状态前，先停止正在运行的 vLLM：

```bash
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /path/to/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension status org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension check org.vllm-hust.ascend-kvcompress
```

Manager 只在 `vllm-hust-ext run -- ...` 内激活插件。开发时直接运行，可设置
`VLLM_ASCEND_KVCOMPRESS_ENABLED=1`、
`VLLM_ASCEND_KVCOMPRESS_CONFIG=/path/to/triattention.json`，以及
`VLLM_PLUGINS=ascend,ascend_kvcompress`。不要混用两种配置方式而不确认
服务实际读取了哪一份。

禁用时，停止服务，执行
`vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress`，再重启。
`extension forget` 删除已保存的配置；卸载软件包是独立操作。
下一步是[校准](calibration.zh.md)。
