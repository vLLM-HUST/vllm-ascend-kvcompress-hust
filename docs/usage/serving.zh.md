# 启动服务并验证

[English](serving.md) | 简体中文 | [用户指南](README.zh.md)

先配置并启用插件。首次进行 Qwen3.5 服务检查时，暴露两张 NPU，选用混合模型
配置，并通过 Extension Manager 启动：

```bash
export VLLM_PLUGINS=ascend,ascend_kvcompress
vllm-hust-ext run -- vllm serve /path/to/Qwen3.5-35B-A3B \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --no-async-scheduling
```

使用匹配的宿主栈，并等待模型完全加载后再发请求。如果 `stats_path` 不存在，
首次启动可能需要较长时间进行校准。这个初次检查示例没有打开前缀缓存、
异步调度或推测解码；测量这些组合前应检查适用的
[测试规程](../benchmark/README.zh.md)。

检查 OpenAI 兼容接口是否响应：

```bash
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/path/to/Qwen3.5-35B-A3B","prompt":"Explain one useful cache invariant.","max_tokens":128}'
```

短请求只能验证服务，**不能证明压缩已生效**。验证压缩生命周期时，请发送
上下文超过 `kv_budget + recompute_window`，且所设 `max_tokens` 不小于
`min_output_tokens_for_compression` 的请求。检查服务日志中同一请求 ID 的
`KV compression scheduler commit` 和
`KV compression worker commit acknowledged`，并确认提交后仍能继续生成。
负载未达到门槛时，预期不会出现 commit。

报告结果时记录精确模型 revision、配置、宿主版本、NPU 布局、请求长度和日志。
性能或质量结论须选择适用的[测试规程](../benchmark/README.zh.md)，
遵循其对照与验收规则；一次压缩请求成功仅是集成检查。
