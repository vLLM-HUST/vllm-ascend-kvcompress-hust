# Serve and verify

English | [简体中文](serving.zh.md) | [User guide](README.md)

Configure and enable the plugin first. For a simple initial Qwen3.5 service,
expose two NPUs, select the hybrid configuration, and start the server through
Extension Manager:

```bash
export VLLM_PLUGINS=ascend,ascend_kvcompress
vllm-hust-ext run -- vllm serve /path/to/Qwen3.5-35B-A3B \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --block-size 128 \
  --no-enable-prefix-caching \
  --no-async-scheduling
```

Use the matching host stack and confirm that the model is fully loaded before
sending traffic. The first start may spend time calibrating if `stats_path`
does not exist. The example keeps optional prefix caching, async scheduling,
and speculative decoding out of the initial smoke test. Check the applicable
[benchmark protocols](../benchmark/README.md) before measuring combinations
of these features.

Check that the OpenAI-compatible endpoint responds:

```bash
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/path/to/Qwen3.5-35B-A3B","prompt":"Explain one useful cache invariant.","max_tokens":128}'
```

This short request verifies serving, **not compression**. To verify the
compression lifecycle, send a request whose context exceeds
`kv_budget + recompute_window` and set `max_tokens` at least as high as
`min_output_tokens_for_compression`. Inspect server logs for both
`KV compression scheduler commit` and
`KV compression worker commit acknowledged` for the same request ID.
Check that the service continues generating after the commit. If the workload
never crosses the threshold, no commit is expected.

Capture the exact model revision, config, host versions, NPU layout, request
length, and logs when reporting a result. For a performance or quality claim,
choose the applicable [benchmark protocol](../benchmark/README.md) and follow
its comparison and acceptance rules. One successful compressed response is
an integration check only.
