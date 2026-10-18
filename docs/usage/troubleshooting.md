# Troubleshooting

English | [简体中文](troubleshooting.zh.md) | [User guide](README.md)

| Symptom | Check and action |
| --- | --- |
| `extension validate` or `extension check` rejects the host | Compare vLLM, vLLM-Ascend, Manager, Python, and Triton-Ascend versions with the [validated line](installation.md). Provision a matched stack; do not bypass the manifest. |
| Qwen3.5 worker cannot import `fla_npu` | Install the architecture-matched FLA NPU artifact specified by the host release, verify its hash, and test `import fla_npu.ops.ascendc` in the serving environment. |
| Plugin appears installed but inactive | Check `extension status` and `extension check`, launch with `vllm-hust-ext run --`, and set `VLLM_PLUGINS=ascend,ascend_kvcompress` when restricting plugin discovery. Restart after state changes. |
| Startup reports budget/page misalignment | Read the reported full-attention page size. Use a budget divisible by that size and 128; the checked-in Qwen3.5 example uses 4608 for 1152 pages. An 8192 budget is aligned for 2048 pages but lacks paired performance evidence. |
| Calibration fails to load model or write file | Confirm the model/tokenizer path, revision, visible NPUs, directory permissions, and `calibration_local_files_only` choice. Qwen3.5 BF16 needs two devices. |
| Existing statistics are rejected | Check model/checkpoint binding and artifact integrity. Generate a new file at a new path for the intended model; do not reuse a mismatched `.pt` file. |
| Server runs but no compression commit appears | Confirm the plugin is enabled, the context exceeds `kv_budget + recompute_window`, and requested `max_tokens` meets the minimum-output guard. Inspect both scheduler and worker logs for the same request ID. |
| NPU memory exhaustion | Reduce service KV allocation or concurrency, verify no stray model process is holding a card, and ensure the calibration copy is released before serving weights load. |
| A combination is rejected at startup | Check the [conflict matrix](../../README.md#conflict-matrix). Start with the simple serving profile before enabling prefix cache, async scheduling, or MTP2. |

Keep the full startup error and `extension check` output when filing an
issue. Include host/package versions, model revision, config without secrets,
and the first failing request. See the [design notes](../design/methods.md)
for runtime boundaries.
