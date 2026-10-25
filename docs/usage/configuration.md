# Configuration and lifecycle

English | [简体中文](configuration.zh.md) | [User guide](README.md)

Copy [the general example](../../examples/triattention.json) for a supported
dense attention model. For Qwen3.5-35B-A3B, copy
[the hybrid example](../../examples/qwen3.5-35b-a3b-triattention.json).
Edit `stats_path` to an absolute writable artifact path. For representative
calibration, set `calibration_input_path` to a licensed UTF-8 text file.
Keep the JSON outside the installed package so upgrades do not replace it.

## Budget and page alignment

`kv_budget` is the compressed physical KV target; `recompute_window` is the
recent span retained for recomputation. Compression can begin only after
the context exceeds their sum and the request's configured `max_tokens` is at
least `min_output_tokens_for_compression`. This guard checks the requested
output limit, not how many tokens have already been generated. Both values
must be positive multiples of 128. `protected_prefix_window` plus
`protected_recent_window` must fit within `kv_budget`.

The budget must also be divisible by the **actual full-attention page size**
reported by the host. The checked-in Qwen3.5 example uses `kv_budget=16128` for a
1,152-token page (`16128 = 14 × 1152`). For a host reporting 2,048-token
pages, `8192` is aligned (`8192 = 4 × 2048`), but this alternative has not
passed the [recorded paired performance test](../optimization/triattention-private-copy-and-v3-selection-20261023.md).
Check the startup layout report
and change the budget before serving if the host reports a different size.
Keep `--block-size 128` for the attention kernel; it is distinct from the
logical attention page size.

`score_layer_stride` controls how many supported attention layers are scored.
The checked-in Qwen3.5 example uses stride 2 and
`min_output_tokens_for_compression=2048`, with 1,024-token recent protection.
Their measurement and earlier candidate settings are in the
[optimization record](../optimization/triattention-private-copy-and-v3-selection-20261023.md).
These settings are workload choices, not universal quality guarantees. Keep
`position_policy=v3` unless evaluating another method
deliberately.

## Enable and inspect

Stop running vLLM processes before changing manager state:

```bash
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /path/to/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension status org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension check org.vllm-hust.ascend-kvcompress
```

The manager activates the plugin only for `vllm-hust-ext run -- ...`.
For direct development invocation, set
`VLLM_ASCEND_KVCOMPRESS_ENABLED=1` and
`VLLM_ASCEND_KVCOMPRESS_CONFIG=/path/to/triattention.json`, plus
`VLLM_PLUGINS=ascend,ascend_kvcompress`. Do not mix direct and manager
configuration without checking which one the service reads.

To disable, stop the server, run
`vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress`, and
restart. `extension forget` removes saved configuration; uninstalling the
package is a separate operation. Continue with [calibration](calibration.md).
