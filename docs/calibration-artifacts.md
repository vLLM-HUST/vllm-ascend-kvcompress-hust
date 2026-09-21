# TriAttention Calibration Artifacts

English | [简体中文](calibration-artifacts.zh.md)

TriAttention needs model-specific query statistics to score cached keys.
Version 0.5 generates the structured `.pt` artifact inside this plugin; an
external TriAttention checkout is no longer required.

## Automatic generation during service startup

`auto_calibrate` defaults to `true`. When `stats_path` is missing, the Ascend
worker performs this sequence:

1. acquire an exclusive lock beside `stats_path` and check the path again;
2. load the same model, tokenizer, revision, dtype, and trust policy selected
   by vLLM through Hugging Face Transformers;
3. run one calibration forward pass and reduce each layer's unrotated
   `q_proj` output directly into per-head frequency statistics;
4. validate the complete payload, flush a temporary file, and atomically
   rename it to `stats_path`;
5. release the temporary model and device cache; and
6. let the original `NPUModelRunner.load_model()` load the serving weights.

The calibration and serving copies of the model are therefore not resident at
the same time. Direct online reduction also avoids retaining full query
sequences for every layer. If another worker created the file while this worker
was waiting, the completed artifact is reused. Partial files are never exposed
at `stats_path`.

The minimum configuration is:

```json
{
  "method_config": {
    "stats_path": "/srv/vllm/calibration/qwen-stats.pt",
    "auto_calibrate": true
  }
}
```

The output directory must exist or be creatable by the service account and
must have enough space for the artifact. The plugin manifest declares
`filesystem_read`, `filesystem_write`, `device_access`, and `network_egress`;
use `calibration_local_files_only=true` when all model files are local and
network access is forbidden.

An existing artifact is never overwritten during service startup. It is loaded
with `torch.load(..., weights_only=True)` and checked before model loading. New
plugin-generated artifacts also bind to the configured model identifier,
revision, and a local checkpoint-manifest fingerprint. A corrupt, structurally
incompatible, or differently bound artifact fails startup. Set
`auto_calibrate=false` to retain the earlier strictly pre-generated workflow.

## Calibration options

| Option | Default | Purpose |
| --- | --- | --- |
| `stats_path` | required | Artifact destination and later runtime input |
| `auto_calibrate` | `true` | Generate only when `stats_path` is absent |
| `calibration_input_path` | built-in corpus | Licensed UTF-8 calibration text |
| `calibration_max_length` | `4096` | Tokenizer truncation limit; minimum 128 |
| `calibration_device` | `auto` | Worker device, or an explicit value such as `npu:0` |
| `calibration_attn_implementation` | `eager` | Transformers backend: `eager`, `sdpa`, or `flash_attention_2` |
| `calibration_local_files_only` | `false` | Forbid model/tokenizer downloads |

The bundled multilingual systems text makes first startup self-contained and
is suitable for bootstrap and integration validation. Production operators
should provide an independent, licensed, representative corpus, record its
hash, and run the workload's quality gate. Calibration is generally
domain-tolerant, but it is not quality-free.

## Generate before service startup

The installed wheel exposes the same generator as a CLI. This is useful when
startup time must be predictable or the service account may not write the
artifact directory:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

vllm-ascend-kvcompress-calibrate \
  --model /path/to/Qwen2.5-14B-Instruct \
  --input /path/to/licensed-calibration.txt \
  --output /srv/vllm/calibration/qwen-stats.pt \
  --max-length 32768 \
  --device npu:0 \
  --dtype bfloat16 \
  --attn-implementation eager \
  --local-files-only
```

For Qwen3.5-35B-A3B, expose two devices and add `--device-map auto`. The model
is too large for one 910B2 in BF16. When an enabled service uses TP>1, the
startup hook selects this automatic distribution itself; the file lock still
ensures that only one worker generates the artifact.

Omit `--input` to use the bundled corpus. The command reuses an existing valid
payload. `--force` is deliberately CLI-only and atomically replaces it; use
that flag only when creating a candidate at a controlled path. Prefer a new
filename until correctness and long-context quality checks pass.

## Generation algorithm

For each supported full-attention decoder layer, the generator hooks the
post-projection `q_norm` when available and otherwise `q_proj`. The output is reshaped to
`[tokens, query_heads, head_dim]`. With the verified half-split RoPE layout,
the first and second halves form the real and imaginary frequency components.
The generator accumulates:

- mean real query value;
- mean imaginary query value; and
- mean complex magnitude.

Partial-RoPE models additionally store the mean query value for every
pass-through dimension. Hybrid layers without `self_attn` are skipped only when
their indices exactly match the model configuration; an unexpected layer set
fails closed.

Reduction happens immediately on the device and only the small sums move to
CPU. Applying RoPE and then numerically inverting it is unnecessary because
`q_proj` already exposes the required unrotated query. For default RoPE, exact
inverse frequencies are read from the model when available or derived from
`rope_theta`. For scaled/non-default RoPE, the model must expose exact
`inv_freq` and attention scaling; otherwise generation fails closed.

The currently validated shape families are Llama, Mistral, Qwen2/Qwen2-MoE,
Qwen3/Qwen3-MoE, and Qwen3.5/Qwen3.5-MoE with the half-split layout.
Fused/custom attention modules without a usable `q_norm` or `q_proj` are
rejected instead of guessed.

## Structured payload

The current generator writes schema 3. In addition to schema 2 fields it records
`rotary_dim`, `attention_layer_indices`, and per-layer `q_pass_mean` when RoPE
is partial:

```python
{
    "metadata": {
        "schema_version": 3,
        "generator": "vllm-ascend-kvcompress-hust",
        "generator_version": "0.6.0",
        "model": "/path/or/hf-id",
        "model_revision": "default",
        "model_source_fingerprint": "local-manifest:...",
        "model_config_sha256": "...",
        "input_source": "/path/to/calibration.txt",
        "input_sha256": "...",
        "token_count": 4096,
        "model_type": "qwen2",
        "num_layers": 48,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rotary_dim": 128,
        "attention_layer_indices": [0, 1, "...", 47],
        "rope_theta": 1000000.0,
        "rope_style": "half",
        "rope_type": "default",
    },
    "layer_stats": {
        "0": {
            "q_mean_real": Tensor[40, 64],
            "q_mean_imag": Tensor[40, 64],
            "q_abs_mean": Tensor[40, 64],
            "freq_scale_sq": Tensor[1, 64],
            "inv_freq": Tensor[64],
            # "q_pass_mean": Tensor[40, pass_dim] for partial RoPE,
        },
        # Every layer is present.
    },
}
```

The loader remains backward compatible with the original flat TriAttention
`stats` mapping and structured payloads that use integer layer keys. Legacy
payloads cannot provide the stronger model-source binding because those fields
were never recorded.

## Legacy repository artifacts

The two files under `artifacts/` are retained for development and historical
reproducibility and are excluded from wheels and sdists:

| File | Intended use | SHA-256 |
| --- | --- | --- |
| `qwen2.5-coder-14b-stats.pt` | Historical benchmark artifact | `d1f43bf5de3ab7d464a0a906060bbc15b828b872af79795bf5d26a7266c8a47a` |
| `qwen2.5-coder-14b-stats-smoke.pt` | Historical startup smoke | `016965e5d638f467fbb1b2ccc458becab1cc3129e3e30b040b131252d7ea9ab3` |

Their model revision, calibration-input hash, and actual token count are
incomplete. Do not infer model/dataset redistribution rights from this
repository's Apache-2.0 license.

## Inspect and accept an artifact

Never use unrestricted pickle loading for an untrusted `.pt` file:

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import torch

path = Path("/srv/vllm/calibration/qwen-stats.pt")
payload = torch.load(path, map_location="cpu", weights_only=True)
print("sha256:", hashlib.sha256(path.read_bytes()).hexdigest())
print("metadata:", payload.get("metadata", {}))
print("layers:", len(payload.get("layer_stats", {})))
PY
```

Structural validation is necessary but insufficient. Before production, keep
the generator/package version, model and tokenizer revisions, checkpoint and
input fingerprints, CANN/PyTorch/torch-npu/Transformers versions, command line,
artifact hash and size, and the matched correctness, long-context quality,
throughput, latency, and HBM evidence. Regenerate when weights, tokenizer,
query shape, RoPE parameters, corpus, or generator algorithm changes.

Changing only the KV budget or runtime score/copy chunk sizes does not
mechanically require regeneration, but the changed policy still requires its
own quality and performance acceptance.
