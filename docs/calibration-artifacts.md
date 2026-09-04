# TriAttention Calibration Artifacts

English | [简体中文](calibration-artifacts.zh.md)

TriAttention requires model-specific query statistics before it can score and
select cached keys. The `.pt` files under `artifacts/` contain those statistics.
They are generated offline, selected at service startup through `stats_path`,
and treated as read-only for the lifetime of the service.

Two legacy calibration files are committed for development and historical
reproducibility, but they are excluded from the Python wheel. Their generation
provenance is incomplete. Operators must generate or securely distribute a
fully traced artifact for the exact model revision they serve and must not
infer model/dataset redistribution rights from this repository's license.

## Committed Legacy Artifacts

The repository contains these two Qwen2.5-Coder-14B-Instruct artifacts:

| File | Intended use | Size | SHA-256 |
| --- | --- | ---: | --- |
| `artifacts/qwen2.5-coder-14b-stats.pt` | Validated serving and benchmark artifact | 3,285,943 bytes | `d1f43bf5de3ab7d464a0a906060bbc15b828b872af79795bf5d26a7266c8a47a` |
| `artifacts/qwen2.5-coder-14b-stats-smoke.pt` | Fast startup/integration smoke artifact; not a quality reference | 3,320,539 bytes | `016965e5d638f467fbb1b2ccc458becab1cc3129e3e30b040b131252d7ea9ab3` |

Both files use the flat payload format emitted by the reference TriAttention
calibration script. Each contains 48 layers x 40 query heads, with 64 frequency
components for a head dimension of 128. At runtime, the 40 query-head rows are
grouped into eight KV heads for the model's GQA layout.

The `-smoke` suffix is only a file-naming convention. The loader does not assign
it different semantics and never selects it automatically. Whichever file is
passed as `stats_path` becomes the active scoring artifact.

The full artifact above is the one identified by the SHA-256 recorded in the
current benchmark evidence. The existing payloads do not record the model
revision, calibration-input hash, or actual token count, so their complete
generation provenance cannot be reconstructed from the `.pt` files alone.

## What the Artifact Does

Calibration summarizes unrotated query vectors in the complex frequency domain
for every transformer layer and query head:

| Field | Meaning | Runtime use |
| --- | --- | --- |
| `q_mean_real` | Real part of the mean complex query at each RoPE frequency | Estimates alignment between a cached post-RoPE key and future queries |
| `q_mean_imag` | Imaginary part of the same mean | Completes the phase-sensitive trigonometric score |
| `q_abs_mean` | Mean magnitude of the complex query | Supplies the magnitude-regression correction term |
| `freq_scale_sq` | Optional squared frequency scale | Required for scaled RoPE and used as a positive score multiplier |
| `inv_freq` | Optional exact inverse RoPE frequencies | Required per layer for scaled or non-default RoPE |
| `metadata` | Shape, RoPE, and generation descriptors | Used for fail-closed compatibility checks when fields are present |

The artifact does not contain model weights, a KV cache, prompts, generated
tokens, or request state. It contains aggregate floating-point statistics
derived from calibration text. It is small enough to load on CPU at startup and
is then converted to FP32 device tensors.

During cache binding, the implementation:

1. loads the payload on CPU with `torch.load(..., weights_only=True)`;
2. validates layer coverage, head shape, head dimension, supported model type,
   RoPE style, RoPE theta, and required scaled-RoPE fields;
3. groups query-head rows according to the model's GQA ratio;
4. transfers statistics to the NPU as FP32;
5. derives frequency scales and the magnitude-correction coefficient; and
6. precomputes future-offset cosine/sine means used by the Ascend scoring
   kernel.

At each compression transaction, those fixed tensors score the request's
current post-RoPE K cache. The artifact influences which tokens survive; it
does not change the scheduler budget, write KV data itself, or alter semantic
RoPE positions.

## Generation Source

This plugin consumes calibration artifacts but does not duplicate the model
instrumentation used to generate them. Generate the flat format with
`scripts/calibrate.py` from a matching checkout of the
[reference TriAttention repository](https://github.com/WeianMao/triattention).

Keep the generator revision with the artifact provenance. The current files
match the payload schema produced by that script: it loads the Hugging Face
model, runs one forward pass over plain text, captures every attention layer's
query projection, inverts RoPE, converts pairs to complex values, reduces over
the token dimension, and saves `metadata` plus `stats`.

## Calibration Input

Use a UTF-8 plain-text file containing natural, coherent text:

- provide enough text to approach the chosen `--max-length`;
- include representative language and structure, but keep calibration data
  separate from benchmark/evaluation prompts;
- avoid corrupted text, empty input, and long mechanically repeated loops; and
- record the input SHA-256 and the tokenizer used to produce the token stream.

Calibration is normally domain-tolerant, but it is not quality-free. A new
artifact must pass the target model's quality guardrail before production use.

The generator holds model weights and captured query tensors during one forward
pass. Start with a short smoke run to confirm compatibility, then run the
production length on a device with sufficient memory.

## Generate a Smoke Artifact

From this plugin repository, point `TRIATTENTION_CHECKOUT` at the reference
checkout and select an idle NPU:

```bash
export TRIATTENTION_CHECKOUT=/path/to/triattention
export MODEL=/path/to/Qwen2.5-Coder-14B-Instruct
export CALIBRATION_TEXT=/path/to/calibration.txt
export ASCEND_RT_VISIBLE_DEVICES=5

python "$TRIATTENTION_CHECKOUT/scripts/calibrate.py" \
  --model "$MODEL" \
  --input "$CALIBRATION_TEXT" \
  --output artifacts/qwen2.5-coder-14b-stats-smoke.pt \
  --max-length 2048 \
  --device npu \
  --attn-implementation eager
```

The 2048-token limit is a recommended smoke convention, not a property encoded
in the output file. Confirm the script reports the expected tokenized length
and all expected layer/head entries before using the artifact.

## Generate a Production Artifact

Use the same model, tokenizer, generator revision, attention implementation,
and device software stack. Supply a longer independent calibration text:

```bash
python "$TRIATTENTION_CHECKOUT/scripts/calibrate.py" \
  --model "$MODEL" \
  --input "$CALIBRATION_TEXT" \
  --output artifacts/qwen2.5-coder-14b-stats.pt \
  --max-length 32768 \
  --device npu \
  --attn-implementation eager
```

The script truncates to `--max-length`; it does not pad a shorter input. Treat
the output as a candidate until structural validation, service startup, matched
A/B capacity/performance tests, and the model-specific quality check pass.

Do not overwrite the last validated artifact before the candidate has passed.
Write to a temporary artifact name, record hashes, validate it, and then update
the deployed `stats_path` atomically.

## Default and Scaled RoPE

The two current Qwen2.5 artifacts declare `rope_style=half` and
`rope_type=default`. They omit `freq_scale_sq` and exact `inv_freq`, so the
loader uses a unit frequency scale and derives standard inverse frequencies
from the serving model's `rope_theta=1000000`.

The reference flat generator shown above is not sufficient for YaRN, LongRoPE,
or another scaled/non-default RoPE configuration. For those models, the
artifact must use the structured `layer_stats` format and include exact
per-layer `inv_freq` and positive `freq_scale_sq`. The provider deliberately
rejects a scaled-RoPE artifact when either value is missing; do not substitute
values from a different model or silently fall back to the default formula.

## Accepted Payload Schemas

### Flat reference format

```python
{
    "metadata": {
        "head_dim": 128,
        "rope_style": "half",
        "rope_type": "default",
        # Additional provenance and model-shape fields are recommended.
    },
    "stats": {
        "layer00_head00": {
            "q_mean_real": Tensor[64],
            "q_mean_imag": Tensor[64],
            "q_abs_mean": Tensor[64],
        },
        # Every head of every layer must be present and contiguous.
    },
}
```

### Structured format

```python
{
    "metadata": {
        "model_type": "qwen2",
        "num_layers": 48,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rope_style": "half",
        "rope_type": "default",
        "rope_theta": 1000000.0,
    },
    "layer_stats": {
        0: {
            "q_mean_real": Tensor[40, 64],
            "q_mean_imag": Tensor[40, 64],
            "q_abs_mean": Tensor[40, 64],
            "freq_scale_sq": Tensor[40, 64],  # scaled RoPE when applicable
            "inv_freq": Tensor[64],           # exact per-layer values
        },
        # Every model layer must be present.
    },
}
```

Query statistics may have either `num_attention_heads` or `num_kv_heads` rows.
The latter is expanded across each GQA query group. Complex query means may also
be supplied as `q_mean_complex`, either as a complex tensor or a real tensor
whose final dimension is two.

## Inspect and Validate an Artifact

Never inspect an untrusted artifact with unrestricted pickle loading. Use
`weights_only=True`:

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import torch

path = Path("artifacts/qwen2.5-coder-14b-stats.pt")
payload = torch.load(path, map_location="cpu", weights_only=True)
stats = payload.get("stats", {})
print("sha256:", hashlib.sha256(path.read_bytes()).hexdigest())
print("metadata:", payload.get("metadata", {}))
print("flat entries:", len(stats))
print("first keys:", list(stats)[:3])
PY
```

For the current Qwen2.5-Coder-14B shape, run the plugin's full loader and
compatibility validation:

```bash
python - <<'PY'
from pathlib import Path
from vllm_ascend_kvcompress.methods.base import ModelShape
from vllm_ascend_kvcompress.methods.triattention.stats import CalibrationStats

path = Path("artifacts/qwen2.5-coder-14b-stats.pt")
model = ModelShape(
    model_type="qwen2",
    num_layers=48,
    num_attention_heads=40,
    num_kv_heads=8,
    head_dim=128,
    rope_theta=1_000_000.0,
    has_rope_scaling=False,
)
reasons = CalibrationStats.load(path).validate(model)
if reasons:
    raise SystemExit("\n".join(reasons))
print("calibration artifact is structurally compatible")
PY
```

Structural compatibility is necessary but does not prove that an artifact was
generated from the intended weights or that its token selection preserves
quality. The final check is a real service startup followed by matched quality
and capacity tests.

## Provenance Checklist

Store a dated manifest under `docs/dev/` next to development evidence. Record:

- model path or repository ID, immutable revision, and model-config hash;
- tokenizer revision and `tokenizer.json` hash;
- reference TriAttention generator revision and local patch state;
- calibration-input source, license, SHA-256, and actual token count;
- device type, PyTorch, torch-npu, Transformers, dtype, attention
  implementation, and command line;
- artifact filename, byte size, SHA-256, schema, layer/head/frequency counts;
- structural-validation output and the quality/benchmark evidence that approved
  it.

The `.pt` file remains under `artifacts/`; the dated provenance and experimental
logs remain under `docs/dev/`. If an artifact is distributed outside the local
workspace, distribute its manifest and checksum with it.

## When to Regenerate

Generate and revalidate a new artifact when any of these changes:

- model weights, fine-tune, merge, quantized model implementation, or revision;
- tokenizer or prompt-processing policy used for calibration;
- number of layers, query/KV heads, head dimension, RoPE style/theta/scaling;
- calibration corpus or generation algorithm; or
- a quality regression suggests the current statistics are not representative.

Changing only `kv_budget`, `recompute_window`, `protected_recent_window`,
`score_chunk_size`, or `score_layer_stride` does not mechanically require new
statistics, but every new policy still requires quality and performance
validation.
