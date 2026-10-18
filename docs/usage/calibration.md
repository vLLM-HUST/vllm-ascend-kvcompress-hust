# Calibrate query statistics

English | [简体中文](calibration.zh.md) | [User guide](README.md)

TriAttention scores cached keys using query statistics bound to the model.
The statistics file is generated per model, tokenizer, revision, and local
checkpoint manifest. Do not copy a file from another checkpoint just because
the architecture name is the same. Model weights, calibration text, and the
resulting `.pt` artifact are not included in the plugin distribution.

## Automatic first startup

Keep `auto_calibrate=true` and set `stats_path` to a missing path under a
writable directory. The enabled plugin locks the path, loads a temporary
Transformers model, creates and validates the statistics, writes the file
atomically, releases the temporary model, then loads the serving model.
Subsequent starts reuse the artifact after checking its binding. An existing
invalid or mismatched file causes startup to fail; automatic mode does not
silently overwrite it.

For Qwen3.5-35B-A3B BF16, expose two NPUs; automatic calibration distributes
the temporary model across them. With local model files, set
`calibration_local_files_only=true`. The built-in text supports initial
integration checks. For real deployments, set `calibration_input_path` to a
licensed, representative UTF-8 corpus and record its hash.

## Pre-generate an artifact

Pre-generation avoids a long first service startup and allows a read-only
service account. For Qwen3.5-35B-A3B:

```bash
vllm-ascend-kvcompress-calibrate \
  --model /path/to/Qwen3.5-35B-A3B \
  --input /path/to/licensed-calibration.txt \
  --output /path/to/qwen35-stats.pt \
  --max-length 4096 \
  --device-map auto \
  --dtype bfloat16 \
  --attn-implementation eager \
  --local-files-only
```

Point `stats_path` at this file and set `auto_calibrate=false` if startup
must never generate data. Keep `--model` and service model identity consistent.
Create a new candidate path when changing checkpoint or corpus. The CLI
`--force` flag replaces an existing file; use it only when that replacement
is intentional. Review the [artifact format and binding rules](../development/calibration-artifacts.md)
before distributing a calibrated deployment, and run an application quality
check after changing the corpus.
