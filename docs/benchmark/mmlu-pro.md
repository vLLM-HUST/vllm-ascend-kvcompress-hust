# MMLU-Pro preparation and evaluation

English | [简体中文](mmlu-pro.zh.md)

No NPU or host package is needed for client installation, dataset download or
freezing tokenizer prompts. Model inference requires an independently started
service; this wrapper never starts a model server.

## Frozen inputs

- Client: `vLLM-HUST/hust-eval-client-mmlu-pro` commit `b6215bf76808b1e9d351295e2ecb401542317372`.
- Dataset: `TIGER-Lab/MMLU-Pro`, revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`.
  Test: 12,032 rows; validation: 70 rows for subject-matched five-shot examples.
- Reference: `TIGER-AI-Lab/MMLU-Pro` commit `f418b116db00b065c2aea046518d8fcf74d39872`.
- Tokenizer: transformers 5.14.1, pyarrow 19.0.1, Jinja2 3.1.6. Use the
  **exact served tokenizer/chat template**; record its hashes. Preparation uses
  no weights and does not calibrate the compression method on test data.

## CPU preparation

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

Downloads default to `.benchmarks/pujiang/`; `pujiang` denotes evaluations
designated by Pujiang Laboratory and is independent of plugin releases.
Both parquet files are checked
against fixed upstream SHA256 values, then the client validates the 12,032/70
row counts and upstream prompt/scorer references. `data-manifest.json` records
hashes, sizes, revisions and download sources. On HF connection failure the
script tries ModelScope's mirror; it accepts it only when bytes match the
same fixed upstream hash. Mirror `master` is not trusted as a revision.

The client freezes all 12,032 test cases, five-shot prompts, token IDs,
reference answers, 32,768-token context and 2,048-token output budget with
thinking disabled. It hashes the plan and scorer and refuses existing output
plans. A plan frozen before NPU allocation has unknown server provenance.
When available, attach an operator-supplied **actual** runtime contract with
`prepare --runtime-contract /path/to/runtime-contract.json` into a new plan;
never generate server evidence from client packages.

## Evaluation after NPU allocation

First satisfy the [host requirements](../development/environment-installation.md).
Use a base URL **without `/v1`**; the service must accept token-ID prompts at
`/v1/completions`. Supply credentials through `--api-key-env MODEL_API_KEY`.
The same frozen plan must be used for both arms of each paired run.

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

Every run retains per-request replies and writes `summary.json`. Transport
failures remain incorrect in the denominator and cause nonzero exit status;
unparsed answers are incorrect rather than randomly guessed. An interrupted
run may use `--resume`; summaries are preserved under new names. Partial
summaries are explicitly not a complete dataset score. Keep errors and
length-limited outputs. One-item smoke is not a benchmark result.

Report overall/per-category accuracy, failures, invalid/truncated answers and
client TTFT/E2E/tokens; capture server throughput, peak HBM and actual compression
transactions independently. The 2,048-token output budget can meet the example's
compression guard, but only runtime telemetry proves compression occurred.

## Paired lifecycle runner

For the fixed HUST stack, `scripts/kvcompress_benchmark_host.py -m
vllm.entrypoints.cli.main serve ...` applies ABI compatibility bridges explicitly
to **both** arms, including spawned workers. Invoke it through
`scripts/kvcompress_ascend_env.sh`. It does not enable compression; B0 keeps
`VLLM_ASCEND_KVCOMPRESS_ENABLED=0`, while B1 uses `1` and a configuration path.
This includes forwarding the newer core's Mamba `prefill_end` boundary while
preserving Ascend's external-KV reservation. Keep the same launcher and host
bridges in both arms; do not compare an adapted B1 with a broken B0.

The [short-context Qwen3.5 candidate](../../examples/mmlu-pro-qwen35-triattention.json)
is an activation diagnostic, not a deployment recommendation. It requires
independent quality validation and uses a 1,152-token KV target and 128-token
recomputation span, with the existing
2,048-token requested-output gate and protected windows. It is a separate
workload configuration, not a change to the general Qwen3.5 example. Check the
actual full-attention page size: this candidate requires 1,152-token pages.
Choose the budget before reading test accuracy. The general example's
5,632-token trigger can exceed every prompt plus its output budget; an enabled
plugin then provides no active-compression quality evidence. Verify scheduler
and worker commit acknowledgements in B1 logs rather than inferring activity
from configuration alone. Built-in calibration text is disjoint integration
material; its deployment representativeness remains unestablished.

Create an executable launcher accepting `b0` or `b1`, then run:

```bash
python scripts/kvcompress_mmlu_pair.py \
  --launcher /path/to/launch-server.sh \
  --client-python "$EVAL_PYTHON" \
  --plan .benchmarks/pujiang/plan \
  --output /path/to/external/benchmark-submission \
  --served-model YOUR_SERVED_MODEL --concurrency 16 --pairs 3
```

The runner restarts the service for each arm, runs identical 16-case warmups,
then evaluates all tasks. Concurrency 16 is recorded as a deviation from the
frozen client contract's concurrency 1. Each arm retains lifecycle metadata,
server/client logs, raw responses, accuracy summaries, Prometheus snapshots
and sampled NPU/HBM inventory. Sampled HBM is an observed peak, not an exact
allocator high-water mark. Startup, warmup or client failures stop the pair and
preserve its failed lifecycle; rerun into a new directory after fixing the cause.
Do not resume a failed attempt into a fresh service lifecycle and call it an
independent uninterrupted run. The runner stops its service on failure or
interruption. Keep other NPU workloads idle during measurement.
