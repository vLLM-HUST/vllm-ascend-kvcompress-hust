# Pujiang datasets: preparation, smoke and B0/B1 comparison

English | [简体中文](pujiang.zh.md) | [Protocol index](README.md)

The user entry points are `scripts/run_benchmark.sh` and `scripts/set_env.sh`; their absolute paths work from any directory.
MMLU-Pro, HLE-Verified and FrontierScience delegate to the [project-designated clients](https://vllm-hust.sage.org.ai/dataset-introductions.html).
SWE-bench-Pro and Terminal-Bench files remain empty; selecting them fails explicitly. No external evaluator source is patched or copied.

## Sources and client environments

Run examples from the plugin repository root. Source checkouts default to `../pujiang-benchmark/`,
relative to that root. Override with `--client-root` consistently.
Client environments are separate from both servers: workspace `.envs/kvcompress-mmlu-pro`,
`.envs/kvcompress-hle-verified`, and `.envs/kvcompress-frontierscience`.

```bash
scripts/run_benchmark.sh mmlu-pro download-client
scripts/run_benchmark.sh hle-verified download-client
scripts/run_benchmark.sh frontier-science download-client
scripts/set_env.sh install client --dataset mmlu-pro --python /path/to/python3.11
scripts/set_env.sh install client --dataset hle-verified --python /path/to/python3.11
scripts/set_env.sh install client --dataset frontier-science --python /path/to/python3.11
scripts/set_env.sh check client --dataset mmlu-pro
```

The installer uses verified local source. MMLU-Pro is installed with its `[prepare]` extra,
so preparation dependencies come from the external client’s `pyproject.toml`; HLE/Frontier use the base package.
MMLU environment checks validate these declared dependencies and run `pip check`.
Dependency versions without upstream pins are resolved by pip; actual installed versions are recorded in `kvcompress-client.json`. Matching existing environments are reused; checkouts must be clean and at the fixed commit.
Mismatches fail without resetting or overwriting. Existing Git object caches may be reused.
Failed setup directories remain for diagnosis; retry with a new `--prefix`.
For a custom client environment, pass `--client-python /path/to/env/bin/python` to evaluation commands.

| Dataset | Repository | Fixed commit |
| --- | --- | --- |
| MMLU-Pro | `vLLM-HUST/hust-eval-client-mmlu-pro` | `b6215bf76808b1e9d351295e2ecb401542317372` |
| HLE-Verified | `vLLM-HUST/hust-eval-client-hle-verified` | `534c22cd2d826bed2e8cbcf674bbebf4c3004bc1` |
| FrontierScience | `vLLM-HUST/hust-eval-client-frontierscience` | `ff7b4315876258b5d0ac9b9413ef79ae27146fa4` |

## Server environment and plugin modes

B0 and B1 use the same [pinned HUST stack](../../requirements/ascend-host.json),
Python environment, toolchain and compatibility bridge. B0 disables the compression plugin;
B1 enables it with an explicit configuration. Both default to `../.envs/kvcompress-host`.
The plugin package remains installed in the shared environment in both modes.
Before installation, prepare the existing HUST base and SDK and select
`KVCOMPRESS_BASE_PYTHON` as described in the [environment guide](../development/environment-installation.md).

```bash
scripts/set_env.sh install b1
scripts/set_env.sh check b1
# b0 is an alias for installing/checking the same host environment.
scripts/set_env.sh check b0
```

See [environment installation](../development/environment-installation.md) for fresh setup or reuse options.
`install/check b0` and `install/check b1` accept the same installer options, including `--prefix`.
No second server environment is needed. Both `run` modes load the same host/toolchain launcher:

```bash
scripts/set_env.sh run b0 -- python scripts/env/benchmark_host.py \
  -m vllm.entrypoints.cli.main serve /path/to/model --served-model-name qwen35-eval
VLLM_ASCEND_KVCOMPRESS_CONFIG=examples/qwen3.5-35b-a3b-triattention.json \
  scripts/set_env.sh run b1 -- python scripts/env/benchmark_host.py \
  -m vllm.entrypoints.cli.main serve /path/to/model --served-model-name qwen35-eval
```

Use identical server arguments and device settings. B0 clears inherited compression/manager
configuration and loads only the Ascend plugin. B1 requires an existing configuration file,
explicitly enables compression and loads Ascend plus the compression plugin.
Both modes clear inherited vLLM settings except `VLLM_API_KEY`; pass server settings as CLI arguments.

## Data and immutable plans

Data and plans default to `../.benchmarks/pujiang/<dataset>/`; local results default to
`../.benchmarks/results/<dataset>/<action>-<arm-or-pair>/`. Defaults are anchored to the plugin
repository, independently of the shell working directory; explicit CLI paths are relative to the shell working directory.
Existing data moved to `../.benchmarks/` can be selected using `--data-root` and `--plan`;
existing result receipts retain their recorded historical paths and are not rewritten. `download-client` acquires code; `download` acquires fixed data.
Preparation reads tokenizer/model small files, not full weights. Evaluation questions are not compression calibration input.

```bash
scripts/run_benchmark.sh mmlu-pro download
scripts/run_benchmark.sh mmlu-pro prepare --model-path /path/to/model
scripts/run_benchmark.sh hle-verified download
scripts/run_benchmark.sh hle-verified prepare --model-path /path/to/model \
  --solver-config examples/benchmark/hle-verified-solver.json \
  --judge-config examples/benchmark/hle-verified-judge.json
scripts/run_benchmark.sh frontier-science download
scripts/run_benchmark.sh frontier-science prepare --model-path /path/to/model \
  --solver-config examples/benchmark/frontier-science-solver.json \
  --judge-config examples/benchmark/frontier-science-judge.json
```

**Edit the JSON examples first**: actual endpoints, served names, model artifacts, budgets and key environment variable names.
Solver examples use port18180; the independent fixed Judge uses port18181 and must already be running.
Keys stay in environment variables. HLE images require a multimodal solver; Judge requires Structured Outputs.
The default concurrency is 1. HLE keeps `strict_json`; Frontier keeps `strict_final` and Appendix B prompts.
Gold/Revision and Olympiad/Research remain separate; HLE excludes Uncertain by default.
Research uses0–10 scores and success at≥7. Upstream unresolved/denominator rules remain unchanged.
MMLU retains all five validation examples and local extraction; unparsed answers are incorrect, never random guesses.

Existing plans cannot be overwritten. Use separate `--plan` paths when arm endpoints, aliases or solver configs differ.
Pairs check model small files, task identities, request budgets and the fixed Judge.
Sequential servers using the same address/name and request settings may share a plan.
`--runtime-contract` and `--judge-runtime-contract` bind actual owner-provided receipts.
Managed pairs record the base identity, plugin activation and launch settings automatically.
External-service comparisons without complete server receipts have `runtime_verification.status=unverified`;
complete matching receipts give `verified`. Known base or launch-setting differences are rejected.
To capture an external receipt, run `scripts/env/runtime_identity.py` through the same `set_env.sh run`
mode and environment used for the server, and add a `server` object with `model_path`, `served_model`,
`server_args` and `server_env` matching the launch. Pass it using `--runtime-contract`; the client cannot
independently attest an external process’s settings.

## Frozen random smoke

```bash
scripts/run_benchmark.sh mmlu-pro run --arm b0 \
  --base-url http://127.0.0.1:18180 --served-model qwen35-eval \
  --smoke --output ../.benchmarks/results/mmlu/b0
scripts/run_benchmark.sh mmlu-pro run --arm b1 \
  --base-url http://127.0.0.1:18180 --served-model qwen35-eval \
  --smoke --output ../.benchmarks/results/mmlu/b1
scripts/run_benchmark.sh mmlu-pro compare \
  --baseline ../.benchmarks/results/mmlu/b0 --candidate ../.benchmarks/results/mmlu/b1 \
  --output ../.benchmarks/results/mmlu/comparison
```

These connect to existing services; restart between arms yourself. URLs exclude `/v1`.
HLE/Frontier use the same run flags; their endpoint/name must match the frozen solver config.
The adapter invokes upstream solver, independent Judge and summary, retaining all errors and truncations.

`--smoke` randomly selects up to 100 cases per split/track; change the count using `--sample-size N`.
The initial seed is 17. MMLU samples test; HLE samples Gold/Revision separately; Frontier samples Olympiad/Research separately.
Default Frontier smoke therefore selects 100+60 cases and remains labeled smoke. Execution retains frozen plan order.
Versioned `samples/selection-*.json` plus `current.json` preserve the IDs, source identity, seed and hashes.
Use a common `--selection-dir` to override storage. Subsequent arms/optimizations reuse the same selection.
Changes to source, size or explicit seed require `--resample`.

```bash
scripts/run_benchmark.sh mmlu-pro run --arm b0 \
  --base-url http://127.0.0.1:18180 --served-model qwen35-eval \
  --smoke --sample-size 50 --resample --seed 23 \
  --output ../.benchmarks/results/mmlu/b0-new-sample
```

Resampling creates a new version and preserves old files. Without an explicit seed it records a new random seed.
Do not resample between paired arms. Omitting `--smoke` evaluates the full selected scope; `--track` explicitly narrows tracks.
External-service `--resume` requires identical plan/selection and retains prior reports. Resuming is not an independent cold lifecycle.

## Managed cold pairs

Edit `examples/benchmark/<dataset>-pair.json`: model path, devices, arguments and B1 plugin config.
File paths in the JSON resolve relative to that JSON: the supplied examples use
`../../../.benchmarks/pujiang/<dataset>/plan` to reach the sibling data directory.
Default source/client paths remain relative to the plugin repository parent.
`server_args` is an argument array. `server_env` is for devices/thread settings, not secret values.
Both arms must use the same host prefix, model alias, arguments and effective server environment;
only B1 supplies `plugin_config`. The supplied examples set one shared `host_prefix`, resolved relative to the JSON file.
The default bench client also uses this prefix.
Runtime preflight verifies base packages, host configuration, bridge and plugin modes before starting either server.
Each arm rechecks its receipt before launch; a changed runtime aborts the pair.

```bash
scripts/run_benchmark.sh mmlu-pro pair \
  --config examples/benchmark/mmlu-pro-pair.json \
  --smoke --pairs 1 --with-bench --output ../.benchmarks/results/mmlu-pair
```

Default: one pair; use `--pairs 3` for repeats. Inputs are frozen before either service starts.
Each arm gets a new service, health wait, quality evaluation and optional independent bench. Judge time is excluded from performance.
Healthy occupied endpoints fail before launch. Only owned process groups are stopped, including failure/timeout/interruption paths.
Results include `pair-N/{b0,b1}/runtime.json`, `server.log`, `lifecycle.json`, `telemetry.jsonl.gz`,
`accuracy/`, optional `bench/`, and `pair-N/comparison/comparison.json`.
No extra quality warmup is inserted. Bench retains upstream warmup behavior: HLE warmups=0,
with vLLM's initial test request outside measured N. This is not a cold-cache claim.

## Independent performance

```bash
scripts/run_benchmark.sh hle-verified bench --arm b1 \
  --base-url http://127.0.0.1:18180 --served-model qwen35-eval \
  --model-path /path/to/model --smoke --server-log /path/to/actual-server.log \
  --output ../.benchmarks/results/hle-bench
```

The default `scripts/env/bench_vllm.sh` uses one fixed B1 vLLM **client** against both servers, without client-side compression.
Override with `--bench-vllm-bin`. Capabilities, including HLE’s `custom_image` support, are checked before sending requests.
The launcher uses the host Python module, so a separate `vllm` console script is unnecessary.
Client identity uses full CLI help from `usage:` onward and normalizes generated request IDs and unordered choices while preserving option/default changes. Raw help, including startup diagnostics, is retained as `bench-client-help.txt`.
`--dry-run` exports inputs/commands and checks capabilities without sending bench requests; actual execution needs a new output directory.

Quality and performance are distinct: MMLU bench sends reconstructed text; Frontier retains server-default template/thinking;
HLE sends one user message with original images, omits the quality system prompt, explicitly disables thinking and applies the template once.
Defaults: output budget 256, temperature 0, seed 0, request-rate=inf, no client concurrency cap. TTFT includes queueing.
Server max_num_seqs is not client concurrency. Quality `--concurrency` does not alter bench protocol.

Non-smoke bench defaults: MMLU 200 evenly spaced; HLE 200 text per split and all images; Frontier 160.
Smoke bench reuses the frozen random IDs, splitting HLE by modality and reporting empty groups without filling/repeating cases.
Exports bind the current verified plan; they do not bypass historical sealed-accuracy imports or source-hash checks.
HLE uses upstream image hashes/input validation and compares detailed input_lens against server `/tokenize`.
If this cannot be verified, raw values remain but estimated total-token throughput is excluded from trusted comparison.

**Verify actual compression separately.** Current Qwen examples require an output budget of 2048 for eligibility;
bench's default 256 bypasses compression. The tool never changes that gate automatically.
Explicitly configure `--output-len` and a preselected plugin config for optimization checks, recording protocol changes.
`--server-log` captures scheduler commits/worker acknowledgements in the measurement interval; absent evidence and no observed commits are distinct.
NPU/HBM telemetry is sampled and is not an exact allocator high-water mark.

Use new output directories. Preserve upstream raw JSON, requests, details, errors and hashes.
Smoke is a development check, not a formal full score. Follow [original custody](../usage/benchmark-originals.md)
for publication; private questions and answers must not become public evidence payloads.

To compare independent performance runs, use the same `compare --baseline ... --candidate ... --output ...` command with the two bench output directories. It rejects different clients, tokenizers, inputs or budgets.

HLE/Frontier concurrency and request timeouts come from the frozen Solver config; explicitly conflicting CLI values fail rather than silently overriding the plan. MMLU defaults are 1 and 1800 seconds. MMLU resume archives previous Solver outputs before upstream writes. Upstream complete refers to the selected plan; the adapter report determines smoke scope.

## Local Judge in a separate phase

When Solver and a local Judge share devices, generate answers before releasing the Solver
and starting the fixed Judge. For HLE/Frontier, `run --stage solver` saves answers with
status `awaiting_judge`. Then use `run --stage judge --resume` with the same plan,
selection, arm, runtime receipt and output directory. The Judge phase verifies answer
file hashes and never calls the Solver. Run the existing `compare` after grading.
The default `--stage all` retains immediate grading. MMLU uses local scoring and does
not accept stage options.

Use `pair --stage solver` to defer grading without requiring an online Judge or producing
premature accuracy comparisons. Once both servers stop, grade `pair-1/b0/accuracy` and
`pair-1/b1/accuracy` separately, passing the original `--plan`, `--arm`,
`--runtime-contract`, Solver URL/model and smoke options. Compare the two arms afterward;
missing or failed grading prevents comparison. Disable compression for the fixed Judge
and keep its model, request budgets and scoring policies identical across arms.
Local Judge scores support development checks and are not interchangeable with formal
scores produced by another Judge model.
