# MMLU-Pro preparation and evaluation

English | [简体中文](mmlu-pro.zh.md)

See the [unified Pujiang guide](pujiang.md) for installation, data, smoke, pairs and performance commands.
B0/B1 share the pinned HUST stack and host environment; B0 disables compression and B1 enables it with the selected configuration.

Fixed client: `b6215bf76808b1e9d351295e2ecb401542317372`.
Fixed data: `b189ec765aa7ed75c8acfea42df31fdae71f97be` (test 12032, validation 70).
Reference: `f418b116db00b065c2aea046518d8fcf74d39872`.
Retain five-shot, thinking-disabled token-ID prompts, 32768 context, 2048 output budget and upstream local answer extraction.
Unparsed/out-of-range answers and request failures are incorrect, never random guesses.

```bash
scripts/run_benchmark.sh mmlu-pro download-client
scripts/set_env.sh install client --dataset mmlu-pro --python /path/to/python3.11
scripts/run_benchmark.sh mmlu-pro download
scripts/run_benchmark.sh mmlu-pro prepare --model-path /path/to/model
scripts/run_benchmark.sh mmlu-pro run --arm b0 \
  --base-url http://127.0.0.1:18180 --served-model qwen35-eval \
  --smoke --output ../.benchmarks/results/mmlu/b0
```

B1 uses the same frozen sample and a new output directory. `--sample-size` defaults to 100; `--resample` creates another version.
Omit `--smoke` for the full selected scope. Clients are separate from server environments; both arms use the same compatibility bridge and server arguments.
Default performance is a separate 200-case, 256-output-token `vllm bench serve` chat workload, not accuracy token-ID replay.
Both arms must use the same bench client; actual compression requires server commit/ack evidence.

Small-file hashes do not verify full weights. Smoke is not a formal score. Unknown runtime provenance remains unknown.
