# Benchmarking and Result Interpretation

English | [简体中文](benchmarking.zh.md)

KV-cache compression must be evaluated as a capacity, latency, throughput,
and quality trade-off. A lower physical KV footprint alone does not establish
an end-to-end improvement.

## Matched A/B Contract

Use the same model, revision, tokenizer, NPU, dtype, block size, KV-pool size,
execution mode, prompt set, arrival pattern, concurrency, sampling parameters,
and output length for both runs. The only intended difference is
`--kv-cache-compression-config`.

For the built-in method, report all method options, especially `kv_budget`,
`recompute_window`, `protected_recent_window`, and `score_chunk_size`. The
score chunk is a performance/memory tuning parameter: larger values reduce
kernel-launch overhead and increase temporary device memory.

Recommended vLLM-HUST Benchmark scenarios are:

- `prefix-repetition-online` for repeated-prefix cache behavior;
- `random-online` for a small synthetic online smoke workload;
- `knorm-kv-compression-longctx` for sustained long-context serving;
- `kv-pressure-online` for a simultaneous workload near the KV capacity
  boundary.

Disable prefix caching and the independent Knorm owner when isolating this
plugin. Use eager mode, block size 128, and the same `max_model_len` and
`gpu_memory_utilization` in both runs.

## Reproduction Environment

Run commands from the `vllm-hust-benchmark` checkout with the target conda
environment activated. Set these paths for the local machine:

```bash
export MODEL=/path/to/Qwen2.5-Coder-14B-Instruct
export STATS=/path/to/triattention-stats.pt
export RESULT_ROOT=/path/to/ab-results
export ASCEND_RT_VISIBLE_DEVICES=5
export VLLM_KNORM_ENABLED=0
export VLLM_ASCEND_TORCH_PREFLIGHT=0
```

Check `npu-smi info` first. The selected physical device must show no process
and its idle HBM baseline before either service is launched.

## Start the Service

Start the baseline service without a compression configuration:

```bash
vllm serve "$MODEL" \
  --enforce-eager \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --block-size 128 \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.8
```

For the enabled half of the A/B pair, stop the baseline service, confirm NPU
release, and start the otherwise identical service with:

```bash
vllm serve "$MODEL" \
  --enforce-eager \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --block-size 128 \
  --max-model-len 12288 \
  --gpu-memory-utilization 0.8 \
  --kv-cache-compression-config "{\
\"schema_version\":1,\
\"provider\":\"ascend_kvcompress\",\
\"provider_config\":{\
\"method\":\"triattention\",\
\"stats_path\":\"$STATS\",\
\"kv_budget\":2048,\
\"recompute_window\":128,\
\"protected_recent_window\":128,\
\"score_aggregation\":\"mean\",\
\"layer_aggregation\":\"mean\",\
\"score_chunk_size\":8192}}"
```

Wait for `Application startup complete` and verify
`curl -f http://127.0.0.1:8000/health` before starting a client.

## Run the Four Benchmark Clients

Set `MODE=baseline` for the compression-off service or
`MODE=triattention` for the enabled service. Run the same four commands for
both modes while the corresponding service remains running.

### prefix-repetition-online

```bash
python -m vllm_hust_benchmark.cli run prefix-repetition-online \
  --model "$MODEL" \
  --set num_prompts=4 \
  --set prefix_repetition_num_prefixes=2 \
  --set prefix_repetition_prefix_len=2304 \
  --set prefix_repetition_suffix_len=256 \
  --set prefix_repetition_output_len=64 \
  --set custom_output_len=64 \
  --set request_rate=1 \
  --set max_concurrency=1 \
  --set temperature=0 \
  --set ignore_eos=true \
  --set save_result=true \
  --set save_detailed=true \
  --set result_dir="$RESULT_ROOT/prefix-repetition-online/$MODE" \
  --set result_filename=raw.json \
  --execute
```

### random-online

```bash
python -m vllm_hust_benchmark.cli run random-online \
  --model "$MODEL" \
  --set num_prompts=2 \
  --set input_len=2560 \
  --set output_len=32 \
  --set request_rate=1 \
  --set max_concurrency=1 \
  --set temperature=0 \
  --set ignore_eos=true \
  --set save_result=true \
  --set save_detailed=true \
  --set result_dir="$RESULT_ROOT/random-online/$MODE" \
  --set result_filename=raw.json \
  --execute
```

### knorm-kv-compression-longctx

```bash
python -m vllm_hust_benchmark.cli run knorm-kv-compression-longctx \
  --model "$MODEL" \
  --set num_prompts=8 \
  --set prefix_repetition_num_prefixes=2 \
  --set prefix_repetition_prefix_len=7168 \
  --set prefix_repetition_suffix_len=1024 \
  --set prefix_repetition_output_len=128 \
  --set custom_output_len=128 \
  --set request_rate=2 \
  --set max_concurrency=4 \
  --set temperature=0 \
  --set ignore_eos=true \
  --set save_result=true \
  --set save_detailed=true \
  --set result_dir="$RESULT_ROOT/knorm-kv-compression-longctx/$MODE" \
  --set result_filename=raw.json \
  --execute
```

### kv-pressure-online

```bash
python -m vllm_hust_benchmark.cli run kv-pressure-online \
  --model "$MODEL" \
  --set num_prompts=16 \
  --set input_len=8192 \
  --set output_len=64 \
  --set request_rate=inf \
  --set max_concurrency=16 \
  --set temperature=0 \
  --set ignore_eos=true \
  --set save_result=true \
  --set save_detailed=true \
  --set result_dir="$RESULT_ROOT/kv-pressure-online/$MODE" \
  --set result_filename=raw.json \
  --execute
```

The prefix-repetition client in the validated benchmark snapshot may still use
its 256-token generic output default even when the dedicated output override is
smaller. Always compare the exact `input_lens` and `output_lens` persisted in
both raw JSON files; reject a pair if they differ.

After the fourth client, stop the service and check that port 8000 has no
listener, no vLLM process remains, and `npu-smi info` reports the selected NPU
at its pre-run idle baseline.

## Required Metrics

Record at least:

- completed and failed requests;
- exact input/output token counts and concurrency;
- request, output-token, and total-token throughput;
- mean and P99 TTFT, TPOT, and ITL;
- source, destination, and released KV blocks per compression commit;
- peak active KV-pool usage, running requests, and waiting requests;
- model-specific quality or task-accuracy results for lossy compression;
- NPU selection, process exit, and post-run resource release.

## Memory Interpretation

vLLM preallocates the KV pool at startup. Returning physical blocks to the
scheduler increases reusable capacity but normally does not reduce the
process-level HBM allocation shown by `npu-smi`. Report active KV-pool usage
and committed physical tokens or blocks as the compression-capacity signal.
Do not claim allocator-level HBM savings from an unchanged preallocated pool.

For block size 128, compressing an 8192-token prompt from 64 blocks to 16
blocks retains 2048 physical tokens and releases 48 blocks, a 75% prompt-KV
reduction. Semantic positions remain at 8192; only physical cache occupancy is
reduced.

## Publication Policy

Keep stable methodology and carefully scoped summary results in public docs.
Put raw JSON, full commands with machine paths, failed attempts, profiler logs,
and tuning notes under git-ignored `docs/dev/`. Treat a single run as
engineering evidence rather than a universal performance guarantee.
