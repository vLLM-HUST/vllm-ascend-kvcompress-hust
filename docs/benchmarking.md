# Version 0.3 Acceptance Protocol

English | [简体中文](benchmarking.zh.md)

This protocol applies to the current independently packaged plugin. The old
fork-only `--kv-cache-compression-config` flag no longer exists and must not be
used. Baseline and compression runs must use identical host commits, model,
artifact, prompts, request order, warm-up, device, and environment.

## 1. Freeze provenance

Record before each run:

- vLLM-HUST, vLLM-Ascend-HUST, Extension Manager, plugin, Triton-Ascend,
  PyTorch, torch-npu, CANN, driver, and firmware versions;
- model path/ID and immutable revision, tokenizer revision, dtype, RoPE config,
  and model-file hashes;
- calibration artifact path, SHA-256, generator revision, input provenance,
  and metadata;
- NPU model/ID, available HBM, power/frequency mode, and other processes;
- plugin JSON, full launch arguments, benchmark command, prompt-set hash, and
  raw-output directory.

Do not publish rows whose provenance cannot be reconstructed.

## 2. Validate package lifecycle

From a clean virtual environment, install the wheel and manager, then run:

```bash
vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /absolute/path/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension status org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension forget org.vllm-hust.ascend-kvcompress
python -m pip uninstall vllm-ascend-kvcompress-hust
```

Also verify that importing the plugin while disabled is inert, enable/disable
takes effect after process restart, and uninstall removes manager discovery.

## 3. Kernel numerical acceptance

Run the NPU smoke test against a PyTorch reference for direct paged scoring,
fused aggregation, selection, and overlapping K/V materialization. Include
empty/small tails, non-power-of-two lengths, all supported dtypes, repeated
compression, and multiple layer/head shapes. Report maximum absolute and
relative error, not only pass/fail.

```bash
python tests/run_npu_kernel_smoke.py
python tests/run_npu_kernel_benchmark.py
```

## 4. Service correctness and quality

Test sequences below, at, and above the first and repeated compression
thresholds. Assert:

- no crash, invalid slot, leaked block, or cross-request contamination;
- semantic positions remain monotonic after one and several transactions;
- block counts fall at the next scheduling barrier and return after request
  completion/cancellation;
- deterministic outputs under a deterministic decoding configuration; and
- an agreed long-context quality suite stays within its predeclared threshold.

Record exact task names, sample count, seeds, scoring code revision, baseline
score, compressed score, and permitted delta. A smoke prompt is not a quality
acceptance test.

## 5. Matched performance matrix

Use at least three prompt-length/concurrency cells that force compression, for
example 8K/c=1, 32K/c=4, and 64K/c=8 where the model supports them. Run at
least one warm-up and three measured repetitions per cell, alternating
baseline and compression order.

Report median and range for:

- input, output, and total token throughput;
- TTFT and TPOT p50/p90/p99;
- end-to-end latency p50/p90/p99;
- peak and steady-state HBM plus cache block usage;
- compression count and compression-time p50/p90/p99; and
- OOM/rejection/failure counts.

Baseline means the plugin is disabled through Extension Manager and the host
is restarted. Compression means the same host command is launched through:

```bash
export VLLM_PLUGINS=ascend,ascend_kvcompress
vllm-hust-ext run -- vllm serve /path/to/model \
  --block-size 128 --no-enable-prefix-caching --no-async-scheduling
```

Do not combine results with prefix caching, speculative decoding, KV transfer,
quantized KV, BidKV, or removed host optimizations. They are unsupported, not
independent tuning variables.

## 6. Publication gate

A release may claim manager/package compatibility after the lifecycle and CPU
suite pass. It may claim NPU support only after kernel and service correctness
pass on the declared snapshots. It may claim a throughput/HBM improvement only
when the matched matrix and raw provenance are published. Historical 0.2
measurements are not a substitute for this gate.
