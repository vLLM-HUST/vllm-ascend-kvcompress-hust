# KV Compression Test and Acceptance Requirements

English | [简体中文](kv-compress-test-requirements.zh.md)

This document specializes the controlled **vLLM-HUST Standard Delivery Test
Plan V4.6** for this plugin. The source PDF is
`vllm-hust-benchmark/docs/assets/vLLM-HUST标准交付测试方案_V4.6.pdf`
(63 pages, 980,035 bytes, SHA-256
`ab2ceacf25f9ba82da39dfd7bd457d6b620b6ad33d121aee46fa5c044f58c370`).
If this document conflicts with the controlled PDF, the PDF and the independent
evaluator's frozen run declaration take precedence.

## Claim levels

| Level | Purpose | May be called formal V4.6 acceptance? |
| --- | --- | --- |
| Unit/contract | PR checks for schema, package, host seams and numerical references | No |
| `kv-pressure-online` | Fast random-token KV pressure smoke against the current host | No; its registry entry is provisional and based on a vLLM 0.18 baseline |
| Repository commissioning | Deterministic A2/A3-shaped service checks using the generated fixture | No; the fixture is explicitly `commissioning_only` |
| Formal delivery | Independent, signed, immutable B0/B1 execution with approved LONG-PUBLIC data | Yes |

Never publish commissioning results as official throughput, quality, HBM, or
V4.6 acceptance claims.

## Frozen target and lifecycle

- Model: `Qwen/Qwen2.5-14B-Instruct`, immutable model and tokenizer revisions.
- Hardware: one Ascend 910B2 in one node; FP16 model and effective FP16 KV.
- Topology: TP=PP=DP=EP=DCP=1, unified non-PD service, FCFS, multiprocessing,
  no CPU offload or swap.
- Common server settings: block size 128, GPU memory utilization 0.85,
  prefix cache off, speculative decoding off, KV transfer off, async scheduling
  off, chunked prefill on, and no silent input truncation.
- Graph mode for A2/A3: `FULL_DECODE_ONLY`, capture sizes matching the profile;
  `--enforce-eager` is not formal A2/A3 evidence.
- Sampling: temperature 0, top-p 1, top-k -1, min-p 0, all penalties 0, n=1,
  no beam search, empty stop list, seed 0, streaming with usage, and special
  tokens enabled.
- Execute at least three independent cold-start lifecycles per B0 and B1:
  start, warm, measure, stop. Retain failures, timeouts and retries. The
  predeclared primary result is the median; up to two additional runs may be
  reported but cannot replace failed runs.

Formal V4.6 B0 is the fixed official vLLM 0.18 / matching official
vLLM-Ascend baseline declared by the controlled plan. For plugin engineering,
“B0/plugin disabled” versus “B1/plugin enabled” on the current HUST host is a
useful paired comparison, but is not a replacement for that formal B0.

## Selected delivery scenarios

### A2-LONG-FP16 — long-context quality and serving

Use at least 64 authorized LONG-PUBLIC documents with immutable provenance,
license, revision and checksum. Prompts must contain verifiable facts and an
oracle, not random tokens, and must not be truncated.

| Cell | Requests | Rendered input | Maximum output | Arrival rates | Max concurrency |
| --- | ---: | ---: | ---: | --- | ---: |
| LONG-8K | 16 | 8,192 | 512 | 0.05, 0.1, 0.2, 0.4 RPS | 4 |
| LONG-16K | 16 | 16,384 | 1,024 | 0.05, 0.1, 0.2, 0.4 RPS | 4 |

Server limits are `max_num_seqs=4`, `max_num_batched_tokens=16384`, GPU memory
utilization 0.85, and `FULL_DECODE_ONLY` capture sizes `[1,2,4]`. Per-request
timeout is 1,800 seconds and capacity-cell timeout is 7,200 seconds.

Required results include request/input/output/total throughput, TTFT/TPOT/E2E
mean/p95/p99, success rate, oracle accuracy, OOM, silent truncation, B1/B0
ratio and delta, HBM peak, KV block reduction, NPU utilization and three-run
variation. B1 quality may fall by at most one percentage point from B0.

### A3-32K-FP16 — 32K stability

- Exact total context: 30,720 rendered input tokens plus 2,048 forced output
  tokens (`ignore_eos=true`) equals 32,768.
- Closed loop, concurrency 1, seed 0, `max_num_seqs=1`,
  `max_num_batched_tokens=16384`, capture size `[1]`.
- Warm for 5 minutes, measure for 30 minutes, report six 5-minute windows;
  stop at 64 requests and require at least 24 completions.
- Per-request timeout 1,800 seconds; lifecycle timeout 3,600 seconds.
- Zero request failures, OOM, deadlock, process exit or silent truncation.
- Window throughput CV at most 5%; median TTFT/TPOT drift at most 10%; p99
  drift at most 20%.

### M3 compression-specific gate

Quality must pass and either physical per-token KV use must fall by at least
20%, or whole-system per-token cost must fall by at least 15%. Scheduler and
worker commit counts must match. Static pool allocation alone is not proof of
KV reduction; retain block-release transactions and 1-second HBM samples.

## Applicability and conflicts

Prefix cache, speculative decoding, KV transfer/disaggregated P/D, quantized
KV, hybrid/MLA/local attention, async scheduling and parallel degrees above
one are rejected at startup. A1's non-chunked configuration and A4's
prefix-enabled configuration therefore do not apply to this plugin as
currently implemented. Former Prefix Router, KV Tiering, KNorm, PyramidKV
Ascend and SliceGPT code is absent from the current hosts and is not assumed.

## Data preparation

For a formal evaluator-approved JSONL, download and pin it without embedding
credentials in commands or logs:

```bash
python scripts/kvcompress_prepare_dataset.py download \
  --url https://approved.example/LONG-PUBLIC-v1.jsonl \
  --sha256 <approved-sha256> \
  --source-revision <immutable-revision> \
  --license-id <SPDX-or-license-name> \
  --output .benchmarks/data/long-public-v1.jsonl
```

The downloader requires HTTPS, verifies the exact SHA-256 before replacing the
destination, validates the JSONL contract, and writes a provenance manifest.
For local commissioning only:

```bash
python scripts/kvcompress_prepare_dataset.py generate-commissioning \
  --tokenizer /data/shared_models/Qwen--Qwen2.5-14B-Instruct \
  --profile all \
  --output .benchmarks/data/kvcompress-commissioning.jsonl
```

## Execution and comparison

Run `kv-pressure-online` once as a quick pressure smoke. Then execute all
frozen A2 cells and A3 independently with plugin disabled (B0 engineering
control) and enabled (B1), restarting the service every time. One A2 example:

```bash
python scripts/kvcompress_long_context_run.py \
  --dataset .benchmarks/data/long-public-v1.jsonl \
  --profile A2-LONG-FP16-16K \
  --model /data/shared_models/Qwen--Qwen2.5-14B-Instruct \
  --request-rate 0.1 --concurrency 4 --timeout 1800 \
  --npu-id 0 --run-label B1 --server-log .benchmarks/b1-r1/server.log \
  --result .benchmarks/b1-r1/a2-16k.json
```

Compare exactly three results from each side:

```bash
python scripts/kvcompress_acceptance_compare.py \
  --baseline .benchmarks/b0-r{1,2,3}/a2-16k.json \
  --plugin .benchmarks/b1-r{1,2,3}/a2-16k.json \
  --output .benchmarks/a2-16k-comparison.json
```

## Evidence package and decision

Retain exact Git commits, package versions, image digest, model/tokenizer/data
hashes, commands and environment, hardware topology, startup configuration,
raw requests/results/logs, 1-second device samples, failures, manifests and
artifact hashes. Timing uses a monotonic clock; percentiles use nearest rank;
warm-up and cool-down are excluded. With fewer than 1,000 samples, zero errors
are allowed (otherwise at most 0.1%). Structured schema checks must be 100%
when applicable, and tool/structured success at least 95%.

A release passes only when every predeclared gate is true on the exact release
candidate. Missing, mismatched, edited-after-run, or unsigned formal evidence
fails closed. Ordinary PR CI runs only static/unit/build/contract/schema/hash
checks; performance acceptance belongs on an isolated evaluator.
