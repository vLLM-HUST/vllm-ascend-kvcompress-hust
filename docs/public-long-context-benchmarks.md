# Public Long-Context Benchmarks

English | [简体中文](public-long-context-benchmarks.zh.md)

Date: 2026-09-12. This record supplements the synthetic commissioning test
with three public A3 long-context/KV scenarios. It is a same-host engineering
comparison, not the formal V4.6 official baseline.

## Scenarios and frozen inputs

| Scenario | Purpose | Evaluated scope | Input tokens | Prepared SHA-256 |
| --- | --- | ---: | ---: | --- |
| LongBench-v2 | Long-context multiple-choice reasoning | 116 eligible short-stratum cases | 10,171–31,869 | `06255143...b538be6c` |
| LongBench `passage_retrieval_en` | Passage retrieval | 200/200 cases | 10,412–15,670 | `cc015530...98a2cb6` |
| LongBench `qasper` | Scientific-paper QA | 93/200 cases at or above the 5,120-token pressure floor | 5,168–22,074 | `20b04612...b8a6da94` |

LongBench-v2 is pinned to Hugging Face revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`; its downloaded `data.json`
hash is `15d61c22...604c7fe2`. The LongBench archive is pinned to revision
`5e628be450b7e67fb7ae6e201bd6d8f7056f7672`; `data.zip` hashes to
`cb45b11a...857f7f64`.

No accepted prompt is truncated. The service limit is 32,768 tokens, so the
387 LongBench-v2 medium/long-stratum cases outside that scope are recorded in
the generated unsupported file rather than shortened. Qasper cases below
5,120 tokens are excluded because they do not exercise the original 4K
compression threshold. `--limit 0` means all eligible cases, not a sample.

The downloader verifies the pins and hashes before extraction. LongBench-v2's
data card declares Apache-2.0. LongBench combines source datasets with their
own terms, so the scripts download it from upstream but neither the package nor
this repository redistributes benchmark text. Users remain responsible for
the terms of each source dataset.

## Reproduction

Download and prepare with the exact model tokenizer:

```bash
MODEL=/path/to/Qwen2.5-Coder-14B-Instruct
python scripts/kvcompress_benchmark_data.py download \
  --source all --root .benchmarks/datasets

python scripts/kvcompress_benchmark_data.py prepare \
  --benchmark longbench-v2 --root .benchmarks/datasets \
  --tokenizer "$MODEL" --local-files-only \
  --max-model-len 32768 --min-input-tokens 5120 --limit 0 \
  --output .benchmarks/public/longbench-v2.jsonl
python scripts/kvcompress_benchmark_data.py prepare \
  --benchmark longbench-passage-retrieval-en --root .benchmarks/datasets \
  --tokenizer "$MODEL" --local-files-only \
  --max-model-len 32768 --min-input-tokens 5120 --limit 0 \
  --output .benchmarks/public/longbench-passage-retrieval-en.jsonl
python scripts/kvcompress_benchmark_data.py prepare \
  --benchmark longbench-qasper --root .benchmarks/datasets \
  --tokenizer "$MODEL" --local-files-only \
  --max-model-len 32768 --min-input-tokens 5120 --limit 0 \
  --output .benchmarks/public/longbench-qasper.jsonl
```

Start one cold service per arm with the settings below. B0 is the same plugin
and host with a 32,768-token budget/32,896-token threshold, so no request can
compress. B1 uses [the example configuration](../examples/triattention.json):
8,192-token budget, 1,024-token recompute window, 512 recent protected tokens,
8,192-token scoring chunks, layer stride 4, and a 64-token minimum requested
output before compression is worthwhile.

```bash
export ASCEND_RT_VISIBLE_DEVICES=6
export VLLM_PLUGINS=ascend,ascend_kvcompress
vllm-hust-ext run -- vllm serve "$MODEL" \
  --served-model-name qwen2.5-coder-14b --dtype float16 \
  --block-size 128 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --max-num-batched-tokens 16384 --max-num-seqs 4 \
  --no-enable-prefix-caching --enable-chunked-prefill --no-async-scheduling \
  --generation-config vllm
```

Run each prepared file against each service arm, changing paths and labels as
needed:

```bash
python scripts/kvcompress_benchmark_run.py \
  --dataset .benchmarks/public/longbench-v2.jsonl \
  --base-url http://127.0.0.1:8000 --model qwen2.5-coder-14b \
  --result .benchmarks/results/b1-longbench-v2.json --run-label B1 \
  --request-rate 0 --concurrency 4 --timeout 1800 --npu-id 6 \
  --server-log .benchmarks/logs/b1.log

# passage_retrieval_en has max_output_tokens=32 and must exercise the bypass:
python scripts/kvcompress_benchmark_run.py \
  --dataset .benchmarks/public/longbench-passage-retrieval-en.jsonl \
  --base-url http://127.0.0.1:8000 --model qwen2.5-coder-14b \
  --result .benchmarks/results/b1-passage-retrieval-en.json --run-label B1 \
  --request-rate 0 --concurrency 4 --timeout 1800 --npu-id 6 \
  --server-log .benchmarks/logs/b1.log \
  --compression-expectation forbidden

python scripts/kvcompress_benchmark_score.py \
  --baseline .benchmarks/results/b0-longbench-v2.json \
  --plugin .benchmarks/results/b1-longbench-v2.json \
  --quality-tolerance-pp 1 \
  --output .benchmarks/results/paired-longbench-v2.json
```

The runner fixes `temperature=0`, `top_p=1`, `top_k=-1`, `min_p=0`, and
`seed=0`. It saves per-request predictions, TTFT/TPOT/E2E, package versions,
device samples, server-reported token counts, and scheduler/worker compression
evidence. The scorer implements the official LongBench-v2 answer extraction,
LongBench retrieval score, and LongBench QA F1 normalization. A B1 result fails
if a request fails, prompt token counts reveal truncation, the declared
compression expectation is violated, an acknowledgement is missing, or quality
drops by more than one percentage point. The default expectation is `required`;
the short-output retrieval run declares `forbidden` to prove the bypass rather
than silently accepting an uncompressed run.

## Results

Model: local Qwen2.5-Coder-14B-Instruct, FP16. Device: one Ascend 910B2.
Each arm is one cold-service run, unlimited arrival rate, concurrency 4. B0 is
the same-host no-compression engineering control described above.

| Scenario | B0 → B1 quality | Requests / compression | Request throughput | Mean TTFT | Mean TPOT | Mean E2E | Physical blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2 | 34.48 → 34.48 accuracy (0.00 pp) | 116 / 116 | **+2.13%** | **−3.84%** | **−7.80%** | **−2.17%** | −61.96% |
| Passage retrieval | 100.00 → 100.00 (0.00 pp) | 200 / 0 (bypassed) | −0.04% | **−0.28%** | +0.27% | +0.01% | n/a |
| Qasper | 42.51 → 42.03 F1 (−0.48 pp) | 93 / 11 | −0.11% | **−2.89%** | +11.31% | +0.41% | −38.73% on compressed cases |

All 818 arm-requests completed with zero silent truncations. B1 produced 127
scheduler commits and 127 worker acknowledgements. Across compressed cases,
20,666 source blocks became 8,128 destination blocks, a 60.67% reduction.
LongBench-v2 dynamic KV use peaked at 39.8% versus B0's 90.4%. Device HBM
still peaked at 87% in both arms because vLLM reserves the KV pool at startup.

The workload boundary is material: compression improves the broad,
10K–32K-input LongBench-v2 run. The 10K–16K retrieval task requests at most 32
output tokens, so the optimized 64-token gate bypasses scoring and copy work;
all 200 runs show zero scheduler commits/worker acknowledgements and effectively
neutral performance. Qasper is mostly below the 9,216-token B1 threshold, so
only 11 cases compress and aggregate performance is also effectively neutral.
These single runs support directional engineering conclusions, not a confidence
interval for throughput.

The initial 4,096-token candidate is retained as a negative tuning result: it
dropped Qasper F1 by 4.71 pp and therefore failed the 1 pp quality gate. The
8,192-token budget is the recommended public-benchmark setting; the earlier
4K synthetic result remains useful only for its stated fixed workload.

Machine-readable evidence is in
[kvcompress-v0.4.0-public-long-context-summary.json](evidence/kvcompress-v0.4.0-public-long-context-summary.json).

## Remaining limits

- The performance comparison has one run per arm, rather than three cold runs.
- B0 is not the prescribed official V4.6 host baseline.
- The local model snapshot and calibration file match the Qwen2.5-Coder-14B
  family but lack complete upstream revision/generation provenance.
- LongBench-v2 medium and long strata require a service/model with a context
  limit above 32K and remain untested here.
- This is not the 30-minute/six-window A3 stability test.
