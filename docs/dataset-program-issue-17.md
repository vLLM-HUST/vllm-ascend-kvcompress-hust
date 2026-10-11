# Pujiang five-dataset adaptation — issue #17

The [candidate matrix](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/6ef387046bfa13cfb6d0b7a7e33f0f95a9f917b3/reports/kvcompress-five-dataset-applicability-20261011) is proposed in [benchmark PR #276](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/276), with review tracked in [this MOD's issue #17](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/issues/17) and the [program ledger #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254). Maintainer review is pending. There is no five-dataset task-quality result or optimization claim.

| Dataset | Status | Gate |
| --- | --- | --- |
| MMLU-Pro | `not-exercised` | Under the audited reference five-shot Qwen rendering, the longest of 12,032 test prompts is 2,910 tokens. A 2,048-token output cap yields at most 4,958 total tokens, below the current Qwen3.5 TriAttention V3 threshold of 9,216. This is an upper-bound comparison, not a MOD run. |
| HLE-Verified | `blocked` | Rights review, set choice, images, judge and runtime contract. |
| SWE-bench-Pro | `blocked` | Rights, validated task/repository/image mapping, fresh sandbox and regrader. Existing SWE prefix-reuse throughput traces are not task resolution. |
| FrontierScience | `blocked` | Separate Olympiad and Research graders, final solver prompts and output budget. Candidate problem-only lengths do not prove final applicability. |
| Terminal-Bench 2.1 | `blocked` | Pinned Harbor images, agent/tool/budget/scorer contract and task rights. |

The current B1 configuration is [`.benchmarks/config/qwen3.5-35b-a3b-v3-batched-stride2-min512.json`](../.benchmarks/config/qwen3.5-35b-a3b-v3-batched-stride2-min512.json). `kv_budget + recompute_window` sets the threshold; `min_output_tokens_for_compression` is another gate. If an evaluator freezes a changed task rendering, run the exact tokenized manifest through:

```sh
python scripts/kvcompress_dataset_applicability.py \
  --manifest /path/to/frozen-task-prompts.jsonl \
  --method-config .benchmarks/config/qwen3.5-35b-a3b-v3-batched-stride2-min512.json \
  --output /path/to/applicability-audit.json
```

Each JSONL row must contain a unique `case_id`, exact `prompt_token_ids`, and `max_tokens`; it may also carry `input_len`. The output pins input/config SHA-256 hashes and identifies requests whose budget *might* reach compression. A possible request still needs a live B1 scheduler commit and worker acknowledgement. Do not pad prompts or lower the threshold to manufacture applicability. When a dataset becomes executable, freeze the scorer and full B0/B1 execution contract before measuring. Task quality is primary; throughput and latency are supporting telemetry. Publish all outcomes and raw hashes in the benchmark repository before updating the website. PyPI release is out of scope for this issue work.
