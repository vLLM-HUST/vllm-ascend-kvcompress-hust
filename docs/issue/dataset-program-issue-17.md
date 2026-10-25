# Pujiang five-dataset adaptation — issue #17

English | [简体中文](dataset-program-issue-17.zh.md)

The [candidate matrix](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/6c49429c3211de0363c534a83fcffef5267c5d9d/reports/kvcompress-five-dataset-applicability-20261011) is proposed in [benchmark PR #276](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/276), with review tracked in [this MOD's issue #17](https://github.com/vLLM-HUST/vllm-ascend-kvcompress-hust/issues/17) and the [program ledger #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254). Maintainer review is pending. There is no five-dataset task-quality result or optimization claim.

| Dataset | Status | Gate |
| --- | --- | --- |
| MMLU-Pro | `not-exercised` | Under the audited reference five-shot Qwen rendering, the longest of 12,032 test prompts is 2,910 tokens. A 2,048-token output cap yields at most 4,958 total tokens, below the current Qwen3.5 TriAttention V3 threshold of 9,216. This is an upper-bound comparison, not a MOD run. |
| HLE-Verified | `blocked` | Rights review, set choice, images, judge and runtime contract. |
| SWE-bench-Pro | `blocked` | Rights, validated task/repository/image mapping, fresh sandbox and regrader. Existing SWE prefix-reuse throughput traces are not task resolution. |
| FrontierScience | `blocked` | Separate Olympiad and Research graders, final solver prompts and output budget. Candidate problem-only lengths do not prove final applicability. |
| Terminal-Bench 2.1 | `blocked` | Pinned Harbor images, agent/tool/budget/scorer contract and task rights. |

## Storage planning estimates

The following user-provided figures help plan dataset downloads and evaluation environments; they have not been independently verified in this repository. Actual use depends on the downloaded scope, image cache policy, and runtime. Dataset download size alone is not the size of a complete evaluation environment.

| Dataset | Dataset download | Possible full environment size | Suggested capacity |
| --- | ---: | ---: | ---: |
| **MMLU-Pro** | **≈ 4.2 MB** | Little additional storage | **< 100 MB** |
| **HLE-Verified** | **≈ 335 MB** including all images | Small | **0.5–1 GB** |
| **SWE-bench-Pro** | **≈ 15 MB** | About **1–2+ TB** if all task images are cached | **2–3 TB** |
| **FrontierScience** | **≈ 518 KB** | Little additional local storage | **< 100 MB** |
| **Terminal-Bench 2.1** | **≈ 58.4 MB** | Docker environment: **tens of GB** | **≈ 80–150 GB** |

## Token-scale screening against the 9,216-token threshold

These user-provided input and output figures are planning estimates without linked raw artifacts, not matched results for this MOD on the five datasets. `Total` means different things for one request and a multi-turn agent task: a task's cumulative tokens exceeding the threshold **does not prove that any single request reached 9,216 physical KV tokens**. Applicability still requires an audit of each frozen request and B1 scheduler commits with worker acknowledgements.

| Benchmark | Input size | Output size | Representative total | `total ≥ 9,216` | Confidence |
| --- | ---: | ---: | ---: | --- | --- |
| **MMLU-Pro** | Current five-shot maximum **2,910** | Current cap **2,048**; Qwen3.5 reference measured mean **1,179** | Current maximum **4,958** | **0% under current configuration** | **High** |
| **HLE-Verified** | Question maximum 16.9k characters, roughly several thousand tokens | HLE-like Qwen3.5 reference measured mean **6,625** | May approach or exceed 9k | **Unknown, potentially material** | Medium |
| **SWE-bench-Pro** | Static issue prompt usually hundreds to thousands of tokens | Multi-turn agent; reference **6.1k–8.9k output/task** | Fresh input + output ≈ **9.4k–14.4k/task** | **Many task totals likely exceed; per-request status unknown** | Medium-high |
| **FrontierScience** | Problem length 408–6,590 characters | Strongly reasoning-model dependent; reference range **3.3k–29k** | Can be below or far above 9,216 | **Highly model-dependent; dataset alone cannot decide** | High |
| **Terminal-Bench 2.1** | Initial instruction usually <~1–1.5k tokens | Multi-turn agent; reference **12k–18k output/task** | Fresh input + output ≈ **12.5k–47k/task** | **Most task totals expected to exceed; per-request status unknown** | Medium-high |

The current B1 configuration is [`.benchmarks/config/qwen3.5-35b-a3b-v3-batched-stride2-min512.json`](../../.benchmarks/config/qwen3.5-35b-a3b-v3-batched-stride2-min512.json). `kv_budget + recompute_window` sets the threshold; `min_output_tokens_for_compression` is another gate. If an evaluator freezes a changed task rendering, run the exact tokenized manifest through:

```sh
python scripts/kvcompress_dataset_applicability.py \
  --manifest /path/to/frozen-task-prompts.jsonl \
  --method-config .benchmarks/config/qwen3.5-35b-a3b-v3-batched-stride2-min512.json \
  --output /path/to/applicability-audit.json
```

Each JSONL row must contain a unique `case_id`, exact `prompt_token_ids`, and `max_tokens`; it may also carry `input_len`. The output pins input/config SHA-256 hashes and identifies requests whose budget *might* reach compression. A possible request still needs a live B1 scheduler commit and worker acknowledgement. Do not pad prompts or lower the threshold to manufacture applicability. When a dataset becomes executable, freeze the scorer and full B0/B1 execution contract before measuring. Task quality is primary; throughput and latency are supporting telemetry.
