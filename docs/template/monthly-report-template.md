# Monthly project report template

English | [简体中文](monthly-report-template.zh.md)

Write a `YYYY-MM.md` report under `dev/reports/monthly/`. The `dev/` directory
contains personal work records and is excluded from Git. Fill all fields marked
`*`; use the actual month and state rather than leaving examples as results.

```md
# Monthly project report — YYYY-MM
## Model selection
## Optimization scenarios and datasets
## Research direction
## Repository stage
## Optimization goals
## Progress stage
## Research outputs
## Coordination and other work
## Self-score (0–20)
## Additional notes
## Token costs
## Invoice
## Payment receipt
```

## Field guidance

- **Model selection***: Choose all applicable models: Qwen2.5-14B,
  Qwen3.6-27B, DeepSeek V4 Flash, Kimi K2.6, Qwen3.5-35B-A3B, or another
  approved work item. Record the exact model revision when reporting tests.
- **Optimization scenarios and datasets***: Name the workloads actually
  studied. The project catalog in `vllm-hust-dev-hub/docs/` groups 44 candidate
  datasets. Additional scenarios require project review. Choices:
  AI2-ARC, AIME-2024, BBH, BFCL-V3, BFCL-V4, BURSTGPT-TRACE, EVOSCIENTIST, GPQA, GSM8K, HOTPOT-QA, HOTPOTQA-REACT-100, HUMANEVAL, INFINITEBENCH, INSTRUCTCODER, JSONSCHEMABENCH, LIVECODEBENCH-CODE-GENERATION, LIVECODEBENCH-EXECUTION, LIVECODEBENCH-TEST-GENERATION, LONGBENCH, LONGBENCH-V2, MATH-500, MBPP, MMLU-PRO, OASST1, PAIO-CHAT-1000, PAIO-CODE-EVAL-1000, PAIO-JOSN-500, PAIO-LONG-PREFILL-5000, PAIO-LONGTEST-4000, PAIO-PREFIX-SHARED-5000, PAIO-REUSE-CONV-5000, SCHEMASTORE, SHAREGPT-V3, SONNET, STRUCTEVAL, SZYN-OPENCODE-SWEBENCH-VERIFIED-500, SYFI-CODING-TRACE, TAU2-BENCH, TOOLBENCH, ULTRACHAT-200k, VISIONARENE, WILDCHAT, WILDCHAT-4.8M, trie-workloads, vllm-hust-bidkv, 通用场景, 无
- **Research direction***: State the concrete question or bottleneck.
- **Repository stage***: Incubating, experimental, demonstrable, or ready for
  delivery. Plugin code belongs in an independent repository under vLLM-HUST.
- **Optimization goals***: Select measurable outcomes such as request/input/
  output/total throughput, TTFT, TPOT, E2E latency, quality, success rate,
  OOM/truncation rate, B1/B0 ratio, HBM, NPU utilization, energy, SLO goodput,
  cache hit rate, or scheduling and KV lifecycle observability. Define units
  and the B0/B1 comparison; do not conceal negative overall effects.
- **Progress stage***: Ideation, research, reproduction, validation, or
  delivery. Summarize this month's actual progress.
- **Research outputs***: None, drafting, finished but unpublished, or
  published; identify the paper, journal, or conference if applicable.
- **Coordination and other work***: List project activities outside the core
  research, such as coordination, documents, and presentations.
- **Self-score***: Give an evidence-based score from 0 to 20.
- **Additional notes***: Summarize completed work, results, blockers, help
  needed, and the next month's plan. Link reviewable evidence where applicable.
- **Token costs, invoice, payment receipt**: Fill when applicable and keep
  financial or personal records outside Git.
