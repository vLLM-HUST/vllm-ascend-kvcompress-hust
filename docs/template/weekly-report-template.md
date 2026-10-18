# Weekly project report template

English | [简体中文](weekly-report-template.zh.md)

Write a `YYYY-MM-DD.md` report under `dev/reports/weekly/`. The `dev/`
directory is excluded from Git. Complete every field marked `*`.

```md
# Weekly project report — YYYY-MM-DD
## Model selection
## Optimization scenarios and datasets
## Research direction
## Research outputs
## Optimization goals
## Progress stage
## Additional notes
## Attachments (optional)
```

## Field guidance

- **Model selection***: Choose from Qwen2.5-14B, Qwen3.6-27B, DeepSeek V4
  Flash, Kimi K2.6, or another approved work item. Record the exact revision
  for any result.
- **Optimization scenarios and datasets***: Name the datasets actually used;
  the project catalog in `vllm-hust-dev-hub/docs/` contains 44 candidate
  datasets. Additional scenarios require review. Choices:
  AI2-ARC, AIME-2024, BBH, BFCL-V3, BFCL-V4, BURSTGPT-TRACE, EVOSCIENTIST, GPQA, GSM8K, HOTPOT-QA, HOTPOTQA-REACT-100, HUMANEVAL, INFINITEBENCH, INSTRUCTCODER, JSONSCHEMABENCH, LIVECODEBENCH-CODE-GENERATION, LIVECODEBENCH-EXECUTION, LIVECODEBENCH-TEST-GENERATION, LONGBENCH, LONGBENCH-V2, MATH-500, MBPP, MMLU-PRO, OASST1, PAIO-CHAT-1000, PAIO-CODE-EVAL-1000, PAIO-JOSN-500, PAIO-LONG-PREFILL-5000, PAIO-LONGTEST-4000, PAIO-PREFIX-SHARED-5000, PAIO-REUSE-CONV-5000, SCHEMASTORE, SHAREGPT-V3, SONNET, STRUCTEVAL, SZYN-OPENCODE-SWEBENCH-VERIFIED-500, SYFI-CODING-TRACE, TAU2-BENCH, TOOLBENCH, ULTRACHAT-200k, VISIONARENE, WILDCHAT, WILDCHAT-4.8M, trie-workloads, vllm-hust-bidkv, 通用场景, 无
- **Research direction***: Describe the concrete bottleneck or question.
- **Research outputs***: None, drafting, finished but unpublished, or
  published; identify any publication.
- **Optimization goals***: Use measurable metrics, such as throughput,
  TTFT/TPOT, E2E latency, quality, cache hit rate, HBM, NPU utilization,
  energy, SLO goodput, or KV lifecycle observability. Keep B0/B1 evidence.
- **Progress stage***: Ideation, research, reproduction, validation, or
  delivery. State what changed this week.
- **Additional notes***: List completed work, results, blockers, help needed,
  and next steps. Do not omit negative experiments.
- **Attachments**: Link reviewable evidence of a solved bottleneck or a
  significant contribution where relevant.
