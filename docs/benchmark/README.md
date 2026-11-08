# Benchmark protocols

English | [简体中文](README.zh.md) | [Documentation](../README.md)

The current evaluation uses the dataset specified by the Pujiang Laboratory, as described on the [project-specified dataset page](https://vllm-hust.sage.org.ai/dataset-introductions.html), with fixed clients and respective scoring agreements retained.

| Dataset | Evaluation | Preparation in this version |
| --- | --- | --- |
| MMLU-Pro | Token-ID completions; local answer extraction | [Client, download, frozen plan and runner](mmlu-pro.md) |
| HLE-Verified | Gold / Revision; solver and separate structured-output judge | [Unified client workflow](pujiang.md) |
| SWE-bench-Pro | Agent sandbox and task tests | Not implemented; empty adapter |
| FrontierScience | Olympiad / Research; separate judge and rubric | [Unified client workflow](pujiang.md) |
| TerminalBench 2.1 | ARM-compatible task sandbox and task grader | Not implemented; empty adapter |

[MMLU-Pro procedure](mmlu-pro.md).

[Unified setup, frozen smoke and B0/B1 comparison](pujiang.md).
