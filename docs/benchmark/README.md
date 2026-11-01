# Benchmark protocols

English | [简体中文](README.zh.md) | [Documentation](../README.md)

The current evaluation uses the dataset specified by the Pujiang Laboratory, as described on the [project-specified dataset page](https://vllm-hust.sage.org.ai/dataset-introductions.html), with fixed clients and respective scoring agreements retained.

| Dataset | Evaluation | Preparation in this version |
| --- | --- | --- |
| MMLU-Pro | Token-ID completions; local answer extraction | [Client, download, frozen plan and runner](mmlu-pro.md) |
| HLE-Verified | Gold / Revision; solver and separate structured-output judge | Protocol entry only |
| SWE-bench-Pro | Agent sandbox and task tests | Protocol entry only |
| FrontierScience | Olympiad / Research; separate judge and rubric | Protocol entry only |
| TerminalBench 2.1 | ARM-compatible task sandbox and task grader | Protocol entry only |

[MMLU-Pro procedure](mmlu-pro.md).