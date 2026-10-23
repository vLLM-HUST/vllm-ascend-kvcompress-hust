# Documentation

Verified historical originals: [retrieval guide](usage/benchmark-originals.md).

English | [简体中文](README.zh.md)

Use this map to find user instructions, design, benchmark methods,
optimization records, and dated results. Read the paired English and Chinese
documents together. The package version is declared in `pyproject.toml`;
release and measurement status belongs in [results](results/README.md).
Dated measurements retain their original version, model, and host identity
after archival.

## Current work

| Topic | Documents |
| --- | --- |
| User guides | [Install, configure, calibrate, serve, and troubleshoot](usage/README.md) |
| Benchmark protocols | [Choose a protocol for the workload and claim](benchmark/README.md) |
| Implementation | [Method API and TriAttention](design/methods.md), [Ascend adaptation](design/ascend-adaptation-vs-triattention-vllm.md), [Qwen3.5 hybrid model](design/qwen3.5-35b-a3b-adaptation.md) |
| Optimization history | [Dated implementation and tuning records](optimization/README.md) |
| Development | [Environment](development/environment-installation.md), [calibration](development/calibration-artifacts.md), [ownership and licensing](development/ownership-and-licensing.md), [commit conventions](development/git-commit.md), [packaging and release](development/packaging-and-release.md) |
| Results | [Result scope and evidence](results/README.md), [HTML leaderboard](results/benchmark-leaderboard.html) |
| Project issues | [Five-dataset applicability, issue #17](issue/dataset-program-issue-17.md) |
| Reporting | [Weekly](template/weekly-report-template.md) and [monthly](template/monthly-report-template.md) templates |

## Historical archive

Browse the [categorized archive](archive/README.md) by benchmark protocol,
results, design, or development history.

The [0.9 Frontier campaign](archive/results/frontier-benchmarking-0.9.md)
contains the full sequence of positive and negative local pairs, including the
2026-10-08 release checks. The [V4.6 requirements](archive/benchmark/kv-compress-test-requirements.md),
[A2/A3 procedure](archive/benchmark/benchmarking.md),
[public long-context experiments](archive/results/public-long-context-benchmarks.md),
[2026-09-20 validation](archive/results/validation.md), and
[0.2 results](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/b36edbc7698e3ca188904d0e20495ed2924a8525/archive/mod-targeted-20261023/compress/snapshot/docs/archive/results/results-0.2.md) describe older workloads and
host stacks. Do not pool measurements from different cohorts.
