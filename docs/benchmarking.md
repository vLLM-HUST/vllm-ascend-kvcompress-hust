# Version 0.6 Candidate Acceptance Protocol

English | [简体中文](benchmarking.zh.md)

The normative project-specific requirements extracted from the HUST V4.6
delivery plan are in [KV-compression test requirements](kv-compress-test-requirements.md).
That document takes precedence over the abbreviated workflow below.

1. Freeze all host/plugin commits, Python/NPU stack, model/tokenizer revision,
   calibration provenance and SHA-256, device state, configuration, commands,
   dataset hash, and raw output paths.
2. Exercise clean wheel install, manager discovery/configure/enable/check,
   disabled inertness, disable/forget/uninstall, disappearance, and reinstall.
3. Run `tests/run_npu_kernel_smoke.py` and benchmark direct scoring, K/V copy,
   aggregation, offsets, repeated compression, edge lengths, and supported
   dtypes against numerical references.
4. Use deterministic authorized data. Assert no failure, OOM, deadlock,
   contamination, incorrect answer, short output, or silent truncation. Match
   every scheduler compression commit with a worker acknowledgement.
5. For A2, run fixed 8,192+512 and 16,384+1,024 cells at 0.05/0.1/0.2/0.4 RPS,
   concurrency 4, and three independent cold B0/B1 lifecycles. For A3, run
   exact 30,720+2,048, concurrency 1, five-minute warm-up and 30-minute
   measurement split into six windows.
6. Report request/input/output/total throughput, TTFT/TPOT/E2E distributions,
   failures, peak HBM, dynamic KV block use, compression count/latency, quality,
   run-level values, median, range, CV, and hashes.

Generate deterministic commissioning data, run one service arm, and compare
paired result sets with the repository tools:

```bash
python scripts/kvcompress_prepare_dataset.py --output .benchmarks/data/a2.jsonl
python scripts/kvcompress_long_context_run.py --help
python scripts/kvcompress_acceptance_compare.py --help
```

For complementary public long-context evidence, the repository also pins and hash-verifies LongBench-v2
and LongBench, prepares prompts with the exact model tokenizer without
truncation, runs the OpenAI-compatible service, and applies benchmark-specific
quality scorers:

```bash
python scripts/kvcompress_benchmark_data.py download --help
python scripts/kvcompress_benchmark_data.py prepare --help
python scripts/kvcompress_benchmark_run.py --help
python scripts/kvcompress_benchmark_score.py --help
```

The selected tasks, complete commands, source revisions, license boundary, and
measured results are in
[Public long-context benchmarks](public-long-context-benchmarks.md). An 8K KV
budget with `score_layer_stride=8` is the quality-qualified public-benchmark
default; the 4K setting is an aggressive workload-specific option and failed
the Qasper quality gate. Runs
must also declare whether compression is required, optional, or forbidden so
an intended short-output bypass cannot be confused with missing activation.

Random data, truncation, prompt reuse, and a warm service restart are not valid
substitutes. `kv-pressure-online` is a quick pressure smoke only. Formal B0 is
the prescribed official vLLM 0.18 plus matching official Ascend baseline; a
current-host run that loads the same plugin but forbids compression with an
above-limit threshold must be labeled an engineering compatibility control.
Host repositories must not be modified to execute the tests.

Prefix caching, speculative decoding, KV transfer, quantized KV, BidKV, and
removed host optimizations are unsupported combinations, not benchmark tuning
variables. Publication claims must stay within the completed matrix and the
limitations in [Validation](validation.md).
