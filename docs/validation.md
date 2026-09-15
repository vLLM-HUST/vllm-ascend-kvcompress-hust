# Version 0.5 Validation Record

English | [简体中文](validation.zh.md)

Date: 2026-09-15. Verdict: **engineering PASS, not formal V4.6 acceptance**.
The package and current-host integration are usable for continued evaluation;
the limitations below prohibit a production or official-baseline claim.

## Version 0.5 calibration acceptance

Version 0.5 adds in-package calibration generation. On physical NPU 6, the
final generator loaded local Qwen2.5-Coder-14B-Instruct and produced all 48 x
40 query-head statistics at the default 4,096-token length. The 1,565,471-byte
schema-2 artifact passed model shape, RoPE, finite-value, source identifier,
revision, and checkpoint-manifest fingerprint checks. Its SHA-256 is
`f53898b153fe8b3177b7e10e3c0f979eb6b470a9e5fc335a3f0ed3258f42846b`.

A clean service startup with a missing `stats_path` demonstrated that
calibration finished before the original serving-weight load, after which the
temporary model was released, 48 KV layers were bound, `/health` returned 200,
and a completion succeeded. A second service used the newly generated 4,096
artifact for one exact 3,072-token retrieval case: it was correct, recorded one
scheduler commit and one worker acknowledgement, and compacted 24 source
blocks to 16 destination blocks. Ascend numerical kernel smoke also passed.

The built-in corpus is a bootstrap default. This single use case does not
replace the prior repeated performance and public-quality evidence below;
production operators must generate from a representative licensed corpus and
rerun their quality/performance/HBM matrix. The machine-readable record is
[kvcompress-v0.5.0-auto-calibration-summary.json](evidence/kvcompress-v0.5.0-auto-calibration-summary.json).
The final automated suite reports 77 passed and 1 skipped; Ruff check and
format check pass.

## Frozen environment

| Component | Validated value |
| --- | --- |
| Plugin | 0.5.0, working tree based on `7c0d21144d` |
| vLLM-HUST | `6cdc0304a8`, `0.28.1.post1.dev260` |
| vLLM-Ascend-HUST | `5901bedbb7`, `0.25.1rc2.dev232+hust.20260903.4.g5901bedbb` |
| Extension Manager | `cf1ea71e3e`, `0.2.0.dev0` |
| Triton Ascend | `8f0a4de84`, wheel `3.6.0+git8f0a4de8` |
| Python / torch / torch-npu | 3.11.16 / 2.10 / 2.10.post2 |
| Device | one Ascend 910B2 |
| Model | local Qwen2.5-14B-Instruct, FP16 |

The existing quickstart attempted the generic Triton `setup.py`, which does
not build the Ascend backend. For this validation, the same unmodified
Triton-Ascend checkout was built through `setup_ascend.py` in a temporary
directory and reinstalled. The source and recovery procedure are documented in
[Environment installation](environment-installation.md).

## Package and Extension Manager lifecycle

The 0.5.0 wheel was installed without a source checkout on `PYTHONPATH`.
Discovery, manifest parsing, compatibility, configuration, validation,
enablement, status/check/plan/env rendering, and `run --dry-run` passed. The
rendered environment contained both:

```text
VLLM_ASCEND_KVCOMPRESS_ENABLED=1
VLLMHUST_EXT_ENABLED_BUNDLES=org.vllm-hust.ascend-kvcompress
```

The 0.5 check covered disable, uninstall, disappearance from `extension list`,
reinstall, rediscovery, check, and re-enable while preserving the prior
configuration. The earlier 0.4 check also covered `forget`. Disabled imports
remain inert. The current manager reports no native extension API for this
host, so the manifest deliberately declares no `api_range` and relies on the
exact package range plus runtime contract checks.

## Automated and NPU checks

The final candidate is required to reproduce:

- `pytest`: 77 passed and 1 skipped;
- Ruff check and format check: pass;
- package metadata and wheel/sdist inspection: pass;
- Ascend numerical kernel smoke: pass;
- the following NPU kernel benchmark ratios are inherited from the unchanged
  version 0.4 runtime kernels and were not remeasured for 0.5:

| Kernel path | Optimized | Reference | Ratio |
| --- | ---: | ---: | ---: |
| Paged K/V copy | 0.592 ms | 0.325 ms | 1.82x |
| Direct paged scoring | 1.306 ms | 0.464 ms | 2.81x |
| Fused aggregation | 0.158 ms | 0.118 ms | 1.34x |
| Offset update | 0.048 ms | 0.058 ms | 0.83x |

The ratio is reference time divided by optimized time. Offset update regressed
and is not represented as an improvement.

## Long-context engineering control

The primary completed cell was A2-LONG-FP16-16K: 4 deterministic requests per
cold service lifecycle, 16,384 input tokens, 1,024 forced output tokens,
ignore-EOS, 0.4 RPS, concurrency 4, block size 128, prefix cache and speculative
decoding disabled. Each arm used three independent cold lifecycles.

| Median of three cold runs | Same-host control | Compression | Change |
| --- | ---: | ---: | ---: |
| Request throughput | 0.06489 req/s | 0.08222 req/s | +26.7% |
| Input throughput | 1063.1 tok/s | 1347.1 tok/s | +26.7% |
| Output throughput | 66.44 tok/s | 84.19 tok/s | +26.7% |
| Total throughput | 1129.5 tok/s | 1431.3 tok/s | +26.7% |
| Mean TTFT | 4167.7 ms | 4326.3 ms | +3.8% (worse) |
| Mean TPOT | 52.43 ms | 39.16 ms | -25.3% |
| Mean E2E | 57.81 s | 44.37 s | -23.2% |
| p99 E2E | 61.45 s | 48.03 s | -21.8% |

All 24 requests (12 per arm) completed with the required output length and
retrieval answer. Compression recorded 12 scheduler commits and 12 worker
acknowledgements. Every compressed request changed from 128 to 32 physical
blocks, a 75% reduction. Device-wide HBM stayed at the vLLM-reserved 87%; the
service log's dynamic KV block use peaked around 16.5% with compression and
56% in the control, so this record claims reduced physical KV pressure, not a
smaller reserved HBM pool.

The machine-readable summary is
[kvcompress-v0.4.0-a2-16k-c4-summary.json](evidence/kvcompress-v0.4.0-a2-16k-c4-summary.json).
The deterministic fixture SHA-256 is
`3f68a54ee4028aa63418534466e3763d8e7c1da308af58cec2506cc16b067ab0`.

Single-request commissioning smokes also passed at 8,192+512 and
16,384+1,024. They verify boundary behavior but are not substituted for the
three-run matrix.

## Public A3 long-context quality sweep

To supplement the synthetic fixture, three public workloads were run
sequentially in one cold B0 and one cold B1 service lifecycle on local
Qwen2.5-Coder-14B-Instruct FP16. B0 used the same host and plugin with a
32,768-token no-compression budget. B1 used the recommended 8,192-token budget.
Every accepted prompt was tokenized without truncation; over-limit inputs were
recorded as unsupported.

| All eligible cases | Quality B0 → B1 | Request throughput | Mean TTFT | Mean TPOT | Physical blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2, 116 cases (10K–32K) | 34.48 → 34.48 accuracy | +2.13% | -3.84% | -7.80% | -61.96% |
| LongBench passage retrieval, 200 cases (10K–16K) | 100.00 → 100.00 | -0.04% | -0.28% | +0.27% | bypassed (0 commits) |
| LongBench Qasper, 93 cases (5K–22K) | 42.51 → 42.03 F1 (-0.48 pp) | -0.11% | -2.89% | +11.31% | -38.73% on 11 compressed cases |

All 818 arm-requests completed with zero silent truncations. B1 recorded 127
scheduler commits and 127 acknowledgements; physical blocks across compressed
cases fell from 20,666 to 8,128 (-60.67%). LongBench-v2 dynamic KV use peaked
at 39.8%, versus 90.4% for B0, while device HBM remained 87% because the pool
is reserved at startup.

The 4,096-token tuning candidate was rejected after Qasper F1 fell 4.71 pp.
The 8K results meet the one-percentage-point quality tolerance, but performance
is workload-dependent: compression improved LongBench-v2, while the optimized
64-token requested-output gate bypassed compression for the 32-token retrieval
workload and reduced its prior overhead to effectively neutral. As one run per arm, these numbers
are directional engineering evidence rather than repeated-run performance
statistics. Dataset pins, licenses, commands, scorers, exact metrics, negative
result, and hashes are in
[Public long-context benchmarks](public-long-context-benchmarks.md) and its
[machine-readable evidence](evidence/kvcompress-v0.4.0-public-long-context-summary.json).

## V4.6 gaps and publication boundary

This record is not a formal V4.6 PASS because:

1. B0 used the current host with compression disabled/no-op for compatibility,
   not the prescribed official vLLM 0.18 plus matching official Ascend stack;
2. public quality tasks now supplement the deterministic synthetic fixture,
   but the public performance arms have only one cold run each and are not an
   authorized production dataset;
3. the public model and statistics are from the same Qwen2.5-Coder-14B family,
   but their exact upstream revision and generation provenance remain
   incomplete; the earlier synthetic run used a different Qwen2.5-14B variant;
4. only the 16K/0.4-RPS/concurrency-4 A2 cell received three cold runs; the
   complete A2 rate matrix and A3 30-minute/six-window stability run remain;
5. percentage HBM reservation did not fall because the host preallocates its
   KV pool, and system cost-per-token was not independently measured.

Therefore the repository may claim package/manager compatibility, NPU
correctness on the declared snapshots, bounded public quality preservation at
the recommended 8K budget, and the workload-specific engineering measurements
above. It must not claim formal V4.6 acceptance, universal model-quality or
throughput preservation, or production readiness.
