# Version 0.4 Validation Record

English | [简体中文](validation.zh.md)

Date: 2026-09-11. Verdict: **engineering PASS, not formal V4.6 acceptance**.
The package and current-host integration are usable for continued evaluation;
the limitations below prohibit a production or official-baseline claim.

## Frozen environment

| Component | Validated value |
| --- | --- |
| Plugin | 0.4.0, working tree based on `db18568aa8` |
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

The 0.4.0 wheel was installed without a source checkout on `PYTHONPATH`.
Discovery, manifest parsing, compatibility, configuration, validation,
enablement, status/check/plan/env rendering, and `run --dry-run` passed. The
rendered environment contained both:

```text
VLLM_ASCEND_KVCOMPRESS_ENABLED=1
VLLMHUST_EXT_ENABLED_BUNDLES=org.vllm-hust.ascend-kvcompress
```

Disable, forget, uninstall, disappearance from `extension list`, reinstall,
and re-enable are included in the release lifecycle check. Disabled imports
remain inert. The current manager reports no native extension API for this
host, so the manifest deliberately declares no `api_range` and relies on the
exact package range plus runtime contract checks.

## Automated and NPU checks

The final candidate is required to reproduce:

- `pytest`: 58 passed, 1 skipped;
- Ruff check and format check: pass;
- package metadata and wheel/sdist inspection: pass;
- Ascend numerical kernel smoke: pass;
- NPU kernel benchmark versus generic references:

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

## V4.6 gaps and publication boundary

This record is not a formal V4.6 PASS because:

1. B0 used the current host with compression disabled/no-op for compatibility,
   not the prescribed official vLLM 0.18 plus matching official Ascend stack;
2. the test fixture is deterministic synthetic retrieval data, not an
   authorized frozen production/quality dataset;
3. the available Qwen2.5-Coder-14B aggregate statistics do not match the tested
   Qwen2.5-14B-Instruct model and lack complete generation provenance;
4. only the 16K/0.4-RPS/concurrency-4 A2 cell received three cold runs; the
   complete A2 rate matrix and A3 30-minute/six-window stability run remain;
5. percentage HBM reservation did not fall because the host preallocates its
   KV pool, and system cost-per-token was not independently measured.

Therefore the repository may claim package/manager compatibility, NPU
correctness on the declared snapshots, a 75% physical KV-block reduction, and
the bounded engineering-control performance above. It must not claim formal
V4.6 acceptance, model-quality preservation, or production readiness.
