# Version 0.3 Validation Record

English | [简体中文](validation.zh.md)

Date: 2026-09-04

## Declared snapshots

| Component | Revision/version |
| --- | --- |
| vLLM-HUST | `5b343ed52`, `0.17.2rc1.dev5941+g5b343ed52.empty` |
| vLLM-Ascend-HUST | `4e57439`, `0.25.1rc1+hust.20260903.4` |
| Extension Manager | `9fb467e`, `0.2.0.dev0` |
| vLLM Ascend KV Compression | working tree based on `e4a4864`, package `0.3.0` |
| Triton-Ascend source | `ee4b0ecef`, required package line `3.6.0` |
| Python | 3.11.16 |

The host, manager, documentation, and Triton repositories were read or
reinstalled as needed but no source file in them was edited. All source and
documentation changes are confined to this plugin repository.

## Package and static acceptance

| Area | Result | Evidence |
| --- | --- | --- |
| CPU/unit suite | PASS | 52 tests; configuration, calibration, selection, provider/scheduler transactions and preemption reset, exact scheduler admission, lazy runner hook, normalized cache-plan binding, version consistency, manifest, and registry coverage |
| Lint/format | PASS | Ruff check and format check on `src` and `tests`; `git diff --check` passes |
| Host interface review | PASS | Current `Scheduler`, inert Ascend `BalanceScheduler`, `KVCacheManager`, `BlockTable`, and `NPUModelRunner` hooks were checked against the declared snapshots |
| Manifest schema/discovery | PASS | `0.2-experimental`; active carrier, extension ID/version, activation, provider, permissions, and entry points are discoverable without importing the runtime |
| Installed-host manager check | PASS | `vllm-ascend 0.25.1rc1+hust.20260903.4` satisfies the declared range and launch configuration renders |
| Manager lifecycle | PASS | Configure, enable, status/environment render, disable, forget, and uninstall succeed in isolation |
| Disabled behavior | PASS | Registration is inert unless direct or manager activation is explicit |
| Lazy worker import | PASS | The control process can patch scheduler/cache seams without importing the NPU runner early |

The tested candidate wheel is
`vllm_ascend_kvcompress_hust-0.3.0-py3-none-any.whl`; the paired sdist passed
metadata and content checks. Capture both hashes from the protected tagged
build before upload. An isolated environment without the host correctly
reports compatibility as unverified and refuses `run --dry-run` for this
trusted in-process extension. This is expected fail-closed behavior.

## Environment recovery finding

The development environment initially contained obsolete duplicate
vLLM-Ascend distributions and both official Triton and editable Triton-Ascend,
causing duplicate platform discovery and a stale `triton._C.libtriton` binary.
The packages were reconciled and the current vLLM-Ascend source was reinstalled
without changing either checkout. Missing `numba` and `torchvision` runtime
packages were installed, after which the current service could start.

Formal dependency acceptance remains open. Current `vllm-ascend` metadata
requires Torch 2.13.0, Torch-NPU 2.13.0rc1, Triton-Ascend 3.6.0, and
Transformers 5.14.1. The available dev environment uses Torch/Torch-NPU 2.10,
published Triton-Ascend 3.2.2, and Transformers 5.15.1. The unchanged
Triton-Ascend 3.6.0 source build requires an approximately 1.2 GB Ascend LLVM
download that did not complete in this validation window. The NPU results
below are therefore compatibility evidence from a fallback environment, not
formal acceptance of the exact declared dependency stack.

## NPU kernel acceptance

Device: Ascend 910B2, physical device 3. Numerical smoke passed for slot
shifting, overlap-safe K/V materialization, direct paged scoring against a
PyTorch reference, and fused normalization/query-head-max/cross-layer
aggregation against a PyTorch reference.

| Hot path | Generic/reference | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| K/V paged copy | 0.564 ms | 0.332 ms | 1.70x |
| Direct paged score | 1.239 ms | 0.501 ms | 2.47x |
| Fused aggregate | 0.155 ms | 0.149 ms | 1.04x |
| Position/slot offset wrapper | 0.043 ms | 0.053 ms | 0.82x |

The slower offset wrapper is not selected over the generic path. These are
isolated kernel timings, not request-level throughput claims.

JIT specialization was also measured over a full 48-layer compression
transaction. Before round/length arguments were marked dynamic, a new request
length incurred an observed 5--6 second compile. In a fresh process after the
change, one cold compile covered alternating 2,176- and 6,311-token shapes;
subsequent transactions took 22--26 ms without per-length recompilation.

## Current-host service acceptance

The manager-wrapped service started with Qwen2.5-Coder-14B-Instruct, FP16,
block size 128, maximum model length 12,288, prefix caching disabled, eager
execution, and the example 2,048-token physical budget. A 6,311-token prompt
with 300 forced output tokens completed and exercised repeated compression.
During single-request generation, the server logged approximately 5% KV-cache
usage, consistent with the bounded physical window; the matched baseline
logged approximately 15%.

### Limited matched performance cell

The following cell used the same model, prompt text, generation settings,
device, host arguments, and four concurrent requests. The tiny prompt-token
count difference is reported rather than normalized away.

| Mode | Prompt + output tokens | Elapsed | Output tok/s | Total tok/s | Steady generation KV usage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plugin | 25,244 + 1,200 | 29.461 s | 40.731 | 897.578 | about 20% |
| Baseline | 25,232 + 1,200 | 28.634 s | 41.908 | 923.086 | about 60% |

The plugin is 2.8% below baseline total throughput in this cell. It must not be
described as an end-to-end speedup. For a single 6,311-token prompt plus 300
output tokens, two optimized plugin runs averaged 25.709 seconds versus
25.216 seconds for two baseline runs (plugin +2.0%). One pre-optimization
plugin observation was 35.972 seconds, showing the JIT change removed a large
plugin overhead, but it is not a statistically complete before/after matrix.

Static KV allocation was essentially unchanged (7.98 GiB plugin and 7.97 GiB
baseline) because the host reserves the cache pool at startup. The lower
request-time cache-use percentage is block-pressure evidence, not a peak-HBM
claim. TTFT/TPOT percentiles, compression-time percentiles, peak/steady HBM,
and the required multi-cell repetitions were not captured.

### Quality guardrail

A needle retrieval prompt containing 11,569 input tokens was tested. Baseline
returned the complete target `BLUE-HERON-4729`; the plugin using the example
2,048-token budget returned only `BLUE-`. A short 61-token plugin control
returned the complete target. The long-context guardrail therefore **fails**
for the default example configuration. Model-matched calibration and/or a
larger budget must be evaluated before publication or production use.

| Required row | Status |
| --- | --- |
| Score/aggregate/copy numerical smoke | PASS on 910B2 fallback environment |
| Manager-wrapped startup and repeated-compression smoke | PASS on fallback environment |
| Long-context quality guardrail | **FAIL** for default 2,048-token budget |
| Matched throughput/latency | PARTIAL — one single-request and one c=4 cell; no speedup versus baseline |
| Cache-block pressure | PASS as service-log evidence, about 20% versus 60% at c=4 |
| TTFT/TPOT, peak/steady HBM, full matrix | NOT RUN |
| Exact current dependency stack | BLOCKED by unavailable Triton-Ascend 3.6 wheel/incomplete source dependency download |

## Release decision

The repository is suitable for Extension Manager integration review as an
experimental source plugin. It is **not ready for PyPI alpha publication or a
production NPU/quality/performance claim** under the BidKV release guide: the
default long-context quality gate failed, the exact dependency stack is not
validated, and the complete performance/HBM matrix is missing. Historical 0.2
results must not be represented as 0.3 acceptance evidence.
