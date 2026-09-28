# Qwen3.5 fixed-host handoff (2026-09-28)

This bundle preserves the complete compatibility/qualification trail for
KVCompress on the unified Qwen3.5 Frontier runtime. It is not a formal
Frontier result: the passing window is 60 seconds and another independently
owned experiment was active on the other NPU pair.

## Exact source provenance

- vLLM: `d0f22d2bda562156e4dbf433ce645e1769b4f804`
- vLLM-Ascend: `03766ac696fde5ab1980d80ca0b8543d3580c989`
- KVCompress tested working-tree content, committed immediately afterward:
  `2ca0f9335399a698342285870df1514e9c519bd4`
- Extension Manager: `9c487b02d088f2cbe95cd8f0ce1024c93d9dcc64`
- model revision: `712cf74392b05026a6db2bf213d343747d1f6d45`
- workload SHA256: `8044561ffa1bb430bea8f778ef814d96649321e1a92654b95f64263b996d5e85`
- tokenizer fingerprint: `3f9ca78537850303ee04bfa6640c020be89723c62f37121c0f27a4c0babc53e0`

## Passing mechanism qualification

`raw/qualification/retry5/c4-60s/summary.json` is a valid C4/60-second unified
SWE qualification: 8,065 output tokens, 134.41666666666666 output tok/s,
P90 decode 70.4122576763652 tok/s/user, 27 requests completed in-window,
4 drained, and 0 failed. This throughput is diagnostic only.

The matching server log records three scheduler compression commits and six
worker acknowledgements (three on each TP rank). It also records APC activity,
real MTP draft/accept counters, async scheduling, Mamba align, exact per-card KV
budget, and successful FULL plus PIECEWISE graph capture. The post-window
metrics record 79,872 prefix-cache hit tokens of 194,863 queried tokens, 8,714
MTP draft tokens, and 8,083 accepted tokens.

The copied raw server metadata still declares the earlier port commit
`0e0fcdc3864819ccf77cb1757ca6e972b22c7548`, because that was the file supplied
to the harness. The server actually imported the then-uncommitted working tree;
that exact code content was committed as `2ca0f9335399a698342285870df1514e9c519bd4`
immediately after the run. Repeat qualification from the clean commit before
formal measurement so future evidence has no such provenance mismatch.

## Failure trail retained intentionally

- `live`: missing Accelerate on the multi-device auto-calibration path.
- `retry1`: same dependency failure before the host was provisioned.
- `retry2`: generic `AutoModel` instantiated the unused Qwen3.5 vision tower.
- `retry3`: reusing the outer multimodal config did not avoid the worker-side
  vision config defect.
- `retry4`: service reached health and graph capture, then the first real
  requests exposed the fixed-host coordinator signature difference.
- `retry5`: passed after text-only calibration loading and signature-adaptive
  coordinator hooks.

## Resume conditions

Use this commit (or the merged replacement), install the optional calibration
extra/Accelerate in the pre-provisioned host without re-resolving the Ascend
stack, and adjust the copied absolute paths in `raw/manager-config.json` and
`raw/start-qualification-server.sh` for the new container. The generated
calibration artifact is included only for audit; reuse it only if its model
fingerprint validates against the same exact checkpoint.

Before any formal window, require all four NPUs to be idle, repeat Manager
inspect/check/plan/status, run a clean 60-second qualification, and then run
C1/C2/C4/C8/C16 for exactly 900 seconds each. Compare only with the published
unified Native series; do not treat the passing 60-second number as Frontier
data.
