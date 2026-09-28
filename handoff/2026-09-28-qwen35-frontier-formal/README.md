# Qwen3.5 KVCompress Frontier formal evidence

This directory records the clean qualification and five independent 900-second
Frontier windows executed on 2026-09-28. The candidate uses the tested source
commit `2ca0f9335399a698342285870df1514e9c519bd4`; the branch evidence commit is
separate from the imported plugin source.

## Fixed contract

- Model: Qwen3.5-35B-A3B, revision
  `712cf74392b05026a6db2bf213d343747d1f6d45`
- Workload: `swe-prefix-reuse/v1`, prepared SHA-256
  `8044561ffa1bb430bea8f778ef814d96649321e1a92654b95f64263b996d5e85`
- Tokenizer fingerprint:
  `3f9ca78537850303ee04bfa6640c020be89723c62f37121c0f27a4c0babc53e0`
- Hardware/runtime: Ascend 910B2, BF16, TP2/PP1/DP1, expert parallel off,
  vLLM `d0f22d2bda562156e4dbf433ce645e1769b4f804`, vLLM-Ascend
  `03766ac696fde5ab1980d80ca0b8543d3580c989`
- Serving: max model length 262144, max sequences 16, max batched tokens
  4096, APC on, async scheduling on, Mamba cache `align`, native MTP with two
  draft tokens, thinking on, temperature 0, `FULL_AND_PIECEWISE`, and
  26038239232 bytes of KV cache per chip

The 14 checkpoint shard hashes were checked against the ModelScope response
metadata for the required revision. The container-local checkpoint manifest
includes five read-only mounted shards and nine local shards. A new calibration
artifact was generated from that verified checkpoint because the earlier
artifact correctly rejected the new container path/mtime manifest. The earlier
artifact and failed validation log are preserved without modification.

## Qualification

The clean C4/60 qualification is valid: 2,991 output tokens, 49.85 output
tokens/s, 18 requests completed in the window, four drained, and zero failed.
It recorded one scheduler compression commit with exactly one acknowledgement
from each TP rank. APC queried 109,440 tokens and hit 32,768; native MTP drafted
3,572 tokens and accepted 3,168. The low qualification throughput includes a
cold first-request kernel compile and is not a formal result.

## Formal results

The only comparison baseline is `swe-unified-native-20260927`.

| Concurrency | Candidate output tok/s | Native output tok/s | Point delta | P90 decode tok/s/user | Completed + drain | Failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 94.57222222222222 | 91.54555555555555 | +3.3061863553057957% | 110.35405668201226 | 151 + 1 | 0 |
| 2 | 136.70555555555555 | 152.7211111111111 | -10.486798739896253% | 88.41768829712161 | 188 + 2 | 0 |
| 4 | 206.5988888888889 | 217.47555555555556 | -5.001328374070136% | 77.34354504472113 | 295 + 4 | 0 |
| 8 | 286.1011111111111 | 291.1066666666667 | -1.7194919006397136% | 61.13461820415613 | 421 + 8 | 0 |
| 16 | 347.75333333333333 | 359.75 | -3.3347231873986583% | 34.93487505412306 | 548 + 16 | 0 |

The dynamically computed five-point geometric-mean throughput gain is
**-3.5518795617593635%**, using
`(geomean(candidate / native) - 1) * 100`. This is a valid negative Frontier
result, not an optimization claim.

Across the formal service lifetime, all 434 scheduler compression commits had
exactly two acknowledgements: 434 from TP0 and 434 from TP1. Every window has
positive APC and native-MTP counter deltas and zero preemptions. All summaries
are valid, cover exactly 900 measured seconds, report zero request failures,
and retain raw request/token timing data.

## Evidence map

- `raw/formal/`: launch identity, service log, five request streams, summaries,
  metrics, NPU snapshots, window/drain boundaries, analyses, shutdown and
  resource-release evidence
- `raw/qualification-clean/`: clean C4/60 qualification and release evidence
- `raw/calibration/`: regenerated artifact, metadata, logs, NPU boundaries, and
  checksum
- `raw/model/`: target-revision shard manifest and full verification output
- `raw/manager/`: Extension Manager inspect/check/plan/configure/enable/status
- `raw/safety/`: pre-run host, process, port, source, memory, disk, model, and
  four-NPU audit
- `raw/qualification-calibration-mismatch-attempt/`: preserved fail-closed
  validation attempt that motivated regeneration
- `raw/SHA256SUMS`: checksum manifest for the complete raw evidence bundle
