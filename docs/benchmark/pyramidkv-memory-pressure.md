# PyramidKV fixed-budget memory-pressure protocol

English | [简体中文](pyramidkv-memory-pressure.zh.md)

This is a separate exploratory scenario for PyramidKV issue #1, following the
[metadata net-cost protocol](pyramidkv-net-cost.md). It tests whether recycled
request-owned KV blocks yield a service benefit under a tighter fixed budget.
It does not replace the original 8 GiB series or establish maximum capacity.

Freeze exactly one budget before this series: **2 GiB KV per worker**, in both
native OFF and PyramidKV ON. Retain the qualified Qwen3.5-35B-A3B BF16/TP2,
910B2, 32768 context, four sequences, 4096 batched tokens, MTP2, APC, async,
FULL_AND_PIECEWISE, 1368/beta-2 method and selected-page materialization. Both
arms use the same frozen opt-in batched metadata source and flag. Change no
selection or generation parameters, do not search for a favorable budget, and
stop for investigation on startup failure or failed requests without an eager
fallback or silently reduced context.

Use four fresh processes in OFF/ON/ON/OFF order, with the same saved 137 synthetic
cases and 12 QA smoke cases as the 8 GiB series. Keep the 38 predefined synthetic
exclusions, 99 measured requests and identical groups. Each synthetic case has
128 forced output tokens; inputs are 1024/8192/24576 with concurrency 1/4 and
three cohorts per cell. Unique API cache salts isolate different requests. Reuse
exact input IDs/salts across processes. Because preemption can recompute a
request, preserve and report cached-token counters instead of requiring them to
remain zero after resumption. Nonzero counters require investigation of the
recomputation path and must not be described as an entirely uncached run.

The preregistered primary comparison is **24576 input tokens / concurrency 4**:
pooled output tokens over cohort wall time. Report all six cells, individual
process/cohort values, TTFT, TPOT, end-to-end latency, failures, counter deltas for preemption and
MTP acceptance, observed HBM, and KV-pool usage. Do not select the best cell after
measurement. A benefit in this budget-constrained scenario does not imply a
benefit with 8 GiB, lower device HBM at equal reservation, an SLO guarantee, or a
measured maximum concurrency. A negative result remains a result.

Verify scheduler transactions and both TP acknowledgements. In a pressure run,
preemption may create multiple transaction lifecycles for one API request; retain
and inspect all of them rather than requiring exactly one or deleting duplicates.
For each committed lifecycle, require matching semantic/physical lengths and both
worker acknowledgements; explain any extra recomputation. Preserve raw SSE,
server logs, metrics, process/source identities, startup observations and exact
contracts in benchmark. The 12 QA examples are only a smoke; the existing
300-example quality evaluation remains separate. No tests, lint, tokenization,
or data audits may overlap measured requests.
