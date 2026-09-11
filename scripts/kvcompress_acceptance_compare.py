#!/usr/bin/env python3
"""Compare three cold-start B0/B1 lifecycles for long-context acceptance."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

PRIMARY_METRICS = (
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_e2e_ms",
    "p95_e2e_ms",
    "p99_e2e_ms",
)


def _load(paths: list[Path], label: str) -> list[dict[str, Any]]:
    if len(paths) != 3:
        raise ValueError(f"{label} requires exactly three independent lifecycles")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(row["run_label"] != label for row in rows):
        raise ValueError(f"one or more {label} files have the wrong run_label")
    return rows


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row["metrics"].get(key) for row in rows]
    if any(value is None for value in values):
        return None
    return float(statistics.median(values))


def _compression_reduction(rows: list[dict[str, Any]]) -> float | None:
    reductions = []
    for row in rows:
        for commit in row["compression"]["scheduler_commits"]:
            semantic = commit["semantic_tokens"]
            physical = commit["physical_tokens"]
            if semantic > 0:
                reductions.append(1 - physical / semantic)
    return min(reductions) if reductions else None


def _all_requests_clean(rows: list[dict[str, Any]]) -> bool:
    return all(
        row["metrics"]["failed"] == 0
        and row["metrics"]["silent_truncations"] == 0
        and row["metrics"]["completed"] == row["metrics"]["requests"]
        for row in rows
    )


def _a3_stable(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        stability = row.get("stability")
        if stability is None or row["metrics"]["completed"] < 24:
            return False
        if row["metrics"]["duration_s"] < 1800:
            return False
        if stability["throughput_cv"] is None or stability["throughput_cv"] > 0.05:
            return False
        for field in ("median_ttft_drift", "median_tpot_drift"):
            if stability[field] is None or stability[field] > 0.10:
                return False
        for field in ("p99_ttft_drift", "p99_tpot_drift"):
            if stability[field] is None or stability[field] > 0.20:
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--plugin", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-throughput-regression-percent", type=float, default=1.0)
    args = parser.parse_args()
    baseline = _load(args.baseline, "B0")
    plugin = _load(args.plugin, "B1")
    all_rows = baseline + plugin
    identity = {
        (row["profile"], row["model"], row["dataset_sha256"]) for row in all_rows
    }
    if len(identity) != 1:
        raise ValueError("B0/B1 profile, model, or dataset hash differs")
    medians = {
        label: {metric: _median(rows, metric) for metric in PRIMARY_METRICS}
        for label, rows in (("B0", baseline), ("B1", plugin))
    }
    throughput_ratios = {
        metric: medians["B1"][metric] / medians["B0"][metric]
        for metric in (
            "request_throughput",
            "input_throughput",
            "output_throughput",
            "total_token_throughput",
        )
        if medians["B0"][metric] and medians["B1"][metric] is not None
    }
    quality = {
        label: float(statistics.median(row["metrics"]["accuracy"] for row in rows))
        for label, rows in (("B0", baseline), ("B1", plugin))
    }
    reduction = _compression_reduction(plugin)
    minimum_ratio = 1 - args.max_throughput_regression_percent / 100
    is_a3 = all(row["profile"].startswith("A3") for row in all_rows)
    checks = {
        "three_independent_lifecycles_each": True,
        "frozen_workload_identity": len(identity) == 1,
        "baseline_requests_clean": _all_requests_clean(baseline),
        "plugin_requests_clean": _all_requests_clean(plugin),
        "quality_drop_within_one_percentage_point": quality["B0"] - quality["B1"]
        <= 0.01,
        "total_throughput_within_declared_regression_budget": throughput_ratios.get(
            "total_token_throughput", 0
        )
        >= minimum_ratio,
        "physical_kv_reduction_at_least_twenty_percent": reduction is not None
        and reduction >= 0.20,
        "scheduler_worker_commit_counts_match": all(
            len(row["compression"]["scheduler_commits"])
            == row["compression"]["worker_acks"]
            for row in plugin
        ),
        "a3_window_stability": not is_a3
        or (_a3_stable(baseline) and _a3_stable(plugin)),
    }
    output = {
        "schema_version": 1,
        "test_id": "kvcompress-v4.6-paired-comparison-v1",
        "identity": list(identity)[0],
        "medians": medians,
        "throughput_ratios": throughput_ratios,
        "quality_accuracy": quality,
        "minimum_observed_physical_kv_reduction": reduction,
        "checks": checks,
        "verdict": "pass" if all(checks.values()) else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
