#!/usr/bin/env python3
"""Aggregate the three-lifecycle C0/C1 KV-compression M3 results."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any

METRICS = (
    "duration",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p99_itl_ms",
)
LOG_TIMESTAMP = re.compile(r"(?:INFO )?(\d\d-\d\d \d\d:\d\d:\d\d)")
KV_USAGE = re.compile(r"GPU KV cache usage: ([0-9.]+)%")
COMMIT = re.compile(
    r"semantic_tokens=(\d+) physical_tokens=(\d+) "
    r"source_blocks=(\d+) destination_blocks=(\d+) "
    r"released_blocks=\(([^)]*)\)"
)
EXPECTED_COMPRESSION_OPTIONS = (
    "provider='ascend_kvcompress'",
    "'method': 'triattention'",
    "'kv_budget': 2048",
    "'recompute_window': 128",
    "'protected_recent_window': 128",
    "'score_aggregation': 'mean'",
    "'layer_aggregation': 'mean'",
    "'score_chunk_size': 8192",
)


def _log_time(line: str, year: int) -> dt.datetime | None:
    match = LOG_TIMESTAMP.search(line)
    if match is None:
        return None
    return dt.datetime.strptime(f"{year}-{match.group(1)}", "%Y-%m-%d %H:%M:%S")


def _pressure_log_evidence(
    server_log: Path, pressure: dict[str, Any]
) -> dict[str, Any]:
    end = dt.datetime.strptime(pressure["date"], "%Y%m%d-%H%M%S")
    start = end - dt.timedelta(seconds=float(pressure["duration"]) + 15)
    finish = end + dt.timedelta(seconds=10)
    usages: list[float] = []
    commits: list[dict[str, int]] = []
    acknowledgements = 0
    for line in server_log.read_text(errors="replace").splitlines():
        timestamp = _log_time(line, end.year)
        if timestamp is None or not start <= timestamp <= finish:
            continue
        if usage_match := KV_USAGE.search(line):
            usages.append(float(usage_match.group(1)))
        if commit_match := COMMIT.search(line):
            released = [
                value for value in commit_match.group(5).split(",") if value.strip()
            ]
            commits.append(
                {
                    "semantic_tokens": int(commit_match.group(1)),
                    "physical_tokens": int(commit_match.group(2)),
                    "source_blocks": int(commit_match.group(3)),
                    "destination_blocks": int(commit_match.group(4)),
                    "released_blocks": len(released),
                }
            )
        if "KV cache compression commit ack" in line:
            acknowledgements += 1
    return {
        "window_start": start.isoformat(),
        "window_end": finish.isoformat(),
        "sampled_kv_peak_percent": max(usages) if usages else None,
        "compression_commits": len(commits),
        "compression_acknowledgements": acknowledgements,
        "compression_shapes": [
            dict(items)
            for items in sorted({tuple(sorted(commit.items())) for commit in commits})
        ],
    }


def _startup_config_evidence(server_log: Path, mode: str) -> dict[str, Any]:
    line = next(
        line
        for line in server_log.read_text(errors="replace").splitlines()
        if "non-default args: " in line
    )
    raw = line.split("non-default args: ", 1)[1]
    compression_present = "'kv_cache_compression_config':" in raw
    base = re.sub(
        r", 'kv_cache_compression_config': KVCacheCompressionConfig\(.*\)\}$",
        "}",
        raw,
    )
    return {
        "base_args_sha256": hashlib.sha256(base.encode()).hexdigest(),
        "compression_config_present": compression_present,
        "compression_config_frozen": (
            mode == "c1"
            and compression_present
            and all(option in raw for option in EXPECTED_COMPRESSION_OPTIONS)
        ),
    }


def _load_run(root: Path, mode: str, repeat: int) -> dict[str, Any]:
    run_dir = root / mode / f"repeat-{repeat}"
    quality = json.loads((run_dir / "quality.json").read_text())
    pressure = json.loads((run_dir / "pressure.json").read_text())
    server_log = run_dir / "server.log"
    return {
        "mode": mode,
        "repeat": repeat,
        "quality": {
            key: quality[key]
            for key in (
                "completed",
                "failed",
                "correct",
                "accuracy",
                "prompt_lengths_valid",
                "ordered_prompt_token_sha256",
                "duration_s",
                "mean_latency_ms",
                "p99_latency_ms",
            )
        },
        "pressure": {
            "completed": pressure["completed"],
            "failed": pressure["failed"],
            "total_input_tokens": pressure["total_input_tokens"],
            "total_output_tokens": pressure["total_output_tokens"],
            "input_lengths": sorted(set(pressure["input_lens"])),
            "output_lengths": sorted(set(pressure["output_lens"])),
            **{metric: pressure[metric] for metric in METRICS},
            **_pressure_log_evidence(server_log, pressure),
        },
        "startup_config": _startup_config_evidence(server_log, mode),
        "npu_released": "No running processes found in NPU 5"
        in (run_dir / "npu-after.txt").read_text(),
    }


def _median(values: list[float | int]) -> float:
    return float(statistics.median(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()

    runs = [
        _load_run(args.result_root, mode, repeat)
        for mode in ("c0", "c1")
        for repeat in (1, 2, 3)
    ]
    by_mode = {
        mode: [run for run in runs if run["mode"] == mode] for mode in ("c0", "c1")
    }
    medians: dict[str, Any] = {}
    for mode, mode_runs in by_mode.items():
        medians[mode] = {
            "quality_accuracy": _median(
                [run["quality"]["accuracy"] for run in mode_runs]
            ),
            "quality_duration_s": _median(
                [run["quality"]["duration_s"] for run in mode_runs]
            ),
            "sampled_kv_peak_percent": _median(
                [run["pressure"]["sampled_kv_peak_percent"] for run in mode_runs]
            ),
            **{
                metric: _median([run["pressure"][metric] for run in mode_runs])
                for metric in METRICS
            },
        }

    changes = {
        metric: (medians["c1"][metric] / medians["c0"][metric] - 1) * 100
        for metric in (
            "duration",
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "mean_ttft_ms",
            "p99_ttft_ms",
            "mean_tpot_ms",
            "p99_tpot_ms",
            "mean_itl_ms",
            "p99_itl_ms",
            "sampled_kv_peak_percent",
        )
    }
    expected_shape = {
        "semantic_tokens": 8192,
        "physical_tokens": 2048,
        "source_blocks": 64,
        "destination_blocks": 16,
        "released_blocks": 48,
    }
    c1_runs = by_mode["c1"]
    physical_kv_reduction = 1 - (
        expected_shape["physical_tokens"] / expected_shape["semantic_tokens"]
    )
    checks = {
        "base_service_config_identical": len(
            {run["startup_config"]["base_args_sha256"] for run in runs}
        )
        == 1,
        "c0_has_no_compression_config": all(
            not run["startup_config"]["compression_config_present"]
            for run in by_mode["c0"]
        ),
        "c1_compression_config_frozen": all(
            run["startup_config"]["compression_config_frozen"] for run in c1_runs
        ),
        "all_quality_requests_succeeded": all(
            run["quality"]["completed"] == 64 and run["quality"]["failed"] == 0
            for run in runs
        ),
        "all_quality_prompt_lengths_valid": all(
            run["quality"]["prompt_lengths_valid"] for run in runs
        ),
        "quality_requests_identical": len(
            {run["quality"]["ordered_prompt_token_sha256"] for run in runs}
        )
        == 1,
        "quality_drop_within_one_percentage_point": (
            medians["c0"]["quality_accuracy"] - medians["c1"]["quality_accuracy"]
            <= 0.01
        ),
        "all_pressure_requests_succeeded": all(
            run["pressure"]["completed"] == 16
            and run["pressure"]["failed"] == 0
            and run["pressure"]["input_lengths"] == [8192]
            and run["pressure"]["output_lengths"] == [64]
            for run in runs
        ),
        "all_pressure_compression_commits_acknowledged": all(
            run["pressure"]["compression_commits"] == 16
            and run["pressure"]["compression_acknowledgements"] == 16
            and run["pressure"]["compression_shapes"] == [expected_shape]
            for run in c1_runs
        ),
        "physical_kv_reduction_at_least_twenty_percent": physical_kv_reduction >= 0.20,
        "npu_released_after_every_lifecycle": all(run["npu_released"] for run in runs),
    }
    output = {
        "schema_version": 1,
        "test_id": "ascend-kv-compress-m3-v3.8-2026-08-28",
        "result_root": str(args.result_root.resolve()),
        "runs": runs,
        "medians": medians,
        "c1_vs_c0_percent": changes,
        "compression_shape": expected_shape,
        "physical_kv_reduction_percent": physical_kv_reduction * 100,
        "checks": checks,
        "verdict": "pass" if all(checks.values()) else "fail",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
