# SPDX-License-Identifier: Apache-2.0
"""Read-only comparison of two pinned AgentX 900-second smoke runs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def _load_arm(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run = json.loads(path.read_text())
    report_path = path.parent / "aiperf" / "profile_export_aiperf.json"
    report = json.loads(report_path.read_text())
    if run.get("status") != "completed" or run.get("returncode") != 0:
        raise ValueError(f"incomplete AgentX run: {path}")
    upstream = run.get("upstream_reports", [])
    if (
        len(upstream) != 1
        or upstream[0].get("submission_valid") is not True
        or upstream[0].get("submission_invalid_reasons")
        or report.get("metadata", {}).get("submission_valid") is not True
        or report.get("was_cancelled") is not False
        or report.get("error_summary")
    ):
        raise ValueError(f"AgentX validity gate failed: {path}")
    coverage = report["metadata"].get("metric_duration_coverage", [])
    if len(coverage) != 1 or coverage[0].get("expected_duration_seconds") != 900:
        raise ValueError(f"not an AgentX 900-second measurement: {path}")
    if run.get("profile") != "smoke" or run.get("one_hour_measurement") is not False:
        raise ValueError(f"not an AgentX 900-second smoke profile: {path}")
    return run, report


def _matched_command(run: dict[str, Any]) -> str:
    launch = run["target"]["engine"]["launch_command_without_secrets"]
    return re.sub(
        r"VLLM_(?:ASCEND_KVCOMPRESS_CONFIG|HUST_EXT_CONFIG)=\S+",
        "VLLM_HUST_EXT_CONFIG=<arm-config>",
        launch,
    )


def _client_command(run: dict[str, Any]) -> list[str]:
    command = list(run["command"])
    if "--artifact-dir" in command:
        index = command.index("--artifact-dir")
        del command[index : index + 2]
    return command


def _metric(report: dict[str, Any], name: str, field: str) -> float:
    value = float(report[name][field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite AgentX metric {name}.{field}")
    return value


def compare(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline, b_report = _load_arm(baseline_path)
    candidate, c_report = _load_arm(candidate_path)
    for field in (
        "protocol",
        "profile",
        "concurrent_agent_clients",
        "dataset_receipt",
        "wrapper_commit",
        "wrapper_tracked_dirty",
    ):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"AgentX pair differs in {field}")
    for field in (
        "url",
        "model",
        "model_revision",
        "tokenizer",
        "tokenizer_revision",
        "hardware",
        "max_model_len",
        "host_kv_budget_gib",
        "precision",
        "speculative_decoding",
        "plugin",
        "runtime_environment",
    ):
        if baseline["target"].get(field) != candidate["target"].get(field):
            raise ValueError(f"AgentX pair differs in target.{field}")
    for field in ("name", "revision"):
        if baseline["target"]["engine"].get(field) != candidate["target"]["engine"].get(
            field
        ):
            raise ValueError(f"AgentX pair differs in engine {field}")
    if _matched_command(baseline) != _matched_command(candidate):
        raise ValueError("AgentX pair differs in server launch beyond arm config")
    if _client_command(baseline) != _client_command(candidate):
        raise ValueError("AgentX pair differs in client command")
    b_serving = dict(baseline["target"]["serving"])
    c_serving = dict(candidate["target"]["serving"])
    b_budget = int(b_serving.pop("kv_budget_tokens"))
    c_budget = int(c_serving.pop("kv_budget_tokens"))
    b_serving.pop("compression_active", None)
    c_serving.pop("compression_active", None)
    if b_serving != c_serving:
        raise ValueError("AgentX pair differs in serving settings beyond compression")
    max_len = int(baseline["target"]["max_model_len"])
    if b_budget < max_len or c_budget >= max_len:
        raise ValueError("AgentX B0/B1 compression budgets are not a valid pair")
    if baseline["concurrent_agent_clients"] != 4:
        raise ValueError("this comparison expects the frozen C4 cohort")
    if b_report["metadata"]["dataset"] != c_report["metadata"]["dataset"]:
        raise ValueError("AgentX reports differ in dataset")

    names = {
        "output_tokens_per_second": ("output_token_throughput", "avg"),
        "requests_per_second": ("request_throughput", "avg"),
        "output_tokens_per_second_per_user_p50": (
            "output_token_throughput_per_user",
            "p50",
        ),
        "decode_p90_tokens_per_second": (
            "output_token_throughput_per_user",
            "p90",
        ),
        "time_to_first_token_p95_ms": ("time_to_first_token", "p95"),
        "inter_token_latency_p90_ms": ("inter_token_latency", "p90"),
        "request_latency_p95_ms": ("request_latency", "p95"),
        "request_count": ("request_count", "avg"),
        "max_input_tokens": ("input_sequence_length", "max"),
    }
    metrics: dict[str, dict[str, float]] = {}
    for label, (name, field) in names.items():
        b_value = _metric(b_report, name, field)
        c_value = _metric(c_report, name, field)
        metrics[label] = {
            "baseline": b_value,
            "candidate": c_value,
            "delta_percent": 100 * (c_value / b_value - 1) if b_value else float("nan"),
        }
    chip_count = int(baseline["target"]["hardware"]["device_count"])
    if chip_count <= 0:
        raise ValueError("AgentX pair must declare positive allocated device_count")
    total = metrics["output_tokens_per_second"]
    metrics["output_tokens_per_second_per_chip"] = {
        "baseline": total["baseline"] / chip_count,
        "candidate": total["candidate"] / chip_count,
        "delta_percent": total["delta_percent"],
    }
    return {
        "cohort": "agentx-256k-qwen3.5-35b-a3b-bf16-tp2-c4-900s-no-mtp",
        "baseline_run": str(baseline_path),
        "candidate_run": str(candidate_path),
        "candidate_valid": True,
        "metrics": metrics,
        "limitations": [
            "900-second smoke, not the formal one-hour AgentX result",
            "different closed-loop request mixes may complete in equal windows",
            "synthetic content does not assess semantic answer quality",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
