#!/usr/bin/env python3
"""Validate and compare two unpooled official SWE Prefix Reuse runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_run(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads((directory / "summary.json").read_text(encoding="utf-8")),
        json.loads((directory / "config.json").read_text(encoding="utf-8")),
    )


def _valid(summary: dict[str, Any], config: dict[str, Any]) -> bool:
    return (
        summary.get("valid") is True
        and summary.get("aborted") is False
        and summary.get("failed_requests") == 0
        and summary.get("measurement_seconds") == 900
        and summary.get("planned_measurement_seconds") == 900
        and summary.get("decode_speed_samples", 0) > 0
        and config.get("duration") == 900
        and config.get("schema") == "swe-prefix-reuse/v1"
    )


def _server_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Read the original SWE handoff metadata without rewriting the raw report."""
    if isinstance(metadata.get("mod"), dict):
        return metadata
    required = (
        "runtime_base_commits",
        "model",
        "model_path",
        "model_revision",
        "tokenizer_fingerprint",
        "precision",
        "hardware",
        "serving_chips",
        "serving_npus",
        "runtime_environment",
        "server_configuration",
        "mod_repository",
        "mod_revision",
        "mod_wheel_sha256",
        "calibration_sha256",
        "launch_command_redacted",
    )
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"SWE server metadata missing {', '.join(missing)}")
    config = metadata["server_configuration"]
    for field in ("kvcompress_kv_budget", "kvcompress_recompute_window"):
        if field not in config:
            raise ValueError(f"SWE server configuration missing {field}")
    environment = dict(metadata["runtime_environment"])
    environment.pop("VLLM_HUST_EXT_CONFIG", None)
    return {
        "engine": {"revision": metadata["runtime_base_commits"]["vllm"]},
        "backend": {"revision": metadata["runtime_base_commits"]["vllm-ascend"]},
        "model": {
            "name": metadata["model"],
            "path": metadata["model_path"],
            "revision": metadata["model_revision"],
        },
        "tokenizer": {
            "revision": metadata["model_revision"],
            "fingerprint": metadata["tokenizer_fingerprint"],
        },
        "precision": metadata["precision"],
        "hardware": {
            "accelerator": metadata["hardware"],
            "accelerator_count": metadata["serving_chips"],
            "serving_npus": metadata["serving_npus"],
        },
        "runtime": {
            "extension_manager_revision": metadata.get("extension_manager_revision"),
            "environment": environment,
            "worker_class": metadata.get("worker_class"),
        },
        "mod": {
            "repository": metadata["mod_repository"],
            "revision": metadata["mod_revision"],
            "wheel_sha256": metadata["mod_wheel_sha256"],
            "calibration_sha256": metadata["calibration_sha256"],
            "calibration_path": metadata.get("calibration_artifact"),
            "kv_budget_tokens": config["kvcompress_kv_budget"],
            "recompute_window_tokens": config["kvcompress_recompute_window"],
        },
        "server_configuration": config,
        "launch_command_without_secrets": metadata["launch_command_redacted"],
    }


def compare(baseline: Path, candidate: Path) -> dict[str, Any]:
    b0, c0 = _read_run(baseline)
    b1, c1 = _read_run(candidate)
    if not _valid(b0, c0) or not _valid(b1, c1):
        raise ValueError("both arms need complete, valid 900-second SWE reports")

    same_fields = (
        "tool_version",
        "schema",
        "model",
        "workload_sha256",
        "tokenizer",
        "concurrency",
        "data_parallel_size",
        "routing_policy",
        "duration",
        "chips",
        "seed",
        "timeout",
        "server_max_context",
        "cache_policy",
        "window_policy",
    )
    for field in same_fields:
        if c0.get(field) != c1.get(field):
            raise ValueError(f"SWE arms differ in {field}")
    m0 = _server_metadata(c0["server_metadata"])
    m1 = _server_metadata(c1["server_metadata"])
    for field in (
        "engine",
        "backend",
        "model",
        "tokenizer",
        "precision",
        "hardware",
        "runtime",
    ):
        if m0.get(field) != m1.get(field):
            raise ValueError(f"SWE server arms differ in {field}")
    settings0 = dict(m0.get("server_configuration", {}))
    settings1 = dict(m1.get("server_configuration", {}))
    for settings in (settings0, settings1):
        for field in (
            "kvcompress_kv_budget",
            "kvcompress_recompute_window",
            "compression_eligible",
        ):
            settings.pop(field, None)
    if settings0 != settings1:
        raise ValueError("SWE server arms differ in serving configuration")
    mod0, mod1 = dict(m0["mod"]), dict(m1["mod"])
    for mod in (mod0, mod1):
        for field in (
            "kv_budget_tokens",
            "recompute_window_tokens",
            "compression_active",
            "baseline_reason",
        ):
            mod.pop(field, None)
    if mod0 != mod1:
        raise ValueError("SWE server arms differ in MOD identity or source")
    launch0 = m0.get("launch_command_without_secrets")
    launch1 = m1.get("launch_command_without_secrets")
    if launch0 is not None or launch1 is not None:
        if not isinstance(launch0, str) or not isinstance(launch1, str):
            raise ValueError("both SWE arms must declare a server launch command")
        normalized0 = re.sub(
            r"VLLM_ASCEND_KVCOMPRESS_CONFIG=\S+",
            "VLLM_ASCEND_KVCOMPRESS_CONFIG=<arm-config>",
            launch0,
        )
        normalized1 = re.sub(
            r"VLLM_ASCEND_KVCOMPRESS_CONFIG=\S+",
            "VLLM_ASCEND_KVCOMPRESS_CONFIG=<arm-config>",
            launch1,
        )
        if normalized0 != normalized1:
            raise ValueError("SWE server arms differ beyond compression config")
    if c0["run_id"] == c1["run_id"]:
        raise ValueError("SWE arms reused a run ID")
    if c0["chips"] != m0["hardware"]["accelerator_count"]:
        raise ValueError("client chip count differs from deployed hardware")
    capacity = c0["server_max_context"]
    threshold_b0 = m0["mod"]["kv_budget_tokens"] + m0["mod"]["recompute_window_tokens"]
    threshold_b1 = m1["mod"]["kv_budget_tokens"] + m1["mod"]["recompute_window_tokens"]
    if threshold_b0 <= capacity or threshold_b1 >= capacity:
        raise ValueError("arms do not separate compression-off and compression-on")

    metrics = {
        "output_tps": "output_tokens_per_second",
        "decode_p90_tps": "decode_tokens_per_second_p90",
        "ttft_p95_ms": "ttft_seconds_p95",
    }
    result: dict[str, Any] = {
        "workload_sha256": c0["workload_sha256"],
        "concurrency": c0["concurrency"],
        "chips": c0["chips"],
        "measurement_seconds": 900,
        "arms": {},
        "deltas": {},
    }
    for arm, summary, config in (("B0", b0, c0), ("B1", b1, c1)):
        result["arms"][arm] = {
            "run_id": config["run_id"],
            "requests_started": summary["requests_started"],
            "requests_completed_in_window": summary["requests_completed_in_window"],
            "max_prompt_tokens_observed": summary["max_prompt_tokens_observed"],
            "mean_client_inflight": summary["mean_client_inflight"],
            "full_concurrency_fraction": summary["full_concurrency_fraction"],
            "output_tps": summary[metrics["output_tps"]],
            "output_tps_per_chip": summary["output_tokens_per_second_per_chip"],
            "decode_p90_tps": summary[metrics["decode_p90_tps"]],
            "ttft_p95_ms": summary[metrics["ttft_p95_ms"]] * 1000,
        }
    for key in ("output_tps", "decode_p90_tps", "ttft_p95_ms"):
        base = result["arms"]["B0"][key]
        other = result["arms"]["B1"][key]
        result["deltas"][f"{key}_pct"] = (other / base - 1) * 100
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(compare(args.baseline, args.candidate), indent=2, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
