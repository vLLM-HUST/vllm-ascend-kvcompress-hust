# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from scripts.kvcompress_frontier_agentx_pair import compare


def _run(root: Path, arm: str, budget: int) -> Path:
    directory = root / arm
    (directory / "aiperf").mkdir(parents=True)
    run = {
        "status": "completed",
        "returncode": 0,
        "upstream_reports": [
            {"submission_valid": True, "submission_invalid_reasons": []}
        ],
        "profile": "smoke",
        "one_hour_measurement": False,
        "protocol": {"dataset": "pinned"},
        "concurrent_agent_clients": 4,
        "dataset_receipt": {"revision": "same"},
        "wrapper_commit": "same",
        "wrapper_tracked_dirty": False,
        "target": {
            "url": "http://127.0.0.1:8012",
            "model": "qwen3.5-35b-a3b",
            "model_revision": "same",
            "tokenizer": "/models/qwen",
            "tokenizer_revision": "same",
            "hardware": {"device_count": 2},
            "max_model_len": 262144,
            "host_kv_budget_gib": 0,
            "precision": {"weights": "bfloat16"},
            "speculative_decoding": {"enabled": False},
            "engine": {
                "revision": "same",
                "launch_command_without_secrets": (
                    "vllm serve VLLM_ASCEND_KVCOMPRESS_CONFIG=/configs/" + arm
                ),
            },
            "serving": {
                "tensor_parallel_size": 2,
                "prefix_caching": True,
                "kv_budget_tokens": budget,
            },
        },
        "command": ["aiperf", "--benchmark-duration", "900", "--artifact-dir", arm],
    }
    scale = 1 if arm == "B0" else 1.1
    report = {
        "metadata": {
            "submission_valid": True,
            "dataset": {"name": "pinned"},
            "metric_duration_coverage": [{"expected_duration_seconds": 900}],
        },
        "was_cancelled": False,
        "error_summary": [],
        "output_token_throughput": {"avg": 100 * scale},
        "request_throughput": {"avg": 1 * scale},
        "output_token_throughput_per_user": {
            "p50": 50 * scale,
            "p90": 60 * scale,
        },
        "time_to_first_token": {"p95": 1000 / scale},
        "inter_token_latency": {"p90": 20 / scale},
        "request_latency": {"p95": 5000 / scale},
        "request_count": {"avg": 50 * scale},
        "input_sequence_length": {"max": 100000 * scale},
    }
    path = directory / "run.json"
    path.write_text(json.dumps(run), encoding="utf-8")
    (directory / "aiperf" / "profile_export_aiperf.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return path


def test_compare_maps_agentx_metrics_and_pair_identity(tmp_path: Path) -> None:
    b0 = _run(tmp_path, "B0", 262144)
    b1 = _run(tmp_path, "B1", 8192)
    result = compare(b0, b1)
    assert result["metrics"]["output_tokens_per_second"][
        "delta_percent"
    ] == pytest.approx(10)
    assert result["metrics"]["time_to_first_token_p95_ms"]["delta_percent"] < 0

    candidate = json.loads(b1.read_text(encoding="utf-8"))
    candidate["target"]["model_revision"] = "different"
    b1.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="target.model_revision"):
        compare(b0, b1)


def test_compare_rejects_invalid_agentx_arm(tmp_path: Path) -> None:
    b0 = _run(tmp_path, "B0", 262144)
    b1 = _run(tmp_path, "B1", 8192)
    report_path = b1.parent / "aiperf" / "profile_export_aiperf.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["metadata"]["submission_valid"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="validity gate"):
        compare(b0, b1)
