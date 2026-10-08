# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from scripts.kvcompress_frontier_swe_pair import compare


def _run(root: Path, arm: str, budget: int) -> Path:
    directory = root / arm
    directory.mkdir()
    summary = {
        "valid": True,
        "aborted": False,
        "failed_requests": 0,
        "measurement_seconds": 900,
        "planned_measurement_seconds": 900,
        "decode_speed_samples": 10,
        "requests_started": 14,
        "requests_completed_in_window": 10,
        "max_prompt_tokens_observed": 20000,
        "mean_client_inflight": 3.9,
        "full_concurrency_fraction": 0.9,
        "output_tokens_per_second": 100 if arm == "B0" else 110,
        "output_tokens_per_second_per_chip": 50 if arm == "B0" else 55,
        "decode_tokens_per_second_p90": 50 if arm == "B0" else 55,
        "ttft_seconds_p95": 2 if arm == "B0" else 1.5,
    }
    config = {
        "schema": "swe-prefix-reuse/v1",
        "run_id": arm,
        "duration": 900,
        "workload_sha256": "same-prepared-file",
        "concurrency": 4,
        "chips": 2,
        "server_max_context": 262144,
        "server_metadata": {
            "engine": {"commit": "a"},
            "backend": {"commit": "b"},
            "model": {"revision": "c"},
            "tokenizer": {"revision": "c"},
            "precision": {"weights": "bfloat16"},
            "hardware": {"accelerator_count": 2},
            "runtime": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
            "mod": {"kv_budget_tokens": budget, "recompute_window_tokens": 128},
            "launch_command_without_secrets": (
                "VLLM_ASCEND_KVCOMPRESS_CONFIG=/configs/" + arm + " vllm serve model"
            ),
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def test_compare_maps_official_metrics_and_pairs_arms(tmp_path: Path) -> None:
    b0 = _run(tmp_path, "B0", 262144)
    b1 = _run(tmp_path, "B1", 8192)
    result = compare(b0, b1)
    assert result["arms"]["B0"]["output_tps"] == 100
    assert result["arms"]["B1"]["output_tps_per_chip"] == 55
    assert result["arms"]["B1"]["ttft_p95_ms"] == 1500
    assert result["deltas"]["decode_p90_tps_pct"] == pytest.approx(10)

    candidate_config = b1 / "config.json"
    config = json.loads(candidate_config.read_text(encoding="utf-8"))
    config["workload_sha256"] = "different-file"
    candidate_config.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="workload_sha256"):
        compare(b0, b1)

    config["workload_sha256"] = "same-prepared-file"
    config["server_metadata"]["mod"]["scorer_source_sha256"] = "different-code"
    candidate_config.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="MOD identity or source"):
        compare(b0, b1)

    config["server_metadata"]["mod"].pop("scorer_source_sha256")
    config["server_metadata"]["launch_command_without_secrets"] += " --different"
    candidate_config.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="beyond compression config"):
        compare(b0, b1)


def test_compare_rejects_invalid_or_unclear_compression_arms(tmp_path: Path) -> None:
    b0 = _run(tmp_path, "B0", 262144)
    b1 = _run(tmp_path, "B1", 8192)
    baseline_summary = b0 / "summary.json"
    summary = json.loads(baseline_summary.read_text(encoding="utf-8"))
    summary["failed_requests"] = 1
    baseline_summary.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="valid 900-second"):
        compare(b0, b1)

    summary["failed_requests"] = 0
    baseline_summary.write_text(json.dumps(summary), encoding="utf-8")
    baseline_config = b0 / "config.json"
    config = json.loads(baseline_config.read_text(encoding="utf-8"))
    config["server_metadata"]["mod"]["kv_budget_tokens"] = 8192
    baseline_config.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="compression-off"):
        compare(b0, b1)


def test_compare_accepts_original_flat_handoff_metadata(tmp_path: Path) -> None:
    b0 = _run(tmp_path, "B0", 262144)
    b1 = _run(tmp_path, "B1", 8192)
    for directory, budget in ((b0, 262144), (b1, 8192)):
        path = directory / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["server_metadata"] = {
            "runtime_base_commits": {"vllm": "a", "vllm-ascend": "b"},
            "model": "Qwen3.5-35B-A3B",
            "model_path": "/models/qwen",
            "model_revision": "c",
            "tokenizer_fingerprint": "d",
            "precision": "bfloat16",
            "hardware": "Ascend 910B2",
            "serving_chips": 2,
            "serving_npus": [2, 7],
            "runtime_environment": {
                "TORCH_CACHING_PRECOMPILE": "0",
                "VLLM_HUST_EXT_CONFIG": f"/configs/{directory.name}",
            },
            "server_configuration": {
                "max_model_len": 262144,
                "kvcompress_kv_budget": budget,
                "kvcompress_recompute_window": 128,
            },
            "mod_repository": "repo",
            "mod_revision": "rev",
            "mod_wheel_sha256": "wheel",
            "calibration_sha256": "cal",
            "launch_command_redacted": "vllm serve model",
        }
        path.write_text(json.dumps(config), encoding="utf-8")
    assert compare(b0, b1)["deltas"]["output_tps_pct"] == pytest.approx(10)
    path = b1 / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["server_metadata"]["runtime_environment"]["TORCH_CACHING_PRECOMPILE"] = "1"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime"):
        compare(b0, b1)

    config["server_metadata"]["runtime_environment"]["TORCH_CACHING_PRECOMPILE"] = "0"
    config["server_metadata"]["server_configuration"]["max_model_len"] = 131072
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="serving configuration"):
        compare(b0, b1)
