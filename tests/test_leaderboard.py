# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from scripts.kvcompress_leaderboard import render, render_frontier

ROOT = Path(__file__).resolve().parents[1]


def test_leaderboard_history_and_browser_bundle_are_in_sync() -> None:
    history = ROOT / "docs/evidence/leaderboard-history.json"
    bundle = ROOT / "docs/evidence/leaderboard-history.js"
    payload = json.loads(history.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert {record["model"] for record in payload["records"]}
    assert bundle.read_text(encoding="utf-8") == render(history)


def test_leaderboard_html_loads_the_history_bundle() -> None:
    page = (ROOT / "docs/benchmark-leaderboard.html").read_text(encoding="utf-8")

    assert 'src="evidence/leaderboard-history.js"' in page
    assert "KV Compression" in page
    assert 'record.status.includes("fail") ? "fail"' in page
    assert 'src="evidence/frontier-results.js"' in page
    assert 'id="frontier-panel"' in page
    assert "P90 解码速度" in page


def test_frontier_bundle_is_separate_from_history_and_has_no_synthetic_points() -> None:
    source = ROOT / "docs/evidence/frontier-results.json"
    bundle = ROOT / "docs/evidence/frontier-results.js"
    payload = json.loads(source.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "leaderboard-frontier/v1"
    assert {cohort["workload"]["id"] for cohort in payload["cohorts"]} == {
        "sweprefix-qwen35-eight-traces-900s-v1",
        "agentx256k-8fecd2fc-smoke900-v1",
    }
    assert {cohort["model"]["revision"] for cohort in payload["cohorts"]} == {
        "59d61f3ce65a6d9863b86d2e96597125219dc754"
    }
    assert {cohort["model"]["id"] for cohort in payload["cohorts"]} == {
        "qwen35-35b-a3b-59d61f3c"
    }
    assert all(point["evidence"]["status"] == "measured" for point in payload["points"])
    assert bundle.read_text(encoding="utf-8") == render_frontier(source)


def test_frontier_bundle_rejects_unmeasured_and_incomplete_points(
    tmp_path: Path,
) -> None:
    source = ROOT / "docs/evidence/frontier-results.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    point = {
        "id": "test-point",
        "pair_id": "test-pair",
        "arm": "B0",
        "cohort_id": payload["cohorts"][0]["id"],
        "configuration": {
            "engine": "vLLM",
            "mods": [],
            "context_capacity_tokens": 262144,
            "hardware": {"label": "Ascend 910B2", "accelerator_count": 2},
        },
        "load": {"concurrency": 2, "max_observed_prompt_tokens": 100000},
        "metrics": {"decode_p90_tps": 50, "output_tps": 100},
        "evidence": {
            "status": "measured",
            "run_ids": ["run-1"],
            "aggregation": "unpooled 900s observation",
            "measurement_seconds": 900,
            "url": "evidence/run-1.json",
        },
    }
    other = json.loads(json.dumps(point))
    other["id"] = "test-point-b1"
    other["arm"] = "B1"
    other["evidence"]["run_ids"] = ["run-2"]
    other["evidence"]["url"] = "evidence/run-2.json"
    payload["points"] = [point, other]
    candidate = tmp_path / "frontier.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    assert "window.KV_FRONTIER_RESULTS=" in render_frontier(candidate)

    point["evidence"]["status"] = "planned"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete measured"):
        render_frontier(candidate)

    point["evidence"]["status"] = "measured"
    point["metrics"]["decode_p90_tps"] = None
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete measured"):
        render_frontier(candidate)

    point["metrics"]["decode_p90_tps"] = 50
    other["load"]["concurrency"] = 4
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="comparable B0/B1"):
        render_frontier(candidate)
