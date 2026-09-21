# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from scripts.kvcompress_leaderboard import render

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
