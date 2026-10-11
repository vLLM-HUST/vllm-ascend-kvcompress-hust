from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dataset_applicability", ROOT / "scripts/kvcompress_dataset_applicability.py"
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_token_budget_boundary_and_duplicate_rejection(tmp_path: Path) -> None:
    config = ROOT / ".benchmarks/config/qwen3.5-35b-a3b-v3-batched-stride2-min512.json"
    manifest = tmp_path / "tasks.jsonl"
    rows = [
        {
            "case_id": "short",
            "prompt_token_ids": [1] * 2910,
            "input_len": 2910,
            "max_tokens": 2048,
        },
        {"case_id": "possible", "prompt_token_ids": [1] * 8704, "max_tokens": 512},
        {
            "case_id": "budget-too-small",
            "prompt_token_ids": [1] * 9216,
            "max_tokens": 511,
        },
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = module.audit(manifest, config)
    assert result["compression_threshold_tokens"] == 9216
    assert result["potentially_exercised_case_ids"] == ["possible"]
    assert result["status"] == "requires-live-commit-audit"

    manifest.write_text("\n".join(json.dumps(rows[0]) for _ in range(2)))
    with pytest.raises(ValueError, match="duplicate"):
        module.audit(manifest, config)
