from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import kvcompress_dataset_applicability as module


def test_token_budget_boundary_and_duplicate_rejection(tmp_path: Path) -> None:
    config = tmp_path / "method.json"
    config.write_text(
        json.dumps(
            {
                "provider": "ascend_kvcompress",
                "method": "triattention",
                "method_config": {
                    "kv_budget": 8192,
                    "recompute_window": 1024,
                    "min_output_tokens_for_compression": 512,
                },
            }
        )
    )
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
