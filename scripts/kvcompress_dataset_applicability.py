#!/usr/bin/env python3
"""Audit whether a frozen tokenized task manifest can reach TriAttention."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_profile(path: Path) -> tuple[int, int]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        config.get("provider") != "ascend_kvcompress"
        or config.get("method") != "triattention"
    ):
        raise ValueError("expected an ascend_kvcompress TriAttention method config")
    options = config["method_config"]
    budget = options["kv_budget"]
    window = options["recompute_window"]
    minimum_output = options.get("min_output_tokens_for_compression", 0)
    if any(
        type(value) is not int or value < 0
        for value in (budget, window, minimum_output)
    ):
        raise ValueError("invalid compression threshold or output budget")
    if budget == 0 or window == 0:
        raise ValueError("compression threshold must be positive")
    return budget + window, minimum_output


def audit(manifest: Path, config: Path) -> dict[str, Any]:
    threshold, minimum_output = load_profile(config)
    seen: set[str] = set()
    possible: list[str] = []
    max_prompt = 0
    max_total = 0
    count = 0
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("case_id")
            tokens = row.get("prompt_token_ids")
            output_budget = row.get("max_tokens")
            if not isinstance(case_id, str) or not case_id or case_id in seen:
                raise ValueError(f"line {line_number}: missing or duplicate case_id")
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(type(x) is not int or x < 0 for x in tokens)
            ):
                raise ValueError(
                    f"line {line_number}: expected frozen prompt_token_ids"
                )
            if type(output_budget) is not int or output_budget <= 0:
                raise ValueError(f"line {line_number}: invalid max_tokens")
            if "input_len" in row and row["input_len"] != len(tokens):
                raise ValueError(
                    f"line {line_number}: input_len disagrees with token IDs"
                )
            seen.add(case_id)
            count += 1
            max_prompt = max(max_prompt, len(tokens))
            total = len(tokens) + output_budget
            max_total = max(max_total, total)
            if output_budget >= minimum_output and total >= threshold:
                possible.append(case_id)
    if count == 0:
        raise ValueError("empty task manifest")
    return {
        "schema_version": 1,
        "manifest_sha256": sha256(manifest),
        "method_config_sha256": sha256(config),
        "case_count": count,
        "compression_threshold_tokens": threshold,
        "minimum_output_budget_tokens": minimum_output,
        "maximum_prompt_tokens": max_prompt,
        "maximum_prompt_plus_output_budget_tokens": max_total,
        "potentially_exercised_case_ids": possible,
        "status": "not-exercised" if not possible else "requires-live-commit-audit",
        "note": (
            "Token budgets provide only an upper bound. A possible case requires "
            "B1 scheduler commit and worker acknowledgement logs to prove actual "
            "compression."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.method_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("case_count", "compression_threshold_tokens", "status")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
