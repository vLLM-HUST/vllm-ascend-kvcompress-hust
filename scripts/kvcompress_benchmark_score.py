#!/usr/bin/env python3
"""Score paired public LongBench/LongBench-v2 service results."""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, answer: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(answer).split()
    if not predicted or not expected:
        return float(predicted == expected)
    same = sum((Counter(predicted) & Counter(expected)).values())
    if same == 0:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(expected)
    return 2 * precision * recall / (precision + recall)


def retrieval_score(prediction: str, answer: str) -> float:
    match = re.search(r"Paragraph (\d+)", answer)
    if match is None:
        raise ValueError(f"invalid LongBench retrieval answer: {answer!r}")
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    return sum(number == match.group(1) for number in numbers) / len(numbers)


def longbench_v2_answer(prediction: str) -> str | None:
    cleaned = prediction.replace("*", "")
    for pattern in (
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])",
    ):
        if match := re.search(pattern, cleaned, flags=re.IGNORECASE):
            return match.group(1).upper()
    isolated = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", cleaned.upper())
    return isolated[-1] if len(set(isolated)) == 1 else None


def case_score(row: dict[str, Any]) -> float:
    task = row["task"]
    prediction = row["prediction"]
    answers = row["answers"]
    if task == "longbench-v2":
        return float(longbench_v2_answer(prediction) == answers[0].upper())
    if task == "passage_retrieval_en":
        return max(retrieval_score(prediction, answer) for answer in answers)
    if task == "qasper":
        return max(token_f1(prediction, answer) for answer in answers)
    raise ValueError(f"unsupported scorer task: {task}")


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_id") != "kvcompress-public-long-context-v1":
        raise ValueError(f"{path} is not a public benchmark result")
    return payload


def scored_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for row in payload["results"]:
        if row["error"] is not None or row["silent_truncation"]:
            score = 0.0
        else:
            score = case_score(row)
        output[row["case_id"]] = {**row, "quality_score": score}
    return output


def bootstrap_delta(
    baseline: list[float], plugin: list[float], *, iterations: int = 5000
) -> list[float]:
    generator = random.Random(20260912)
    paired = [right - left for left, right in zip(baseline, plugin, strict=True)]
    draws = []
    for _ in range(iterations):
        draws.append(sum(generator.choice(paired) for _ in paired) / len(paired))
    draws.sort()
    return [draws[int(0.025 * iterations)], draws[int(0.975 * iterations)]]


def relative_change(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return new / old - 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-tolerance-pp", type=float, default=1.0)
    args = parser.parse_args()
    baseline = load_result(args.baseline)
    plugin = load_result(args.plugin)
    if baseline["dataset_sha256"] != plugin["dataset_sha256"]:
        raise ValueError("B0/B1 request-set hashes differ")
    left = scored_rows(baseline)
    right = scored_rows(plugin)
    if left.keys() != right.keys():
        raise ValueError("B0/B1 case IDs differ")
    case_ids = sorted(left)
    left_scores = [left[case_id]["quality_score"] for case_id in case_ids]
    right_scores = [right[case_id]["quality_score"] for case_id in case_ids]
    baseline_quality = 100 * sum(left_scores) / len(left_scores)
    plugin_quality = 100 * sum(right_scores) / len(right_scores)
    quality_delta = plugin_quality - baseline_quality
    left_metrics = baseline["metrics"]
    right_metrics = plugin["metrics"]
    output = {
        "schema_version": 1,
        "test_id": "kvcompress-public-long-context-paired-v1",
        "benchmark": next(iter(left.values()))["benchmark"],
        "task": next(iter(left.values()))["task"],
        "case_count": len(case_ids),
        "dataset_sha256": baseline["dataset_sha256"],
        "quality": {
            "baseline": baseline_quality,
            "plugin": plugin_quality,
            "delta_percentage_points": quality_delta,
            "paired_bootstrap_delta_95ci_percentage_points": [
                100 * value for value in bootstrap_delta(left_scores, right_scores)
            ],
            "tolerance_percentage_points": args.quality_tolerance_pp,
        },
        "performance": {
            "baseline": left_metrics,
            "plugin": right_metrics,
            "request_throughput_change": relative_change(
                right_metrics["request_throughput"], left_metrics["request_throughput"]
            ),
            "total_token_throughput_change": relative_change(
                right_metrics["total_token_throughput"],
                left_metrics["total_token_throughput"],
            ),
            "mean_ttft_change": relative_change(
                right_metrics["mean_ttft_ms"], left_metrics["mean_ttft_ms"]
            ),
            "mean_tpot_change": relative_change(
                right_metrics["mean_tpot_ms"], left_metrics["mean_tpot_ms"]
            ),
            "mean_e2e_change": relative_change(
                right_metrics["mean_e2e_ms"], left_metrics["mean_e2e_ms"]
            ),
        },
        "compression": plugin["compression"],
        "device": {"baseline": baseline["device"], "plugin": plugin["device"]},
        "cases": [
            {
                "case_id": case_id,
                "baseline_score": left[case_id]["quality_score"],
                "plugin_score": right[case_id]["quality_score"],
            }
            for case_id in case_ids
        ],
    }
    expectation = plugin.get("compression_expectation", "required")
    commit_count = len(plugin["compression"]["scheduler_commits"])
    compression_expectation_met = (
        expectation == "optional"
        or (expectation == "required" and commit_count > 0)
        or (
            expectation == "forbidden"
            and commit_count == 0
            and plugin["compression"]["worker_acks"] == 0
        )
    )
    gates = {
        "same_cases": True,
        "baseline_zero_failures": left_metrics["failed"] == 0,
        "plugin_zero_failures": right_metrics["failed"] == 0,
        "baseline_zero_silent_truncations": left_metrics["silent_truncations"] == 0,
        "plugin_zero_silent_truncations": right_metrics["silent_truncations"] == 0,
        "quality_within_tolerance": quality_delta >= -args.quality_tolerance_pp,
        "compression_expectation_met": compression_expectation_met,
        "worker_acknowledgements_match": plugin["compression"]["worker_acks"]
        >= len(plugin["compression"]["scheduler_commits"]),
    }
    output["gates"] = gates
    output["verdict"] = "PASS" if all(gates.values()) else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    compact = {
        "benchmark": output["benchmark"],
        "task": output["task"],
        "case_count": output["case_count"],
        "quality": output["quality"],
        "performance_changes": {
            key: value
            for key, value in output["performance"].items()
            if key.endswith("_change")
        },
        "compression": {
            "scheduler_commits": len(output["compression"]["scheduler_commits"]),
            "worker_acks": output["compression"]["worker_acks"],
        },
        "gates": output["gates"],
        "verdict": output["verdict"],
    }
    print(json.dumps(compact, indent=2))
    return 0 if output["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
