#!/usr/bin/env python3
"""Run the deterministic long-context quality guard for KV compression M3.

The client sends exact token-id prompts to the OpenAI-compatible Completions
endpoint.  Each prompt contains one authoritative station/code record at a
frozen depth and asks the model to return only that code.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

FACT_POSITIONS = (512, 2048, 4096, 6144)
FILLER = (
    "Routine telemetry record: systems nominal; temperature stable; "
    "inspection complete.\n"
)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _repeat_to_length(pattern: list[int], length: int) -> list[int]:
    if length < 0:
        raise ValueError("requested filler length is negative")
    repeats, remainder = divmod(length, len(pattern))
    return pattern * repeats + pattern[:remainder]


def _build_case(tokenizer: Any, case_index: int, input_len: int) -> dict[str, Any]:
    station = f"KVC-{case_index:03d}"
    code = str(10_000_000 + ((case_index * 7_919 + 1_337) % 89_999_999))
    header = tokenizer.encode(
        "Read the audit log and answer the final question. Ignore routine "
        "telemetry. Return only the requested 8-digit code, with no words.\n",
        add_special_tokens=False,
    )
    fact = tokenizer.encode(
        f"AUTHORITATIVE RECORD: The access code for station {station} is {code}.\n",
        add_special_tokens=False,
    )
    question = tokenizer.encode(
        f"FINAL QUESTION: What is the access code for station {station}? "
        "Return only the 8-digit code.\nANSWER:",
        add_special_tokens=False,
    )
    filler = tokenizer.encode(FILLER, add_special_tokens=False)
    fact_position = FACT_POSITIONS[case_index % len(FACT_POSITIONS)]
    if fact_position < len(header):
        raise ValueError("fact position overlaps the header")
    before = _repeat_to_length(filler, fact_position - len(header))
    after_len = input_len - len(header) - len(before) - len(fact) - len(question)
    after = _repeat_to_length(filler, after_len)
    prompt_ids = header + before + fact + after + question
    if len(prompt_ids) != input_len:
        raise AssertionError(
            f"prompt has {len(prompt_ids)} tokens, expected {input_len}"
        )
    prompt_bytes = json.dumps(prompt_ids, separators=(",", ":")).encode()
    return {
        "case_id": f"m3-quality-{case_index:03d}",
        "station": station,
        "expected_code": code,
        "fact_position": fact_position,
        "prompt_ids": prompt_ids,
        "prompt_token_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
    }


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _run_case(
    base_url: str,
    model: str,
    case: dict[str, Any],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": case["prompt_ids"],
        "max_tokens": max_tokens,
        "min_tokens": 0,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "ignore_eos": False,
        "seed": 0,
        "logprobs": None,
    }
    started = time.monotonic()
    try:
        response = _post_json(
            f"{base_url.rstrip('/')}/v1/completions", payload, timeout
        )
        elapsed = time.monotonic() - started
        text = response.get("choices", [{}])[0].get("text", "")
        match = re.search(r"(?<!\d)(\d{8})(?!\d)", text)
        predicted = match.group(1) if match else None
        usage = response.get("usage") or {}
        return {
            "case_id": case["case_id"],
            "station": case["station"],
            "expected_code": case["expected_code"],
            "predicted_code": predicted,
            "correct": predicted == case["expected_code"],
            "fact_position": case["fact_position"],
            "prompt_token_sha256": case["prompt_token_sha256"],
            "reported_prompt_tokens": usage.get("prompt_tokens"),
            "reported_completion_tokens": usage.get("completion_tokens"),
            "latency_s": elapsed,
            "generated_text": text,
            "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
            "error": None,
        }
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        KeyError,
    ) as error:
        return {
            "case_id": case["case_id"],
            "station": case["station"],
            "expected_code": case["expected_code"],
            "predicted_code": None,
            "correct": False,
            "fact_position": case["fact_position"],
            "prompt_token_sha256": case["prompt_token_sha256"],
            "reported_prompt_tokens": None,
            "reported_completion_tokens": None,
            "latency_s": time.monotonic() - started,
            "generated_text": "",
            "finish_reason": None,
            "error": f"{type(error).__name__}: {error}",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen2.5-14b-instruct")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()

    if args.num_cases <= 0 or args.input_len <= 0 or args.concurrency <= 0:
        parser.error("num-cases, input-len, and concurrency must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=False,
    )
    cases = [
        _build_case(tokenizer, index, args.input_len) for index in range(args.num_cases)
    ]
    ordered_hash = hashlib.sha256(
        "".join(case["prompt_token_sha256"] for case in cases).encode()
    ).hexdigest()

    started = time.monotonic()
    if args.build_only:
        results: list[dict[str, Any]] = []
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = [
                pool.submit(
                    _run_case,
                    args.base_url,
                    args.model,
                    case,
                    args.max_tokens,
                    args.timeout,
                )
                for case in cases
            ]
            results = [future.result() for future in futures]
    duration = time.monotonic() - started

    successful = [item for item in results if item["error"] is None]
    correct = [item for item in successful if item["correct"]]
    latencies_ms = [item["latency_s"] * 1000 for item in successful]
    prompt_lengths_valid = bool(successful) and all(
        item["reported_prompt_tokens"] == args.input_len for item in successful
    )
    output = {
        "schema_version": 1,
        "test_id": "kvcompress-m3-long-context-quality-v1",
        "base_url": args.base_url,
        "model": args.model,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "num_cases": args.num_cases,
        "input_len": args.input_len,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "seed": 0,
        "fact_positions": list(FACT_POSITIONS),
        "ordered_prompt_token_sha256": ordered_hash,
        "duration_s": duration,
        "completed": len(successful),
        "failed": len(results) - len(successful),
        "correct": len(correct),
        "accuracy": len(correct) / len(results) if results else None,
        "prompt_lengths_valid": prompt_lengths_valid,
        "mean_latency_ms": statistics.fmean(latencies_ms) if latencies_ms else None,
        "p99_latency_ms": _percentile(latencies_ms, 0.99),
        "results": results,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps({key: output[key] for key in output if key != "results"}, indent=2)
    )
    return (
        0 if args.build_only or (output["failed"] == 0 and prompt_lengths_valid) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
