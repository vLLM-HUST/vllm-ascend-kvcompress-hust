#!/usr/bin/env python3
"""Run a frozen public benchmark request set against an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(percentile * len(ordered) / 100)) - 1]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError("request set is empty")
    required = {
        "case_id",
        "benchmark",
        "task",
        "prompt_token_ids",
        "input_len",
        "max_tokens",
        "answers",
    }
    for case in cases:
        missing = required - case.keys()
        if missing:
            raise ValueError(f"{case.get('case_id')} missing {sorted(missing)}")
        if len(case["prompt_token_ids"]) != case["input_len"]:
            raise ValueError(f"{case['case_id']} has mismatched prompt length")
    return cases


def run_case(
    base_url: str,
    model: str,
    case: dict[str, Any],
    timeout: float,
    sequence: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": case["prompt_token_ids"],
        "max_tokens": case["max_tokens"],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": False,
        "seed": 0,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    first_token: float | None = None
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = None
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                event = json.loads(body)
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices:
                    choice = choices[0]
                    piece = choice.get("text") or ""
                    if piece:
                        first_token = first_token or time.monotonic()
                        pieces.append(piece)
                    finish_reason = choice.get("finish_reason") or finish_reason
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as caught:
        error = f"{type(caught).__name__}: {caught}"
    ended = time.monotonic()
    prompt_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    ttft = first_token - started if first_token is not None else None
    tpot = None
    if ttft is not None and isinstance(output_tokens, int) and output_tokens > 1:
        tpot = (ended - started - ttft) / (output_tokens - 1)
    return {
        "sequence": sequence,
        "case_id": case["case_id"],
        "benchmark": case["benchmark"],
        "task": case["task"],
        "answers": case["answers"],
        "metadata": case.get("metadata", {}),
        "input_len": case["input_len"],
        "max_tokens": case["max_tokens"],
        "reported_prompt_tokens": prompt_tokens,
        "reported_output_tokens": output_tokens,
        "silent_truncation": prompt_tokens != case["input_len"],
        "prediction": "".join(pieces),
        "finish_reason": finish_reason,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_s": ended - started,
        "error": error,
    }


class DeviceSampler:
    def __init__(self, npu_id: int | None) -> None:
        self.npu_id = npu_id
        self.samples: list[dict[str, float]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.npu_id is not None:
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def _sample(self) -> None:
        origin = time.monotonic()
        while not self.stop_event.is_set():
            result = subprocess.run(
                ["npu-smi", "info", "-t", "usages", "-i", str(self.npu_id), "-c", "0"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            sample: dict[str, float] = {"offset_s": time.monotonic() - origin}
            for key, field in (
                ("hbm_percent", "HBM Usage Rate"),
                ("npu_percent", "NPU Utilization"),
                ("hbm_bandwidth_percent", "HBM Bandwidth Usage Rate"),
            ):
                if match := re.search(rf"{field}\(\%\)\s*:\s*([0-9.]+)", result.stdout):
                    sample[key] = float(match.group(1))
            self.samples.append(sample)
            self.stop_event.wait(1)


def run_open_loop(
    cases: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    futures = []
    origin = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for sequence, case in enumerate(cases):
            if args.request_rate > 0:
                delay = origin + sequence / args.request_rate - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            futures.append(
                pool.submit(
                    run_case,
                    args.base_url,
                    args.model,
                    case,
                    args.timeout,
                    sequence,
                )
            )
        return [future.result() for future in futures]


def summarize(results: list[dict[str, Any]], duration: float) -> dict[str, Any]:
    completed = [row for row in results if row["error"] is None]
    metrics: dict[str, Any] = {
        "requests": len(results),
        "completed": len(completed),
        "failed": len(results) - len(completed),
        "silent_truncations": sum(row["silent_truncation"] for row in completed),
        "duration_s": duration,
        "request_throughput": len(completed) / duration,
    }
    input_tokens = sum(row["reported_prompt_tokens"] or 0 for row in completed)
    output_tokens = sum(row["reported_output_tokens"] or 0 for row in completed)
    metrics.update(
        {
            "input_throughput": input_tokens / duration,
            "output_throughput": output_tokens / duration,
            "total_token_throughput": (input_tokens + output_tokens) / duration,
        }
    )
    for name in ("ttft", "tpot", "e2e"):
        values = [row[f"{name}_s"] for row in completed if row[f"{name}_s"] is not None]
        metrics[f"mean_{name}_ms"] = statistics.fmean(values) * 1000 if values else None
        for percentile in (50, 95, 99):
            value = nearest_rank(values, percentile)
            metrics[f"p{percentile}_{name}_ms"] = (
                value * 1000 if value is not None else None
            )
    return metrics


def compression_evidence(path: Path | None, start_offset: int) -> dict[str, Any]:
    if path is None:
        return {"server_log": None, "scheduler_commits": [], "worker_acks": 0}
    end_offset = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(min(start_offset, end_offset))
        segment = stream.read().decode(errors="replace")
    pattern = re.compile(
        r"KV compression scheduler commit request_id=(\S+) semantic_tokens=(\d+) "
        r"physical_tokens=(\d+) source_blocks=(\d+) destination_blocks=(\d+) "
        r"released_blocks=(\d+)"
    )
    commits = [
        {
            "request_id": match.group(1),
            "semantic_tokens": int(match.group(2)),
            "physical_tokens": int(match.group(3)),
            "source_blocks": int(match.group(4)),
            "destination_blocks": int(match.group(5)),
            "released_blocks": int(match.group(6)),
        }
        for match in pattern.finditer(segment)
    ]
    return {
        "server_log": str(path.resolve()),
        "server_log_sha256": sha256(path),
        "server_log_byte_range": [min(start_offset, end_offset), end_offset],
        "scheduler_commits": commits,
        "worker_acks": segment.count("KV compression worker commit acknowledged"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-label", choices=("B0", "B1"), required=True)
    parser.add_argument("--request-rate", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--npu-id", type=int)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument(
        "--compression-expectation",
        choices=("required", "optional", "forbidden"),
        default="required",
        help="B1 compression evidence gate; ignored for B0",
    )
    args = parser.parse_args()
    if args.request_rate < 0 or args.concurrency <= 0:
        parser.error("request-rate must be nonnegative and concurrency positive")
    cases = load_cases(args.dataset)
    log_offset = (
        args.server_log.stat().st_size
        if args.server_log is not None and args.server_log.exists()
        else 0
    )
    sampler = DeviceSampler(args.npu_id)
    sampler.start()
    started = time.monotonic()
    try:
        results = run_open_loop(cases, args)
    finally:
        sampler.stop()
    duration = time.monotonic() - started
    metrics = summarize(results, duration)
    hbm = [
        sample["hbm_percent"] for sample in sampler.samples if "hbm_percent" in sample
    ]
    npu = [
        sample["npu_percent"] for sample in sampler.samples if "npu_percent" in sample
    ]
    output = {
        "schema_version": 1,
        "test_id": "kvcompress-public-long-context-v1",
        "run_label": args.run_label,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "model": args.model,
        "base_url": args.base_url,
        "command": sys.argv,
        "versions": {
            name: package_version(name)
            for name in (
                "vllm",
                "vllm-ascend",
                "vllm-hust-ext",
                "vllm-ascend-kvcompress-hust",
                "torch",
                "torch-npu",
                "triton-ascend",
            )
        },
        "metrics": metrics,
        "device": {
            "npu_id": args.npu_id,
            "hbm_peak_percent": max(hbm) if hbm else None,
            "npu_mean_percent": statistics.fmean(npu) if npu else None,
            "samples": sampler.samples,
        },
        "compression": compression_evidence(args.server_log, log_offset),
        "compression_expectation": args.compression_expectation,
        "results": results,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "test_id": output["test_id"],
                "run_label": output["run_label"],
                "dataset_sha256": output["dataset_sha256"],
                "metrics": output["metrics"],
                "device": {
                    key: value
                    for key, value in output["device"].items()
                    if key != "samples"
                },
                "compression": {
                    "scheduler_commits": len(
                        output["compression"]["scheduler_commits"]
                    ),
                    "worker_acks": output["compression"]["worker_acks"],
                },
            },
            indent=2,
        )
    )
    compression = output["compression"]
    commits = len(compression["scheduler_commits"])
    expectation_met = (
        args.compression_expectation == "optional"
        or (args.compression_expectation == "required" and commits > 0)
        or (
            args.compression_expectation == "forbidden"
            and commits == 0
            and compression["worker_acks"] == 0
        )
    )
    passed = (
        metrics["completed"] == len(cases)
        and metrics["failed"] == 0
        and metrics["silent_truncations"] == 0
        and (
            args.run_label == "B0"
            or (
                expectation_met
                and compression["worker_acks"] >= len(compression["scheduler_commits"])
            )
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
