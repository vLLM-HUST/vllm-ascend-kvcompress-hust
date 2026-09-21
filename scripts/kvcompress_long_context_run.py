#!/usr/bin/env python3
"""Run deterministic streaming A2-LONG/A3-32K commissioning or formal inputs."""

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

FROZEN_SAMPLING_PARAMS: dict[str, Any] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    # Repetition penalty is multiplicative; 1.0 is its neutral value.
    "repetition_penalty": 1.0,
    "n": 1,
    "use_beam_search": False,
    "stop": [],
    "seed": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
    "add_special_tokens": True,
}


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return ordered[rank - 1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cases(path: Path, profile: str) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = [case for case in cases if case["profile"] == profile]
    if not selected:
        raise ValueError(f"dataset has no profile {profile!r}")
    for case in selected:
        if len(case["prompt_token_ids"]) != int(case["input_len"]):
            raise ValueError(f"case {case['case_id']!r} has a mismatched input length")
    return selected


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _run_case(
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
        **FROZEN_SAMPLING_PARAMS,
        "ignore_eos": bool(case["ignore_eos"]),
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    first_token: float | None = None
    token_times: list[float] = []
    text_parts: list[str] = []
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
                if not choices:
                    continue
                choice = choices[0]
                piece = choice.get("text") or ""
                if piece:
                    now = time.monotonic()
                    first_token = now if first_token is None else first_token
                    token_times.append(now)
                    text_parts.append(piece)
                finish_reason = choice.get("finish_reason") or finish_reason
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as caught:
        error = f"{type(caught).__name__}: {caught}"
    end = time.monotonic()
    text = "".join(text_parts)
    prompt_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    ttft = first_token - start if first_token is not None else None
    tpot = None
    if isinstance(output_tokens, int) and output_tokens > 1 and ttft is not None:
        tpot = (end - start - ttft) / (output_tokens - 1)
    inter_token = [
        later - earlier
        for earlier, later in zip(token_times, token_times[1:], strict=False)
    ]
    expected = str(case["expected"])
    correct = re.search(rf"(?<!\d){re.escape(expected)}(?!\d)", text) is not None
    silent_truncation = prompt_tokens != int(case["input_len"])
    short_forced_output = bool(case["ignore_eos"]) and output_tokens != int(
        case["max_tokens"]
    )
    return {
        "sequence": sequence,
        "case_id": case["case_id"],
        "prompt_sha256": case.get("prompt_sha256"),
        "expected": expected,
        "correct": correct,
        "reported_prompt_tokens": prompt_tokens,
        "reported_output_tokens": output_tokens,
        "e2e_s": end - start,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "itl_s": inter_token,
        "finish_reason": finish_reason,
        "silent_truncation": silent_truncation,
        "short_forced_output": short_forced_output,
        "completed_at_s": end,
        "generated_text": text,
        "error": error,
    }


class _DeviceSampler:
    def __init__(self, npu_id: int | None) -> None:
        self.npu_id = npu_id
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.npu_id is None:
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _read(self, category: str) -> str:
        result = subprocess.run(
            ["npu-smi", "info", "-t", category, "-i", str(self.npu_id), "-c", "0"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout

    @staticmethod
    def _percent(text: str, field: str) -> float | None:
        match = re.search(rf"{re.escape(field)}\(%\)\s*:\s*([0-9.]+)", text)
        return float(match.group(1)) if match else None

    def _sample(self) -> None:
        origin = time.monotonic()
        while not self._stop.is_set():
            output = self._read("usages")
            sample = {
                "offset_s": time.monotonic() - origin,
                "hbm_percent": self._percent(output, "HBM Usage Rate"),
                "npu_percent": self._percent(output, "NPU Utilization"),
                "hbm_bandwidth_percent": self._percent(
                    output, "HBM Bandwidth Usage Rate"
                ),
            }
            self.samples.append(
                {key: value for key, value in sample.items() if value is not None}
            )
            self._stop.wait(1.0)


def _run_open_loop(
    cases: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    futures = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for sequence, case in enumerate(cases):
            target = started + sequence / args.request_rate
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            futures.append(
                pool.submit(
                    _run_case,
                    args.base_url,
                    args.model,
                    case,
                    args.timeout,
                    sequence,
                )
            )
        return [future.result() for future in futures]


def _run_closed_loop(
    cases: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    if args.concurrency != 1:
        raise ValueError("A3 closed-loop acceptance requires concurrency=1")
    results = []
    start = time.monotonic()
    sequence = 0
    while sequence < args.max_requests:
        if sequence and time.monotonic() - start >= args.duration_s:
            break
        case = cases[sequence % len(cases)]
        results.append(
            _run_case(args.base_url, args.model, case, args.timeout, sequence)
        )
        sequence += 1
    return results


def _summary(
    results: list[dict[str, Any]], duration: float, measure_start: float
) -> dict[str, Any]:
    successful = [item for item in results if item["error"] is None]
    input_tokens = sum(item["reported_prompt_tokens"] or 0 for item in successful)
    output_tokens = sum(item["reported_output_tokens"] or 0 for item in successful)
    ttft = [item["ttft_s"] for item in successful if item["ttft_s"] is not None]
    tpot = [item["tpot_s"] for item in successful if item["tpot_s"] is not None]
    e2e = [item["e2e_s"] for item in successful]
    itl = [value for item in successful for value in item["itl_s"]]
    metric: dict[str, Any] = {
        "requests": len(results),
        "completed": len(successful),
        "failed": len(results) - len(successful),
        "correct": sum(item["correct"] for item in successful),
        "accuracy": (
            sum(item["correct"] for item in successful) / len(successful)
            if successful
            else None
        ),
        "silent_truncations": sum(item["silent_truncation"] for item in results),
        "short_forced_outputs": sum(item["short_forced_output"] for item in results),
        "duration_s": duration,
        "request_throughput": len(successful) / duration if duration else None,
        "input_throughput": input_tokens / duration if duration else None,
        "output_throughput": output_tokens / duration if duration else None,
        "total_token_throughput": (
            (input_tokens + output_tokens) / duration if duration else None
        ),
    }
    for name, values in (("ttft", ttft), ("tpot", tpot), ("e2e", e2e), ("itl", itl)):
        metric[f"mean_{name}_ms"] = statistics.fmean(values) * 1000 if values else None
        metric[f"p95_{name}_ms"] = _nearest_rank(values, 95)
        metric[f"p99_{name}_ms"] = _nearest_rank(values, 99)
        if metric[f"p95_{name}_ms"] is not None:
            metric[f"p95_{name}_ms"] *= 1000
            metric[f"p99_{name}_ms"] *= 1000
    for item in results:
        item["completed_at_s"] -= measure_start
    return metric


def _stability_windows(
    results: list[dict[str, Any]], duration: float, window_s: float = 300.0
) -> dict[str, Any] | None:
    if duration < 6 * window_s:
        return None
    windows = []
    for index in range(6):
        start = index * window_s
        end = start + window_s
        rows = [
            row
            for row in results
            if row["error"] is None and start <= row["completed_at_s"] < end
        ]
        output_tokens = sum(row["reported_output_tokens"] or 0 for row in rows)
        ttft = [row["ttft_s"] for row in rows if row["ttft_s"] is not None]
        tpot = [row["tpot_s"] for row in rows if row["tpot_s"] is not None]
        windows.append(
            {
                "index": index + 1,
                "start_s": start,
                "end_s": end,
                "completions": len(rows),
                "output_throughput": output_tokens / window_s,
                "median_ttft_ms": statistics.median(ttft) * 1000 if ttft else None,
                "p99_ttft_ms": (_nearest_rank(ttft, 99) * 1000 if ttft else None),
                "median_tpot_ms": statistics.median(tpot) * 1000 if tpot else None,
                "p99_tpot_ms": (_nearest_rank(tpot, 99) * 1000 if tpot else None),
            }
        )
    throughput = [window["output_throughput"] for window in windows]
    mean_throughput = statistics.fmean(throughput)

    def drift(field: str) -> float | None:
        values = [window[field] for window in windows]
        if any(value is None for value in values):
            return None
        center = statistics.median(values)
        return (max(values) - min(values)) / center if center else None

    return {
        "window_seconds": window_s,
        "windows": windows,
        "throughput_cv": (
            statistics.pstdev(throughput) / mean_throughput if mean_throughput else None
        ),
        "median_ttft_drift": drift("median_ttft_ms"),
        "p99_ttft_drift": drift("p99_ttft_ms"),
        "median_tpot_drift": drift("median_tpot_ms"),
        "p99_tpot_drift": drift("p99_tpot_ms"),
    }


def _compression_evidence(path: Path | None, start_offset: int = 0) -> dict[str, Any]:
    if path is None:
        return {"server_log": None, "scheduler_commits": [], "worker_acks": 0}
    commit_pattern = re.compile(
        r"KV compression scheduler commit request_id=(\S+) "
        r"semantic_tokens=(\d+) physical_tokens=(\d+) source_blocks=(\d+) "
        r"destination_blocks=(\d+) released_blocks=(\d+)"
    )
    commits = []
    acknowledgements = 0
    end_offset = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(min(start_offset, end_offset))
        log_segment = stream.read().decode("utf-8", errors="replace")
    for line in log_segment.splitlines():
        if match := commit_pattern.search(line):
            commits.append(
                {
                    "request_id": match.group(1),
                    "semantic_tokens": int(match.group(2)),
                    "physical_tokens": int(match.group(3)),
                    "source_blocks": int(match.group(4)),
                    "destination_blocks": int(match.group(5)),
                    "released_blocks": int(match.group(6)),
                }
            )
        if "KV compression worker commit acknowledged" in line:
            acknowledgements += 1
    return {
        "server_log": str(path.resolve()),
        "server_log_sha256": _sha256(path),
        "server_log_byte_range": [min(start_offset, end_offset), end_offset],
        "scheduler_commits": commits,
        "worker_acks": acknowledgements,
    }


def _compression_evidence_passes(
    evidence: dict[str, Any], expectation: str, run_label: str
) -> bool:
    if expectation == "auto":
        expectation = "required" if run_label == "B1" else "optional"
    commits = len(evidence["scheduler_commits"])
    acknowledgements = int(evidence["worker_acks"])
    if expectation == "required":
        return commits > 0 and acknowledgements >= commits
    if expectation == "forbidden":
        return commits == 0 and acknowledgements == 0
    if expectation == "optional":
        return acknowledgements >= commits
    raise ValueError(f"unknown compression evidence expectation: {expectation!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--request-rate", type=float, default=0.1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--max-requests", type=int, default=64)
    parser.add_argument("--npu-id", type=int)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--run-label", choices=("B0", "B1"), required=True)
    parser.add_argument(
        "--compression-evidence",
        choices=("auto", "required", "optional", "forbidden"),
        default="auto",
        help=(
            "compression transaction expectation; use optional for workloads "
            "that do not cross the configured threshold"
        ),
    )
    args = parser.parse_args()
    if args.request_rate <= 0 or args.concurrency <= 0 or args.max_requests <= 0:
        parser.error("request-rate, concurrency, and max-requests must be positive")
    cases = _load_cases(args.dataset, args.profile)[: args.max_requests]
    server_log_offset = (
        args.server_log.stat().st_size
        if args.server_log is not None and args.server_log.exists()
        else 0
    )
    sampler = _DeviceSampler(args.npu_id)
    sampler.start()
    measure_start = time.monotonic()
    try:
        results = (
            _run_closed_loop(cases, args)
            if args.duration_s > 0
            else _run_open_loop(cases, args)
        )
    finally:
        sampler.stop()
    duration = time.monotonic() - measure_start
    metrics = _summary(results, duration, measure_start)
    hbm_values = [
        sample["hbm_percent"] for sample in sampler.samples if "hbm_percent" in sample
    ]
    npu_values = [
        sample["npu_percent"] for sample in sampler.samples if "npu_percent" in sample
    ]
    output = {
        "schema_version": 1,
        "test_id": "kvcompress-v4.6-long-context-v1",
        "run_label": args.run_label,
        "profile": args.profile,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "model": args.model,
        "base_url": args.base_url,
        "command": sys.argv,
        "versions": {
            name: _package_version(name)
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
        "stability": _stability_windows(results, duration),
        "device": {
            "npu_id": args.npu_id,
            "sample_interval_s": 1,
            "hbm_peak_percent": max(hbm_values) if hbm_values else None,
            "npu_mean_percent": statistics.fmean(npu_values) if npu_values else None,
            "samples": sampler.samples,
        },
        "compression": _compression_evidence(args.server_log, server_log_offset),
        "compression_evidence_expectation": args.compression_evidence,
        "results": results,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "results"},
            ensure_ascii=False,
            indent=2,
        )
    )
    zero_error_required = len(results) < 1000
    compression = output["compression"]
    forced_output_required = any(bool(case.get("ignore_eos")) for case in cases)
    passed = (
        metrics["completed"] == len(results)
        and metrics["correct"] == len(results)
        and metrics["silent_truncations"] == 0
        and (not zero_error_required or metrics["failed"] == 0)
        and (not forced_output_required or metrics["short_forced_outputs"] == 0)
        and _compression_evidence_passes(
            compression,
            args.compression_evidence,
            args.run_label,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
