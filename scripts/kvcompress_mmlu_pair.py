#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run independent MMLU-Pro server lifecycle pairs and retain raw telemetry."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(url: str) -> str:
    with OPENER.open(url, timeout=5) as response:
        return response.read().decode()


def sample(root: Path, base_url: str, stop: threading.Event) -> None:
    with gzip.open(root / "telemetry.jsonl.gz", "wt") as stream:
        while not stop.is_set():
            record: dict = {"time_unix": time.time()}
            try:
                record["metrics"] = fetch(base_url + "/metrics")
            except OSError as error:
                record["metrics_error"] = repr(error)
            try:
                device = subprocess.run(
                    ["npu-smi", "info"], capture_output=True, text=True, timeout=15
                )
                record["npu_smi"] = device.stdout
                record["npu_smi_returncode"] = device.returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                record["npu_smi_error"] = repr(error)
            record["memory_current"] = (
                Path("/sys/fs/cgroup/memory.current").read_text().strip()
            )
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            stop.wait(5)


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=30)


def run_arm(args: argparse.Namespace, pair: int, arm: str) -> None:
    root = args.output / f"pair-{pair}" / arm
    root.mkdir(parents=True)  # Never overwrite a previous attempt.
    command = [str(args.launcher.resolve()), arm]
    metadata = {
        "pair": pair,
        "arm": arm,
        "server_command": command,
        "started_unix": time.time(),
        "plan": str(args.plan.resolve()),
        "concurrency": args.concurrency,
        "base_url": args.base_url,
        "status": "starting",
    }
    server = None
    stop = threading.Event()
    sampler = None
    try:
        with (root / "server.log").open("w") as log:
            server = subprocess.Popen(
                command, stdout=log, stderr=log, start_new_session=True
            )
            metadata["server_pid"] = server.pid
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"server exited during startup: {server.returncode}"
                    )
                try:
                    fetch(args.base_url + "/health")
                    break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("server startup timeout")
            metadata["ready_unix"] = time.time()
            # Both arms get the same warmup; its partial score is never a full score.
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            env["PYTHONNOUSERSITE"] = "1"
            base = [
                str(args.client_python.absolute()),
                str(Path(__file__).with_name("kvcompress_mmlu_pro.py")),
                "run",
                "--plan",
                str(args.plan.resolve()),
                "--base-url",
                args.base_url,
                "--served-model",
                args.served_model,
                "--concurrency",
                str(args.concurrency),
            ]
            with (root / "warmup.log").open("w") as client_log:
                subprocess.run(
                    [*base, "--output", str(root / "warmup"), "--limit", "16"],
                    env=env,
                    stdout=client_log,
                    stderr=client_log,
                    check=True,
                )
            sampler = threading.Thread(target=sample, args=(root, args.base_url, stop))
            sampler.start()
            metadata["measurement_started_unix"] = time.time()
            (root / "metrics-before.prom").write_text(fetch(args.base_url + "/metrics"))
            with (root / "client.log").open("w") as client_log:
                subprocess.run(
                    [*base, "--output", str(root / "full")],
                    env=env,
                    stdout=client_log,
                    stderr=client_log,
                    check=True,
                )
            (root / "metrics-after.prom").write_text(fetch(args.base_url + "/metrics"))
            metadata["measurement_finished_unix"] = time.time()
            metadata["status"] = "completed"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = repr(error)
        raise
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=30)
        try:
            if server is not None:
                stop_server(server)
                metadata["server_exit_code"] = server.returncode
        except Exception as error:
            metadata["status"] = "failed"
            metadata["cleanup_error"] = repr(error)
            raise
        finally:
            metadata["finished_unix"] = time.time()
            (root / "lifecycle.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launcher", type=Path, required=True, help="Executable accepting b0|b1"
    )
    parser.add_argument("--client-python", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--served-model", default="qwen35-mmlu")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    args.output = args.output.resolve()
    if args.concurrency < 1 or args.pairs < 1:
        parser.error("concurrency and pairs must be positive")
    try:
        fetch(args.base_url + "/health")
    except OSError:
        pass
    else:
        parser.error("base URL is already occupied by a healthy server")
    args.output.mkdir(parents=True, exist_ok=True)
    for pair in range(1, args.pairs + 1):
        for arm in ("b0", "b1"):
            print(f"pair={pair} arm={arm} starting", flush=True)
            run_arm(args, pair, arm)
            print(f"pair={pair} arm={arm} completed", flush=True)


if __name__ == "__main__":
    main()
