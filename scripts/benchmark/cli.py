#!/usr/bin/env python3
"""Pujiang evaluation entry point; heavy dependencies live in external clients."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmark.common.clients import (  # noqa: E402
    check_installed,
    clean_client_env,
)
from scripts.benchmark.common.files import WORKSPACE  # noqa: E402

DATASETS = (
    "mmlu-pro",
    "hle-verified",
    "frontier-science",
    "swe-bench-pro",
    "terminal-bench",
)


def adapter(name):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name}")
    spec = importlib.util.spec_from_file_location(
        "pujiang_" + name.replace("-", "_"),
        ROOT / "scripts/benchmark/pujiang" / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CLIENT"):
        raise NotImplementedError(f"{name}: not implemented yet")
    return module


def parser():
    p = argparse.ArgumentParser(prog="run_benchmark.sh", description=__doc__)
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument(
        "action",
        choices=(
            "download-client",
            "download",
            "prepare",
            "run",
            "bench",
            "pair",
            "compare",
            "inspect",
        ),
    )
    p.add_argument("--client-root", type=Path, default=WORKSPACE / "pujiang-benchmark")
    p.add_argument("--client-python", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--model-path", type=Path)
    p.add_argument("--solver-config", type=Path)
    p.add_argument("--judge-config", type=Path)
    p.add_argument("--runtime-contract")
    p.add_argument("--judge-runtime-contract")
    p.add_argument("--track", action="append")
    p.add_argument("--modality", choices=("all", "text-only"), default="all")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--seed", type=int)
    p.add_argument("--resample", action="store_true")
    p.add_argument("--selection-dir", type=Path)
    p.add_argument("--arm", choices=("b0", "b1"))
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--served-model")
    p.add_argument("--api-key-env")
    p.add_argument(
        "--concurrency",
        type=int,
        help="MMLU default 1; otherwise match frozen Solver config",
    )
    p.add_argument(
        "--timeout",
        type=int,
        help="MMLU/tokenize default 1800; otherwise match frozen config",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stage", choices=("all", "solver", "judge"), default="all")
    p.add_argument("--server-log", type=Path)
    p.add_argument("--bench-vllm-bin", type=Path)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--config", type=Path, help="pair JSON configuration")
    p.add_argument("--pairs", type=int, default=1)
    p.add_argument("--startup-timeout", type=int, default=1800)
    p.add_argument("--with-bench", action="store_true")
    p.add_argument("--baseline", type=Path)
    p.add_argument("--candidate", type=Path)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        mod = adapter(args.dataset)
        client = mod.CLIENT
        if args.stage != "all":
            if args.action not in ("run", "pair") or client.dataset == "mmlu-pro":
                p.error("--stage is supported for HLE/Frontier run and pair only")
            if args.action == "pair" and args.stage == "judge":
                p.error("use run --stage judge --resume for each completed solver arm")
            if args.stage == "judge" and not args.resume:
                p.error("--stage judge requires --resume and the original output")
        if args.track and (
            set(args.track) - set(client.tracks)
            or len(args.track) != len(set(args.track))
        ):
            p.error(f"track must be unique values from {client.tracks}")
        positive = (
            args.sample_size,
            args.concurrency,
            args.timeout,
            args.output_len,
            args.pairs,
            args.startup_timeout,
        )
        if any(value is not None and value <= 0 for value in positive):
            p.error("sample size, concurrency, limits and timeouts must be positive")
        if args.model_path and not args.model_path.is_dir():
            p.error("--model-path must be an existing local model/tokenizer directory")
        if args.resample and not args.smoke:
            p.error("--resample requires --smoke")
        if args.resume and args.resample:
            p.error("cannot resample an existing run during resume")
        if args.dry_run and args.action != "bench":
            p.error("--dry-run is available for bench only")
        if args.server_log and not args.server_log.is_file():
            p.error("--server-log must be an existing actual server log")
        if args.action not in ("run", "bench", "pair", "inspect") and args.resample:
            p.error("--resample is used by run, bench or pair")
        args.data_root = (
            args.data_root or WORKSPACE / ".benchmarks/pujiang" / args.dataset
        ).absolute()
        args.plan = (args.plan or args.data_root / "plan").absolute()
        args.selection_dir = (
            args.selection_dir or args.data_root / "samples"
        ).absolute()
        args.output = (
            args.output
            or WORKSPACE
            / ".benchmarks/results"
            / args.dataset
            / f"{args.action}-{args.arm or 'pair'}"
        ).absolute()
        if args.action == "download-client":
            print(mod.acquire(args.client_root))
            return
        if args.action in ("pair", "compare"):
            if args.action == "pair":
                from scripts.benchmark.common.lifecycle import pair

                pair(client, args)
            else:
                from scripts.benchmark.common.reporting import compare

                if not args.baseline or not args.candidate:
                    p.error("compare requires --baseline and --candidate")
                print(
                    json.dumps(
                        compare(args.baseline, args.candidate, args.output), indent=2
                    )
                )
            return
        python = (args.client_python or client.python()).absolute()
        if not python.is_file():
            p.error(
                f"missing client interpreter: {python}; run set_env.sh install client"
            )
        if Path(sys.prefix).absolute() != python.parent.parent:
            # Do not resolve the venv symlink: its interpreter prefix is intentional.
            command = [
                str(python),
                str(Path(__file__).resolve()),
                *(argv or sys.argv[1:]),
            ]
            subprocess.run(command, env=clean_client_env(), check=True)
            return
        check_installed(client, args.client_root)
        if args.action == "inspect":
            from scripts.benchmark.common.evaluation import selection, semantic_identity
            from scripts.benchmark.common.files import write

            _, _, manifest = selection(client, args)
            write(
                args.output,
                {
                    "selection": manifest,
                    "identity": semantic_identity(client, args.plan),
                },
            )
        elif args.action == "download":
            from scripts.benchmark.common.evaluation import download

            download(client, args)
        elif args.action == "prepare":
            from scripts.benchmark.common.evaluation import prepare

            prepare(client, args)
        else:
            if not args.arm:
                p.error("run/bench requires --arm b0|b1")
            u = urlsplit(args.base_url)
            if (
                u.scheme not in ("http", "https")
                or not u.hostname
                or u.username
                or u.password
                or u.query
                or u.fragment
                or u.path not in ("", "/")
            ):
                p.error("base-url must be HTTP(S) origin without /v1 or credentials")
            args.base_url = args.base_url.rstrip("/")
            if args.api_key_env and not os.environ.get(args.api_key_env):
                p.error("API key environment variable is unset")
            if args.action == "run":
                from scripts.benchmark.common.evaluation import evaluate

                if client.dataset == "mmlu-pro" and not args.served_model:
                    p.error("MMLU run requires --served-model")
                evaluate(client, args)
            else:
                from scripts.benchmark.common.performance import bench

                bench(client, args)
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        NotImplementedError,
    ) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
