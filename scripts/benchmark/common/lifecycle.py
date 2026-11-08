"""Own only the model server process groups created for a configured pair."""

import gzip
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .clients import clean_client_env
from .files import ROOT, WORKSPACE, digest, read, write
from .performance import compression_evidence
from .runtime import compare_runtime

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch(url, key_env=None):
    headers = {}
    if key_env:
        headers["Authorization"] = "Bearer " + os.environ[key_env]
    with OPENER.open(
        urllib.request.Request(url, headers=headers), timeout=5
    ) as response:
        return response.read().decode()


def stop_server(server):
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=30)


def sample(root, base_url, stop, key_env=None):
    with gzip.open(root / "telemetry.jsonl.gz", "wt") as stream:
        while not stop.is_set():
            record = {"time_unix": time.time()}
            try:
                record["metrics"] = fetch(base_url + "/metrics", key_env)
            except OSError as error:
                record["metrics_error"] = repr(error)
            try:
                proc = subprocess.run(
                    ["npu-smi", "info"], capture_output=True, text=True, timeout=15
                )
                record.update(npu_smi=proc.stdout, npu_smi_returncode=proc.returncode)
            except (OSError, subprocess.TimeoutExpired) as error:
                record["npu_smi_error"] = repr(error)
            memory = Path("/sys/fs/cgroup/memory.current")
            if memory.exists():
                record["memory_current"] = memory.read_text().strip()
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            stop.wait(5)


def run_arm(
    command,
    client_commands,
    root,
    base_url,
    env,
    timeout,
    key_env=None,
    runtime_command=None,
    runtime_context=None,
    expected_runtime=None,
):
    """One cold lifecycle; all failure paths retain evidence and release the server."""
    root.mkdir(parents=True, exist_ok=False)
    meta = {"command": command, "started_unix": time.time(), "status": "starting"}
    stop = threading.Event()
    server = sampler = None
    try:
        if runtime_command:
            raw = subprocess.check_output(runtime_command, env=env, text=True)
            receipt = json.loads(raw.splitlines()[-1])
            if runtime_context is not None:
                receipt["server"] = runtime_context
            if expected_runtime is not None and receipt != expected_runtime:
                raise ValueError("server runtime changed after pair preflight")
            write(root / "runtime.json", receipt)
        with (root / "server.log").open("w") as log:
            server = subprocess.Popen(
                command, env=env, stdout=log, stderr=log, start_new_session=True
            )
            meta["pid"] = server.pid
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"server exited during startup: {server.returncode}"
                    )
                try:
                    fetch(base_url + "/health", key_env)
                    break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("model server startup timeout")
            meta["ready_unix"] = time.time()
            sampler = threading.Thread(
                target=sample, args=(root, base_url, stop, key_env)
            )
            sampler.start()
            for index, client_command in enumerate(client_commands):
                with (root / f"client-{index}.log").open("w") as client_log:
                    subprocess.run(
                        client_command,
                        env={
                            **clean_client_env(),
                            **(
                                {"KVCOMPRESS_HOST_ENV": env["KVCOMPRESS_HOST_ENV"]}
                                if env.get("KVCOMPRESS_HOST_ENV")
                                else {}
                            ),
                        },
                        check=True,
                        stdout=client_log,
                        stderr=client_log,
                    )
            if server.poll() is not None:
                raise RuntimeError("server exited during measurement")
            meta["status"] = "completed"
    except BaseException as error:
        meta.update(status="failed", error=repr(error))
        raise
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=30)
        try:
            if server is not None:
                stop_server(server)
                meta["server_exit_code"] = server.returncode
        except Exception as error:
            meta.update(status="failed", cleanup_error=repr(error))
            raise
        finally:
            meta["finished_unix"] = time.time()
            if (root / "server.log").exists():
                meta["compression"] = compression_evidence(root / "server.log")
            write(root / "lifecycle.json", meta)


def pair(client, args):
    stage = getattr(args, "stage", "all")
    if not args.config:
        raise ValueError("pair requires --config")
    cfg = read(args.config)
    base = args.config.resolve().parent

    def path(value):
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else base / p).absolute())

    model = path(cfg["model_path"])
    arms = cfg["arms"]
    if set(arms) != {"b0", "b1"}:
        raise ValueError("pair config requires exactly b0 and b1")
    for arm in arms.values():
        if not isinstance(arm["server_args"], list) or not all(
            isinstance(x, str) for x in arm["server_args"]
        ):
            raise ValueError("server_args must be an argument array")
        if any(x.startswith("--api-key") for x in arm["server_args"]):
            raise ValueError(
                "use service API key environment configuration, not argv secrets"
            )
    default_prefix = (
        path(cfg["host_prefix"])
        if cfg.get("host_prefix")
        else os.environ.get(
            "KVCOMPRESS_HOST_ENV", str(WORKSPACE / ".envs/kvcompress-host")
        )
    )
    prefixes = {
        name: str(
            Path(path(arm["prefix"]) if arm.get("prefix") else default_prefix)
            .expanduser()
            .resolve()
        )
        for name, arm in arms.items()
    }
    if prefixes["b0"] != prefixes["b1"]:
        raise ValueError("B0/B1 must share one host environment prefix")
    settings = {}
    for name, arm in arms.items():
        server_env = {**cfg.get("server_env", {}), **arm.get("server_env", {})}
        forbidden = {"VLLM_PLUGINS", "VLLM_HUST_EXT_CONFIG"}
        if any(
            k in forbidden or k.startswith("VLLM_ASCEND_KVCOMPRESS") for k in server_env
        ):
            raise ValueError("set plugin activation through the arm, not server_env")
        if any(x.startswith("--kv-compression") for x in arm["server_args"]):
            raise ValueError(
                "set compression through B1 plugin_config, not server_args"
            )
        settings[name] = {
            "model_path": model,
            "served_model": arm["served_model"],
            "server_args": arm["server_args"],
            "server_env": {k: str(v) for k, v in server_env.items()},
        }
    if settings["b0"] != settings["b1"]:
        raise ValueError("B0/B1 model, server arguments or environment settings differ")
    if not arms["b1"].get("plugin_config"):
        raise ValueError("B1 requires plugin_config")

    def launch_settings(name):
        arm = arms[name]
        env = clean_client_env()
        for key in list(env):
            if key.startswith("VLLM_") and key != "VLLM_API_KEY":
                env.pop(key)
        env.update(settings[name]["server_env"])
        env["KVCOMPRESS_HOST_ENV"] = prefixes[name]
        if name == "b1":
            env["VLLM_ASCEND_KVCOMPRESS_CONFIG"] = path(arm["plugin_config"])
        launch = [
            str(ROOT / "scripts/set_env.sh"),
            "run",
            name,
            "--prefix",
            prefixes[name],
        ]
        probe = [
            *launch,
            "--",
            "python",
            "-I",
            str(ROOT / "scripts/env/runtime_identity.py"),
        ]
        return env, launch, probe

    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "pair-config.json", cfg)
    common = [sys.executable, str(ROOT / "scripts/benchmark/cli.py"), client.dataset]
    flags = [
        "--client-root",
        str(args.client_root),
        "--data-root",
        str(args.data_root),
        "--selection-dir",
        str(args.selection_dir),
        "--sample-size",
        str(args.sample_size),
    ]
    python = args.client_python or (
        Path(path(cfg["client_python"]))
        if cfg.get("client_python")
        else client.python()
    )
    flags += ["--client-python", str(python)]
    if args.smoke:
        flags.append("--smoke")
    if args.seed is not None:
        flags += ["--seed", str(args.seed)]
    for track in args.track or []:
        flags += ["--track", track]
    identities = []
    for index, name in enumerate(("b0", "b1")):
        arm = arms[name]
        inspected = args.output / f"{name}-input.json"
        resample = ["--resample"] if args.resample and index == 0 else []
        subprocess.run(
            [
                *common,
                "inspect",
                *flags,
                *resample,
                "--plan",
                path(arm["plan"]),
                "--output",
                str(inspected),
            ],
            check=True,
            env=clean_client_env(),
        )
        identities.append(read(inspected))
        if client.dataset != "mmlu-pro":
            c = read(Path(path(arm["plan"])) / "contract.json")
            if (
                c["solver"]["base_url"].rstrip("/") != arm["base_url"].rstrip("/")
                or c["solver"]["model"] != arm["served_model"]
            ):
                raise ValueError("pair service endpoint/model differs from frozen plan")
        check = [
            str(ROOT / "scripts/set_env.sh"),
            "check",
            name,
            "--prefix",
            prefixes[name],
        ]
        subprocess.run(check, check=True, env=clean_client_env())
    if identities[0] != identities[1]:
        raise ValueError("B0/B1 input, model, requests or fixed Judge differ")
    if client.dataset != "mmlu-pro":
        judge = identities[0]["identity"]["judge"]
        if judge["base_url"].rstrip("/") in {
            a["base_url"].rstrip("/") for a in arms.values()
        }:
            raise ValueError("pair requires an independent fixed Judge service")
        if stage != "solver":
            fetch(
                judge["base_url"].rstrip("/") + "/v1/models", judge.get("api_key_env")
            )
    bench_binary = args.bench_vllm_bin or cfg.get("bench_vllm_bin")
    if args.with_bench:
        from .performance import check_bench

        check_bench(
            bench_binary or ROOT / "scripts/env/bench_vllm.sh",
            image=client.dataset == "hle-verified",
            host_prefix=prefixes["b0"],
        )
    runtimes = {}
    for name, arm in arms.items():
        try:
            fetch(arm["base_url"].rstrip("/") + "/health", arm.get("api_key_env"))
        except OSError:
            pass
        else:
            raise ValueError(f"healthy server already occupies {arm['base_url']}")
        env, _, probe = launch_settings(name)
        raw = subprocess.check_output(probe, env=env, text=True)
        runtimes[name] = json.loads(raw.splitlines()[-1])
        runtimes[name]["server"] = settings[name]
    if compare_runtime(runtimes["b0"], runtimes["b1"])["status"] != "verified":
        raise ValueError("pair requires verified same-base runtime receipts")
    write(args.output / "runtime-preflight.json", runtimes)
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for number in range(1, args.pairs + 1):
            for name in ("b0", "b1"):
                arm = arms[name]
                url = arm["base_url"].rstrip("/")
                key_env = arm.get("api_key_env")
                try:
                    fetch(url + "/health", key_env)
                except OSError:
                    pass
                else:
                    raise ValueError(f"healthy server already occupies {url}")
                root = args.output / f"pair-{number}" / name
                env, launch, runtime_command = launch_settings(name)
                launch += [
                    "--",
                    "python",
                    str(ROOT / "scripts/env/benchmark_host.py"),
                    "-m",
                    "vllm.entrypoints.cli.main",
                    "serve",
                    model,
                    *arm["server_args"],
                ]
                run_flags = [
                    *flags,
                    "--arm",
                    name,
                    "--plan",
                    path(arm["plan"]),
                    "--base-url",
                    url,
                    "--served-model",
                    arm["served_model"],
                    "--runtime-contract",
                    str(root / "runtime.json"),
                    "--server-log",
                    str(root / "server.log"),
                ]
                for key in ("concurrency", "timeout"):
                    if getattr(args, key) is not None:
                        run_flags += ["--" + key, str(getattr(args, key))]
                if key_env:
                    run_flags += ["--api-key-env", key_env]
                commands = [
                    [
                        *common,
                        "run",
                        *run_flags,
                        "--stage",
                        stage,
                        "--output",
                        str(root / "accuracy"),
                    ]
                ]
                if args.with_bench:
                    perf = [
                        *common,
                        "bench",
                        *run_flags,
                        "--model-path",
                        model,
                        "--output",
                        str(root / "bench"),
                        "--output-len",
                        str(args.output_len),
                        "--server-log",
                        str(root / "server.log"),
                    ]
                    if bench_binary:
                        perf += ["--bench-vllm-bin", path(bench_binary)]
                    commands.append(perf)
                print(
                    f"pair={number} arm={name} input={digest(identities[0])}",
                    flush=True,
                )
                run_arm(
                    launch,
                    commands,
                    root,
                    url,
                    env,
                    args.startup_timeout,
                    key_env,
                    runtime_command,
                    settings[name],
                    runtimes[name],
                )
            from .reporting import compare

            if stage == "solver":
                write(
                    args.output / f"pair-{number}/pending-judge.json",
                    {"status": "awaiting_judge", "arms": ["b0", "b1"]},
                )
                continue
            compare(
                args.output / f"pair-{number}/b0",
                args.output / f"pair-{number}/b1",
                args.output / f"pair-{number}/comparison",
            )
    finally:
        signal.signal(signal.SIGTERM, previous)
