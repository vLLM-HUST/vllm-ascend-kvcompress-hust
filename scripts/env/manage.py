#!/usr/bin/env python3
"""Configure external CPU clients and run plugin-off/on arms on one HUST host."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmark.cli import adapter  # noqa: E402
from scripts.benchmark.common.clients import (  # noqa: E402
    check_source,
    clean_client_env,
)
from scripts.benchmark.common.files import WORKSPACE, write  # noqa: E402


def run(*command, env=None, capture=False):
    result = subprocess.run(
        [str(c) for c in command],
        check=True,
        text=True,
        env=env,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else None


def prefix_for(target, dataset=None):
    name = {
        "b0": "host",
        "b1": "host",
        "client": (dataset or "").replace("frontier-science", "frontierscience"),
    }[target]
    if not name:
        raise ValueError("client requires --dataset")
    return WORKSPACE / ".envs" / f"kvcompress-{name}"


def inventory(python):
    return json.loads(
        run(
            python,
            "-I",
            "-c",
            """
import importlib.metadata as m,json,sys
print(json.dumps({'python':list(sys.version_info[:2]),'prefix':sys.prefix,
 'packages':{d.metadata['Name'].lower().replace('_','-'):d.version
             for d in m.distributions() if d.metadata.get('Name')}}))
""",
            env=clean_client_env(),
            capture=True,
        )
    )


def arm_environment(arm, prefix, inherited=None):
    """Both arms share the host; clear inherited activation before setting the mode."""
    env = dict(os.environ if inherited is None else inherited)
    for key in list(env):
        if key.startswith("VLLM_") and key != "VLLM_API_KEY":
            env.pop(key)
    env["KVCOMPRESS_HOST_ENV"] = str(prefix)
    env["VLLM_ASCEND_KVCOMPRESS_ENABLED"] = "1" if arm == "b1" else "0"
    env["VLLM_PLUGINS"] = "ascend,ascend_kvcompress" if arm == "b1" else "ascend"
    if arm == "b1":
        config = (os.environ if inherited is None else inherited).get(
            "VLLM_ASCEND_KVCOMPRESS_CONFIG"
        )
        if not config:
            raise ValueError("B1 run requires VLLM_ASCEND_KVCOMPRESS_CONFIG")
        config = Path(config).expanduser().resolve()
        if not config.is_file():
            raise ValueError(f"missing B1 plugin configuration: {config}")
        env["VLLM_ASCEND_KVCOMPRESS_CONFIG"] = str(config)
    return env


def client_command(client, root, python):
    return [
        python,
        "-c",
        (
            "import sys;sys.path.insert(0,sys.argv[1]);"
            "from scripts.benchmark.cli import adapter;"
            "from scripts.benchmark.common.clients import check_installed;"
            "check_installed(adapter(sys.argv[2]).CLIENT,sys.argv[3])"
        ),
        ROOT,
        client.dataset,
        root,
    ]


def install_client(args, prefix):
    client = adapter(args.dataset).CLIENT
    source = check_source(client, args.client_root)
    python = prefix / "bin/python"
    if prefix.exists():
        run(*client_command(client, args.client_root, python), env=clean_client_env())
        if client.dataset == "mmlu-pro":
            check_mmlu_dependencies(python)
        print(f"Reusing client environment: {prefix}")
        return
    run(args.python, "-m", "venv", prefix, env=clean_client_env())
    requirement = f"{source}[prepare]" if client.dataset == "mmlu-pro" else str(source)
    run(python, "-m", "pip", "install", requirement, env=clean_client_env())
    run(*client_command(client, args.client_root, python), env=clean_client_env())
    if client.dataset == "mmlu-pro":
        check_mmlu_dependencies(python)
    write(
        prefix / "kvcompress-client.json",
        {
            "dataset": client.dataset,
            "source": str(source),
            "commit": client.commit,
            "packages": inventory(python),
        },
    )


def check_mmlu_dependencies(python):
    # Read the installed metadata already verified against the pinned local source.
    run(
        python,
        "-I",
        "-c",
        """
from importlib.metadata import requires, version
from packaging.requirements import Requirement
requirements = requires("hust-eval-client-mmlu-pro") or []
for declaration in requirements:
    requirement = Requirement(declaration)
    if requirement.marker and not requirement.marker.evaluate({"extra": "prepare"}):
        continue
    actual = version(requirement.name)
    if not requirement.specifier.contains(actual, prereleases=True):
        raise ValueError(f"{requirement}: installed {actual}")
""",
        env=clean_client_env(),
    )
    run(python, "-m", "pip", "check", env=clean_client_env())


def main():
    p = argparse.ArgumentParser(prog="set_env.sh", description=__doc__)
    p.add_argument("action", choices=("install", "check", "run"))
    p.add_argument("target", choices=("b0", "b1", "client"))
    p.add_argument(
        "--dataset", choices=("mmlu-pro", "hle-verified", "frontier-science")
    )
    p.add_argument("--prefix", type=Path)
    p.add_argument("--client-root", type=Path, default=WORKSPACE / "pujiang-benchmark")
    p.add_argument("--python", default="python3.11")
    # Split -- explicitly: argparse REMAINDER would swallow environment options.
    argv = sys.argv[1:]
    boundary = argv.index("--") if "--" in argv else len(argv)
    command = argv[boundary + 1 :]
    args = p.parse_args(argv[:boundary])
    try:
        default_prefix = (
            Path(os.environ["KVCOMPRESS_HOST_ENV"])
            if args.target in ("b0", "b1") and os.environ.get("KVCOMPRESS_HOST_ENV")
            else prefix_for(args.target, args.dataset)
        )
        prefix = (args.prefix or default_prefix).expanduser().absolute()
        if args.action == "install":
            if args.target == "client":
                install_client(args, prefix)
            else:
                p.error("use scripts/set_env.sh install b0|b1")
        elif args.action == "check":
            if args.target == "client":
                client = adapter(args.dataset).CLIENT
                run(
                    *client_command(client, args.client_root, prefix / "bin/python"),
                    env=clean_client_env(),
                )
                if client.dataset == "mmlu-pro":
                    check_mmlu_dependencies(prefix / "bin/python")
            print(
                "Environment identity verified; "
                "NPU service qualification not performed."
            )
        else:
            if not command:
                p.error("run requires -- COMMAND [ARGS...]")
            if args.target in ("b0", "b1"):
                env = arm_environment(args.target, prefix)
                os.execvpe(
                    "bash",
                    ["bash", str(ROOT / "scripts/env/ascend_env.sh"), *command],
                    env,
                )
            else:
                client = adapter(args.dataset).CLIENT
                run(
                    *client_command(client, args.client_root, prefix / "bin/python"),
                    env=clean_client_env(),
                )
                env = clean_client_env()
                env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
                os.execvpe(command[0], command, env)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
