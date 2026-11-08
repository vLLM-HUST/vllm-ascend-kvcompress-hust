#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the plugin overlay on an already-built, pinned HUST Python base."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
LOCK = json.loads((REPO / "requirements/ascend-host.json").read_text())


def run(*args: str | Path, capture: bool = False) -> str:
    result = subprocess.run(
        [str(a) for a in args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def packages(python: Path) -> dict:
    return json.loads(
        run(
            python,
            "-I",
            "-c",
            """
import importlib.metadata as m,json,sys
rows={}
for d in m.distributions():
    name=d.metadata.get('Name','').lower().replace('_','-')
    if name and name not in rows:
        rows[name]={'version':d.version,'root':str(d.locate_file('')),
                    'direct_url':json.loads(d.read_text('direct_url.json') or 'null')}
print(json.dumps({'packages':rows,'base_python':sys._base_executable,
                  'python':'.'.join(map(str,sys.version_info[:2]))}))
""",
            capture=True,
        )
    )


def check_versions(inventory: dict, expected: dict) -> None:
    for name, version in expected.items():
        actual = inventory["packages"].get(name, {}).get("version")
        if actual != version:
            raise ValueError(f"{name}: expected {version}, found {actual}")


def source_paths(inventory: dict) -> dict:
    paths = {}
    for name, expected in LOCK["sources"].items():
        direct = inventory["packages"][name]["direct_url"] or {}
        url = urlparse(direct.get("url", ""))
        if url.scheme != "file" or not direct.get("dir_info", {}).get("editable"):
            raise ValueError(f"{name}: the pinned HUST editable source is required")
        path = Path(unquote(url.path)).resolve()
        head = run("git", "-C", path, "rev-parse", "HEAD", capture=True)
        if head != expected["commit"]:
            raise ValueError(
                f"{name}: source HEAD {head} differs from the pinned commit"
            )
        paths[name] = {
            "path": str(path),
            "commit": head,
            "status": run("git", "-C", path, "status", "--short", capture=True),
            "diff_sha256": hashlib.sha256(
                subprocess.check_output(
                    ["git", "-C", str(path), "diff", "HEAD", "--binary"]
                )
            ).hexdigest(),
        }
    return paths


def toolchain_identity(cann: Path, atb: Path) -> dict:
    result = {}
    for name, setup, relative, pattern in (
        (
            "cann",
            cann,
            "share/info/asc-devkit/version.info",
            r"(?im)^\s*Version\s*=\s*([\d.]+)",
        ),
        ("atb", atb, "latest/version.info", r"(?im)^\s*Ascend-cann-atb\s*:\s*([\d.]+)"),
    ):
        metadata = setup.parent / relative
        if not metadata.is_file():
            raise ValueError(f"missing {name} version metadata: {metadata}")
        match = re.search(pattern, metadata.read_text())
        if match is None or match.group(1) != LOCK[name]:
            raise ValueError(f"{name} must be {LOCK[name]}: {metadata}")
        result[name] = {
            "version": match.group(1),
            "metadata": str(metadata),
            "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        }
    return result


def check_overlay(prefix: Path, base_python: Path, sources: dict) -> dict:
    inventory = packages(prefix / "bin/python")
    check_versions(
        inventory,
        {
            **LOCK["base"],
            **LOCK["overlay"],
            LOCK["fla"]["distribution"]: LOCK["fla"]["version"],
        },
    )
    if Path(inventory["base_python"]).resolve() != base_python:
        raise ValueError("overlay uses a different Python base")
    if (
        "include-system-site-packages = true"
        not in (prefix / "pyvenv.cfg").read_text().lower()
    ):
        raise ValueError("the plugin overlay must inherit the verified HUST base")
    current_sources = source_paths(inventory)
    if any(current_sources[name]["path"] != sources[name]["path"] for name in sources):
        raise ValueError("overlay HUST sources differ from the verified base")
    direct = (
        inventory["packages"].get("vllm-ascend-kvcompress-hust", {}).get("direct_url")
        or {}
    )
    if direct.get("url") != REPO.as_uri() or not direct.get("dir_info", {}).get(
        "editable"
    ):
        raise ValueError("the plugin must be editable from this repository")
    return inventory


def copy_seed(seed: Path, prefix: Path) -> None:
    # Copy the Python overlay, never the inherited base or CANN tree. Recreate
    # launchers with the new prefix; do not move a venv with stale shebangs.
    source = seed / "lib/python3.11/site-packages"
    target = prefix / "lib/python3.11/site-packages"
    target.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if "vllm_ascend_kvcompress" in entry.name or entry.name == "__pycache__":
            continue
        dest = target / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dest, dirs_exist_ok=True, symlinks=True)
        else:
            shutil.copy2(entry, dest)
    old_python = str(seed / "bin/python")
    for entry in (seed / "bin").iterdir():
        if (
            entry.is_symlink()
            or not entry.is_file()
            or entry.name.startswith("activate")
        ):
            continue
        if entry.name in {"Activate.ps1", "python", "python3", "python3.11"}:
            continue
        data = entry.read_bytes()
        if data.startswith(b"#!"):
            data = data.replace(
                old_python.encode(), str(prefix / "bin/python").encode()
            )
        dest = prefix / "bin" / entry.name
        dest.write_bytes(data)
        dest.chmod(entry.stat().st_mode)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prefix",
        type=Path,
        default=Path(
            os.environ.get("KVCOMPRESS_HOST_ENV", WORKSPACE / ".envs/kvcompress-host")
        ),
    )
    p.add_argument(
        "--seed-env",
        type=Path,
        help="copy a matched existing plugin overlay without downloading",
    )
    p.add_argument(
        "--cann-env", type=Path, default=os.environ.get("KVCOMPRESS_CANN_ENV")
    )
    p.add_argument("--atb-env", type=Path, default=os.environ.get("KVCOMPRESS_ATB_ENV"))
    p.add_argument(
        "--fla-wheel", type=Path, default=os.environ.get("KVCOMPRESS_FLA_WHEEL")
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="validate inputs and installed versions without installing",
    )
    a = p.parse_args()
    prefix = a.prefix.expanduser().absolute()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        p.error("the pinned environment requires Linux aarch64")
    # The locked A2 FLA artifact is tagged manylinux_2_34_aarch64.
    libc, libc_version = platform.libc_ver()
    if libc != "glibc" or tuple(map(int, libc_version.split(".")[:2])) < (2, 34):
        p.error("the locked A2 FLA wheel requires glibc >= 2.34")
    config_file = prefix / "kvcompress-host.json"
    saved_config = json.loads(config_file.read_text()) if config_file.is_file() else {}
    if saved_config.get("mode") == "standalone":
        p.error("standalone provisioning is unsupported; use a base-backed overlay")
    candidate = saved_config.get("base_python")
    if (
        candidate
        and not os.environ.get("KVCOMPRESS_BASE_PYTHON")
        and Path(sys._base_executable).resolve() != Path(candidate).resolve()
    ):
        os.execv(candidate, [candidate, str(Path(__file__).resolve()), *sys.argv[1:]])
    if sys.version_info[:2] != (3, 11):
        p.error(
            "provide an already-built Python 3.11 HUST base via KVCOMPRESS_BASE_PYTHON"
        )
    # sys._base_executable strips any plugin overlay from the base validation.
    base_python = Path(sys._base_executable).resolve()
    base = packages(base_python)
    check_versions(base, LOCK["base"])
    sources = source_paths(base)
    config_path = prefix / "kvcompress-host.json"
    saved = json.loads(config_path.read_text()) if config_path.is_file() else {}
    cann = (
        a.cann_env
        or (Path(saved["cann_env"]) if "cann_env" in saved else None)
        or WORKSPACE / ".envs/kvcompress-toolchain/Ascend/cann-9.1.0/set_env.sh"
    ).resolve()
    atb = (
        a.atb_env
        or (Path(saved["atb_env"]) if "atb_env" in saved else None)
        or WORKSPACE / ".envs/kvcompress-toolchain/Ascend/nnal/atb/set_env.sh"
    ).resolve()
    for path in (cann, atb):
        if not path.is_file():
            p.error(f"missing vendor setup: {path}; provide --cann-env and --atb-env")
    toolchain = toolchain_identity(cann, atb)
    seed = a.seed_env.resolve() if a.seed_env else None
    if seed:
        if seed == prefix or prefix.exists():
            p.error("--seed-env requires a new, distinct target directory")
        seed_inventory = packages(seed / "bin/python")
        check_versions(
            seed_inventory,
            {
                **LOCK["base"],
                **LOCK["overlay"],
                LOCK["fla"]["distribution"]: LOCK["fla"]["version"],
            },
        )
        if Path(seed_inventory["base_python"]).resolve() != base_python:
            p.error("seed and target must share the same persistent Python base")
    vendor = (
        Path(sources["vllm-ascend"]["path"])
        / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
    )
    if not vendor.is_file():
        p.error(f"missing built Ascend vendor setup: {vendor}")
    for name in ("bishengir-compile", "bishengir-opt"):
        executable = cann.parent / "tools/bishengir/bin" / name
        if not executable.is_file() or not os.access(executable, os.X_OK):
            p.error(f"missing SDK executable: {executable}")
    if a.check:
        if not seed:
            check_overlay(prefix, base_python, sources)
            if not config_file.is_file():
                p.error("missing host installation record; run install b1 first")
            if (
                saved_config.get("lock_sha256")
                != hashlib.sha256(
                    (REPO / "requirements/ascend-host.json").read_bytes()
                ).hexdigest()
            ):
                p.error("host lock changed; reinstall the overlay")
        print(
            "Pinned HUST base, overlay, source origins and vendor identities verified."
        )
        return
    if prefix.exists() and not (prefix / "pyvenv.cfg").is_file():
        p.error("target exists and is not a virtual environment")
    if not prefix.exists():
        run(base_python, "-m", "venv", "--system-site-packages", prefix)
    python = prefix / "bin/python"
    if seed:
        copy_seed(seed, prefix)
    else:
        # The hardware host owns its dependency set. Installing only this
        # overlay avoids selecting incompatible public host wheels.
        inventory = packages(python)
        for name, version in LOCK["overlay"].items():
            if inventory["packages"].get(name, {}).get("version") == version:
                continue
            spec = f"{name}=={version}"
            args = [python, "-m", "pip", "install", "--no-deps", spec]
            if name in {"torch", "torchvision"}:
                args += ["--index-url", "https://download.pytorch.org/whl/cpu"]
            elif name == "vllm-hust-ext":
                args[-1] = (
                    "https://github.com/vLLM-HUST/extension-manager/releases/"
                    "download/v0.3.0/vllm_hust_ext-0.3.0-py3-none-any.whl"
                )
            run(*args)
        inventory = packages(python)
        if (
            inventory["packages"].get(LOCK["fla"]["distribution"], {}).get("version")
            != LOCK["fla"]["version"]
        ):
            wheel = (
                a.fla_wheel or WORKSPACE / ".cache/fla-npu" / LOCK["fla"]["filename"]
            )
            if (
                not wheel.is_file()
                or hashlib.sha256(wheel.read_bytes()).hexdigest()
                != LOCK["fla"]["sha256"]
            ):
                p.error("provide the hash-matched A2 FLA wheel with --fla-wheel")
            run(python, "-m", "pip", "install", "--no-deps", wheel)
    run(python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", REPO)
    final = check_overlay(prefix, base_python, sources)
    config = {
        "lock_sha256": hashlib.sha256(
            (REPO / "requirements/ascend-host.json").read_bytes()
        ).hexdigest(),
        "base_python": str(base_python),
        "toolchain": toolchain,
        "cann_env": str(cann),
        "atb_env": str(atb),
        "triton_source": str(Path(sources["triton-ascend"]["path"]) / "python"),
        "vendor_env": str(vendor),
        "sources": sources,
        "packages": final["packages"],
    }
    (prefix / "kvcompress-host.json").write_text(json.dumps(config, indent=2) + "\n")
    # Keep inherited optional host dependency gaps visible. This is a report,
    # not proof of an NPU-ready runtime, and never changes the pinned versions.
    report = subprocess.run(
        [str(python), "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (prefix / "dependency-check.txt").write_text(report.stdout)
    print(f"Plugin environment configured: {prefix}")
    print(
        f"Dependency audit exit status: {report.returncode}; "
        f"see {prefix / 'dependency-check.txt'}"
    )
    print(
        "No NPU inference performed; "
        "run the documented hardware checks after allocation."
    )


if __name__ == "__main__":
    try:
        main()
    except (
        ValueError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(f"B1 installation failed: {error}") from error
