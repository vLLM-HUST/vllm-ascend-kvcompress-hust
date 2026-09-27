#!/usr/bin/env python3
"""Read-only readiness check for the frozen Qwen3.5 Frontier campaign.

This is not a substitute for an NPU service qualification or 900-second run.
It deliberately does not import torch, torch_npu, vLLM, or device libraries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from importlib import metadata
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
PLUGIN = Path(__file__).resolve().parents[1]
PREPARED_SHA256 = "8105957b4001e7fd21373150fb7b844a9931abe9f88c699b3db191b84a8596d3"
AGENTX_REVISION = "8fecd2fc56694469f758f0afbbb6335ad3043740"
AGENTX_FINGERPRINT = "0d8fdac271289f80"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_version(requirements: Path, package: str) -> str:
    pattern = re.compile(rf"^{re.escape(package)}==([^\s#]+)")
    for line in requirements.read_text(encoding="utf-8").splitlines():
        if match := pattern.match(line.strip()):
            return match.group(1)
    raise ValueError(f"{package} is not pinned in {requirements}")


def installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def cann_version(root: Path) -> str | None:
    info = root / "share/info/asc-devkit/version.info"
    if not info.is_file():
        return None
    match = re.search(r"^Version=(\S+)$", info.read_text(), re.MULTILINE)
    return match.group(1) if match else None


def model_shards(model: Path) -> tuple[int, int]:
    index = model / "model.safetensors.index.json"
    if not index.is_file():
        return 0, 0
    filenames = set(
        json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()
    )
    return len(filenames), sum((model / filename).is_file() for filename in filenames)


def inspect(
    *,
    requirements: Path,
    cann_root: Path,
    model: Path,
    swe_prepared: Path,
    agentx_receipt: Path,
) -> dict:
    checks: list[dict] = []

    def add(name: str, ok: bool, actual: object, expected: object) -> None:
        checks.append(
            {"name": name, "ok": bool(ok), "actual": actual, "expected": expected}
        )

    for package in ("torch", "torch-npu"):
        expected = pinned_version(requirements, package)
        actual = installed_version(package)
        # PEP 440 permits a local build label such as ``2.13.0+cpu`` to satisfy
        # the upstream ``==2.13.0`` pin. The CPU wheel is the Ascend-compatible
        # choice here: the generic PyPI wheel requires CUDA shared libraries.
        matches = actual is not None and actual.split("+", 1)[0] == expected
        add(package, matches, actual, expected)

    # TorchNPU 2.13.0rc1 is an unreleased host pin. For this campaign, require
    # the CANN 9.1 family used by the official prerelease image, not 9.0.x.
    actual_cann = cann_version(cann_root)
    add(
        "CANN",
        bool(actual_cann and actual_cann.startswith("9.1.")),
        actual_cann,
        "9.1.x",
    )

    shard_count, present = model_shards(model)
    add(
        "model shards",
        shard_count > 0 and shard_count == present,
        {"expected_count": shard_count, "present_count": present},
        "all indexed shards",
    )
    add(
        "model tokenizer",
        (model / "tokenizer.json").is_file(),
        str(model / "tokenizer.json"),
        "existing local tokenizer",
    )

    actual_swe = sha256(swe_prepared) if swe_prepared.is_file() else None
    add("SWE prepared file", actual_swe == PREPARED_SHA256, actual_swe, PREPARED_SHA256)

    receipt = json.loads(agentx_receipt.read_text()) if agentx_receipt.is_file() else {}
    add(
        "AgentX dataset receipt",
        receipt.get("revision") == AGENTX_REVISION
        and receipt.get("fingerprint") == AGENTX_FINGERPRINT
        and receipt.get("sessions") == 393,
        {key: receipt.get(key) for key in ("revision", "fingerprint", "sessions")},
        {
            "revision": AGENTX_REVISION,
            "fingerprint": AGENTX_FINGERPRINT,
            "sessions": 393,
        },
    )

    return {
        "schema": "kvcompress/frontier-preflight/v1",
        "scope": "read-only metadata and local-input check; no NPU service test",
        "ready": all(check["ok"] for check in checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=WORKSPACE / "vllm-ascend-hust/requirements.txt",
    )
    parser.add_argument(
        "--cann-root",
        type=Path,
        default=Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann")),
    )
    parser.add_argument(
        "--model", type=Path, default=Path("/workspace/models/Qwen--Qwen3.5-35B-A3B")
    )
    parser.add_argument(
        "--swe-prepared",
        type=Path,
        default=PLUGIN / ".benchmarks/swe-qwen35-262144.json",
    )
    parser.add_argument(
        "--agentx-receipt",
        type=Path,
        default=WORKSPACE / f"agentx-bench/.cache/{AGENTX_REVISION}/prepared.json",
    )
    args = parser.parse_args()
    result = inspect(
        requirements=args.requirements,
        cann_root=args.cann_root,
        model=args.model,
        swe_prepared=args.swe_prepared,
        agentx_receipt=args.agentx_receipt,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
