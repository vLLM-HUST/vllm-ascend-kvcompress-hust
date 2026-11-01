#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pinned MMLU-Pro preparation and client entry point (no NPU required)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

DATA_REV = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
REF_REV = "f418b116db00b065c2aea046518d8fcf74d39872"
CLIENT_REV = "b6215bf76808b1e9d351295e2ecb401542317372"
FILES = {
    "test": "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
    "validation": "139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078",
}
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / ".benchmarks/pujiang"


def call(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "hust_eval_mmlu_pro", *args], check=True)


def download(root: Path) -> None:
    from hust_eval_mmlu_pro.mmlu_pro.plan import load_assets, load_instruction

    assets = root / "assets/mmlu-pro/data"
    assets.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "data-manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "revision": DATA_REV,
        "reference_revision": REF_REV,
        "client_revision": CLIENT_REV,
        "files": {},
    }
    for split, digest in FILES.items():
        name = f"{split}-00000-of-00001.parquet"
        path = assets / name
        url = f"https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/{DATA_REV}/data/{name}"
        existed = path.exists()
        source = url
        if not path.exists():
            temp = path.with_suffix(".part")
            mirror = (
                "https://modelscope.cn/datasets/TIGER-Lab/MMLU-Pro/"
                f"resolve/master/data/{name}"
            )
            for source in (url, mirror):
                try:
                    with (
                        urllib.request.urlopen(source, timeout=30) as src,
                        temp.open("wb") as dst,
                    ):
                        while block := src.read(1024 * 1024):
                            dst.write(block)
                    if hashlib.sha256(temp.read_bytes()).hexdigest() != digest:
                        raise ValueError(f"download checksum mismatch: {source}")
                    temp.replace(path)
                    break
                except (urllib.error.URLError, TimeoutError, ValueError):
                    temp.unlink(missing_ok=True)
                    if source == mirror:
                        raise
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"checksum mismatch: {path}")
        manifest["files"][name] = {
            "sha256": digest,
            "bytes": path.stat().st_size,
            "canonical_url": url,
            "download_url": (
                source
                if not existed
                else previous.get("files", {})
                .get(name, {})
                .get("download_url", "existing-checksum-verified")
            ),
        }
    ref = root / "reference/MMLU-Pro"
    if not ref.exists():
        ref.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/TIGER-AI-Lab/MMLU-Pro.git", str(ref)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(ref), "checkout", "--detach", REF_REV], check=True
        )
    head = subprocess.check_output(
        ["git", "-C", str(ref), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != REF_REV:
        raise ValueError("reference revision mismatch; use a separate checkout")
    test, validation = load_assets(assets)
    load_instruction(ref)
    if (len(test), len(validation)) != (12032, 70):
        raise ValueError("unexpected dataset row counts")
    manifest["rows"] = {"test": len(test), "validation": len(validation)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Pujiang Laboratory designated evaluation inputs (default: %(default)s)",
    )
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser(
        "download", help="download and verify both splits and upstream reference"
    )
    prep = sub.add_parser("prepare", help="freeze full token-ID plan locally")
    prep.add_argument("--model-path", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--runtime-contract")
    run = sub.add_parser("run", help="evaluate ONE arm against an existing server")
    run.add_argument("--plan", required=True)
    run.add_argument("--base-url", required=True, help="service URL without /v1")
    run.add_argument("--served-model", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--limit", type=int)
    run.add_argument("--api-key-env")
    run.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.action == "download":
        download(a.root)
    elif a.action == "prepare":
        args = [
            "mmlu-pro",
            "prepare",
            "--assets",
            str(a.root / "assets/mmlu-pro/data"),
            "--reference",
            str(a.root / "reference/MMLU-Pro"),
            "--model-path",
            a.model_path,
            "--output",
            a.output,
        ]
        if a.runtime_contract:
            args.extend(["--runtime-contract", a.runtime_contract])
        call(*args)
    else:
        if a.base_url.rstrip("/").endswith("/v1"):
            p.error("--base-url must exclude /v1")
        if a.concurrency < 1 or (a.limit is not None and a.limit < 1):
            p.error("concurrency and limit must be positive")
        args = [
            "mmlu-pro",
            "run",
            "--plan",
            a.plan,
            "--base-url",
            a.base_url,
            "--served-model",
            a.served_model,
            "--output",
            a.output,
            "--concurrency",
            str(a.concurrency),
        ]
        for key in ["limit", "api_key_env"]:
            if getattr(a, key) is not None:
                args.extend(["--" + key.replace("_", "-"), str(getattr(a, key))])
        if a.resume:
            args.append("--resume")
        # The pinned client overwrites summary.json during run(), including resume.
        # Preserve it before invoking the client rather than after it returns.
        previous_summary = Path(a.output) / "summary.json"
        if a.resume and previous_summary.exists():
            backup = previous_summary.with_name(f"summary.previous.{uuid4().hex}.json")
            backup.write_bytes(previous_summary.read_bytes())
        failed = None
        try:
            call(*args)
        except subprocess.CalledProcessError as error:
            failed = error
        # Keep a summary even when transport failed. The client keeps failures in
        # the denominator and labels incomplete runs; never mask its exit status.
        summary_path = Path(a.output) / "summary.json"
        if summary_path.exists():
            summary_path = summary_path.with_name(f"summary.{uuid4().hex}.json")
        call(
            "mmlu-pro",
            "summary",
            "--plan",
            a.plan,
            "--accuracy-dir",
            a.output,
            "--out",
            str(summary_path),
        )
        if failed:
            raise failed


if __name__ == "__main__":
    main()
