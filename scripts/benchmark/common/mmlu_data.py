#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Acquire fixed MMLU inputs; validation reuses the external client's API."""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

DATA_REV = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
REF_REV = "f418b116db00b065c2aea046518d8fcf74d39872"
CLIENT_REV = "b6215bf76808b1e9d351295e2ecb401542317372"
FILES = {
    "test": "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
    "validation": "139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078",
}


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
