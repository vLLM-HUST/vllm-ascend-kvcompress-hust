"""Resolve migrated inputs externally; preserve isolated negative-test roots."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def original_path(root: Path, relative: str) -> Path:
    if root.resolve() != ROOT:
        return root / relative
    config = json.loads((ROOT / "BENCHMARK_ORIGINALS.json").read_text())
    expected = config.get("migrated_sha256", {}).get(relative)
    if expected is None:
        return root / relative
    value = os.environ.get("VLLM_HUST_BENCHMARK_ORIGINALS")
    if not value:
        raise ValueError("Download originals and set VLLM_HUST_BENCHMARK_ORIGINALS")
    external = Path(value).resolve()
    if external == ROOT or ROOT in external.parents:
        raise ValueError("Originals must be outside the source checkout")
    path = (external / relative).resolve()
    if external not in path.parents:
        raise ValueError("Unsafe original input path")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Original input checksum mismatch")
    return path
