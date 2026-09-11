#!/usr/bin/env python3
"""Prepare or acquire immutable long-context acceptance inputs.

The generated fixture is intentionally labelled commissioning-only.  A formal
V4.6 run must instead download an evaluator-approved LONG-PUBLIC JSONL with a
declared license, revision, and checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        required = {
            "case_id",
            "profile",
            "prompt_token_ids",
            "expected",
            "input_len",
            "max_tokens",
        }
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"line {line_number} is missing: {', '.join(missing)}")
        token_ids = row["prompt_token_ids"]
        if not isinstance(token_ids, list) or len(token_ids) != row["input_len"]:
            raise ValueError(f"line {line_number} has an invalid prompt length")
        rows.append(row)
    if not rows:
        raise ValueError("dataset contains no cases")
    return rows


def _write_manifest(
    dataset: Path,
    *,
    source: str,
    source_revision: str,
    license_id: str,
    commissioning_only: bool,
) -> None:
    rows = _load_rows(dataset)
    profiles: dict[str, int] = {}
    for row in rows:
        profiles[row["profile"]] = profiles.get(row["profile"], 0) + 1
    manifest = {
        "schema_version": 1,
        "dataset": str(dataset.resolve()),
        "sha256": _sha256(dataset),
        "source": source,
        "source_revision": source_revision,
        "license": license_id,
        "commissioning_only": commissioning_only,
        "formal_v4_6_eligible": not commissioning_only,
        "case_count": len(rows),
        "profiles": profiles,
    }
    manifest_path = dataset.with_suffix(dataset.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _download(args: argparse.Namespace) -> None:
    if not args.url.startswith("https://"):
        raise ValueError("formal dataset downloads require an HTTPS URL")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="kvcompress-dataset-", suffix=".jsonl", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        with urllib.request.urlopen(args.url, timeout=args.timeout) as response:
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
    actual = _sha256(temporary_path)
    if actual != args.sha256.lower():
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"dataset SHA-256 mismatch: expected {args.sha256}, got {actual}"
        )
    temporary_path.replace(args.output)
    _write_manifest(
        args.output,
        source=args.url,
        source_revision=args.source_revision,
        license_id=args.license_id,
        commissioning_only=False,
    )


def _filler_tokens(tokenizer: Any, case_index: int, minimum: int) -> list[int]:
    chunks: list[int] = []
    section = 0
    while len(chunks) < minimum:
        text = (
            f"Archive {case_index:03d}, section {section:05d}. "
            "The observatory checked its instruments, recorded the weather, "
            "reviewed the maintenance ledger, and confirmed routine operation. "
            "This descriptive entry is context, not an authorization record.\n"
        )
        chunks.extend(tokenizer.encode(text, add_special_tokens=False))
        section += 1
    return chunks


def _make_case(
    tokenizer: Any,
    case_index: int,
    profile: str,
    input_len: int,
    max_tokens: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    station = f"HUST-{case_index:04d}"
    expected = f"{10_000_000 + ((case_index * 104_729 + 7_919) % 89_999_999):08d}"
    header = tokenizer.encode(
        "Read the archive. The AUTHORITATIVE RECORD overrides routine entries. "
        "At the final question, return the requested eight-digit code first.\n",
        add_special_tokens=False,
    )
    fact = tokenizer.encode(
        f"AUTHORITATIVE RECORD: station {station} has access code {expected}.\n",
        add_special_tokens=False,
    )
    question = tokenizer.encode(
        f"FINAL QUESTION: Give the access code for station {station}. ANSWER: ",
        add_special_tokens=False,
    )
    anchors = (512, input_len // 4, input_len // 2, input_len - 2048)
    fact_position = anchors[case_index % len(anchors)]
    filler = _filler_tokens(tokenizer, case_index, input_len + 256)
    before_len = fact_position - len(header)
    after_len = input_len - len(header) - before_len - len(fact) - len(question)
    if before_len < 0 or after_len < 0:
        raise ValueError(f"profile {profile} is too short for the fixture template")
    prompt = (
        header
        + filler[:before_len]
        + fact
        + filler[before_len : before_len + after_len]
        + question
    )
    if len(prompt) != input_len:
        raise AssertionError(f"built {len(prompt)} tokens, expected {input_len}")
    return {
        "case_id": f"{profile.lower()}-{case_index:04d}",
        "profile": profile,
        "prompt_token_ids": prompt,
        "prompt_sha256": hashlib.sha256(
            json.dumps(prompt, separators=(",", ":")).encode()
        ).hexdigest(),
        "expected": expected,
        "input_len": input_len,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "fact_position": fact_position,
        "commissioning_only": True,
    }


def _generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    specs = []
    if args.profile in {"a2", "all"}:
        specs.extend(
            [
                # Fixed decode lengths make throughput and latency comparable
                # across B0/B1; early EOS would otherwise turn A2 into a
                # prefill-only measurement for terse answers.
                ("A2-LONG-FP16-8K", 16, 8192, 512, True),
                ("A2-LONG-FP16-16K", 16, 16384, 1024, True),
            ]
        )
    if args.profile in {"a3", "all"}:
        specs.append(("A3-32K-FP16", 64, 30720, 2048, True))
    rows = []
    case_index = 0
    for profile, count, input_len, max_tokens, ignore_eos in specs:
        for _ in range(count):
            rows.append(
                _make_case(
                    tokenizer,
                    case_index,
                    profile,
                    input_len,
                    max_tokens,
                    ignore_eos,
                )
            )
            case_index += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_manifest(
        args.output,
        source="deterministic repository-generated commissioning fixture",
        source_revision="kvcompress_prepare_dataset.py/v1",
        license_id="Apache-2.0",
        commissioning_only=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--url", required=True)
    download.add_argument("--sha256", required=True)
    download.add_argument("--source-revision", required=True)
    download.add_argument("--license-id", required=True)
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--timeout", type=float, default=120.0)
    generate = commands.add_parser("generate-commissioning")
    generate.add_argument("--tokenizer", required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--profile", choices=("a2", "a3", "all"), default="all")
    args = parser.parse_args()
    if args.command == "download":
        _download(args)
    else:
        _generate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
