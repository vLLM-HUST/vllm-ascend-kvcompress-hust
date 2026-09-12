#!/usr/bin/env python3
"""Download and freeze public long-context benchmark requests.

The script uses pinned official snapshots, verifies their SHA-256 values, and
never truncates a benchmark prompt. Cases that do not fit the declared model
context are recorded as unsupported instead of silently disappearing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SOURCES = {
    "longbench-v2": {
        "repo": "THUDM/LongBench-v2",
        "revision": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
        "filename": "data.json",
        "sha256": "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2",
        "license": "Apache-2.0 (dataset card)",
    },
    "longbench": {
        "repo": "THUDM/LongBench",
        "revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
        "filename": "data.zip",
        "sha256": "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64",
        "license": "Composite upstream datasets; retain task-level terms",
    },
}

LONG_BENCH_TASKS = {
    "longbench-passage-retrieval-en": {
        "filename": "passage_retrieval_en.jsonl",
        "task": "passage_retrieval_en",
        "max_tokens": 32,
        "template": (
            "Here are 30 paragraphs from Wikipedia, along with an abstract. "
            "Please determine which paragraph the abstract is from.\n\n"
            "{context}\n\nThe following is an abstract.\n\n{input}\n\n"
            "Please enter the number of the paragraph that the abstract is "
            'from. The answer format must be like "Paragraph 1", "Paragraph '
            '2", etc.\n\nThe answer is: '
        ),
    },
    "longbench-qasper": {
        "filename": "qasper.jsonl",
        "task": "qasper",
        "max_tokens": 128,
        "template": (
            "You are given a scientific article and a question. Answer the "
            "question as concisely as you can, using a single phrase or "
            "sentence if possible. If the question cannot be answered based "
            'on the information in the article, write "unanswerable". If the '
            'question is a yes/no question, answer "yes", "no", or '
            '"unanswerable". Do not provide any explanation.\n\nArticle: '
            "{context}\n\nAnswer the question based on the above article as "
            "concisely as you can, using a single phrase or sentence if "
            "possible. If the question cannot be answered based on the "
            'information in the article, write "unanswerable". If the '
            'question is a yes/no question, answer "yes", "no", or '
            '"unanswerable". Do not provide any explanation.\n\nQuestion: '
            "{input}\n\nAnswer:"
        ),
    },
}

LONGBENCH_V2_TEMPLATE = """Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Format your response as follows: "The correct answer is (insert answer here)"."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(root: Path, source_name: str) -> Path:
    source = SOURCES[source_name]
    return root / source_name / str(source["filename"])


def _download(url: str, destination: Path, timeout: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="kvcompress-public-", suffix=".download", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        with urllib.request.urlopen(url, timeout=timeout) as response:
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
    temporary_path.replace(destination)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def download_sources(args: argparse.Namespace) -> None:
    selected = SOURCES if args.source == "all" else {args.source: SOURCES[args.source]}
    manifests = []
    for name, source in selected.items():
        destination = source_path(args.root, name)
        expected = str(source["sha256"])
        if not destination.exists() or sha256(destination) != expected:
            url = (
                f"https://huggingface.co/datasets/{source['repo']}/resolve/"
                f"{source['revision']}/{source['filename']}"
            )
            _download(url, destination, args.timeout)
        actual = sha256(destination)
        if actual != expected:
            raise ValueError(f"{name} SHA-256 mismatch: {actual} != {expected}")
        if name == "longbench":
            _safe_extract(destination, args.root / name / "extracted")
        manifests.append(
            {
                "source": name,
                **source,
                "path": str(destination.resolve()),
                "verified_sha256": actual,
            }
        )
    manifest_path = args.root / "source-manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "sources": manifests}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "sources": manifests}, indent=2))


def _load_raw(root: Path, benchmark: str) -> list[dict[str, Any]]:
    if benchmark == "longbench-v2":
        data = json.loads(source_path(root, benchmark).read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) != 503:
            raise ValueError("LongBench-v2 must contain exactly 503 cases")
        return data
    task = LONG_BENCH_TASKS[benchmark]
    path = root / "longbench" / "extracted" / "data" / str(task["filename"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 200:
        raise ValueError(f"{benchmark} must contain exactly 200 cases")
    return rows


def _format_case(benchmark: str, row: dict[str, Any]) -> tuple[str, list[str], int]:
    if benchmark == "longbench-v2":
        prompt = LONGBENCH_V2_TEMPLATE.format(**row)
        return prompt, [str(row["answer"])], 128
    task = LONG_BENCH_TASKS[benchmark]
    prompt = str(task["template"]).format(**row)
    return prompt, [str(value) for value in row["answers"]], int(task["max_tokens"])


def _tokenize(tokenizer: Any, prompt: str, limit: int) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        max_length=limit + 1,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    return list(encoded)


def _stable_order(case: dict[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{case['case_id']}".encode()).hexdigest()


def prepare_requests(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=args.local_files_only, trust_remote_code=False
    )
    raw_rows = _load_raw(args.root, args.benchmark)
    eligible: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    short: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows):
        case_id = str(row.get("_id", f"{args.benchmark}-{index:04d}"))
        if args.benchmark == "longbench-v2":
            answers = [str(row["answer"])]
            max_tokens = 128
        else:
            answers = [str(value) for value in row["answers"]]
            max_tokens = int(LONG_BENCH_TASKS[args.benchmark]["max_tokens"])
        record = {
            "case_id": case_id,
            "benchmark": args.benchmark,
            "task": (
                "longbench-v2"
                if args.benchmark == "longbench-v2"
                else LONG_BENCH_TASKS[args.benchmark]["task"]
            ),
            "answers": answers,
            "max_tokens": max_tokens,
            "metadata": {
                key: row[key]
                for key in ("domain", "sub_domain", "difficulty", "length")
                if key in row
            },
        }
        # LongBench-v2 defines medium/long as beyond its <=32K-word short
        # stratum. Conservatively report those strata as unsupported on a 32K
        # token model rather than spending minutes tokenizing multi-megabyte
        # contexts or applying the official middle-truncation recipe.
        if args.benchmark == "longbench-v2" and row.get("length") != "short":
            record["reason"] = "official_length_stratum_exceeds_32k_model_scope"
            unsupported.append(record)
            continue
        prompt, _, _ = _format_case(args.benchmark, row)
        prompt_limit = args.max_model_len - max_tokens
        ids = _tokenize(tokenizer, prompt, prompt_limit)
        if len(ids) > prompt_limit:
            record["reason"] = "prompt_plus_output_exceeds_max_model_len"
            record["observed_prompt_tokens_lower_bound"] = len(ids)
            unsupported.append(record)
            continue
        record["prompt_token_ids"] = ids
        record["input_len"] = len(ids)
        record["prompt_sha256"] = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest()
        if len(ids) < args.min_input_tokens:
            record["reason"] = "below_declared_long_context_minimum"
            short.append(record)
            continue
        eligible.append(record)

    eligible.sort(key=lambda item: _stable_order(item, args.selection_seed))
    selected = eligible if args.limit == 0 else eligible[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
    )
    unsupported_path = args.output.with_suffix(
        args.output.suffix + ".unsupported.jsonl"
    )
    unsupported_path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":")) + "\n" for row in unsupported + short
        ),
        encoding="utf-8",
    )
    source_name = "longbench-v2" if args.benchmark == "longbench-v2" else "longbench"
    manifest = {
        "schema_version": 1,
        "benchmark": args.benchmark,
        "source": SOURCES[source_name],
        "source_file_sha256": sha256(source_path(args.root, source_name)),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "max_model_len": args.max_model_len,
        "min_input_tokens": args.min_input_tokens,
        "raw_cases": len(raw_rows),
        "eligible_cases": len(eligible),
        "unsupported_context_cases": len(unsupported),
        "below_minimum_cases": len(short),
        "selected_cases": len(selected),
        "selection_seed": args.selection_seed,
        "selection": "all eligible" if args.limit == 0 else "stable SHA-256 subset",
        "dataset_sha256": sha256(args.output),
        "unsupported_path": str(unsupported_path.resolve()),
        "no_prompt_truncation": True,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--source", choices=(*SOURCES, "all"), default="all")
    download.add_argument("--root", type=Path, required=True)
    download.add_argument("--timeout", type=float, default=600.0)
    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--benchmark", choices=("longbench-v2", *LONG_BENCH_TASKS), required=True
    )
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--tokenizer", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--max-model-len", type=int, default=32768)
    prepare.add_argument("--min-input-tokens", type=int, default=5120)
    prepare.add_argument("--limit", type=int, default=0)
    prepare.add_argument("--selection-seed", type=int, default=20260912)
    prepare.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.command == "download":
        download_sources(args)
    else:
        prepare_requests(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
