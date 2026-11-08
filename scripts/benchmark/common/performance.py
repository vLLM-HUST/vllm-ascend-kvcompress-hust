"""Invoke upstream bench APIs; never reimplement streaming timing or scoring."""

import hashlib
import importlib
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

from .clients import clean_client_env
from .evaluation import mmlu_subset, pipeline, selection, semantic_identity
from .files import ROOT, digest, read, sha, write


def canonical_help(text):
    """Keep option/default changes while removing generated IDs and set order."""
    # Import warnings are process startup diagnostics, not CLI capabilities.
    match = re.search(r"(?m)^usage: ", text)
    if match:
        text = text[match.start() :]
    text = re.sub(r"\bbench-[0-9a-f]{8}-", "bench-<generated>-", text)
    return re.sub(
        r"\{([A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)+)\}",
        lambda match: "{" + ",".join(sorted(match[1].split(","))) + "}",
        text,
    )


def check_bench(binary, image=False, host_prefix=None, help_path=None):
    env = clean_client_env()
    if host_prefix is not None:
        env["KVCOMPRESS_HOST_ENV"] = str(host_prefix)
    help_text = subprocess.check_output(
        [str(binary), "bench", "serve", "--help=all"],
        text=True,
        env=env,
        stderr=subprocess.STDOUT,
    )
    if help_path is not None:
        Path(help_path).write_text(help_text, encoding="utf-8")
    required = ["--disable-shuffle", "--custom-output-len", "--save-result"]
    if image:
        required += [
            "custom_image",
            "--skip-chat-template",
            "--no-oversample",
            "--extra-body",
            "--save-detailed",
            "--num-warmups",
        ]
    missing = [flag for flag in required if flag not in help_text]
    if missing:
        raise ValueError(f"bench client lacks {missing}; use a compatible fixed client")
    version = (
        subprocess.check_output(
            [str(binary), "--version"],
            text=True,
            env=env,
            stderr=subprocess.STDOUT,
        )
        .strip()
        .splitlines()[-1]
    )
    return {
        "executable": str(Path(binary).absolute()),
        "version": version,
        "executable_sha256": sha(binary),
        "help_sha256": digest(canonical_help(help_text)),
        "help_normalization": "usage-only-generated-request-prefix-and-choice-order/v2",
    }


def compression_evidence(path, offset=0):
    if not path:
        return {"status": "unknown", "scheduler_commits": None, "worker_acks": None}
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        text = stream.read().decode(errors="replace")
    commits = text.count("KV compression scheduler commit request_id=")
    return {
        "status": "observed" if commits else "not_observed",
        "scheduler_commits": commits,
        "worker_acks": text.count("KV compression worker commit acknowledged"),
        "server_log": str(Path(path).resolve()),
        "start_offset": offset,
    }


def _jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_groups(client, args, tasks, manifest, root):
    """Use upstream exporters when possible; otherwise only select and serialize."""
    wanted = set(manifest["case_ids"])
    groups = []
    if client.dataset == "mmlu-pro":
        module = importlib.import_module(client.module + ".mmlu_pro.materialize")
        plan = args.plan
        if args.smoke or args.track:
            plan = mmlu_subset(client, plan, manifest, args.data_root / "subset-plans")
        path = root / "mmlu-pro.jsonl"
        count = module.materialize(
            str(args.data_root / "assets/mmlu-pro/data"),
            str(args.data_root / "reference/MMLU-Pro"),
            str(args.model_path),
            str(plan),
            str(path),
            sample=None if args.smoke else min(200, len(wanted)),
        )
        groups.append(("test", "text", path, count, None))
    elif client.dataset == "frontier-science":
        path = root / "frontier-science.jsonl"
        pipeline(client).materialize(
            type(
                "Options",
                (),
                {
                    "plan": str(args.plan),
                    "output": str(path),
                    "sample": None,
                },
            )()
        )
        provenance = read(str(path) + ".manifest.json")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        selected = [
            (cid, row)
            for cid, row in zip(provenance["case_ids"], rows, strict=True)
            if cid in wanted
        ]
        frozen = root / "frontier-science-selected.jsonl"
        _jsonl(frozen, [row for _, row in selected])
        write(
            str(frozen) + ".manifest.json",
            {
                "case_ids": [cid for cid, _ in selected],
                "parent": provenance,
                "sha256": sha(frozen),
                "selection": manifest,
            },
        )
        groups.append(("all-tracks", "text", frozen, len(selected), None))
    else:
        from hust_eval_hle_verified.bench_inputs import image_info

        for track in sorted({t["track"] for t in tasks}):
            for modality in ("text", "image"):
                eligible = [
                    t
                    for t in tasks
                    if t["case_id"] in wanted
                    and t["track"] == track
                    and bool(t["image"]) == (modality == "image")
                ]
                if not args.smoke and modality == "text" and len(eligible) > 200:
                    eligible = [eligible[i * len(eligible) // 200] for i in range(200)]
                if not eligible:
                    groups.append((track, modality, None, 0, None))
                    continue
                path = root / f"{track}-{modality}.jsonl"
                rows, cases = [], []
                for task in eligible:
                    case = {
                        "case_id": task["case_id"],
                        "text_sha256": hashlib.sha256(
                            task["problem"].encode()
                        ).hexdigest(),
                    }
                    if modality == "image":
                        case["image"] = image_info(task["image"])
                        row = {
                            "content": [
                                {"type": "text", "text": task["problem"]},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": task["image"]},
                                },
                            ]
                        }
                    else:
                        row = {"prompt": task["problem"]}
                    rows.append(row)
                    cases.append(case)
                _jsonl(path, rows)
                manifest_path = Path(str(path) + ".manifest.json")
                write(
                    manifest_path,
                    {
                        "kind": "smoke_or_bench_export_from_verified_current_plan",
                        "modality": modality,
                        "cases": cases,
                        "sha256": sha(path),
                        "source_contract_sha256": sha(args.plan / "contract.json"),
                        "selection": manifest,
                        "note": (
                            "Current verified plan; not an import from sealed history."
                        ),
                    },
                )
                groups.append((track, modality, path, len(rows), manifest_path))
    return groups


def verify_image_tokens(path, result, args):
    """Compare detailed input_lens with server tokenization, without guessing."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    observed = result.get("input_lens", [])
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    checks = []
    for index, row in enumerate(rows):
        payload = {
            "model": args.served_model,
            "messages": [{"role": "user", "content": row["content"]}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if args.api_key_env:
            headers["Authorization"] = "Bearer " + os.environ[args.api_key_env]
        try:
            request = urllib.request.Request(
                args.base_url.rstrip("/") + "/tokenize",
                json.dumps(payload).encode(),
                headers,
            )
            with opener.open(request, timeout=args.timeout or 1800) as response:
                tokenized = json.load(response)
            count = tokenized.get("count")
            actual = observed[index] if index < len(observed) else None
            checks.append(
                {
                    "index": index,
                    "server_count": count,
                    "bench_count": actual,
                    "verified": type(count) is int and count == actual,
                }
            )
        except (OSError, ValueError) as error:
            checks.append({"index": index, "verified": False, "error": str(error)})
    return {
        "verified": len(observed) == len(rows) and all(c["verified"] for c in checks),
        "checks": checks,
    }


def bench(client, args):
    runtime = read(args.runtime_contract) if args.runtime_contract else None
    _, tasks, manifest = selection(client, args)
    if not args.model_path or not args.served_model:
        raise ValueError("bench requires --model-path and --served-model")
    binary = args.bench_vllm_bin or ROOT / "scripts/env/bench_vllm.sh"
    args.output.mkdir(parents=True, exist_ok=False)
    identity = check_bench(
        binary,
        image=client.dataset == "hle-verified",
        help_path=args.output / "bench-client-help.txt",
    )
    if args.api_key_env:
        if not os.environ.get(args.api_key_env):
            raise ValueError("API key environment variable is unset")
        os.environ["OPENAI_API_KEY"] = os.environ[args.api_key_env]
    inputs = args.output / "exports"
    inputs.mkdir()
    groups = export_groups(client, args, tasks, manifest, inputs)
    module = importlib.import_module(
        client.module
        + (".mmlu_pro.bench" if client.dataset == "mmlu-pro" else ".bench")
    )
    offset = args.server_log.stat().st_size if args.server_log else 0
    results = {}
    status = "failed"
    try:
        for track, modality, path, count, input_manifest in groups:
            name = f"{track}-{modality}"
            if not count:
                results[name] = {"selected": 0, "status": "no_coverage"}
                continue
            kwargs = {}
            if client.dataset == "hle-verified":
                kwargs = {"modality": modality, "input_manifest": str(input_manifest)}
            output = args.output / name
            upstream = module.bench(
                str(output),
                name,
                str(path),
                args.base_url,
                args.served_model,
                str(args.model_path),
                num_prompts=count,
                output_len=args.output_len,
                vllm_bin=str(binary),
                dry_run=args.dry_run,
                **kwargs,
            )
            result = {
                "selected": count,
                "input_sha256": sha(path),
                "upstream_summary": upstream,
            }
            results[name] = result
            if not args.dry_run:
                raw = read(output / "outputs" / f"{name}.json")
                if raw.get("completed") != count or raw.get("failed", 0):
                    raise RuntimeError(
                        f"{name}: missing/failed bench requests retained"
                    )
                if modality == "image":
                    check = verify_image_tokens(path, raw, args)
                    write(output / "token-verification.json", check)
                    result["input_tokens_verified"] = check["verified"]
                    if not check["verified"]:
                        # Keep the upstream raw JSON, but do not compare an estimate
                        # as if it were a verified text-plus-vision throughput.
                        upstream.pop("total_token_throughput", None)
                    result["trusted_total_token_throughput"] = (
                        raw.get("total_token_throughput") if check["verified"] else None
                    )
        status = "dry_run" if args.dry_run else "completed"
    finally:
        write(
            args.output / "performance.json",
            {
                "dataset": client.dataset,
                "arm": args.arm,
                "runtime": runtime,
                "status": status,
                "smoke": args.smoke,
                "selection": manifest,
                "bench_client": identity,
                "identity": semantic_identity(client, args.plan),
                "tokenizer_files": {
                    name: sha(args.model_path / name)
                    for name in (
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "chat_template.jinja",
                        "config.json",
                    )
                    if (args.model_path / name).is_file()
                },
                "output_budget": args.output_len,
                "groups": results,
                "compression": compression_evidence(args.server_log, offset),
                "protocol_note": (
                    "Upstream independent bench, not accuracy replay; "
                    "TTFT includes queueing."
                ),
            },
        )
