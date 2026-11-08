"""Thin adapters around upstream plan, solver, Judge and summary APIs."""

import gzip
import importlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from .clients import contract_identity
from .files import digest, read, sha, write
from .sampling import select


def pipeline(client):
    return importlib.import_module(client.module + ".pipeline")


def tasks_for(client, plan):
    if client.dataset == "mmlu-pro":
        mod = importlib.import_module(client.module + ".mmlu_pro.plan")
    else:
        mod = pipeline(client)
    return mod.load_plan(plan)


def download(client, args):
    if client.dataset == "mmlu-pro":
        from .mmlu_data import download as fetch

        fetch(args.data_root)
    else:
        module = pipeline(client)
        assets = args.data_root / "assets"
        if assets.exists():
            # Upstream prepare/load performs checks too; validate reuse now.
            tracks = args.track or list(client.tracks)
            hashes = (
                module.DATA
                if client.dataset == "hle-verified"
                else {t: {"sha256": h} for t, h in module.DATA_HASHES.items()}
            )
            for track in tracks:
                if sha(assets / track / "test.jsonl") != hashes[track]["sha256"]:
                    raise ValueError(f"existing dataset checksum mismatch: {track}")
        else:
            module.fetch(SimpleNamespace(output=str(assets), subset=args.track))


def prepare(client, args):
    if not args.model_path:
        raise ValueError("prepare requires --model-path (tokenizer/model identity)")
    if args.plan.exists():
        raise FileExistsError("plan exists; reuse it or choose a new --plan")
    if client.dataset == "mmlu-pro":
        mod = importlib.import_module(client.module + ".mmlu_pro.plan")
        mod.prepare(
            args.data_root / "assets/mmlu-pro/data",
            args.model_path,
            args.data_root / "reference/MMLU-Pro",
            args.plan,
            runtime_contract=args.runtime_contract,
        )
    else:
        if not args.solver_config or not args.judge_config:
            raise ValueError("prepare requires --solver-config and --judge-config")
        pipeline(client).prepare(
            SimpleNamespace(
                assets=str(args.data_root / "assets"),
                output=str(args.plan),
                solver_config=str(args.solver_config),
                judge_config=str(args.judge_config),
                subset=args.track,
                modality=args.modality,
                solver_runtime_contract=args.runtime_contract,
                judge_runtime_contract=args.judge_runtime_contract,
            )
        )
    model = Path(args.model_path).resolve()
    write(
        args.plan / "adapter.json",
        {
            "dataset": client.dataset,
            "client_commit": client.commit,
            "model_path": str(model),
            "model_files": {
                name: sha(model / name)
                for name in (
                    "config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                )
                if (model / name).is_file()
            },
            "weights_verified": False,
        },
    )


def selection(client, args):
    contract, tasks = tasks_for(client, args.plan)
    if args.track:
        tasks = [t for t in tasks if t.get("track", "test") in args.track]
    if not tasks:
        raise ValueError("no tasks in selected tracks")
    if args.smoke:
        manifest = select(
            client,
            tasks,
            args.selection_dir,
            args.sample_size,
            args.seed,
            args.resample,
        )
    else:
        manifest = {
            "schema": "kvcompress-full/v1",
            "dataset": client.dataset,
            "revision": client.revision,
            "case_ids": [t["case_id"] for t in tasks],
        }
    return contract, tasks, manifest


def mmlu_subset(client, parent, manifest, directory):
    """Use upstream plan writer, retaining all inventory and parent provenance."""
    mod = importlib.import_module(client.module + ".mmlu_pro.plan")
    contract, cases = mod.load_plan(parent)
    wanted = set(manifest["case_ids"])
    selected = [c for c in cases if c["case_id"] in wanted]
    if len(selected) != len(wanted):
        raise ValueError("selection contains unknown MMLU IDs")
    target = Path(directory) / digest(
        {"parent": sha(Path(parent) / "contract.json"), "selection": manifest}
    )
    if target.exists():
        mod.load_plan(target)
        return target
    with gzip.open(Path(parent) / "inventory.jsonl.gz", "rt") as stream:
        inventory = [json.loads(line) for line in stream if line.strip()]
    extra = {
        k: v
        for k, v in contract.items()
        if k not in {"files_sha256", "tasks", "selection"}
    }
    extra.update(
        {
            "parent_contract_sha256": sha(Path(parent) / "contract.json"),
            "selection": "explicit IDs from adapter selection manifest",
            "adapter_selection": manifest,
        }
    )
    mod.write_plan(target, inventory, selected, extra=extra)
    return target


def semantic_identity(client, plan):
    contract, tasks = tasks_for(client, plan)
    identity = contract_identity(plan)
    if client.dataset != "mmlu-pro":
        solver = dict(contract["solver"])
        # Different served aliases/endpoints are allowed; request/scoring is fixed.
        solver.pop("base_url", None)
        solver.pop("model", None)
        identity["solver"] = solver
        identity["judge"] = contract["judge"]
    identity["tasks"] = digest(tasks)
    adapter = Path(plan) / "adapter.json"
    identity["model"] = (
        read(adapter)
        if adapter.exists()
        else {
            "model_files": contract.get("model_files_sha256"),
            "weights_verified": False,
        }
    )
    return identity


def evaluate(client, args):
    stage = getattr(args, "stage", "all")
    contract, tasks, manifest = selection(client, args)
    if client.dataset != "mmlu-pro":
        if contract["solver"]["base_url"].rstrip("/") != args.base_url:
            raise ValueError(
                "base-url differs from frozen solver config; prepare an arm plan"
            )
        if args.served_model and contract["solver"]["model"] != args.served_model:
            raise ValueError("served-model differs from frozen solver config")
        for key in ("concurrency", "timeout"):
            requested = getattr(args, key)
            actual = contract["solver"].get(key, 1 if key == "concurrency" else 1800)
            if requested is not None and requested != actual:
                raise ValueError(
                    f"{key} differs from frozen solver config; prepare again"
                )
    root = args.output
    metadata = {
        "dataset": client.dataset,
        "arm": args.arm,
        "smoke": args.smoke,
        "selection": manifest,
        "selection_sha256": digest(manifest),
        "client_commit": client.commit,
        "identity": semantic_identity(client, args.plan),
        "plan": str(args.plan.resolve()),
        "contract_sha256": sha(args.plan / "contract.json"),
        "runtime": read(args.runtime_contract) if args.runtime_contract else None,
    }
    if args.resume:
        if read(root / "evaluation.json") != metadata:
            raise ValueError(
                "resume requires identical plan, selection, arm and runtime"
            )
    else:
        root.mkdir(parents=True, exist_ok=False)
        write(root / "evaluation.json", metadata)
    receipt_path = root / "solver-receipt.json"
    if stage == "judge" or (stage == "solver" and receipt_path.exists()):
        receipt = read(receipt_path)
        if receipt["evaluation_sha256"] != digest(metadata):
            raise ValueError("solver receipt belongs to another evaluation")
        if not receipt["files"] or receipt["files"] != solver_files(root):
            raise ValueError("solver output changed after generation")
    failure = None
    reports = {}
    log = getattr(args, "server_log", None)
    offset = log.stat().st_size if log else 0
    try:
        if client.dataset == "mmlu-pro":
            plan = args.plan
            if args.smoke or args.track:
                plan = mmlu_subset(
                    client, plan, manifest, args.data_root / "subset-plans"
                )
            runner = importlib.import_module(client.module + ".mmlu_pro.run")
            summary = importlib.import_module(client.module + ".mmlu_pro.summary")
            try:
                if args.resume and (root / "solver").exists():
                    shutil.copytree(
                        root / "solver", root / f"resume-archive-{uuid4().hex}"
                    )
                result = runner.run(
                    str(plan),
                    str(root / "solver"),
                    base_url=args.base_url,
                    served_model=args.served_model,
                    concurrency=args.concurrency or 1,
                    timeout=args.timeout or 1800,
                    resume=args.resume,
                    api_key_env=args.api_key_env,
                )
                if result.get("transport_failures"):
                    failure = "MMLU transport failures retained in denominator"
            finally:
                if (root / "solver/requests").exists():
                    reports["test"] = summary.build_report(
                        str(plan), str(root / "solver")
                    )
        else:
            module = pipeline(client)
            selected = set(manifest["case_ids"])
            # Run each track separately, preserving upstream denominators/policies.
            for track in sorted({t["track"] for t in tasks}):
                ids = [
                    t["case_id"]
                    for t in tasks
                    if t["track"] == track and t["case_id"] in selected
                ]
                if not ids:
                    continue
                base = root / track
                solver, judge = base / "solver", base / "judge"
                stages = [("solver", solver), ("judge", judge)]
                for part, output in stages:
                    if stage != "all" and part != stage:
                        continue
                    module.execute(
                        SimpleNamespace(
                            plan=str(args.plan),
                            output=str(output),
                            responses=str(solver),
                            case_ids=ids,
                            track=track,
                            limit=None,
                            resume=args.resume,
                        ),
                        part,
                    )
                if stage == "solver":
                    continue
                report_dir = base / f"report-{uuid4().hex}"
                reports[track] = module.summary(
                    SimpleNamespace(
                        plan=str(args.plan),
                        responses=str(solver),
                        grades=str(judge),
                        output=str(report_dir),
                    )
                )
                stat = reports[track]["tracks"][track]
                if stat.get("solver_failures", 0) or stat.get(
                    "judge_transport_failures", 0
                ):
                    failure = f"{track}: transport failures retained in report"
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        from .performance import compression_evidence

        compression = (
            receipt["compression"]
            if stage == "judge" or (stage == "solver" and receipt_path.exists())
            else compression_evidence(log, offset)
        )
        if stage == "solver" and not failure and not receipt_path.exists():
            write(
                receipt_path,
                {
                    "evaluation_sha256": digest(metadata),
                    "files": solver_files(root),
                    "compression": compression,
                },
            )
        write(
            root / f"summary-{uuid4().hex}.json",
            {
                **metadata,
                "finished_unix_ns": time.time_ns(),
                "reports": reports,
                "status": (
                    "failed"
                    if failure
                    else "awaiting_judge"
                    if stage == "solver"
                    else "completed"
                ),
                "error": failure,
                "compression": compression,
            },
        )
    if failure:
        raise RuntimeError(failure)
    return reports


def solver_files(root):
    """Bind every upstream solver file before the service is released."""
    return {
        str(path.relative_to(root)): sha(path)
        for path in sorted(root.glob("*/solver/**/*"))
        if path.is_file()
    }
