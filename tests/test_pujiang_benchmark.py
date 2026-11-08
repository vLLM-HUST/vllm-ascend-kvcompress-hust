"""Exercise selection, upstream delegation and process ownership on CPU fixtures."""

import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.benchmark.cli import adapter, main
from scripts.benchmark.common import clients, evaluation, lifecycle, performance
from scripts.benchmark.common.files import read, sha, write
from scripts.benchmark.common.reporting import compare
from scripts.benchmark.common.sampling import select

CLIENT = adapter("hle-verified").CLIENT


def tasks():
    return [
        {"case_id": f"{track}-{i}", "track": track}
        for track in ("gold", "revision")
        for i in range(120)
    ]


def test_sample_is_random_frozen_and_shared_between_arms(tmp_path):
    first = select(CLIENT, tasks(), tmp_path, size=13)
    assert first == select(CLIENT, tasks(), tmp_path, size=13)
    assert first["counts"] == {"gold": 13, "revision": 13}
    assert len(set(first["case_ids"])) == 26
    assert first["case_ids"] != [t["case_id"] for t in tasks()[:26]]
    original = read(tmp_path / "current.json")
    second = select(CLIENT, tasks(), tmp_path, size=13, seed=19, resample=True)
    assert second["case_ids"] != first["case_ids"]
    assert read(tmp_path / original["file"]) == first
    assert len(list(tmp_path.glob("selection-*.json"))) == 2


def test_sample_validates_source_size_seed_and_tampering(tmp_path):
    select(CLIENT, tasks(), tmp_path, size=13)
    for kwargs in ({"size": 12}, {"size": 13, "seed": 99}):
        with pytest.raises(ValueError):
            select(CLIENT, tasks(), tmp_path, **kwargs)
    with pytest.raises(ValueError):
        select(CLIENT, tasks()[:-1], tmp_path, size=13)
    pointer = read(tmp_path / "current.json")
    path = tmp_path / pointer["file"]
    altered = read(path)
    altered["case_ids"] = []
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="changed"):
        select(CLIENT, tasks(), tmp_path, size=13)


def test_small_track_has_no_repetition_and_duplicate_ids_rejected(tmp_path):
    few = tasks()[:3]
    result = select(CLIENT, few, tmp_path)
    assert len(result["case_ids"]) == 3
    with pytest.raises(ValueError, match="duplicate"):
        select(CLIENT, few + few, tmp_path / "other")


def test_placeholder_and_bad_arguments_fail_without_client_calls(tmp_path):
    with pytest.raises(NotImplementedError):
        adapter("terminal-bench")
    with pytest.raises(SystemExit) as error:
        main(["mmlu-pro", "run", "--resample"])
    assert error.value.code == 2


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def test_checkout_reuse_and_no_destructive_reset(tmp_path):
    client = clients.Client("fixture", "fixture-client", "", "fixture", "r", ("test",))
    path = client.source(tmp_path)
    path.mkdir()
    git(path, "init", "--quiet")
    git(path, "remote", "add", "origin", client.url)
    (path / "file").write_text("fixed")
    git(path, "add", "file")
    git(
        path,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    client = clients.Client(
        "fixture",
        "fixture-client",
        git(path, "rev-parse", "HEAD"),
        "fixture",
        "r",
        ("test",),
    )
    assert clients.download_client(client, tmp_path) == path
    (path / "file").write_text("user changes")
    with pytest.raises(ValueError, match="changes"):
        clients.download_client(client, tmp_path)
    assert (path / "file").read_text() == "user changes"


def options(tmp_path):
    return SimpleNamespace(
        plan=tmp_path / "plan",
        output=tmp_path / "run",
        arm="b1",
        smoke=True,
        track=None,
        data_root=tmp_path,
        selection_dir=tmp_path / "samples",
        sample_size=2,
        seed=17,
        resample=False,
        resume=False,
        runtime_contract=None,
        base_url="http://solver.invalid",
        served_model="model",
        api_key_env=None,
        concurrency=1,
        timeout=10,
    )


def test_chat_adapter_delegates_same_ids_to_upstream_solver_and_judge(
    tmp_path, monkeypatch
):
    args = options(tmp_path)
    args.plan.mkdir()
    write(args.plan / "contract.json", {"fixture": True})
    contract = {
        "solver": {
            "base_url": args.base_url,
            "model": "model",
            "concurrency": 1,
            "timeout": 10,
        }
    }
    calls = []

    def execute(a, stage):
        calls.append((stage, a.track, list(a.case_ids)))

    def summary(a):
        return {"tracks": {t: {"total": 2, "accuracy": 0.5} for t in CLIENT.tracks}}

    module = SimpleNamespace(execute=execute, summary=summary)
    monkeypatch.setattr(evaluation, "tasks_for", lambda *a: (contract, tasks()))
    monkeypatch.setattr(evaluation, "semantic_identity", lambda *a: {"fixed": True})
    monkeypatch.setattr(evaluation, "pipeline", lambda *a: module)
    reports = evaluation.evaluate(CLIENT, args)
    assert set(reports) == {"gold", "revision"}
    for index in (0, 2):
        assert calls[index][0] == "solver" and calls[index + 1][0] == "judge"
        assert calls[index][1:] == calls[index + 1][1:]
        assert len(calls[index][2]) == 2
    result = read(next(args.output.glob("summary-*.json")))
    assert result["smoke"] and result["status"] == "completed"


def test_upstream_failure_retained_with_nonzero_failure(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.plan.mkdir()
    write(args.plan / "contract.json", {})
    contract = {
        "solver": {
            "base_url": args.base_url,
            "model": "model",
            "concurrency": 1,
            "timeout": 10,
        }
    }
    monkeypatch.setattr(evaluation, "tasks_for", lambda *a: (contract, tasks()))
    monkeypatch.setattr(evaluation, "semantic_identity", lambda *a: {})

    def fail(*args):
        raise RuntimeError("transport fixture")

    monkeypatch.setattr(
        evaluation, "pipeline", lambda *a: SimpleNamespace(execute=fail)
    )
    with pytest.raises(RuntimeError, match="transport fixture"):
        evaluation.evaluate(CLIENT, args)
    assert read(next(args.output.glob("summary-*.json")))["status"] == "failed"


@pytest.mark.parametrize("startup_failure", [True, False])
def test_lifecycle_failures_stop_only_owned_server(
    tmp_path, monkeypatch, startup_failure
):
    server = SimpleNamespace(
        pid=123,
        returncode=7 if startup_failure else None,
        poll=lambda: 7 if startup_failure else None,
    )
    stopped = []
    monkeypatch.setattr(lifecycle.subprocess, "Popen", lambda *a, **kw: server)
    monkeypatch.setattr(lifecycle, "stop_server", stopped.append)
    monkeypatch.setattr(lifecycle, "fetch", lambda *a: "")
    monkeypatch.setattr(lifecycle, "sample", lambda *a: None)

    def failed_client(*a, **kw):
        raise subprocess.CalledProcessError(1, "client")

    monkeypatch.setattr(lifecycle.subprocess, "run", failed_client)
    root = tmp_path / "arm"
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        lifecycle.run_arm(["fixture"], [["client"]], root, "http://fixture", {}, 10)
    assert stopped == [server]
    assert read(root / "lifecycle.json")["status"] == "failed"


def test_comparison_rejects_wrong_samples_and_keeps_null_scores(tmp_path):
    common = {
        "dataset": "fixture",
        "identity": {},
        "selection": {"ids": [1]},
        "smoke": True,
        "client_commit": "fixed",
        "status": "completed",
        "selection_sha256": "fixture",
        "runtime": None,
        "reports": {"research": {"success_rate": None}},
    }
    write(tmp_path / "b0/summary-1.json", {**common, "arm": "b0"})
    write(tmp_path / "b1/summary-1.json", {**common, "arm": "b1"})
    report = compare(tmp_path / "b0", tmp_path / "b1", tmp_path / "comparison")
    assert report["quality"] == {}
    assert report["runtime_verification"]["status"] == "unverified"
    altered = {**common, "arm": "b1", "selection": {"ids": [2]}}
    (tmp_path / "b1/summary-1.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="selection"):
        compare(tmp_path / "b0", tmp_path / "b1", tmp_path / "other")


def test_bench_preflight_requires_image_capabilities(monkeypatch, tmp_path):
    binary = tmp_path / "vllm"
    binary.write_text("fixture")
    monkeypatch.setattr(
        performance.subprocess,
        "check_output",
        lambda *a, **kw: "--disable-shuffle --custom-output-len --save-result",
    )
    with pytest.raises(ValueError, match="custom_image"):
        performance.check_bench(binary, image=True)


def test_compression_not_observed_is_not_claimed_as_active(tmp_path):
    assert performance.compression_evidence(None)["status"] == "unknown"
    log = tmp_path / "server.log"
    log.write_text("no compression\n")
    assert performance.compression_evidence(log)["status"] == "not_observed"
    log.write_text(
        "KV compression scheduler commit request_id=x\n"
        "KV compression worker commit acknowledged request_id=x\n"
    )
    assert performance.compression_evidence(log)["scheduler_commits"] == 1


def load_env_manager():
    source = Path(__file__).parents[1] / "scripts/env/manage.py"
    spec = importlib.util.spec_from_file_location("env_manage", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arm_modes_share_host_and_clear_inherited_activation(tmp_path):
    manager = load_env_manager()
    config = tmp_path / "config.json"
    config.write_text("{}")
    inherited = {
        "VLLM_HUST_EXT_CONFIG": "stale-manager-config",
        "VLLM_ASCEND_KVCOMPRESS_CONFIG": str(config),
        "VLLM_PLUGINS": "unrelated,ascend_kvcompress",
        "VLLM_USE_V1": "stale",
        "VLLM_API_KEY": "secret",
        "ASCEND_RT_VISIBLE_DEVICES": "5,7",
    }
    assert manager.prefix_for("b0") == manager.prefix_for("b1")
    b0 = manager.arm_environment("b0", tmp_path, inherited)
    b1 = manager.arm_environment("b1", tmp_path, inherited)
    for env in (b0, b1):
        assert env["KVCOMPRESS_HOST_ENV"] == str(tmp_path)
        assert env["ASCEND_RT_VISIBLE_DEVICES"] == "5,7"
        assert env["VLLM_API_KEY"] == "secret"
        assert "VLLM_HUST_EXT_CONFIG" not in env
        assert "VLLM_USE_V1" not in env
    assert b0["VLLM_PLUGINS"] == "ascend"
    assert b0["VLLM_ASCEND_KVCOMPRESS_ENABLED"] == "0"
    assert "VLLM_ASCEND_KVCOMPRESS_CONFIG" not in b0
    assert b1["VLLM_PLUGINS"] == "ascend,ascend_kvcompress"
    assert b1["VLLM_ASCEND_KVCOMPRESS_ENABLED"] == "1"
    assert b1["VLLM_ASCEND_KVCOMPRESS_CONFIG"] == str(config)
    with pytest.raises(ValueError, match="requires"):
        manager.arm_environment("b1", tmp_path, {})


def test_hle_exports_preserve_images_and_never_include_reference_answers(
    tmp_path,
    monkeypatch,
):
    from types import ModuleType

    raw = b"original fixture image bytes"
    image = "data:image/png;base64," + base64.b64encode(raw).decode()
    info = {
        "mime": "image/png",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    upstream = ModuleType("hust_eval_hle_verified.bench_inputs")
    seen = []

    def image_info(url):
        seen.append(url)
        return info

    upstream.image_info = image_info
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    args = options(tmp_path)
    args.plan.mkdir()
    write(args.plan / "contract.json", {})
    rows = [
        {
            "case_id": "gold-text",
            "track": "gold",
            "problem": "Original text?",
            "image": "",
            "answer": "SECRET REFERENCE",
        },
        {
            "case_id": "gold-image",
            "track": "gold",
            "problem": "Original image?",
            "image": image,
            "answer": "SECRET REFERENCE",
        },
    ]
    groups = performance.export_groups(
        CLIENT,
        args,
        rows,
        {"case_ids": [r["case_id"] for r in rows]},
        tmp_path,
    )
    assert seen == [image]
    assert len(groups) == 2
    _, modality, path, count, manifest = groups[1]
    assert modality == "image" and count == 1
    assert json.loads(path.read_text())["content"][1]["image_url"]["url"] == image
    assert "SECRET REFERENCE" not in path.read_text()
    frozen = read(manifest)
    assert frozen["sha256"] == sha(path)
    assert frozen["cases"][0]["image"] == info


def runtime_receipt(arm):
    return {
        "base": {
            "python": "3.11",
            "prefix": "/fixture/host",
            "packages": {
                name: {"version": "fixed"}
                for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "triton")
            },
            "host_config_sha256": "host-hash",
            "bridge_sha256": "bridge-hash",
        },
        "plugin": {
            "enabled": arm == "b1",
            "plugins": "ascend,ascend_kvcompress" if arm == "b1" else "ascend",
            "config_sha256": "config-hash" if arm == "b1" else None,
        },
        "server": {"model_path": "model", "server_args": ["--enforce-eager"]},
    }


def test_runtime_comparison_rejects_changed_base_settings_and_activation():
    from scripts.benchmark.common.runtime import compare_runtime

    assert compare_runtime(None, None)["status"] == "unverified"
    assert compare_runtime({}, {})["status"] == "unverified"
    assert compare_runtime(runtime_receipt("b0"), runtime_receipt("b1")) == {
        "status": "verified"
    }
    for field, value, message in (
        ("base", {"python": "other"}, "bases differ"),
        ("server", {"server_args": []}, "settings differ"),
        ("plugin", runtime_receipt("b0")["plugin"], "activation"),
    ):
        b1 = runtime_receipt("b1")
        b1[field] = value
        with pytest.raises(ValueError, match=message):
            compare_runtime(runtime_receipt("b0"), b1)


def test_standalone_bench_comparison_rejects_different_clients(tmp_path):
    common = {
        "dataset": "fixture",
        "smoke": True,
        "status": "completed",
        "selection": {"case_ids": [1]},
        "bench_client": {"version": "fixed"},
        "output_budget": 256,
        "identity": {},
        "tokenizer_files": {},
        "compression": {"status": "not_observed"},
        "groups": {
            "text": {
                "input_sha256": "same",
                "selected": 1,
                "upstream_summary": {"output_throughput": 10},
            }
        },
    }
    write(tmp_path / "b0/performance.json", {**common, "arm": "b0"})
    write(tmp_path / "b1/performance.json", {**common, "arm": "b1"})
    report = compare(tmp_path / "b0", tmp_path / "b1", tmp_path / "compared")
    metric = report["performance"]["text.upstream_summary.output_throughput"]
    assert metric == {"b0": 10, "b1": 10, "difference": 0, "percent": 0}
    path = tmp_path / "b1/performance.json"
    changed = read(path)
    changed["bench_client"] = {"version": "other"}
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="bench_client"):
        compare(tmp_path / "b0", tmp_path / "b1", tmp_path / "rejected")


def test_managed_pair_refuses_existing_external_server(tmp_path, monkeypatch):
    from scripts.benchmark.common.files import ROOT

    model = tmp_path / "model"
    model.mkdir()
    config = tmp_path / "pair.json"
    write(
        config,
        {
            "model_path": str(model),
            "arms": {
                arm: {
                    "server_args": [],
                    "plan": "plan",
                    "base_url": "http://fixture",
                    "served_model": "model",
                    "plugin_config": "config.json",
                }
                for arm in ("b0", "b1")
            },
        },
    )
    args = options(tmp_path)
    args.config, args.client_root, args.client_python = config, tmp_path, None
    args.with_bench, args.bench_vllm_bin, args.pairs = False, None, 1
    args.startup_timeout = 10
    args.output = tmp_path / "pairs"
    inspected = {"selection": {"case_ids": ["one"]}, "identity": {}}

    def run(command, **kwargs):
        if "inspect" in command:
            write(Path(command[command.index("--output") + 1]), inspected)
        else:
            assert command[:2] == [str(ROOT / "scripts/set_env.sh"), "check"]

    monkeypatch.setattr(lifecycle.subprocess, "run", run)
    monkeypatch.setattr(lifecycle, "fetch", lambda *a: "healthy")
    monkeypatch.setattr(
        lifecycle, "run_arm", lambda *a: pytest.fail("launched a server")
    )
    with pytest.raises(ValueError, match="already occupies"):
        lifecycle.pair(adapter("mmlu-pro").CLIENT, args)


def test_mmlu_resume_archives_outputs_before_upstream_overwrites(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.plan.mkdir()
    write(args.plan / "contract.json", {})
    client = adapter("mmlu-pro").CLIENT
    monkeypatch.setattr(
        evaluation,
        "tasks_for",
        lambda *a: (
            {},
            [
                {"case_id": "mmlu-1"},
                {"case_id": "mmlu-2"},
            ],
        ),
    )
    monkeypatch.setattr(evaluation, "semantic_identity", lambda *a: {})
    monkeypatch.setattr(evaluation, "mmlu_subset", lambda *a: args.plan)
    calls = []

    def run(plan, output, **kwargs):
        calls.append(kwargs)
        directory = Path(output)
        (directory / "requests").mkdir(parents=True, exist_ok=True)
        (directory / "summary.json").write_text(str(len(calls)))
        return {"transport_failures": 0}

    modules = {
        client.module + ".mmlu_pro.run": SimpleNamespace(run=run),
        client.module + ".mmlu_pro.summary": SimpleNamespace(
            build_report=lambda *a: {"accuracy": 0.5},
        ),
    }
    original = evaluation.importlib.import_module
    monkeypatch.setattr(
        evaluation.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else original(name),
    )
    evaluation.evaluate(client, args)
    args.resume = True
    evaluation.evaluate(client, args)
    archive = next(args.output.glob("resume-archive-*/summary.json"))
    assert archive.read_text() == "1"
    assert (args.output / "solver/summary.json").read_text() == "2"
    assert len(list(args.output.glob("summary-*.json"))) == 2


def test_judge_client_rejects_cli_overrides_before_sending(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.concurrency = 16
    contract = {
        "solver": {
            "base_url": args.base_url,
            "model": "model",
            "concurrency": 1,
            "timeout": 10,
        }
    }
    monkeypatch.setattr(evaluation, "tasks_for", lambda *a: (contract, tasks()))
    monkeypatch.setattr(evaluation, "pipeline", lambda *a: pytest.fail("sent request"))
    with pytest.raises(ValueError, match="concurrency differs"):
        evaluation.evaluate(CLIENT, args)
    assert not args.output.exists()


def test_interrupt_never_marks_quality_run_completed(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.plan.mkdir()
    write(args.plan / "contract.json", {})
    contract = {
        "solver": {
            "base_url": args.base_url,
            "model": "model",
            "concurrency": 1,
            "timeout": 10,
        }
    }
    monkeypatch.setattr(evaluation, "tasks_for", lambda *a: (contract, tasks()))
    monkeypatch.setattr(evaluation, "semantic_identity", lambda *a: {})

    def interrupt(*a):
        raise KeyboardInterrupt("fixture")

    monkeypatch.setattr(
        evaluation, "pipeline", lambda *a: SimpleNamespace(execute=interrupt)
    )
    with pytest.raises(KeyboardInterrupt):
        evaluation.evaluate(CLIENT, args)
    result = read(next(args.output.glob("summary-*.json")))
    assert result["status"] == "failed" and "KeyboardInterrupt" in result["error"]


@pytest.mark.parametrize("change", [None, "prefix", "args", "base", "mode", "deferred"])
def test_pair_checks_same_base_and_uses_identical_server_entrypoints(
    tmp_path, monkeypatch, change
):
    from scripts.benchmark.common import reporting

    config = {
        "host_prefix": "host",
        "model_path": "model",
        "server_env": {"ASCEND_RT_VISIBLE_DEVICES": "0,1"},
        "arms": {
            name: {
                "plan": "plan",
                "base_url": "http://fixture",
                "served_model": "model",
                "server_args": ["--enforce-eager"],
                **({"plugin_config": "plugin.json"} if name == "b1" else {}),
            }
            for name in ("b0", "b1")
        },
    }
    if change == "prefix":
        config["arms"]["b1"]["prefix"] = "other-host"
    if change == "args":
        config["arms"]["b1"]["server_args"] = []
    args = options(tmp_path)
    args.config = tmp_path / "pair.json"
    write(args.config, config)
    args.output = tmp_path / "output"
    args.client_root, args.client_python = tmp_path, None
    args.with_bench, args.bench_vllm_bin, args.pairs = True, None, 1
    args.startup_timeout, args.output_len = 10, 256
    launches, compared = [], []
    deferred = change == "deferred"
    if deferred:
        args.stage = "solver"
        args.plan.mkdir()
        write(
            args.plan / "contract.json",
            {"solver": {"base_url": "http://fixture", "model": "model"}},
        )

    def inspect(command, **kwargs):
        if "inspect" in command:
            write(
                Path(command[command.index("--output") + 1]),
                {"identity": {"judge": {"base_url": "http://offline-judge"}}}
                if deferred
                else {"identity": {}},
            )
        else:
            assert "check" in command
            assert command[-1] == str(tmp_path / "host")

    def probe(command, **kwargs):
        name = command[2]
        receipt = runtime_receipt(name)
        receipt.pop("server")
        if name == "b1" and change == "base":
            receipt["base"]["bridge_sha256"] = "changed"
        if name == "b1" and change == "mode":
            receipt["plugin"] = runtime_receipt("b0")["plugin"]
        return json.dumps(receipt)

    def bench_check(*a, **kw):
        assert kw["host_prefix"] == str(tmp_path / "host")

    def launch(*a):
        launches.append(a)
        assert a[4]["KVCOMPRESS_HOST_ENV"] == str(tmp_path / "host")
        assert a[-1]["server"] == a[-2]

    monkeypatch.setattr(lifecycle.subprocess, "run", inspect)
    monkeypatch.setattr(lifecycle.subprocess, "check_output", probe)
    monkeypatch.setattr(lifecycle, "fetch", lambda *a: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(lifecycle, "run_arm", launch)
    monkeypatch.setattr(performance, "check_bench", bench_check)
    monkeypatch.setattr(reporting, "compare", lambda *a: compared.append(a))
    if change and not deferred:
        with pytest.raises(ValueError):
            lifecycle.pair(adapter("mmlu-pro").CLIENT, args)
        assert not launches
    else:
        lifecycle.pair(CLIENT if deferred else adapter("mmlu-pro").CLIENT, args)
        assert len(launches) == 2 and len(compared) == (0 if deferred else 1)
        if deferred:
            assert (
                read(args.output / "pair-1/pending-judge.json")["status"]
                == "awaiting_judge"
            )
        commands = [a[0][a[0].index("--") + 1 :] for a in launches]
        assert commands[0] == commands[1]
        assert "benchmark_host.py" in commands[0][1]
        assert (
            read(args.output / "runtime-preflight.json")["b0"]["plugin"]["enabled"]
            is False
        )


def test_runtime_change_after_preflight_stops_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lifecycle.subprocess,
        "check_output",
        lambda *a, **kw: json.dumps({"changed": True}),
    )
    monkeypatch.setattr(
        lifecycle.subprocess, "Popen", lambda *a, **kw: pytest.fail("started a server")
    )
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="changed after pair preflight"):
        lifecycle.run_arm(
            ["server"],
            [],
            root,
            "http://fixture",
            {},
            1,
            runtime_command=["probe"],
            expected_runtime={"original": True},
        )
    assert read(root / "lifecycle.json")["status"] == "failed"


@pytest.mark.parametrize("tamper", [False, True])
def test_deferred_judge_binds_solver_files_and_keeps_generation_evidence(
    tmp_path, monkeypatch, tamper
):
    args = options(tmp_path)
    args.stage = "solver"
    args.plan.mkdir()
    write(args.plan / "contract.json", {})
    contract = {
        "solver": {
            "base_url": args.base_url,
            "model": "model",
            "concurrency": 1,
            "timeout": 10,
        }
    }
    calls = []

    def execute(a, stage):
        calls.append(stage)
        if stage == "solver":
            write(Path(a.output) / "requests/case.json", {"answer": a.track})

    def summary(a):
        return {"tracks": {t: {"total": 2, "accuracy": 0.5} for t in CLIENT.tracks}}

    monkeypatch.setattr(evaluation, "tasks_for", lambda *a: (contract, tasks()))
    monkeypatch.setattr(evaluation, "semantic_identity", lambda *a: {"fixed": True})
    monkeypatch.setattr(
        evaluation,
        "pipeline",
        lambda *a: SimpleNamespace(execute=execute, summary=summary),
    )
    evaluation.evaluate(CLIENT, args)
    assert calls == ["solver", "solver"]
    receipt = read(args.output / "solver-receipt.json")
    assert len(receipt["files"]) == 2
    from scripts.benchmark.common.reporting import latest

    assert latest(args.output)["status"] == "awaiting_judge"
    args.stage, args.resume = "judge", True
    if tamper:
        (args.output / "gold/solver/requests/case.json").write_text(
            json.dumps({"answer": "changed"})
        )
        with pytest.raises(ValueError, match="solver output changed"):
            evaluation.evaluate(CLIENT, args)
        assert calls == ["solver", "solver"]
    else:
        evaluation.evaluate(CLIENT, args)
        assert calls == ["solver", "solver", "judge", "judge"]
        result = latest(args.output)
        assert result["status"] == "completed"
        assert result["compression"] == receipt["compression"]
        assert set(result["reports"]) == {"gold", "revision"}


def test_judge_stage_requires_existing_run():
    with pytest.raises(SystemExit) as error:
        main(["hle-verified", "run", "--stage", "judge"])
    assert error.value.code == 2


def test_bench_capabilities_use_full_help_and_stable_version(tmp_path, monkeypatch):
    binary = tmp_path / "client"
    binary.write_text("fixed client")
    required = (
        "--disable-shuffle --custom-output-len --save-result custom_image "
        "--skip-chat-template --no-oversample --extra-body --save-detailed "
        "--num-warmups"
    )
    noise = iter(["startup warning at 10:00", "startup warning at 10:01"])
    help_variants = iter(
        [
            "warning one\nusage: client\n"
            + required
            + " bench-1234abcd- {throughput_1k,qualitative}",
            "warning two at 10:01\nusage: client\n"
            + required
            + " bench-5678efab- {qualitative,throughput_1k}",
        ]
    )

    def output(command, **kwargs):
        if command[-1] == "--version":
            return next(noise) + "\n0.29.1\n"
        if command[-1] == "--help=all":
            return next(help_variants)
        return "Config groups; use --help=all for full options"

    monkeypatch.setattr(performance.subprocess, "check_output", output)
    help_path = tmp_path / "help.txt"
    first = performance.check_bench(binary, image=True, help_path=help_path)
    assert help_path.read_text().startswith("warning one\nusage:")
    assert first["version"] == "0.29.1"
    assert first == performance.check_bench(binary, image=True)


def test_bench_help_retains_real_default_changes():
    assert performance.canonical_help(
        "--request-rate inf"
    ) != performance.canonical_help("--request-rate 1")
