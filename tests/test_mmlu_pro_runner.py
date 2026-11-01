# SPDX-License-Identifier: Apache-2.0
"""Check orchestration without a model service or NPU."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/kvcompress_mmlu_pro.py"
spec = importlib.util.spec_from_file_location("mmlu_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def arguments(monkeypatch, output, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "run",
            "--plan",
            "plan",
            "--base-url",
            "http://localhost:8000",
            "--served-model",
            "model",
            "--output",
            str(output),
            *extra,
        ],
    )


def test_transport_failure_is_summarized_and_propagated(monkeypatch, tmp_path):
    calls = []
    failure = subprocess.CalledProcessError(1, "client")

    def call(*args):
        calls.append(args)
        if args[1] == "run":
            raise failure

    monkeypatch.setattr(runner, "call", call)
    arguments(monkeypatch, tmp_path, "--limit", "1", "--api-key-env", "MODEL_KEY")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        runner.main()
    assert caught.value is failure
    assert [args[1] for args in calls] == ["run", "summary"]
    assert calls[0][-4:] == ("--limit", "1", "--api-key-env", "MODEL_KEY")
    assert "--out" in calls[1]


def test_resume_preserves_previous_summary(monkeypatch, tmp_path):
    original = tmp_path / "summary.json"
    original.write_text("previous")
    calls = []

    def call(*args):
        calls.append(args)
        if args[1] == "run":
            original.write_text("new client summary")

    monkeypatch.setattr(runner, "call", call)
    arguments(monkeypatch, tmp_path, "--resume")
    runner.main()
    assert "--resume" in calls[0]
    assert Path(calls[1][-1]) != original
    assert Path(calls[1][-1]).parent == tmp_path
    assert original.read_text() == "new client summary"
    backups = list(tmp_path.glob("summary.previous.*.json"))
    assert len(backups) == 1
    assert backups[0].read_text() == "previous"


def test_rejects_v1_url_before_sending(monkeypatch, tmp_path):
    arguments(monkeypatch, tmp_path)
    sys.argv[sys.argv.index("--base-url") + 1] += "/v1"
    monkeypatch.setattr(runner, "call", lambda *args: pytest.fail("sent a request"))
    with pytest.raises(SystemExit) as caught:
        runner.main()
    assert caught.value.code == 2
