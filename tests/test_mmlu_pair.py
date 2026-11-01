# SPDX-License-Identifier: Apache-2.0
"""Lifecycle failures must preserve evidence and release the model server."""

import json
from types import SimpleNamespace

import pytest

from scripts import kvcompress_mmlu_pair as pair


def options(tmp_path):
    return SimpleNamespace(
        output=tmp_path,
        launcher=tmp_path / "launch",
        plan=tmp_path / "plan",
        concurrency=16,
        base_url="http://localhost:8000",
        startup_timeout=10,
        client_python=tmp_path / "python",
        served_model="model",
    )


def test_startup_failure_preserves_lifecycle_and_stops_server(tmp_path, monkeypatch):
    server = SimpleNamespace(pid=123, returncode=7, poll=lambda: 7)
    stopped = []
    monkeypatch.setattr(pair.subprocess, "Popen", lambda *a, **kw: server)
    monkeypatch.setattr(pair, "stop_server", stopped.append)
    with pytest.raises(RuntimeError, match="startup"):
        pair.run_arm(options(tmp_path), 1, "b0")
    result = json.loads((tmp_path / "pair-1/b0/lifecycle.json").read_text())
    assert result["status"] == "failed"
    assert result["server_exit_code"] == 7
    assert stopped == [server]


def test_warmup_failure_stops_server_without_starting_full_run(tmp_path, monkeypatch):
    underlying = tmp_path / "base-python"
    underlying.touch()
    (tmp_path / "python").symlink_to(underlying)
    server = SimpleNamespace(pid=123, returncode=None, poll=lambda: None)
    stopped, commands = [], []
    monkeypatch.setattr(pair.subprocess, "Popen", lambda *a, **kw: server)
    monkeypatch.setattr(pair, "stop_server", stopped.append)
    monkeypatch.setattr(pair, "fetch", lambda url: "")

    def fail(command, **kwargs):
        commands.append(command)
        raise pair.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(pair.subprocess, "run", fail)
    with pytest.raises(pair.subprocess.CalledProcessError):
        pair.run_arm(options(tmp_path), 1, "b1")
    assert len(commands) == 1 and "--limit" in commands[0]
    assert commands[0][0] == str(tmp_path / "python")
    assert stopped == [server]
    assert not (tmp_path / "pair-1/b1/full").exists()
    result = json.loads((tmp_path / "pair-1/b1/lifecycle.json").read_text())
    assert result["status"] == "failed"
