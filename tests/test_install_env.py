# SPDX-License-Identifier: Apache-2.0
"""Validate seed migration and fail-before-write behavior."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/env/install_b1.py"
spec = importlib.util.spec_from_file_location("install_env", SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_copy_seed_rewrites_scripts_and_excludes_stale_plugin(tmp_path):
    seed = tmp_path / "seed"
    prefix = tmp_path / "target"
    source = seed / "lib/python3.11/site-packages"
    source.mkdir(parents=True)
    (source / "host.pth").write_text("/durable/host/src")
    (source / "__editable__.vllm_ascend_kvcompress_hust-old.pth").write_text("old")
    (seed / "bin").mkdir()
    (prefix / "bin").mkdir(parents=True)
    script = seed / "bin/tool"
    script.write_text(f"#!{seed}/bin/python\nprint('ok')\n")
    script.chmod(0o755)
    binary = b"\x7fELF\0\xff\x01"
    (seed / "bin/binary-tool").write_bytes(binary)
    installer.copy_seed(seed, prefix)
    assert (prefix / "bin/tool").read_text().startswith(f"#!{prefix}/bin/python\n")
    assert (prefix / "bin/tool").stat().st_mode & 0o111
    assert (prefix / "bin/binary-tool").read_bytes() == binary
    assert not list((prefix / "lib/python3.11/site-packages").glob("*kvcompress*"))
    assert script.read_text().startswith(f"#!{seed}/bin/python\n")


def test_mismatched_seed_fails_without_creating_target(monkeypatch, tmp_path):
    seed = tmp_path / "seed"
    target = tmp_path / "target"
    cann = tmp_path / "cann.sh"
    atb = tmp_path / "atb.sh"
    cann.touch()
    atb.touch()
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(installer.platform, "libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setattr(sys, "version_info", (3, 11))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--seed-env",
            str(seed),
            "--prefix",
            str(target),
            "--cann-env",
            str(cann),
            "--atb-env",
            str(atb),
        ],
    )
    base = {
        name: {"version": version} for name, version in installer.LOCK["base"].items()
    }
    monkeypatch.setattr(installer, "source_paths", lambda _: {})
    monkeypatch.setattr(installer, "toolchain_identity", lambda *_: {})

    def inventory(python):
        packages = dict(base)
        if python == seed / "bin/python":
            packages["torch"] = {"version": "0.0.0"}
        return {"packages": packages}

    monkeypatch.setattr(installer, "packages", inventory)
    with pytest.raises(ValueError, match="torch: expected"):
        installer.main()
    assert not target.exists()


def test_unsupported_libc_fails_before_any_environment_is_created(
    monkeypatch, tmp_path, capsys
):
    target = tmp_path / "host"
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(installer.platform, "libc_ver", lambda: ("glibc", "2.31"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--prefix", str(target)])
    with pytest.raises(SystemExit):
        installer.main()
    assert "requires glibc >= 2.34" in capsys.readouterr().err
    assert not target.exists()


def test_vendor_version_is_read_from_metadata_not_directory_name(tmp_path):
    cann = tmp_path / "cann-9.1.0"
    atb = tmp_path / "atb"
    (cann / "share/info/asc-devkit").mkdir(parents=True)
    (atb / "latest").mkdir(parents=True)
    info = cann / "share/info/asc-devkit/version.info"
    info.write_text("Version=9.1.0\n")
    (atb / "latest/version.info").write_text("Ascend-cann-atb : 9.1.0\n")
    identity = installer.toolchain_identity(cann / "set_env.sh", atb / "set_env.sh")
    assert identity["cann"]["version"] == "9.1.0"
    info.write_text("Version=9.0.0\n")
    with pytest.raises(ValueError, match="cann must be"):
        installer.toolchain_identity(cann / "set_env.sh", atb / "set_env.sh")


def test_overlay_cannot_silently_change_python_base(monkeypatch, tmp_path):
    rows = {
        name: {"version": version}
        for name, version in {
            **installer.LOCK["base"],
            **installer.LOCK["overlay"],
            installer.LOCK["fla"]["distribution"]: installer.LOCK["fla"]["version"],
        }.items()
    }
    monkeypatch.setattr(
        installer,
        "packages",
        lambda _: {
            "packages": rows,
            "base_python": str(tmp_path / "unexpected/bin/python"),
        },
    )
    with pytest.raises(ValueError, match="different Python base"):
        installer.check_overlay(tmp_path / "host", tmp_path / "expected/bin/python", {})
