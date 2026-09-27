# SPDX-License-Identifier: Apache-2.0

import json

from scripts import kvcompress_frontier_preflight as preflight


def test_frontier_preflight_checks_pins_and_local_inputs(tmp_path, monkeypatch) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("torch==2.13.0\ntorch-npu==2.13.0rc1\n")
    cann = tmp_path / "cann"
    info = cann / "share/info/asc-devkit/version.info"
    info.parent.mkdir(parents=True)
    info.write_text("Version=9.1.0\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}}
        )
    )
    (model / "shard-1.safetensors").touch()
    (model / "shard-2.safetensors").touch()
    (model / "tokenizer.json").touch()
    swe = tmp_path / "swe.json"
    swe.write_text("frozen fixture")
    monkeypatch.setattr(preflight, "PREPARED_SHA256", preflight.sha256(swe))
    receipt = tmp_path / "prepared.json"
    receipt.write_text(
        json.dumps(
            {
                "revision": preflight.AGENTX_REVISION,
                "fingerprint": preflight.AGENTX_FINGERPRINT,
                "sessions": 393,
            }
        )
    )
    monkeypatch.setattr(
        preflight,
        "installed_version",
        lambda package: {"torch": "2.13.0+cpu", "torch-npu": "2.13.0rc1"}[package],
    )

    kwargs = {
        "requirements": requirements,
        "cann_root": cann,
        "model": model,
        "swe_prepared": swe,
        "agentx_receipt": receipt,
    }
    assert preflight.inspect(**kwargs)["ready"]

    info.write_text("Version=9.0.1\n")
    result = preflight.inspect(**kwargs)
    assert not result["ready"]
    assert (
        next(check for check in result["checks"] if check["name"] == "CANN")["ok"]
        is False
    )


def test_frontier_preflight_detects_missing_model_shard(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "shard-1.safetensors"}})
    )
    assert preflight.model_shards(model) == (1, 0)
