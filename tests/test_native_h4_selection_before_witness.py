"""Only the witness preselection branch; no Torch/device/product simulation."""

import importlib.util
import os
from pathlib import Path

import pytest


@pytest.fixture
def witness():
    path = Path(
        os.environ.get(
            "H4_SELECTOR_KV_WITNESS_SOURCE",
            str(
                Path(__file__).resolve().parents[1]
                / "src/vllm_ascend_kvcompress/native_tensor_witness.py"
            ),
        )
    )
    spec = importlib.util.spec_from_file_location(
        "h4_kv_witness_source_candidate", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unselected_request_stops_before_witness_sync_or_export(witness, monkeypatch):
    calls = []

    def sink(*args):
        raise AssertionError("unselected witness submitted")

    sink.native_request_selected = lambda rid: calls.append(rid) or False
    witness.register_candidate_sink(sink)
    monkeypatch.setattr(
        witness, "_sync", lambda t: pytest.fail("unselected device sync")
    )
    assert (
        witness.capture_before_actual_gather(
            object(),
            object(),
            {"runtime_request_id": "real-opaque"},
            object(),
            object(),
        )
        is None
    )
    assert calls == ["real-opaque"]


@pytest.mark.parametrize("selected_callback", [True, None])
def test_selected_or_default_path_keeps_original_witness_entry(
    witness, monkeypatch, selected_callback
):
    def sink(*args):
        return None

    if selected_callback is not None:
        sink.native_request_selected = lambda rid: True
    witness.register_candidate_sink(sink)

    class OriginalWitnessEntered(Exception):
        pass

    def sync(tensor):
        raise OriginalWitnessEntered

    monkeypatch.setattr(witness, "_sync", sync)
    with pytest.raises(OriginalWitnessEntered):
        witness.capture_before_actual_gather(
            object(),
            object(),
            {"runtime_request_id": "real-opaque"},
            object(),
            object(),
        )
