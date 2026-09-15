# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend_kvcompress.calibration as calibration
from vllm_ascend_kvcompress.calibration import (
    CalibrationRequest,
    generate_calibration_artifact,
)
from vllm_ascend_kvcompress.methods.base import ModelShape
from vllm_ascend_kvcompress.methods.triattention.stats import CalibrationStats


class _FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)


class _FakeLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer(), _FakeLayer()])

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        hidden = torch.nn.functional.one_hot(input_ids % 8, num_classes=8).float()
        for layer in self.layers:
            hidden = layer.self_attn.q_proj(hidden)
        return (hidden,)


class _FakeConfig:
    model_type = "qwen2"
    num_hidden_layers = 2
    num_attention_heads = 2
    num_key_value_heads = 1
    hidden_size = 8
    head_dim = 4
    rope_theta = 10000.0
    rope_scaling = None
    rope_parameters = None

    def to_dict(self):
        return {
            "model_type": self.model_type,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
        }


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        del text, kwargs
        return {
            "input_ids": torch.arange(32).reshape(1, 32),
            "attention_mask": torch.ones((1, 32), dtype=torch.int64),
        }


def test_generator_creates_safe_model_matched_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _FakeConfig(),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _FakeTokenizer(),
    )
    loads = []

    def load_model(*args, **kwargs):
        loads.append((args, kwargs))
        return _FakeModel()

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)
    output = tmp_path / "stats.pt"
    request = CalibrationRequest(
        model="fake/model",
        output=output,
        device="cpu",
        dtype="float32",
        max_length=128,
    )

    assert generate_calibration_artifact(request)
    assert not generate_calibration_artifact(request)
    assert len(loads) == 1
    assert not list(tmp_path.glob("*.tmp"))

    artifact = CalibrationStats.load(output)
    shape = ModelShape("qwen2", 2, 2, 1, 4, 10000.0, False)
    assert artifact.validate(shape) == ()
    assert artifact.metadata["generator"] == "vllm-ascend-kvcompress-hust"
    assert artifact.metadata["token_count"] == 32
    assert artifact.metadata["input_source"].startswith("builtin:")
    assert artifact.layers[0].q_mean_real.shape == (2, 2)
    assert artifact.layers[0].inv_freq is not None


def test_generator_rejects_empty_explicit_input(tmp_path: Path) -> None:
    input_path = tmp_path / "empty.txt"
    input_path.write_text("", encoding="utf-8")
    request = CalibrationRequest(
        model="unused", output=tmp_path / "stats.pt", input_path=input_path
    )

    with pytest.raises(ValueError, match="empty"):
        calibration._load_calibration_text(request)


def test_runner_uses_model_settings_for_missing_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "stats.pt"
    selection = SimpleNamespace(
        method="triattention",
        method_config={
            "stats_path": str(output),
            "calibration_max_length": 1024,
            "calibration_local_files_only": True,
        },
    )
    hf_config = _FakeConfig()
    model_config = SimpleNamespace(
        model="fake/model",
        tokenizer=None,
        revision="model-revision",
        tokenizer_revision=None,
        trust_remote_code=True,
        dtype=torch.float16,
        hf_text_config=hf_config,
    )
    runner = SimpleNamespace(model_config=model_config, device=torch.device("npu:0"))
    captured = []

    def generate(request):
        captured.append(request)
        layer_stats = {}
        for layer_idx in range(2):
            layer_stats[str(layer_idx)] = {
                "q_mean_real": torch.zeros(2, 2),
                "q_mean_imag": torch.zeros(2, 2),
                "q_abs_mean": torch.ones(2, 2),
                "inv_freq": torch.ones(2),
                "freq_scale_sq": torch.ones(1, 2),
            }
        torch.save(
            {
                "metadata": {
                    "model_type": "qwen2",
                    "num_layers": 2,
                    "num_attention_heads": 2,
                    "num_kv_heads": 1,
                    "head_dim": 4,
                    "rope_theta": 10000.0,
                },
                "layer_stats": layer_stats,
            },
            output,
        )
        return True

    monkeypatch.setattr(calibration, "generate_calibration_artifact", generate)

    assert calibration.ensure_calibration_for_runner(runner, selection)
    assert captured[0].model == "fake/model"
    assert captured[0].revision == "model-revision"
    assert captured[0].device == "npu:0"
    assert captured[0].dtype == "float16"
    assert captured[0].max_length == 1024
    assert captured[0].local_files_only


def test_runner_does_not_generate_when_auto_calibration_is_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    selection = SimpleNamespace(
        method="triattention",
        method_config={
            "stats_path": str(tmp_path / "missing.pt"),
            "auto_calibrate": False,
        },
    )
    runner = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=_FakeConfig()))
    monkeypatch.setattr(
        calibration,
        "generate_calibration_artifact",
        lambda request: pytest.fail("generation should not run"),
    )

    assert not calibration.ensure_calibration_for_runner(runner, selection)


def test_plugin_generated_artifact_rejects_a_different_model_source(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stats.pt"
    layer_stats = {
        str(layer_idx): {
            "q_mean_real": torch.zeros(2, 2),
            "q_mean_imag": torch.zeros(2, 2),
            "q_abs_mean": torch.ones(2, 2),
            "inv_freq": torch.ones(2),
            "freq_scale_sq": torch.ones(1, 2),
        }
        for layer_idx in range(2)
    }
    torch.save(
        {
            "metadata": {
                "generator": "vllm-ascend-kvcompress-hust",
                "model": "model/a",
                "model_revision": "default",
                "model_source_fingerprint": calibration._model_source_fingerprint(
                    "model/a", None
                ),
                "model_type": "qwen2",
                "num_layers": 2,
                "num_attention_heads": 2,
                "num_kv_heads": 1,
                "head_dim": 4,
                "rope_theta": 10000.0,
            },
            "layer_stats": layer_stats,
        },
        output,
    )
    shape = ModelShape("qwen2", 2, 2, 1, 4, 10000.0, False)
    model_config = SimpleNamespace(model="model/b", revision=None)

    with pytest.raises(ValueError, match="model_source_fingerprint"):
        calibration._validate_artifact(output, shape, model_config)


def test_concurrent_generation_publishes_one_atomic_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "stats.pt"
    request = CalibrationRequest(model="fake/model", output=output, max_length=128)
    calls = []
    payload = {
        "metadata": {},
        "layer_stats": {
            "0": {
                "q_mean_real": torch.zeros(1, 2),
                "q_mean_imag": torch.zeros(1, 2),
                "q_abs_mean": torch.ones(1, 2),
            }
        },
    }

    def generate(_request):
        calls.append(1)
        time.sleep(0.1)
        return payload

    monkeypatch.setattr(calibration, "_generate_payload", generate)
    monkeypatch.setattr(
        calibration, "_validate_existing_for_request", lambda path, request: None
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(generate_calibration_artifact, (request, request)))

    assert sorted(results) == [False, True]
    assert len(calls) == 1
    assert output.is_file()
    assert not list(tmp_path.glob("*.tmp"))
