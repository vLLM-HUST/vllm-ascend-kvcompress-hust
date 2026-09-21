# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import torch

from vllm_ascend_kvcompress.methods.base import ModelShape
from vllm_ascend_kvcompress.methods.triattention.stats import (
    CalibrationStats,
    LayerCalibrationStats,
)
from vllm_ascend_kvcompress.model import model_shape_from_config


def _write_flat_stats(path: Path, *, layers: int = 2, heads: int = 4) -> None:
    stats = {}
    for layer in range(layers):
        for head in range(heads):
            stats[f"layer{layer:02d}_head{head:02d}"] = {
                "q_mean_real": torch.full((4,), float(layer + head)),
                "q_mean_imag": torch.zeros(4),
                "q_abs_mean": torch.ones(4),
            }
    torch.save(
        {
            "metadata": {
                "head_dim": 8,
                "rope_style": "half",
                "rope_type": "default",
            },
            "stats": stats,
        },
        path,
    )


def test_flat_stats_load_and_gqa_grouping(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.pt"
    _write_flat_stats(stats_path)
    calibration = CalibrationStats.load(stats_path)
    model = ModelShape(
        model_type="qwen3",
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        has_rope_scaling=False,
    )
    assert calibration.validate(model) == ()
    device_stats = calibration.to_device(model, torch.device("cpu"))
    assert device_stats[0].q_mean_real.shape == (2, 2, 4)
    assert device_stats[1].omega.shape == (4,)


def test_incomplete_heads_fail_closed(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.pt"
    _write_flat_stats(stats_path, layers=1, heads=2)
    payload = torch.load(stats_path, weights_only=True)
    del payload["stats"]["layer00_head00"]
    torch.save(payload, stats_path)
    try:
        CalibrationStats.load(stats_path)
    except ValueError as error:
        assert "complete and contiguous" in str(error)
    else:
        raise AssertionError("incomplete calibration heads were accepted")


def test_scaled_rope_requires_exact_scaling_stats(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.pt"
    _write_flat_stats(stats_path, layers=1, heads=2)
    calibration = CalibrationStats.load(stats_path)
    model = ModelShape(
        model_type="qwen3",
        num_layers=1,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=8,
        rope_theta=10000.0,
        has_rope_scaling=True,
    )
    reasons = calibration.validate(model)
    assert any("inv_freq" in reason for reason in reasons)
    assert any("freq_scale_sq" in reason for reason in reasons)


def test_scaled_rope_requires_inv_freq_for_every_layer() -> None:
    layer = LayerCalibrationStats(
        q_mean_real=torch.ones(2, 4),
        q_mean_imag=torch.zeros(2, 4),
        q_abs_mean=torch.ones(2, 4),
        freq_scale_sq=torch.ones(2, 4),
    )
    calibration = CalibrationStats(
        metadata={
            "head_dim": 8,
            "rope_style": "half",
            "layer_inv_freqs": {0: torch.ones(4)},
        },
        layers={0: layer, 1: layer},
    )
    model = ModelShape(
        model_type="qwen3",
        num_layers=2,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=8,
        rope_theta=10000.0,
        has_rope_scaling=True,
    )

    reasons = calibration.validate(model)
    assert any("layer 1" in reason and "inv_freq" in reason for reason in reasons)


def test_invalid_calibration_values_fail_closed() -> None:
    calibration = CalibrationStats(
        metadata={"head_dim": 8, "rope_style": "half"},
        layers={
            0: LayerCalibrationStats(
                q_mean_real=torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]),
                q_mean_imag=torch.zeros(1, 4),
                q_abs_mean=torch.tensor([[-1.0, 1.0, 1.0, 1.0]]),
                freq_scale_sq=torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
            )
        },
    )
    model = ModelShape(
        model_type="qwen3",
        num_layers=1,
        num_attention_heads=1,
        num_kv_heads=1,
        head_dim=8,
        rope_theta=10000.0,
        has_rope_scaling=False,
    )

    reasons = calibration.validate(model)
    assert any("non-finite" in reason for reason in reasons)
    assert any("negative" in reason for reason in reasons)
    assert any("freq_scale_sq must be positive" in reason for reason in reasons)


def test_structured_kv_head_stats_expand_across_gqa_queries() -> None:
    calibration = CalibrationStats(
        metadata={"head_dim": 8, "rope_style": "half"},
        layers={
            0: LayerCalibrationStats(
                q_mean_real=torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
                q_mean_imag=torch.zeros(2, 4),
                q_abs_mean=torch.ones(2, 4),
                freq_scale_sq=torch.full((2, 4), 4.0),
            )
        },
    )
    model = ModelShape(
        model_type="qwen3",
        num_layers=1,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        has_rope_scaling=False,
    )

    assert calibration.validate(model) == ()
    grouped = calibration.to_device(model, torch.device("cpu"))[0]
    assert grouped.q_mean_real.shape == (2, 2, 4)
    torch.testing.assert_close(grouped.q_mean_real[0, 0], grouped.q_mean_real[0, 1])
    torch.testing.assert_close(grouped.q_mean_real[1, 0], grouped.q_mean_real[1, 1])
    torch.testing.assert_close(grouped.freq_scale_sq, torch.full((2, 2, 4), 4.0))


def test_model_shape_reads_transformers_v5_rope_parameters() -> None:
    text_config = SimpleNamespace(
        model_type="qwen2",
        num_hidden_layers=48,
        num_attention_heads=40,
        num_key_value_heads=8,
        hidden_size=5120,
        rope_scaling={
            "rope_theta": 1_000_000.0,
            "rope_type": "default",
        },
        rope_parameters={
            "rope_theta": 1_000_000.0,
            "rope_type": "default",
        },
    )
    model = model_shape_from_config(SimpleNamespace(hf_text_config=text_config))

    assert model.rope_theta == 1_000_000.0
    assert model.has_rope_scaling is False


def test_model_shape_detects_transformers_v5_scaled_rope() -> None:
    text_config = SimpleNamespace(
        model_type="qwen2",
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=32,
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "yarn"},
    )
    model = model_shape_from_config(SimpleNamespace(hf_text_config=text_config))

    assert model.has_rope_scaling is True


def test_model_shape_extracts_qwen35_hybrid_attention_and_partial_rope() -> None:
    text_config = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        hidden_size=2048,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
        * 2,
        rope_parameters={
            "rope_theta": 10_000_000.0,
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
    )

    model = model_shape_from_config(SimpleNamespace(hf_text_config=text_config))

    assert model.model_type == "qwen3_5_moe_text"
    assert model.effective_rotary_dim == 64
    assert model.full_attention_layer_indices == (3, 7)
    assert model.is_hybrid


def test_qwen35_partial_rope_calibration_uses_only_attention_layers() -> None:
    layer = LayerCalibrationStats(
        q_mean_real=torch.ones(16, 32),
        q_mean_imag=torch.zeros(16, 32),
        q_abs_mean=torch.ones(16, 32),
        freq_scale_sq=torch.ones(1, 32),
        inv_freq=torch.ones(32),
        q_pass_mean=torch.ones(16, 192),
    )
    calibration = CalibrationStats(
        metadata={
            "model_type": "qwen3_5_moe_text",
            "num_layers": 8,
            "head_dim": 256,
            "rotary_dim": 64,
            "attention_layer_indices": [3, 7],
            "rope_style": "half",
        },
        layers={3: layer, 7: layer},
    )
    model = ModelShape(
        model_type="qwen3_5_moe_text",
        num_layers=8,
        num_attention_heads=16,
        num_kv_heads=2,
        head_dim=256,
        rope_theta=10_000_000.0,
        has_rope_scaling=False,
        rotary_dim=64,
        attention_layer_indices=(3, 7),
    )

    assert calibration.validate(model) == ()
    device = calibration.to_device(model, torch.device("cpu"))
    assert set(device) == {3, 7}
    assert device[3].q_mean_real.shape == (2, 8, 32)
    assert device[3].q_pass_mean is not None
    assert device[3].q_pass_mean.shape == (2, 8, 192)


def test_qwen35_calibration_shards_kv_heads_for_tensor_parallelism() -> None:
    layer = LayerCalibrationStats(
        q_mean_real=torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32),
        q_mean_imag=torch.zeros(16, 32),
        q_abs_mean=torch.ones(16, 32),
        q_pass_mean=torch.ones(16, 192),
    )
    calibration = CalibrationStats(
        metadata={"rope_style": "half"},
        layers={3: layer},
    )
    model = ModelShape(
        model_type="qwen3_5_moe_text",
        num_layers=4,
        num_attention_heads=16,
        num_kv_heads=2,
        head_dim=256,
        rope_theta=10_000_000.0,
        has_rope_scaling=False,
        rotary_dim=64,
        attention_layer_indices=(3,),
    )

    rank_zero = calibration.to_device(
        model, torch.device("cpu"), tensor_parallel_rank=0, tensor_parallel_size=2
    )[3]
    rank_one = calibration.to_device(
        model, torch.device("cpu"), tensor_parallel_rank=1, tensor_parallel_size=2
    )[3]

    assert rank_zero.q_mean_real.shape == (1, 8, 32)
    assert rank_one.q_mean_real.shape == (1, 8, 32)
    assert not torch.equal(rank_zero.q_mean_real, rank_one.q_mean_real)
