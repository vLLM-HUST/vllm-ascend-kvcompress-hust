# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from vllm_ascend_kvcompress.core.config import (
    CONFIG_ENV,
    ENABLE_ENV,
    EXTENSION_ID,
    MANAGER_ENABLED_ENV,
    PROVIDER_NAME,
    PluginConfig,
    extension_enabled,
    load_runtime_selection,
)
from vllm_ascend_kvcompress.methods.triattention import TriAttentionConfig


def test_provider_selection_accepts_provider_name():
    plugin_config = PluginConfig.from_mapping(
        {
            "provider": PROVIDER_NAME,
            "method_config": {"stats_path": "/tmp/stats.pt"},
        }
    )
    assert plugin_config.provider_name == PROVIDER_NAME
    assert plugin_config.method == "triattention"
    assert plugin_config.method_config == {"stats_path": "/tmp/stats.pt"}


def test_provider_selection_rejects_unsupported_provider():
    with pytest.raises(ValueError, match="provider .* is unsupported"):
        PluginConfig.from_mapping({"provider": "unknown"})


def test_provider_selection_ignores_unknown_fields() -> None:
    plugin_config = PluginConfig.from_mapping(
        {"schema_version": 99, "extra": {"nested": [1, 2]}}
    )
    assert plugin_config == PluginConfig.from_mapping({})


def test_triattention_config_accepts_documented_values() -> None:
    config = TriAttentionConfig.from_method_config(
        {
            "stats_path": "/tmp/stats.pt",
            "kv_budget": 2048,
            "recompute_window": 128,
            "position_policy": "v3",
            "protected_prefix_window": 128,
            "protected_recent_window": 64,
            "position_segments": 8,
            "score_aggregation": "max",
            "layer_aggregation": "mean",
            "score_chunk_size": 512,
            "score_layer_stride": 8,
            "min_output_tokens_for_compression": 64,
            "auto_calibrate": True,
            "calibration_input_path": "/tmp/calibration.txt",
            "calibration_max_length": 8192,
            "calibration_device": "npu:0",
            "calibration_attn_implementation": "eager",
            "calibration_local_files_only": True,
        }
    )
    assert config.stats_path == Path("/tmp/stats.pt")
    assert config.compression_threshold_tokens == 2176
    assert config.position_policy == "v3"
    assert config.protected_prefix_window == 128
    assert config.position_segments == 8
    assert config.min_output_tokens_for_compression == 64
    assert config.auto_calibrate
    assert config.calibration_input_path == Path("/tmp/calibration.txt")
    assert config.calibration_max_length == 8192


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("kv_budget", 2000),
        ("recompute_window", 0),
        ("score_layer_stride", 0),
        ("min_output_tokens_for_compression", -1),
        ("protected_prefix_window", -1),
        ("protected_prefix_window", 4096),
        ("position_segments", 0),
        ("position_policy", "unknown"),
        ("auto_calibrate", 1),
        ("calibration_max_length", 127),
    ],
)
def test_triattention_config_rejects_invalid_values(option: str, value: object) -> None:
    method_config = {"stats_path": "/tmp/stats.pt", option: value}
    with pytest.raises(ValueError):
        TriAttentionConfig.from_method_config(method_config)


def test_triattention_config_ignores_unknown_options() -> None:
    # 嵌套的额外字段也应通过通用配置层，不因旧版标量限制被提前拒绝。
    options = {"stats_path": "/tmp/stats.pt", "kv_budget": 4096}
    plugin_config = PluginConfig.from_mapping(
        {"method_config": {**options, "extra": {"nested": [1, 2]}}}
    )
    config = TriAttentionConfig.from_method_config(plugin_config.method_config)
    assert config == TriAttentionConfig.from_method_config(options)


def test_extension_activation_is_explicit() -> None:
    assert not extension_enabled({})
    assert extension_enabled({ENABLE_ENV: "1"})
    assert extension_enabled({MANAGER_ENABLED_ENV: f"other,{EXTENSION_ID}"})


def test_direct_runtime_config_accepts_json_or_path(tmp_path: Path) -> None:
    payload = '{"method_config":{"stats_path":"/tmp/stats.pt"}}'
    direct = load_runtime_selection({CONFIG_ENV: payload})
    config_path = tmp_path / "config.json"
    config_path.write_text(payload, encoding="utf-8")
    from_path = load_runtime_selection({CONFIG_ENV: str(config_path)})

    assert direct == from_path
    assert direct.method_config["stats_path"] == "/tmp/stats.pt"
