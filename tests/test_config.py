# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend_kvcompress.config import (
    CONFIG_ENV,
    ENABLE_ENV,
    EXTENSION_ID,
    LEGACY_PROVIDER_NAME,
    MANAGER_ENABLED_ENV,
    PROVIDER_NAME,
    ProviderSelection,
    extension_enabled,
    load_runtime_selection,
)
from vllm_ascend_kvcompress.methods.triattention import TriAttentionConfig


def _core_config(provider: str, **options: object) -> SimpleNamespace:
    return SimpleNamespace(
        schema_version=1,
        provider=provider,
        provider_config=options,
    )


def test_provider_selection_routes_canonical_method() -> None:
    selection = ProviderSelection.from_core_config(
        _core_config(
            PROVIDER_NAME,
            method="triattention",
            stats_path="/tmp/stats.pt",
        )
    )
    assert selection.provider_name == PROVIDER_NAME
    assert selection.method == "triattention"
    assert selection.method_config == {"stats_path": "/tmp/stats.pt"}


def test_provider_selection_preserves_legacy_triattention_config() -> None:
    selection = ProviderSelection.from_core_config(
        _core_config(LEGACY_PROVIDER_NAME, stats_path="/tmp/stats.pt")
    )
    assert selection.provider_name == LEGACY_PROVIDER_NAME
    assert selection.method == "triattention"


def test_legacy_provider_rejects_other_methods() -> None:
    with pytest.raises(ValueError, match="legacy provider"):
        ProviderSelection.from_core_config(
            _core_config(LEGACY_PROVIDER_NAME, method="future_method")
        )


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
        ("score_chunk_size", 129),
        ("score_layer_stride", 0),
        ("min_output_tokens_for_compression", -1),
        ("protected_prefix_window", -1),
        ("protected_recent_window", -1),
        ("position_segments", 0),
        ("position_policy", "unknown"),
        ("score_aggregation", "median"),
        ("auto_calibrate", 1),
        ("calibration_max_length", 127),
        ("calibration_attn_implementation", "unknown"),
    ],
)
def test_triattention_config_rejects_invalid_values(option: str, value: object) -> None:
    method_config = {"stats_path": "/tmp/stats.pt", option: value}
    with pytest.raises(ValueError):
        TriAttentionConfig.from_method_config(method_config)  # type: ignore[arg-type]


def test_triattention_config_rejects_unknown_option() -> None:
    with pytest.raises(ValueError, match="unknown"):
        TriAttentionConfig.from_method_config(
            {"stats_path": "/tmp/stats.pt", "typo": 1}
        )


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
