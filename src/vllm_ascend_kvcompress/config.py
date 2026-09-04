# SPDX-License-Identifier: Apache-2.0
"""Provider-level configuration and method selection."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

PROVIDER_NAME = "ascend_kvcompress"
LEGACY_PROVIDER_NAME = "triattention_ascend"
SUPPORTED_PROVIDER_NAMES = frozenset({PROVIDER_NAME, LEGACY_PROVIDER_NAME})
DEFAULT_METHOD = "triattention"
SCHEMA_VERSION = 1
ASCEND_BLOCK_SIZE = 128
EXTENSION_ID = "org.vllm-hust.ascend-kvcompress"
ENABLE_ENV = "VLLM_ASCEND_KVCOMPRESS_ENABLED"
CONFIG_ENV = "VLLM_ASCEND_KVCOMPRESS_CONFIG"
MANAGER_ENABLED_ENV = "VLLMHUST_EXT_ENABLED_BUNDLES"

JsonScalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True)
class ProviderSelection:
    """Validated routing information for one compression method."""

    provider_name: str
    method: str
    method_config: dict[str, JsonScalar]

    @classmethod
    def from_core_config(cls, core_config: Any) -> ProviderSelection:
        if core_config is None:
            raise ValueError("KV cache compression configuration is disabled")
        if core_config.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {core_config.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        provider_name = str(core_config.provider)
        if provider_name not in SUPPORTED_PROVIDER_NAMES:
            supported = ", ".join(sorted(SUPPORTED_PROVIDER_NAMES))
            raise ValueError(
                f"provider {provider_name!r} is unsupported; expected one of: "
                f"{supported}"
            )

        options = _copy_scalar_mapping(core_config.provider_config)
        raw_method = options.pop("method", DEFAULT_METHOD)
        if not isinstance(raw_method, str) or not raw_method.strip():
            raise ValueError("provider option 'method' must be a non-empty string")
        method = raw_method.strip().lower()
        if provider_name == LEGACY_PROVIDER_NAME and method != DEFAULT_METHOD:
            raise ValueError(
                f"legacy provider {LEGACY_PROVIDER_NAME!r} only supports method "
                f"{DEFAULT_METHOD!r}; use provider {PROVIDER_NAME!r} for other methods"
            )
        return cls(
            provider_name=provider_name,
            method=method,
            method_config=options,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProviderSelection:
        """Parse the plugin-owned configuration used by current upstream vLLM."""
        unknown = set(value) - {"schema_version", "provider", "method", "method_config"}
        if unknown:
            raise ValueError(
                "unknown plugin configuration fields: " + ", ".join(sorted(unknown))
            )
        schema_version = value.get("schema_version", SCHEMA_VERSION)
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {schema_version!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        provider_name = value.get("provider", PROVIDER_NAME)
        if provider_name not in SUPPORTED_PROVIDER_NAMES:
            supported = ", ".join(sorted(SUPPORTED_PROVIDER_NAMES))
            raise ValueError(
                f"provider {provider_name!r} is unsupported; "
                f"expected one of: {supported}"
            )
        method = value.get("method", DEFAULT_METHOD)
        if not isinstance(method, str) or not method.strip():
            raise ValueError("'method' must be a non-empty string")
        raw_method_config = value.get("method_config", {})
        if not isinstance(raw_method_config, Mapping):
            raise ValueError("'method_config' must be an object")
        return cls(
            provider_name=str(provider_name),
            method=method.strip().lower(),
            method_config=_copy_scalar_mapping(raw_method_config),
        )


def extension_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """Return true only for explicit direct or Extension Manager activation."""
    env = os.environ if environment is None else environment
    direct = env.get(ENABLE_ENV, "").strip().lower()
    if direct in {"1", "true", "yes", "on"}:
        return True
    enabled = {
        item.strip()
        for item in env.get(MANAGER_ENABLED_ENV, "").split(",")
        if item.strip()
    }
    return EXTENSION_ID in enabled


def load_runtime_selection(
    environment: Mapping[str, str] | None = None,
) -> ProviderSelection:
    """Load direct JSON/path configuration or the manager's stored config."""
    env = os.environ if environment is None else environment
    direct = env.get(CONFIG_ENV)
    if direct:
        candidate = Path(direct).expanduser()
        payload = (
            json.loads(candidate.read_text(encoding="utf-8"))
            if candidate.is_file()
            else json.loads(direct)
        )
        if not isinstance(payload, Mapping):
            raise ValueError(f"{CONFIG_ENV} must resolve to a JSON object")
        return ProviderSelection.from_mapping(payload)

    try:
        from vllm_hust_ext.config import load_config
    except ImportError as error:
        raise ValueError(
            f"{CONFIG_ENV} is required when Extension Manager is not installed"
        ) from error
    state = load_config().extension(EXTENSION_ID)
    if not state.enabled:
        raise ValueError(f"Extension Manager does not mark {EXTENSION_ID!r} enabled")
    return ProviderSelection.from_mapping(state.configuration)


def _copy_scalar_mapping(options: Mapping[str, Any]) -> dict[str, JsonScalar]:
    copied: dict[str, JsonScalar] = {}
    for key, value in options.items():
        if not isinstance(key, str) or not key:
            raise ValueError("provider option names must be non-empty strings")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError(f"provider option {key!r} must be a JSON scalar")
        copied[key] = value
    return copied


def require_int(options: Mapping[str, Any], name: str, default: int) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"method option {name!r} must be an integer")
    return value


def require_choice(
    options: Mapping[str, Any],
    name: str,
    default: str,
    choices: set[str],
) -> str:
    value = options.get(name, default)
    if not isinstance(value, str) or value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"method option {name!r} must be one of: {expected}")
    return value
