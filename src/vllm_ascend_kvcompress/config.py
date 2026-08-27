# SPDX-License-Identifier: Apache-2.0
"""Provider-level configuration and method selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

PROVIDER_NAME = "ascend_kvcompress"
LEGACY_PROVIDER_NAME = "triattention_ascend"
SUPPORTED_PROVIDER_NAMES = frozenset({PROVIDER_NAME, LEGACY_PROVIDER_NAME})
DEFAULT_METHOD = "triattention"
SCHEMA_VERSION = 1
ASCEND_BLOCK_SIZE = 128

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
