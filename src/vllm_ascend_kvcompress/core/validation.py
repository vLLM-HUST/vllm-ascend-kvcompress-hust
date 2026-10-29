# SPDX-License-Identifier: Apache-2.0
"""配置选项的通用类型与取值校验工具。"""

from collections.abc import Mapping
from typing import Any


def require_int(options: Mapping[str, Any], name: str, default: int) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"option {name!r} must be an integer")
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
        raise ValueError(f"option {name!r} must be one of: {expected}")
    return value


def require_bool(options: Mapping[str, Any], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"method option {name!r} must be a boolean")
    return value
