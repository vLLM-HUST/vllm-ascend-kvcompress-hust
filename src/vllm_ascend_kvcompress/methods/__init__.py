# SPDX-License-Identifier: Apache-2.0
"""Built-in and externally registered KV compression methods."""

from .registry import (
    METHOD_ENTRY_POINT_GROUP,
    available_methods,
    create_method,
    register_method,
)
from .triattention import TriAttentionMethod, create_triattention_method

register_method("triattention", create_triattention_method)

__all__ = [
    "METHOD_ENTRY_POINT_GROUP",
    "TriAttentionMethod",
    "available_methods",
    "create_method",
    "register_method",
]
