# SPDX-License-Identifier: Apache-2.0
"""Built-in and externally registered KV compression methods."""

from .registry import (
    METHOD_ENTRY_POINT_GROUP,
    available_methods,
    create_method,
    get_method_runtime_spec,
    register_method,
)
from .triattention import TriAttentionMethod, create_triattention_method
from .triattention.config import triattention_runtime_spec
from .vato import VATOMethod, create_vato_method, vato_runtime_spec

register_method(
    "triattention",
    create_triattention_method,
    runtime_spec_factory=triattention_runtime_spec,
)
register_method("vato", create_vato_method, runtime_spec_factory=vato_runtime_spec)

__all__ = [
    "METHOD_ENTRY_POINT_GROUP",
    "TriAttentionMethod",
    "VATOMethod",
    "available_methods",
    "create_method",
    "get_method_runtime_spec",
    "register_method",
]
