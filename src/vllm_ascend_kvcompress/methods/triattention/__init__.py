# SPDX-License-Identifier: Apache-2.0
"""Built-in TriAttention compression method package."""

from .config import TriAttentionConfig
from .method import (
    METHOD_NAME,
    TriAttentionAscendConfig,
    TriAttentionLayerCache,
    TriAttentionMethod,
    create_triattention_method,
)

__all__ = [
    "METHOD_NAME",
    "TriAttentionAscendConfig",
    "TriAttentionConfig",
    "TriAttentionLayerCache",
    "TriAttentionMethod",
    "create_triattention_method",
]
