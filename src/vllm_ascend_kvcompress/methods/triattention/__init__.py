# SPDX-License-Identifier: Apache-2.0
"""TriAttention 配置与 压缩后端，worker 代码按需加载。"""

from .backend import TriAttentionBackend, create_triattention_backend
from .config import TriAttentionConfig

__all__ = ["TriAttentionBackend", "TriAttentionConfig", "create_triattention_backend"]
