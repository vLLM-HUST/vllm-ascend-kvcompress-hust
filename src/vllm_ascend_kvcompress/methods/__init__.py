# SPDX-License-Identifier: Apache-2.0
"""公共方法 API 与内置工厂装配，具体算法实现按需加载。"""

from .backend import BackendFactory, CompressionBackend
from .builtins import register_builtin_methods
from .registry import (
    METHOD_ENTRY_POINT_GROUP,
    METHOD_REGISTRY,
    available_methods,
    create_backend,
    register_method,
)

# registry 模块加载完成后再装配内置方法，避免通用注册机制依赖具体算法。
# 包初始化只登记工厂，第三方入口点仍在首次查询或创建方法时发现。
register_builtin_methods(METHOD_REGISTRY)

__all__ = [
    "BackendFactory",
    "CompressionBackend",
    "METHOD_ENTRY_POINT_GROUP",
    "available_methods",
    "create_backend",
    "register_method",
]
