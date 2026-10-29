# SPDX-License-Identifier: Apache-2.0
"""内置压缩方法的延迟工厂与注册装配。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backend import CompressionBackend
    from .base import ModelShape
    from .registry import MethodRegistry


def _triattention_backend(
    options: Mapping[str, Any], vllm_config: Any, model_shape: ModelShape
) -> CompressionBackend:
    """选中 TriAttention 时才导入其配置和 backend，不在注册阶段创建实例。"""
    from .triattention import create_triattention_backend

    return create_triattention_backend(options, vllm_config, model_shape)


def register_builtin_methods(registry: MethodRegistry) -> None:
    """
    向指定注册表登记内置方法，由 methods 包初始化时调用。
    注册表只保存延迟工厂，因此公共 API 导入不会加载具体算法实现。
    新增内置方法在这里登记，通用注册表无需增加算法名分支。
    """
    registry.register("triattention", _triattention_backend)
