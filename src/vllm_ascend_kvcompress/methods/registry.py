# SPDX-License-Identifier: Apache-2.0
"""压缩方法注册表，支持可选的 Python 入口点发现。"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Any

from .backend import BackendFactory, CompressionBackend
from .base import ModelShape

METHOD_ENTRY_POINT_GROUP = "vllm_ascend_kvcompress.methods"


class MethodRegistry:
    """维护压缩方法名称到 backend 工厂的进程内映射。

    注册时只保存工厂，不创建 backend 或 worker 算法实例。
    创建时将方法配置、宿主配置和模型描述交给对应工厂，
    并检查返回对象是否满足 CompressionBackend 契约。

    本类不负责入口点发现，也不缓存已创建的 backend；模块级包装函数
    负责发现第三方入口点，具体 backend 再负责后续算法和运行时初始化。
    """

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory) -> None:
        """
        注册一个 BackendFactory，不允许覆盖同名方法。

        名称会去除首尾空白并转成小写，因此不同大小写或外部空白
        不构成不同方法。这里只检查工厂可调用，返回类型在创建时检查。
        """
        normalized = _normalize_name(name)
        # 内置方法和第三方方法使用同一命名空间；重复注册直接报错，
        # 避免加载顺序决定最终启用哪个实现。
        if normalized in self._factories:
            raise ValueError(f"compression method {normalized!r} is already registered")
        if not callable(factory):
            raise TypeError("compression method factory must be callable")
        self._factories[normalized] = factory

    def create(
        self,
        name: str,
        options: Mapping[str, Any],
        vllm_config: Any,
        model_shape: ModelShape,
    ) -> CompressionBackend:
        """
        调用指定方法的工厂，创建并验证一个 backend。

        options 是该方法自己的配置；vllm_config 是宿主配置；
        model_shape 是尚未按 TP 切分的模型描述。本层不解析方法参数，
        参数校验及静态预算构造交由工厂返回的 backend 负责。
        """
        normalized = _normalize_name(name)
        factory = self._factories.get(normalized)
        if factory is None:
            available = ", ".join(self.names()) or "<none>"
            raise ValueError(
                f"unknown compression method {normalized!r}; available: {available}"
            )
        backend = factory(options, vllm_config, model_shape)
        if not isinstance(backend, CompressionBackend):
            raise TypeError(
                f"compression method factory {normalized!r} returned "
                f"{type(backend).__name__}, expected CompressionBackend"
            )
        # backend.name 必须与规范化注册名完全一致，避免配置名称与实际实现错位。
        if backend.name != normalized:
            raise ValueError(
                f"compression method factory {normalized!r} returned method "
                f"named {backend.name!r}"
            )
        return backend

    def names(self) -> tuple[str, ...]:
        """返回当前已注册名称的有序快照。"""
        return tuple(sorted(self._factories))


METHOD_REGISTRY = MethodRegistry()
_ENTRY_POINTS_LOADED = False


def register_method(name: str, factory: BackendFactory) -> None:
    METHOD_REGISTRY.register(name, factory)


def load_method_entry_points() -> None:
    """仅加载一次公共入口点组中声明的第三方方法。"""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    # 进程内只发现一次；在开始加载前置位以避免加载第三方包时递归发现。
    # 名称冲突或 entry point 加载失败直接报错，不通过注册顺序静默覆盖内置算法。
    _ENTRY_POINTS_LOADED = True
    for entry_point in entry_points(group=METHOD_ENTRY_POINT_GROUP):
        METHOD_REGISTRY.register(entry_point.name, entry_point.load())


def create_backend(
    name: str,
    options: Mapping[str, Any],
    vllm_config: Any,
    model_shape: ModelShape,
) -> CompressionBackend:
    load_method_entry_points()
    return METHOD_REGISTRY.create(name, options, vllm_config, model_shape)


def available_methods() -> tuple[str, ...]:
    load_method_entry_points()
    return METHOD_REGISTRY.names()


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("compression method name must be a non-empty string")
    normalized = name.strip().lower()
    if not normalized.replace("_", "").replace("-", "").isalnum():
        raise ValueError(
            "compression method names may contain only letters, digits, '-' and '_'"
        )
    return normalized
