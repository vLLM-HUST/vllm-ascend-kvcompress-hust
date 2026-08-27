# SPDX-License-Identifier: Apache-2.0
"""Compression-method registry with optional Python entry-point discovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from typing import Any, TypeAlias

from ..config import JsonScalar
from .base import KVCompressionMethod, ModelShape

MethodFactory: TypeAlias = Callable[
    [Mapping[str, JsonScalar], Any, ModelShape], KVCompressionMethod
]
METHOD_ENTRY_POINT_GROUP = "vllm_ascend_kvcompress.methods"


class MethodRegistry:
    """Fail-closed name-to-factory registry used by the provider."""

    def __init__(self) -> None:
        self._factories: dict[str, MethodFactory] = {}

    def register(self, name: str, factory: MethodFactory) -> None:
        normalized = _normalize_name(name)
        if normalized in self._factories:
            raise ValueError(f"compression method {normalized!r} is already registered")
        if not callable(factory):
            raise TypeError("compression method factory must be callable")
        self._factories[normalized] = factory

    def create(
        self,
        name: str,
        options: Mapping[str, JsonScalar],
        vllm_config: Any,
        model_shape: ModelShape,
    ) -> KVCompressionMethod:
        normalized = _normalize_name(name)
        factory = self._factories.get(normalized)
        if factory is None:
            available = ", ".join(self.names()) or "<none>"
            raise ValueError(
                f"unknown compression method {normalized!r}; available: {available}"
            )
        method = factory(options, vllm_config, model_shape)
        if not isinstance(method, KVCompressionMethod):
            raise TypeError(
                f"compression method factory {normalized!r} returned "
                f"{type(method).__name__}, expected KVCompressionMethod"
            )
        if method.name != normalized:
            raise ValueError(
                f"compression method factory {normalized!r} returned method "
                f"named {method.name!r}"
            )
        return method

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


METHOD_REGISTRY = MethodRegistry()
_ENTRY_POINTS_LOADED = False


def register_method(name: str, factory: MethodFactory) -> None:
    """Register an in-process compression method factory."""
    METHOD_REGISTRY.register(name, factory)


def load_method_entry_points() -> None:
    """Load third-party methods declared in the public entry-point group once."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    for entry_point in entry_points(group=METHOD_ENTRY_POINT_GROUP):
        METHOD_REGISTRY.register(entry_point.name, entry_point.load())


def create_method(
    name: str,
    options: Mapping[str, JsonScalar],
    vllm_config: Any,
    model_shape: ModelShape,
) -> KVCompressionMethod:
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
