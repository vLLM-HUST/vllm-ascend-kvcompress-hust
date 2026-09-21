# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend_kvcompress.methods.base import (
    CompressionRequest,
    CompressionResult,
    KVCompressionMethod,
    LayerCache,
    MethodRuntimeSpec,
    ModelShape,
)
from vllm_ascend_kvcompress.methods.registry import MethodRegistry


class _Method(KVCompressionMethod):
    @property
    def name(self) -> str:
        return "test_method"

    @property
    def runtime_spec(self) -> MethodRuntimeSpec:
        return MethodRuntimeSpec(True, 256, 128, 128)

    def compatibility_reasons(self, worker: object) -> tuple[str, ...]:
        del worker
        return ()

    def bind_model_runner(
        self, runner: object, layer_caches: tuple[LayerCache, ...]
    ) -> None:
        del runner, layer_caches

    def compress(self, request: CompressionRequest) -> CompressionResult:
        del request
        return CompressionResult(physical_num_tokens=128)


def _shape() -> ModelShape:
    return ModelShape("qwen2", 1, 1, 1, 128, 10_000.0, False)


def test_registry_creates_registered_method() -> None:
    registry = MethodRegistry()
    registry.register("test_method", lambda options, config, shape: _Method())

    method = registry.create("test_method", {}, SimpleNamespace(), _shape())

    assert isinstance(method, _Method)
    assert registry.names() == ("test_method",)


def test_registry_rejects_duplicate_and_unknown_names() -> None:
    registry = MethodRegistry()

    def factory(options, config, shape):
        del options, config, shape
        return _Method()

    registry.register("test_method", factory)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("test_method", factory)
    with pytest.raises(ValueError, match="available: test_method"):
        registry.create("missing", {}, SimpleNamespace(), _shape())


def test_registry_rejects_factory_contract_violation() -> None:
    registry = MethodRegistry()
    registry.register("broken", lambda options, config, shape: object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="expected KVCompressionMethod"):
        registry.create("broken", {}, SimpleNamespace(), _shape())


def test_scheduler_runtime_spec_does_not_construct_worker_method() -> None:
    registry = MethodRegistry()

    def worker_factory(options, config, shape):
        raise AssertionError("scheduler must not initialize worker resources")

    registry.register(
        "test_method",
        worker_factory,
        runtime_spec_factory=lambda options: MethodRuntimeSpec(True, 256, 128, 128),
    )
    assert registry.runtime_spec("test_method", {}).max_physical_num_tokens == 128


def test_scheduler_rejects_method_without_runtime_spec_factory() -> None:
    registry = MethodRegistry()
    registry.register("test_method", lambda options, config, shape: _Method())
    with pytest.raises(ValueError, match="scheduler"):
        registry.runtime_spec("test_method", {})


def test_builtin_scheduler_specs_match_workers_without_calibration(tmp_path) -> None:
    from vllm_ascend_kvcompress.methods import get_method_runtime_spec

    spec = get_method_runtime_spec("vato", {"kv_budget": 128})
    assert spec == MethodRuntimeSpec(True, 256, 128, 128)
    # This path must remain usable before automatic calibration runs in workers.
    spec = get_method_runtime_spec(
        "triattention", {"stats_path": str(tmp_path / "missing.pt")}
    )
    assert spec == MethodRuntimeSpec(True, 2176, 128, 2048)
