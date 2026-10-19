# SPDX-License-Identifier: Apache-2.0
"""Router shims must follow the core ABI without importing absent legacy modules."""

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest
from vllm.model_executor.layers.fused_moe import config

from vllm_ascend_kvcompress.host_compat import install_grouped_topk_router_compat

OPTIONS = dict(
    scoring_func="sigmoid",
    top_k=8,
    renormalize=True,
    num_expert_group=None,
    has_e_score_bias=True,
)


def test_legacy_classifier_does_not_import_new_ascend_router(monkeypatch):
    def legacy(*, routed_scaling_factor=1.0, **kwargs):
        return config.RoutingMethodType.Unspecified

    monkeypatch.setattr(config, "get_routing_method_type", legacy)
    original_import = builtins.__import__

    def reject_router(name, *args, **kwargs):
        if name == "vllm_ascend.ops.fused_moe.router":
            raise AssertionError("legacy core must not import the newer Ascend router")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_router)
    assert install_grouped_topk_router_compat() is False
    assert config.get_routing_method_type is legacy


def test_modern_classifier_keeps_scaling_semantics_and_is_idempotent(monkeypatch):
    calls = []

    def modern(*, scoring_func, top_k, renormalize, num_expert_group, has_e_score_bias):
        calls.append(
            (scoring_func, top_k, renormalize, num_expert_group, has_e_score_bias)
        )
        return config.RoutingMethodType.MiniMax2

    monkeypatch.setattr(config, "get_routing_method_type", modern)
    router = SimpleNamespace(get_routing_method_type=modern)
    module = ModuleType("vllm_ascend.ops.fused_moe.router")
    module.grouped_topk_router = router
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert install_grouped_topk_router_compat() is True
    patched = router.get_routing_method_type
    assert (
        patched(**OPTIONS, routed_scaling_factor=2.0)
        == config.RoutingMethodType.Unspecified
    )
    assert calls == []
    assert (
        patched(**OPTIONS, routed_scaling_factor=1.0)
        == config.RoutingMethodType.MiniMax2
    )
    assert (
        patched(**OPTIONS, routed_scaling_factor=None)
        == config.RoutingMethodType.MiniMax2
    )
    assert len(calls) == 2
    assert install_grouped_topk_router_compat() is False
    assert router.get_routing_method_type is patched
    assert config.get_routing_method_type is modern


def test_modern_router_import_failure_is_not_hidden(monkeypatch):
    def modern(**kwargs):
        return config.RoutingMethodType.Unspecified

    monkeypatch.setattr(config, "get_routing_method_type", modern)
    original_import = builtins.__import__

    def missing_dependency(name, *args, **kwargs):
        if name == "vllm_ascend.ops.fused_moe.router":
            raise ModuleNotFoundError(
                "missing router dependency", name="router_dependency"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="missing router dependency"):
        install_grouped_topk_router_compat()
