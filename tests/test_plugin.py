# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_ascend_kvcompress.plugin import (
    _install_balance_scheduler_hook,
    _install_hooks,
)
from vllm_ascend_kvcompress.provider import PROVIDER_FACTORY_QUALNAME


class _Platform:
    @classmethod
    def get_kv_cache_compression_provider_factory(cls) -> str | None:
        del cls
        return None


class _Worker:
    def validate_kv_cache_compression(self):
        return "original-validate"

    def initialize_from_config(self, config):
        return config


class _Runner:
    def _update_states(self, output):
        return output

    def _prepare_inputs(self, output, scheduled):
        return output, scheduled

    def sample_tokens(self, grammar):
        return grammar


def test_hook_installation_is_idempotent() -> None:
    _install_hooks(_Platform, _Worker, _Runner)
    first_validate = _Worker.validate_kv_cache_compression
    first_prepare = _Runner._prepare_inputs

    _install_hooks(_Platform, _Worker, _Runner)

    assert _Platform.get_kv_cache_compression_provider_factory() == (
        PROVIDER_FACTORY_QUALNAME
    )
    assert _Worker.validate_kv_cache_compression is first_validate
    assert _Runner._prepare_inputs is first_prepare


def test_balance_scheduler_forwards_compression_runtime_spec() -> None:
    class BaseScheduler:
        def __init__(self, **kwargs):
            self.base_kwargs = kwargs

    class BalanceScheduler(BaseScheduler):
        def __init__(self, **kwargs):
            self.original_kwargs = kwargs

    class VictimSelector:
        @classmethod
        def from_vllm_config(cls, config):
            return (cls, config)

    _install_balance_scheduler_hook(
        BalanceScheduler,
        VictimSelector,
        lambda config: True,
    )
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=2))
    runtime_spec = object()
    scheduler = BalanceScheduler(
        vllm_config=config,
        kv_cache_config=object(),
        structured_output_manager=object(),
        block_size=128,
        kv_cache_compression_runtime_spec=runtime_spec,
    )

    assert scheduler.base_kwargs["kv_cache_compression_runtime_spec"] is runtime_spec
    assert len(scheduler.balance_queue) == 2
    assert scheduler.victim_selector == (VictimSelector, config)
