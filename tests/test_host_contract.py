# SPDX-License-Identifier: Apache-2.0
"""使用 CPU 张量验证宿主布局与能力约束。"""

from types import SimpleNamespace

import torch

from vllm_ascend_kvcompress.host.contracts import (
    common_compatibility_reasons,
    find_full_attention_group,
    unpack_layer_cache,
)
from vllm_ascend_kvcompress.runtime.coordinates import expand_scheduler_block_ids


def test_worker_compatibility_admits_apc_and_qualified_mtp2(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_ASCEND_KVCOMPRESS_EXPERIMENTAL_MTP2", raising=False)
    runner_cls = type("NPUModelRunner", (), {})
    runner_cls.__module__ = "vllm_ascend.worker.model_runner_v1"
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            block_size=128,
            cache_dtype="auto",
        ),
        speculative_config=None,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=True),
        model_config=SimpleNamespace(
            is_hybrid=False,
            use_mla=False,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(num_key_value_heads=2),
        ),
        aux_output_config=SimpleNamespace(enable_return_routed_experts=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )

    assert common_compatibility_reasons(config, runner_cls()) == ()

    config.model_config.is_hybrid = True
    config.model_config.hf_text_config.model_type = "qwen3_5_moe_text"
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.block_size = 2048
    assert common_compatibility_reasons(config, runner_cls()) == ()

    config.speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=2,
        num_speculative_tokens_per_batch_size=None,
    )
    assert any(
        "speculative decoding is unqualified" in reason
        for reason in common_compatibility_reasons(config, runner_cls())
    )
    monkeypatch.setenv("VLLM_ASCEND_KVCOMPRESS_EXPERIMENTAL_MTP2", "1")
    assert common_compatibility_reasons(config, runner_cls()) == ()

    config.speculative_config.num_speculative_tokens = 3
    assert any(
        "speculative decoding is unqualified" in reason
        for reason in common_compatibility_reasons(config, runner_cls())
    )
    config.speculative_config.num_speculative_tokens = 2
    config.speculative_config.method = "eagle"
    assert any(
        "speculative decoding is unqualified" in reason
        for reason in common_compatibility_reasons(config, runner_cls())
    )
    config.speculative_config.method = "mtp"

    config.cache_config.mamba_cache_mode = "all"
    assert "hybrid compression requires mamba_cache_mode='none' or 'align'" in (
        common_compatibility_reasons(config, runner_cls())
    )


def test_scheduler_blocks_expand_to_consecutive_kernel_cache_blocks() -> None:
    actual = expand_scheduler_block_ids((2, 0), 16, torch.device("cpu"))

    torch.testing.assert_close(
        actual,
        torch.tensor([*range(32, 48), *range(0, 16)], dtype=torch.long),
    )


def test_current_standardized_cache_uses_ascend_unpacker() -> None:
    k_cache = torch.empty(2, 128, 1, 4)
    v_cache = torch.empty_like(k_cache)

    class Impl:
        def _unpack_kv_cache(self, cache):
            assert cache == "standardized-cache"
            return k_cache, v_cache

    layer = SimpleNamespace(kv_cache=["standardized-cache"], impl=Impl())
    actual_k, actual_v = unpack_layer_cache("layer", layer)
    assert actual_k is k_cache
    assert actual_v is v_cache


def test_hybrid_group_finder_selects_only_full_attention_group() -> None:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    full_attention = object.__new__(FullAttentionSpec)
    recurrent_state = object.__new__(MambaSpec)
    groups = [
        SimpleNamespace(kv_cache_spec=recurrent_state),
        SimpleNamespace(kv_cache_spec=full_attention),
    ]

    assert find_full_attention_group(groups) == 1
