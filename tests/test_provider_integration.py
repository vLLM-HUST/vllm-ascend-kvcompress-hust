# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend_kvcompress.config import PROVIDER_NAME
from vllm_ascend_kvcompress.methods.base import (
    CompressionRequest,
    CompressionResult,
    LayerCache,
    MethodRuntimeSpec,
)
from vllm_ascend_kvcompress.provider import (
    ActiveCompression,
    AscendKVCompressionProvider,
    PendingCompression,
    _plain_full_attention_spec_reasons,
    _unpack_separate_kv_cache,
    _validate_method_layer_lengths,
    _validate_method_runtime_spec,
)


class _RecordingBlockTable:
    def __init__(self) -> None:
        self.added: tuple[tuple[list[int], ...], int] | None = None
        self.physical_positions: torch.Tensor | None = None

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        self.added = (block_ids, row_idx)

    def compute_slot_mapping(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        del num_reqs, query_start_loc
        self.physical_positions = positions.clone()


class _RecordingMethod:
    name = "recording"

    def __init__(self) -> None:
        self.last_request: CompressionRequest | None = None

    def compress(self, request: CompressionRequest) -> CompressionResult:
        self.last_request = request
        return CompressionResult(physical_num_tokens=128)


def _provider_without_init() -> AscendKVCompressionProvider:
    provider = AscendKVCompressionProvider.__new__(AscendKVCompressionProvider)
    provider.provider_name = PROVIDER_NAME
    provider.method = _RecordingMethod()
    provider.method_runtime_spec = MethodRuntimeSpec(
        requires_private_destination=True,
        compression_threshold_tokens=256,
        required_recompute_tokens=128,
        max_physical_num_tokens=128,
    )
    provider.pending = {}
    provider.active = {}
    provider._request_offsets_cpu = None
    provider._request_offsets_device = None
    provider._physical_positions = None
    return provider


def test_current_ascend_direct_tuple_cache_binding_is_accepted() -> None:
    k_cache = torch.empty(2, 128, 1, 4)
    v_cache = torch.empty_like(k_cache)
    direct_k, direct_v = _unpack_separate_kv_cache("layer", (k_cache, v_cache))
    wrapped_k, wrapped_v = _unpack_separate_kv_cache("layer", [(k_cache, v_cache)])

    assert direct_k is k_cache
    assert direct_v is v_cache
    assert wrapped_k is k_cache
    assert wrapped_v is v_cache


def test_unquantized_full_attention_spec_is_not_rejected() -> None:
    from vllm.v1.kv_cache_interface import KVQuantMode

    spec = SimpleNamespace(
        sliding_window=None,
        attention_chunk_size=None,
        non_causal=False,
        head_size=128,
        head_size_v=128,
        kv_quant_mode=KVQuantMode.NONE,
        page_size_padded=None,
        indexes_kv_by_block_stride=False,
    )
    assert _plain_full_attention_spec_reasons(spec) == ()


def test_method_runtime_spec_rejects_non_boolean_destination_flag() -> None:
    spec = MethodRuntimeSpec(
        requires_private_destination=1,
        compression_threshold_tokens=256,
        required_recompute_tokens=128,
        max_physical_num_tokens=128,
    )
    with pytest.raises(ValueError, match="must be a boolean"):
        _validate_method_runtime_spec("bad", spec)


def test_method_layer_lengths_require_exact_bound_layer_set() -> None:
    caches = (
        LayerCache("layer.0", 0, torch.empty(0), torch.empty(0)),
        LayerCache("layer.1", 1, torch.empty(0), torch.empty(0)),
    )

    with pytest.raises(RuntimeError, match="missing=layer.1"):
        _validate_method_layer_lengths(
            "bad",
            (("layer.0", 128),),
            caches,
            128,
        )


def test_final_prefill_builds_plan_after_overlap_safe_materialization() -> None:
    provider = _provider_without_init()
    k_cache = torch.arange(4 * 128, dtype=torch.float32).view(4, 128, 1, 1)
    v_cache = k_cache.clone() + 1000
    provider.layer_caches = (
        LayerCache(
            "model.layers.0.self_attn",
            0,
            k_cache,
            v_cache,
        ),
    )
    request = SimpleNamespace(
        num_computed_tokens=256,
        num_prompt_tokens=300,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"), requests={"r": request}
    )
    provider.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False)
    )
    scheduler_output = SimpleNamespace(
        kv_cache_compression_transaction_ids={"r": 7},
        num_scheduled_tokens={"r": 44},
        kv_cache_compression_destination_block_ids=None,
    )

    plans = provider.compress_transactions(scheduler_output)

    assert len(plans) == 1
    assert plans[0].semantic_num_tokens == 300
    assert plans[0].physical_num_tokens == 128
    assert plans[0].expected_block_ids == ((2, 0, 3),)
    assert provider.pending["r"].destination_block_ids == ((2,),)
    assert provider.method.last_request is not None
    assert provider.method.last_request.semantic_num_tokens == 300


def test_commit_ack_replaces_tables_and_decode_uses_physical_positions() -> None:
    provider = _provider_without_init()
    block_table = _RecordingBlockTable()
    request = SimpleNamespace(block_ids=([1, 2, 3],))
    input_batch = SimpleNamespace(
        req_id_to_index={"compressed": 0, "plain": 1},
        req_ids=["compressed", "plain"],
        num_reqs=2,
        block_table=block_table,
    )
    provider.runner = SimpleNamespace(
        requests={"compressed": request, "plain": SimpleNamespace()},
        input_batch=input_batch,
        req_indices=SimpleNamespace(gpu=torch.tensor([0, 1], dtype=torch.long)),
        query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 1, 2], dtype=torch.int32)),
        positions=torch.tensor([300, 20], dtype=torch.int64),
        seq_lens=torch.tensor([301, 21], dtype=torch.int64),
        optimistic_seq_lens_cpu=torch.tensor([301, 21], dtype=torch.int64),
    )
    provider.pending["compressed"] = PendingCompression(
        semantic_num_tokens=300,
        physical_num_tokens=128,
        destination_block_ids=((5,),),
    )
    provider._request_offsets_cpu = torch.zeros(2, dtype=torch.int64)
    provider._request_offsets_device = torch.zeros(2, dtype=torch.int64)
    provider._physical_positions = torch.empty(2, dtype=torch.int64)

    provider.consume_block_table_updates(
        SimpleNamespace(
            finished_req_ids=set(),
            kv_cache_compression_block_table_updates={"compressed": ([5, 9],)},
        )
    )

    assert request.block_ids == ([5, 9],)
    assert block_table.added == (([5, 9],), 0)
    assert provider.active["compressed"] == ActiveCompression(300, 128)

    provider.apply_physical_decode_state(
        SimpleNamespace(total_num_scheduled_tokens=2),
        np.array([1, 1], dtype=np.int32),
    )
    torch.testing.assert_close(provider.runner.positions, torch.tensor([300, 20]))
    torch.testing.assert_close(block_table.physical_positions, torch.tensor([128, 20]))
    torch.testing.assert_close(provider.runner.seq_lens, torch.tensor([129, 21]))
    torch.testing.assert_close(
        provider.runner.optimistic_seq_lens_cpu, torch.tensor([129, 21])
    )
