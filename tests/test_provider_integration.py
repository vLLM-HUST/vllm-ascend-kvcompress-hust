# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

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
    _unpack_layer_cache,
)


class _RecordingMethod:
    name = "recording"

    def __init__(self) -> None:
        self.last_request: CompressionRequest | None = None

    def compress(self, request: CompressionRequest) -> CompressionResult:
        self.last_request = request
        return CompressionResult(physical_num_tokens=128)


class _RecordingBlockTable:
    def __init__(self) -> None:
        self.added = None

    def add_row(self, block_ids, row_idx) -> None:
        self.added = block_ids, row_idx


def _provider() -> AscendKVCompressionProvider:
    provider = AscendKVCompressionProvider.__new__(AscendKVCompressionProvider)
    provider.method = _RecordingMethod()
    provider.runtime_spec = MethodRuntimeSpec(True, 256, 128, 128)
    provider.layer_caches = (
        LayerCache("model.layers.0.self_attn", 0, torch.empty(0), torch.empty(0)),
    )
    provider.pending = {}
    provider.active = {}
    provider._request_offsets_cpu = None
    provider._request_offsets_device = None
    provider._physical_positions = None
    provider._offset_row_snapshot = ()
    provider._has_active_rows = False
    provider._physical_lengths_applied = False
    return provider


def test_current_standardized_cache_uses_ascend_unpacker() -> None:
    k_cache = torch.empty(2, 128, 1, 4)
    v_cache = torch.empty_like(k_cache)

    class Impl:
        def _unpack_kv_cache(self, cache):
            assert cache == "standardized-cache"
            return k_cache, v_cache

    layer = SimpleNamespace(kv_cache=["standardized-cache"], impl=Impl())
    actual_k, actual_v = _unpack_layer_cache("layer", layer)
    assert actual_k is k_cache
    assert actual_v is v_cache


def test_final_prefill_compresses_and_arms_worker_commit() -> None:
    provider = _provider()
    request = SimpleNamespace(
        num_computed_tokens=256,
        num_prompt_tokens=300,
        max_tokens=128,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"), requests={"r": request}
    )

    provider.compress_scheduled_requests(
        SimpleNamespace(num_scheduled_tokens={"r": 44})
    )

    assert provider.pending["r"] == PendingCompression(300, 128, (2,))
    assert provider.method.last_request is not None
    assert provider.method.last_request.physical_num_tokens == 300


def test_worker_commit_truncates_tables_and_activates_offsets() -> None:
    provider = _provider()
    block_table = _RecordingBlockTable()
    request = SimpleNamespace(block_ids=([2, 0, 3],))
    provider.runner = SimpleNamespace(
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}, block_table=block_table),
    )
    provider.pending["r"] = PendingCompression(300, 128, (2,))
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
    )

    provider.before_update_states(output)

    assert request.block_ids == ([2],)
    assert block_table.added == (([2],), 0)
    assert provider.active["r"] == ActiveCompression(300, 128)


def test_worker_discards_preempted_compression_state() -> None:
    provider = _provider()
    provider.runner = SimpleNamespace(
        requests={},
        input_batch=SimpleNamespace(req_id_to_index={}),
    )
    provider.pending["r"] = PendingCompression(300, 128, (2,))
    provider.active["r"] = ActiveCompression(300, 128)
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids={"r"},
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
    )

    provider.before_update_states(output)

    assert "r" not in provider.pending
    assert "r" not in provider.active


def test_physical_positions_and_lengths_preserve_semantic_rope() -> None:
    provider = _provider()
    provider.active["compressed"] = ActiveCompression(300, 128)
    provider._request_offsets_cpu = torch.tensor([172, 0], dtype=torch.int64)
    provider._request_offsets_device = torch.tensor([172, 0], dtype=torch.int64)
    provider._physical_positions = torch.empty(2, dtype=torch.int64)
    provider._has_active_rows = True
    positions = torch.tensor([300, 20], dtype=torch.int64)
    provider.runner = SimpleNamespace(
        req_indices=SimpleNamespace(gpu=torch.tensor([0, 1], dtype=torch.long)),
        input_batch=SimpleNamespace(num_reqs=2),
        seq_lens=torch.tensor([301, 21], dtype=torch.int64),
        optimistic_seq_lens_cpu=torch.tensor([301, 21], dtype=torch.int64),
    )

    physical = provider.physical_positions_for_slot_mapping(positions)
    provider.apply_physical_attention_lengths()

    torch.testing.assert_close(positions, torch.tensor([300, 20]))
    torch.testing.assert_close(physical, torch.tensor([128, 20]))
    torch.testing.assert_close(provider.runner.seq_lens, torch.tensor([129, 21]))
    torch.testing.assert_close(
        provider.runner.optimistic_seq_lens_cpu, torch.tensor([129, 21])
    )


def test_repeated_compression_uses_current_physical_window() -> None:
    provider = _provider()
    request = SimpleNamespace(
        num_computed_tokens=427,
        num_prompt_tokens=300,
        max_tokens=128,
        block_ids=([5, 9],),
    )
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"), requests={"r": request}
    )
    provider.active["r"] = ActiveCompression(300, 128)

    provider.compress_scheduled_requests(SimpleNamespace(num_scheduled_tokens={"r": 1}))

    assert provider.pending["r"] == PendingCompression(428, 128, (5,))
    assert provider.method.last_request is not None
    assert provider.method.last_request.physical_num_tokens == 256


def test_worker_skips_short_decode_request() -> None:
    provider = _provider()
    provider.runtime_spec = MethodRuntimeSpec(True, 256, 128, 128, 64)
    request = SimpleNamespace(
        num_computed_tokens=256,
        num_prompt_tokens=300,
        max_tokens=32,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"), requests={"r": request}
    )

    provider.compress_scheduled_requests(
        SimpleNamespace(num_scheduled_tokens={"r": 44})
    )

    assert not provider.pending
    assert provider.method.last_request is None


def test_worker_reads_output_limit_from_cached_request_sampling_params() -> None:
    provider = _provider()
    provider.runtime_spec = MethodRuntimeSpec(True, 256, 128, 128, 64)
    request = SimpleNamespace(
        num_computed_tokens=256,
        num_prompt_tokens=300,
        sampling_params=SimpleNamespace(max_tokens=32),
        pooling_params=None,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"), requests={"r": request}
    )

    provider.compress_scheduled_requests(
        SimpleNamespace(num_scheduled_tokens={"r": 44})
    )

    assert not provider.pending
