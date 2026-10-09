# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_ascend_kvcompress.methods.base import (
    CompressionRequest,
    CompressionResult,
    LayerCache,
    MethodRuntimeSpec,
    QueryObservation,
)
from vllm_ascend_kvcompress.provider import (
    ActiveCompression,
    AscendKVCompressionProvider,
    PendingCompression,
    _allowed_scheduler_block_sizes,
    _common_compatibility_reasons,
    _expand_scheduler_block_ids,
    _find_full_attention_group,
    _unpack_layer_cache,
)
from vllm_ascend_kvcompress.transaction import PLAN_ATTRIBUTE, CompressionPlan


def _output_with_plan(
    semantic: int, source: tuple[int, ...], destination: tuple[int, ...], scheduled: int
) -> SimpleNamespace:
    output = SimpleNamespace(num_scheduled_tokens={"r": scheduled})
    setattr(
        output,
        PLAN_ATTRIBUTE,
        {"r": CompressionPlan(semantic, 128, source, destination)},
    )
    return output


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


def test_async_scheduling_is_admitted_by_worker_compatibility(monkeypatch) -> None:
    runner_type = type("NPUModelRunner", (SimpleNamespace,), {})
    runner_type.__module__ = "vllm_ascend.worker.model_runner_v1"
    runner = runner_type(use_sparse=False, use_compress=False)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            block_size=128,
            cache_dtype="auto",
            mamba_cache_mode="none",
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
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )

    assert _common_compatibility_reasons(config, runner) == ()

    config.cache_config.enable_prefix_caching = True
    assert _common_compatibility_reasons(config, runner) == ()

    config.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=2)
    assert "speculative decoding is unqualified" in " ".join(
        _common_compatibility_reasons(config, runner)
    )

    config.model_config.is_hybrid = True
    config.model_config.hf_text_config.model_type = "qwen3_5_moe_text"
    config.cache_config.mamba_cache_mode = "align"
    monkeypatch.setenv("VLLM_ASCEND_KVCOMPRESS_EXPERIMENTAL_MTP2", "1")
    assert _common_compatibility_reasons(config, runner) == ()


def _provider() -> AscendKVCompressionProvider:
    provider = AscendKVCompressionProvider.__new__(AscendKVCompressionProvider)
    provider.method = _RecordingMethod()
    provider.requires_per_layer_physical_state = False
    provider.query_window_tokens = 0
    provider.query_layer_caches = ()
    provider.speculative_cache_layer = None
    provider._query_step_output = None
    provider._query_seen_layers = set()
    provider._query_forward_complete = False
    provider._query_tracked_ids = set()
    provider.runtime_spec = MethodRuntimeSpec(True, 256, 128, 128)
    provider.layer_caches = (
        LayerCache("model.layers.0.self_attn", 0, torch.empty(0), torch.empty(0)),
    )
    provider.pending = {}
    provider.active = {}
    provider.attention_group_index = 0
    provider.scheduler_block_size = 128
    provider.cache_block_size = 128
    provider.cache_blocks_per_scheduler_block = 1
    provider._request_offsets_cpu = None
    provider._request_offsets_device = None
    provider._physical_positions = None
    provider._semantic_seq_lens_device = None
    provider._semantic_seq_lens_cpu = None
    provider._offset_row_snapshot = ()
    provider._has_active_rows = False
    provider._has_per_layer_rows = False
    provider._per_layer_metadata_seen = False
    provider._per_layer_step_requires_metadata = False
    provider._per_layer_slot_buffers = {}
    provider._physical_lengths_applied = False
    provider.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=None,
    )
    return provider


def test_gdn_metadata_retains_semantic_lengths_after_attention_compaction() -> None:
    @dataclass
    class Metadata:
        num_reqs: int
        seq_lens: torch.Tensor
        _seq_lens_cpu: torch.Tensor
        seq_lens_cpu_upper_bound: torch.Tensor

    provider = _provider()
    seq_lens = torch.tensor([300], dtype=torch.int64)
    cpu_lens = torch.tensor([300], dtype=torch.int64)
    provider.runner = SimpleNamespace(
        seq_lens=seq_lens,
        optimistic_seq_lens_cpu=cpu_lens,
        input_batch=SimpleNamespace(num_reqs=1),
    )
    provider._request_offsets_cpu = torch.tensor([172], dtype=torch.int64)
    provider._request_offsets_device = torch.tensor([172], dtype=torch.int64)
    provider._semantic_seq_lens_device = torch.empty_like(seq_lens)
    provider._semantic_seq_lens_cpu = torch.empty_like(cpu_lens)
    provider._has_active_rows = True

    provider.apply_physical_attention_lengths()
    gdn = provider.semantic_gdn_metadata(Metadata(1, seq_lens, cpu_lens, cpu_lens))

    assert seq_lens.tolist() == [128]
    assert cpu_lens.tolist() == [128]
    assert gdn.seq_lens.tolist() == [300]
    assert gdn._seq_lens_cpu.tolist() == [300]
    assert gdn.seq_lens_cpu_upper_bound.tolist() == [300]


def test_qwen35_hybrid_accepts_host_promoted_logical_block_size() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_hybrid=True,
            hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text"),
        )
    )

    assert _allowed_scheduler_block_sizes(config) == frozenset({128, 1152, 2048})


def test_prefix_caching_is_admitted_with_private_destination_protocol(
    monkeypatch,
) -> None:
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
        scheduler_config=SimpleNamespace(async_scheduling=False),
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

    assert _common_compatibility_reasons(config, runner_cls()) == ()

    config.model_config.is_hybrid = True
    config.model_config.hf_text_config.model_type = "qwen3_5_moe_text"
    config.cache_config.mamba_cache_mode = "align"
    assert _common_compatibility_reasons(config, runner_cls()) == ()

    config.scheduler_config.async_scheduling = True
    assert _common_compatibility_reasons(config, runner_cls()) == ()

    config.speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=2,
        num_speculative_tokens_per_batch_size=None,
    )
    assert any(
        "speculative decoding is unqualified" in reason
        for reason in _common_compatibility_reasons(config, runner_cls())
    )
    monkeypatch.setenv("VLLM_ASCEND_KVCOMPRESS_EXPERIMENTAL_MTP2", "1")
    assert _common_compatibility_reasons(config, runner_cls()) == ()

    config.speculative_config.num_speculative_tokens = 3
    assert any(
        "speculative decoding is unqualified" in reason
        for reason in _common_compatibility_reasons(config, runner_cls())
    )
    config.speculative_config.num_speculative_tokens = 2
    config.speculative_config.method = "eagle"
    assert any(
        "speculative decoding is unqualified" in reason
        for reason in _common_compatibility_reasons(config, runner_cls())
    )
    config.speculative_config.method = "mtp"

    config.cache_config.mamba_cache_mode = "all"
    assert "hybrid compression requires mamba_cache_mode='none' or 'align'" in (
        _common_compatibility_reasons(config, runner_cls())
    )


def test_scheduler_blocks_expand_to_consecutive_kernel_cache_blocks() -> None:
    actual = _expand_scheduler_block_ids((2, 0), 16, torch.device("cpu"))

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
    actual_k, actual_v = _unpack_layer_cache("layer", layer)
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

    assert _find_full_attention_group(groups) == 1


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

    provider.compress_scheduled_requests(_output_with_plan(300, (2, 0, 3), (50,), 44))

    assert provider.pending["r"] == PendingCompression(300, 128, (50,))
    assert provider.method.last_request is not None
    assert provider.method.last_request.physical_num_tokens == 300
    assert provider.method.last_request.source_block_ids == ((2, 0, 3),)
    assert provider.method.last_request.destination_block_ids == ((50,),)


def test_worker_uses_scheduler_private_destination_for_prefix_caching() -> None:
    provider = _provider()
    provider.vllm_config.cache_config.enable_prefix_caching = True
    request = SimpleNamespace(
        num_computed_tokens=256,
        num_prompt_tokens=300,
        max_tokens=128,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"), requests={"r": request}
    )
    provider.compress_scheduled_requests(_output_with_plan(300, (2, 0, 3), (100,), 44))

    assert provider.pending["r"] == PendingCompression(300, 128, (100,))
    assert provider.method.last_request is not None
    assert provider.method.last_request.source_block_ids == ((2, 0, 3),)
    assert provider.method.last_request.destination_block_ids == ((100,),)


def test_prefix_worker_skips_request_without_scheduler_transaction() -> None:
    provider = _provider()
    provider.vllm_config.cache_config.enable_prefix_caching = True
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"),
        requests={
            "r": SimpleNamespace(
                num_computed_tokens=256,
                num_prompt_tokens=300,
                max_tokens=128,
                block_ids=([2, 0, 3],),
            )
        },
    )
    output = SimpleNamespace(num_scheduled_tokens={"r": 44})

    provider.compress_scheduled_requests(output)

    assert not provider.pending
    assert provider.method.last_request is None


def test_speculative_worker_requires_scheduler_transaction() -> None:
    provider = _provider()
    provider.vllm_config.speculative_config = SimpleNamespace(num_speculative_tokens=2)
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"),
        requests={
            "r": SimpleNamespace(
                num_computed_tokens=256,
                num_prompt_tokens=300,
                max_tokens=128,
                block_ids=([2, 0, 3],),
            )
        },
    )

    provider.compress_scheduled_requests(
        SimpleNamespace(num_scheduled_tokens={"r": 44})
    )

    assert not provider.pending
    assert provider.method.last_request is None


def test_worker_commit_truncates_tables_and_activates_offsets() -> None:
    provider = _provider()
    block_table = _RecordingBlockTable()
    request = SimpleNamespace(block_ids=([2, 0, 3],))
    provider.runner = SimpleNamespace(
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}, block_table=block_table),
    )
    provider.pending["r"] = PendingCompression(300, 128, (50,))
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"r": 1},
    )

    provider.before_update_states(output)

    assert request.block_ids == ([50],)
    assert block_table.added == (([50],), 0)
    assert provider.active["r"] == ActiveCompression(300, 128)


def test_worker_commit_preserves_non_attention_group_tables() -> None:
    provider = _provider()
    provider.attention_group_index = 1
    block_table = _RecordingBlockTable()
    recurrent_state_blocks = [11]
    request = SimpleNamespace(block_ids=(recurrent_state_blocks, [2, 0, 3]))
    provider.runner = SimpleNamespace(
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}, block_table=block_table),
    )
    provider.pending["r"] = PendingCompression(300, 128, (50,))
    output = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"r": 1},
    )

    provider.before_update_states(output)

    assert request.block_ids == (recurrent_state_blocks, [50])
    assert block_table.added == ((recurrent_state_blocks, [50]), 0)


def test_worker_waits_for_request_reschedule_before_async_table_switch() -> None:
    provider = _provider()
    block_table = _RecordingBlockTable()
    request = SimpleNamespace(block_ids=([2, 0, 3],))
    provider.runner = SimpleNamespace(
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}, block_table=block_table),
    )
    provider.pending["r"] = PendingCompression(300, 128, (50,))

    unrelated = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"other": 1},
    )
    provider.before_update_states(unrelated)
    assert request.block_ids == ([2, 0, 3],)
    assert provider.pending["r"] == PendingCompression(300, 128, (50,))
    assert block_table.added is None

    provider.before_update_states(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            num_scheduled_tokens={"r": 1},
        )
    )
    assert request.block_ids == ([50],)
    assert "r" not in provider.pending


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
        num_scheduled_tokens={},
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
    provider._semantic_seq_lens_device = torch.empty(2, dtype=torch.int64)
    provider._semantic_seq_lens_cpu = torch.empty(2, dtype=torch.int64)
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
    provider.active["r"] = ActiveCompression(
        300, 128, (("model.layers.0.self_attn", 64),)
    )

    provider.compress_scheduled_requests(_output_with_plan(428, (5, 9), (50,), 1))

    assert provider.pending["r"] == PendingCompression(428, 128, (50,))
    assert provider.method.last_request is not None
    assert provider.method.last_request.physical_num_tokens == 256
    assert provider.method.last_request.per_layer_physical_num_tokens == (
        ("model.layers.0.self_attn", 192),
    )


def test_per_layer_result_commits_only_on_acknowledged_request_step() -> None:
    provider = _provider()
    provider.layer_caches += (
        LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),
    )
    provider.runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        device=torch.device("cpu"),
        requests={
            "r": SimpleNamespace(
                num_computed_tokens=256,
                num_prompt_tokens=300,
                max_tokens=128,
                block_ids=([2, 0, 3],),
            )
        },
        input_batch=SimpleNamespace(
            req_id_to_index={}, block_table=_RecordingBlockTable()
        ),
    )
    provider.method.compress = lambda request: CompressionResult(
        128,
        (("model.layers.1.self_attn", 64), ("model.layers.0.self_attn", 128)),
    )
    provider.compress_scheduled_requests(_output_with_plan(300, (2, 0, 3), (50,), 44))
    expected = (
        ("model.layers.0.self_attn", 128),
        ("model.layers.1.self_attn", 64),
    )
    assert provider.pending["r"].per_layer_physical_num_tokens == expected
    other_step = SimpleNamespace(
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        num_scheduled_tokens={"other": 1},
    )
    provider.before_update_states(other_step)
    assert "r" in provider.pending and "r" not in provider.active

    other_step.num_scheduled_tokens = {"r": 1}
    provider.before_update_states(other_step)
    assert provider.active["r"].per_layer_physical_num_tokens == expected
    assert "r" not in provider.pending

    other_step.num_scheduled_tokens = {}
    other_step.preempted_req_ids = {"r"}
    provider.before_update_states(other_step)
    assert "r" not in provider.active


@pytest.mark.parametrize("reset_kind", ["finished", "preempted", "resumed"])
def test_per_layer_state_clears_on_request_reset(reset_kind) -> None:
    provider = _provider()
    layers = (("model.layers.0.self_attn", 64),)
    provider.pending["r"] = PendingCompression(300, 128, (50,), layers)
    provider.active["r"] = ActiveCompression(300, 128, layers)
    provider.runner = SimpleNamespace(
        requests={"r": object()}, input_batch=SimpleNamespace(req_id_to_index={})
    )
    output = SimpleNamespace(
        finished_req_ids={"r"} if reset_kind == "finished" else set(),
        preempted_req_ids={"r"} if reset_kind == "preempted" else set(),
        scheduled_cached_reqs=SimpleNamespace(
            resumed_req_ids={"r"} if reset_kind == "resumed" else set()
        ),
        num_scheduled_tokens={},
    )
    provider.before_update_states(output)
    assert not provider.pending and not provider.active


def test_per_layer_state_clears_if_request_disappears_after_update() -> None:
    provider = _provider()
    layers = (("model.layers.0.self_attn", 64),)
    provider.pending["r"] = PendingCompression(300, 128, (50,), layers)
    provider.active["r"] = ActiveCompression(300, 128, layers)
    provider.runner = SimpleNamespace(requests={})
    provider.after_update_states(None)
    assert not provider.pending and not provider.active


@pytest.mark.parametrize(
    "lengths",
    [
        (),
        (("model.layers.0.self_attn", 128),),
        (("model.layers.0.self_attn", 128), ("model.layers.0.self_attn", 64)),
        (("model.layers.0.self_attn", 128), ("wrong", 64)),
        (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 0)),
        (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 129)),
        (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", True)),
    ],
)
def test_per_layer_result_rejects_invalid_maps(lengths) -> None:
    provider = _provider()
    provider.layer_caches += (
        LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),
    )
    with pytest.raises(RuntimeError, match="per-layer physical lengths"):
        provider._validate_per_layer_lengths(lengths, 128)


def test_uniform_per_layer_result_keeps_shared_metadata_path() -> None:
    provider = _provider()
    provider.layer_caches += (
        LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),
    )
    assert (
        provider._validate_per_layer_lengths(
            (("model.layers.1.self_attn", 128), ("model.layers.0.self_attn", 128)),
            128,
        )
        is None
    )


def test_per_layer_eager_metadata_uses_distinct_write_slots_and_lengths() -> None:
    @dataclass
    class AscendMetadata:
        seq_lens: torch.Tensor
        seq_lens_cpu: torch.Tensor
        seq_lens_list: list[int]
        slot_mapping: torch.Tensor

    AscendMetadata.__module__ = "vllm_ascend.attention.attention_v1"

    class BlockTable:
        def __init__(self):
            self.slot_mapping = SimpleNamespace(gpu=torch.tensor([12928, 2580]))

        def native(self, num_reqs, query_start_loc, positions):
            assert num_reqs == 2
            assert query_start_loc.tolist() == [0, 1, 2]
            # Two private 128-token pages for request 0, one for request 1.
            blocks = ((100, 101), (20,))
            for token, position in enumerate(positions.tolist()):
                request = token
                self.slot_mapping.gpu[token] = (
                    blocks[request][position // 128] * 128 + position % 128
                )

    BlockTable._ascend_kvcompress_patch_v3_slot_mapping_original = BlockTable.native
    table = BlockTable()
    provider = _provider()
    provider.layer_caches += (
        LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),
    )
    provider.active["compressed"] = ActiveCompression(
        300,
        128,
        (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 64)),
    )
    provider._has_per_layer_rows = True
    provider._physical_lengths_applied = True
    provider._request_offsets_cpu = torch.tensor([172, 0])
    provider.runner = SimpleNamespace(
        max_num_tokens=4,
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        input_batch=SimpleNamespace(
            req_ids=["compressed", "plain"],
            block_table=SimpleNamespace(block_tables=[table]),
        ),
        positions=torch.tensor([300, 20]),
        req_indices=SimpleNamespace(gpu=torch.tensor([0, 1])),
        query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 1, 2])),
    )
    shared = AscendMetadata(
        torch.tensor([129, 21]),
        torch.tensor([129, 21]),
        [129, 21],
        table.slot_mapping.gpu,
    )
    metadata, common = provider.per_layer_attention_metadata(
        ({layer.name: shared for layer in provider.layer_caches}, "common"), 2, 2
    )
    first, second = (metadata[layer.name] for layer in provider.layer_caches)
    assert common == "common"
    assert first is not shared
    assert first.seq_lens_list == [129, 21]
    first_slot_address = first.slot_mapping.data_ptr()
    second_slot_address = second.slot_mapping.data_ptr()
    assert second.seq_lens_list == [65, 21]
    assert second.seq_lens.tolist() == [65, 21]
    assert second.slot_mapping.tolist() == [100 * 128 + 64, 20 * 128 + 20]
    assert table.slot_mapping.gpu.tolist() == [12928, 2580]
    for slots in provider._per_layer_slot_buffers.values():
        assert slots[2:].tolist() == [-1, -1]
        slots[2:].fill_(777)

    provider.runner.positions[0] = 305
    table.slot_mapping.gpu[0] = 12933
    next_shared = AscendMetadata(
        torch.tensor([134, 21]),
        torch.tensor([134, 21]),
        [134, 21],
        table.slot_mapping.gpu,
    )
    next_metadata, _ = provider.per_layer_attention_metadata(
        ({layer.name: next_shared for layer in provider.layer_caches}, None), 2, 2
    )
    assert (
        next_metadata["model.layers.0.self_attn"].slot_mapping.data_ptr()
        == first_slot_address
    )
    assert (
        next_metadata["model.layers.1.self_attn"].slot_mapping.data_ptr()
        == second_slot_address
    )
    assert next_metadata["model.layers.1.self_attn"].seq_lens_list == [70, 21]
    assert next_metadata["model.layers.1.self_attn"].slot_mapping.tolist() == [
        100 * 128 + 69,
        20 * 128 + 20,
    ]
    assert table.slot_mapping.gpu.tolist() == [12933, 2580]
    for slots in provider._per_layer_slot_buffers.values():
        assert slots[2:].tolist() == [-1, -1]


def test_undeclared_nonuniform_lengths_fail_closed_under_graph_replay() -> None:
    provider = _provider()
    provider.layer_caches += (
        LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),
    )
    provider.runner = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE")
        )
    )
    with pytest.raises(RuntimeError, match="declare per-layer physical state"):
        provider._validate_per_layer_lengths(
            (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 64)),
            128,
        )


def test_declared_per_layer_state_accepts_layer_aware_full_graph(
    monkeypatch,
) -> None:
    class AscendAttentionBackendImpl:
        _ascend_kvcompress_layer_aware_graph_v1 = True

    ascend = ModuleType("vllm_ascend")
    attention = ModuleType("vllm_ascend.attention")
    attention_v1 = ModuleType("vllm_ascend.attention.attention_v1")
    attention_v1.AscendAttentionBackendImpl = AscendAttentionBackendImpl
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention", attention)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.attention_v1", attention_v1)

    provider = _provider()
    provider.requires_per_layer_physical_state = True
    provider.runner = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE")
        ),
        use_dcp=False,
        pcp_enabled=False,
        use_async_spec_decode=True,
    )
    provider.vllm_config.speculative_config = None

    provider._validate_per_layer_host()


def test_per_layer_full_graph_capture_accepts_synthetic_empty_batch() -> None:
    @dataclass
    class Metadata:
        slot_mapping: torch.Tensor

    provider = _provider()
    provider.requires_per_layer_physical_state = True
    provider._validate_per_layer_host = lambda: None  # type: ignore[method-assign]
    provider.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=[], num_reqs=0),
    )
    provider.layer_caches += (
        LayerCache("mtp.layers.0.self_attn", 0, torch.empty(0), torch.empty(0)),
    )
    provider.speculative_cache_layer = provider.layer_caches[-1]
    provider._per_layer_slot_buffers = {
        layer.name: torch.empty(48, dtype=torch.int64)
        for layer in provider.layer_caches
    }
    shared = Metadata(torch.arange(48))
    metadata, common = provider.per_layer_attention_metadata(
        ({layer.name: shared for layer in provider.layer_caches}, shared), 48, 48
    )
    captured_slots = [
        metadata[layer.name].slot_mapping for layer in provider.layer_caches
    ]
    assert all(
        slots.data_ptr() != shared.slot_mapping.data_ptr() for slots in captured_slots
    )
    assert captured_slots[0].data_ptr() != captured_slots[1].data_ptr()
    assert common.slot_mapping.data_ptr() == captured_slots[1].data_ptr()
    # A captured cache-write kernel keeps this address: replay must update
    # its contents independently for unequal physical layer lengths.
    for index, layer in enumerate(provider.layer_caches):
        provider._per_layer_slot_buffers[layer.name][0] = 64 + index
    assert [int(slots[0]) for slots in captured_slots] == [64, 65]
    assert int(shared.slot_mapping[0]) == 0
    assert provider._per_layer_metadata_seen


def test_per_layer_full_graph_capture_rejects_partial_real_batch() -> None:
    provider = _provider()
    provider.requires_per_layer_physical_state = True
    provider._validate_per_layer_host = lambda: None  # type: ignore[method-assign]
    provider.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["real"], num_reqs=1),
    )

    with pytest.raises(RuntimeError, match="batch dimensions changed"):
        provider.per_layer_attention_metadata(
            ({"model.layers.0.self_attn": object()}, object()),
            48,
            48,
        )


def test_query_observation_can_exclude_auxiliary_mtp_cache() -> None:
    provider = _provider()
    provider.layer_caches += (
        LayerCache("mtp.layers.0.self_attn.attn", 0, torch.empty(0), torch.empty(0)),
        LayerCache("model.layers.3.self_attn", 3, torch.empty(0), torch.empty(0)),
    )
    provider.method = SimpleNamespace(query_layer_indices=(3,))
    provider.query_window_tokens = 2

    selected = provider._resolve_query_layer_caches()

    assert tuple(cache.name for cache in selected) == ("model.layers.3.self_attn",)


def test_mtp_rejected_draft_reuses_layer_specific_physical_slots() -> None:
    @dataclass
    class AscendMetadata:
        seq_lens: torch.Tensor
        seq_lens_cpu: torch.Tensor
        seq_lens_list: list[int]
        slot_mapping: torch.Tensor

    AscendMetadata.__module__ = "vllm_ascend.attention.attention_v1"

    @dataclass
    class CommonMetadata:
        seq_lens: torch.Tensor
        _seq_lens_cpu: torch.Tensor
        seq_lens_cpu_upper_bound: torch.Tensor
        slot_mapping: torch.Tensor
        seq_lens_cpu: torch.Tensor | None = None
        num_computed_tokens_cpu: torch.Tensor | None = None

    class BlockTable:
        def __init__(self) -> None:
            self.slot_mapping = SimpleNamespace(
                gpu=torch.tensor([128, 129, 130], dtype=torch.int64)
            )

        def native(self, num_reqs, query_start_loc, positions):
            assert num_reqs == 1
            assert int(query_start_loc[-1]) == len(positions)
            for token, position in enumerate(positions.tolist()):
                self.slot_mapping.gpu[token] = position

    BlockTable._ascend_kvcompress_patch_v3_slot_mapping_original = BlockTable.native
    table = BlockTable()
    provider = _provider()
    provider.layer_caches += (
        LayerCache("model.layers.1.self_attn", 1, torch.empty(0), torch.empty(0)),
    )
    provider.active["r"] = ActiveCompression(
        300,
        128,
        (("model.layers.0.self_attn", 128), ("model.layers.1.self_attn", 64)),
    )
    provider._has_per_layer_rows = True
    provider._physical_lengths_applied = True
    provider._request_offsets_cpu = torch.tensor([172])
    provider.speculative_cache_layer = provider.layer_caches[1]
    provider.runner = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        input_batch=SimpleNamespace(
            req_ids=["r"],
            block_table=SimpleNamespace(block_tables=[table]),
        ),
        positions=torch.tensor([300, 301, 302]),
        req_indices=SimpleNamespace(gpu=torch.tensor([0, 0, 0])),
        query_start_loc=SimpleNamespace(gpu=torch.tensor([0, 3])),
    )
    first = AscendMetadata(
        torch.tensor([131]),
        torch.tensor([131]),
        [131],
        table.slot_mapping.gpu,
    )
    first_common = CommonMetadata(
        torch.tensor([131]),
        torch.tensor([131]),
        torch.tensor([131]),
        table.slot_mapping.gpu,
        torch.tensor([131]),
        torch.tensor([300]),
    )
    metadata, common = provider.per_layer_attention_metadata(
        ({layer.name: first for layer in provider.layer_caches}, first_common),
        3,
        1,
    )
    assert metadata["model.layers.1.self_attn"].slot_mapping.tolist() == [64, 65, 66]
    assert common.seq_lens.tolist() == [67]
    assert common._seq_lens_cpu.tolist() == [67]
    assert common.num_computed_tokens_cpu.tolist() == [64]
    assert common.slot_mapping.tolist() == [64, 65, 66]

    # Only the first speculative token was accepted. The next iteration writes
    # semantic position 301 again and must overwrite the rejected draft slot.
    provider.runner.positions = torch.tensor([301])
    provider.runner.req_indices.gpu = torch.tensor([0])
    provider.runner.query_start_loc.gpu = torch.tensor([0, 1])
    table.slot_mapping.gpu[0] = 129
    second = AscendMetadata(
        torch.tensor([130]),
        torch.tensor([130]),
        [130],
        table.slot_mapping.gpu[:1],
    )
    second_common = CommonMetadata(
        torch.tensor([130]),
        torch.tensor([130]),
        torch.tensor([130]),
        table.slot_mapping.gpu[:1],
        torch.tensor([130]),
        torch.tensor([301]),
    )
    metadata, common = provider.per_layer_attention_metadata(
        ({layer.name: second for layer in provider.layer_caches}, second_common),
        1,
        1,
    )
    assert metadata["model.layers.1.self_attn"].slot_mapping.tolist() == [65]
    assert metadata["model.layers.1.self_attn"].seq_lens_list == [66]
    assert common.seq_lens.tolist() == [66]
    assert common.num_computed_tokens_cpu.tolist() == [65]
    assert common.slot_mapping.tolist() == [65]


def test_per_layer_metadata_bypass_fails_closed() -> None:
    provider = _provider()
    provider._has_per_layer_rows = True
    provider.begin_per_layer_step(SimpleNamespace(num_scheduled_tokens={"r": 1}))
    with pytest.raises(RuntimeError, match="metadata hook was bypassed"):
        provider.finish_per_layer_step()


def test_per_layer_metadata_allows_async_empty_execution_step() -> None:
    provider = _provider()
    provider._has_per_layer_rows = True

    provider.begin_per_layer_step(SimpleNamespace(num_scheduled_tokens={}))
    provider.finish_per_layer_step()

    assert not provider._per_layer_step_requires_metadata


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


class _ObservingMethod(_RecordingMethod):
    def __init__(self) -> None:
        super().__init__()
        self.buffers: dict[str, torch.Tensor] = {}
        self.counts: dict[str, int] = {}
        self.observation: QueryObservation | None = None
        self.observed_rows: torch.Tensor | None = None
        self.discarded: list[str] = []

    def capture_query(self, layer, query, spans) -> None:
        assert layer.layer_index == 0
        for span in spans:
            buffer = self.buffers.setdefault(span.request_id, torch.empty(2, 2))
            rows = query[span.start : span.end]
            count = self.counts.get(span.request_id, 0)
            if count < 2:
                copy = min(2 - count, len(rows))
                buffer[count : count + copy].copy_(rows[:copy])
                count += copy
                rows = rows[copy:]
            for row in rows:
                buffer[:-1].copy_(buffer[1:].clone())
                buffer[-1].copy_(row)
            self.counts[span.request_id] = count

    def complete_query_observation(self, observation: QueryObservation) -> None:
        assert observation.request_id == "r"
        assert observation.window_tokens == 2
        assert observation.layer_indices == (0,)
        assert self.counts["r"] == 2
        self.observation = observation
        self.observed_rows = self.buffers["r"].clone()

    def compress(self, request: CompressionRequest) -> CompressionResult:
        assert self.observation is not None
        assert request.plan is self.observation.plan
        return super().compress(request)

    def discard_query_observation(self, request_id: str) -> None:
        self.discarded.append(request_id)
        self.buffers.pop(request_id, None)
        self.counts.pop(request_id, None)


def _observing_provider() -> AscendKVCompressionProvider:
    provider = _provider()
    provider.method = _ObservingMethod()
    provider.query_window_tokens = 2
    provider.query_layer_caches = provider.layer_caches
    request = SimpleNamespace(
        num_computed_tokens=298,
        num_prompt_tokens=300,
        max_tokens=128,
        block_ids=([2, 0, 3],),
    )
    provider.runner = SimpleNamespace(
        device=torch.device("cpu"),
        requests={"r": request},
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}),
        query_start_loc=SimpleNamespace(cpu=torch.tensor([0, 1])),
    )
    return provider


def test_query_observation_spans_chunked_prefill_and_matches_plan() -> None:
    provider = _observing_provider()
    first = SimpleNamespace(num_scheduled_tokens={"r": 1})

    provider.begin_query_step(first)
    provider.capture_attention_query(
        "model.layers.0.self_attn", torch.tensor([[1.0, 2.0]])
    )
    provider.finish_query_step()
    stable_address = provider.method.buffers["r"].data_ptr()

    provider.runner.requests["r"].num_computed_tokens = 299
    final = _output_with_plan(300, (2, 0, 3), (50,), 1)
    provider.begin_query_step(final)
    provider.capture_attention_query(
        "model.layers.0.self_attn", torch.tensor([[3.0, 4.0]])
    )
    provider.finish_query_step()
    assert provider.method.buffers["r"].data_ptr() == stable_address
    provider.compress_scheduled_requests(final)

    torch.testing.assert_close(
        provider.method.observed_rows,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    assert provider.method.observation.plan is getattr(final, PLAN_ATTRIBUTE)["r"]
    assert provider.method.discarded == ["r"]
    assert provider.method.buffers == {}


def test_query_observation_accepts_attention_backend_callback() -> None:
    provider = _observing_provider()
    output = SimpleNamespace(num_scheduled_tokens={"r": 1})

    provider.begin_query_step(output)
    provider.capture_attention_query(
        "model.layers.0.self_attn", torch.tensor([[1.0, 2.0]])
    )
    provider.finish_query_step()

    torch.testing.assert_close(
        provider.method.buffers["r"][0], torch.tensor([1.0, 2.0])
    )
    assert provider.method.counts["r"] == 1


def test_query_observation_fails_closed_when_forward_hook_is_bypassed() -> None:
    provider = _observing_provider()
    output = SimpleNamespace(num_scheduled_tokens={"r": 1})

    provider.begin_query_step(output)
    with pytest.raises(RuntimeError, match="graph replay path is unvalidated"):
        provider.finish_query_step()
    provider.abort_query_step(output)

    assert not provider._query_forward_complete
    assert provider.method.discarded == ["r"]


def test_query_observation_is_discarded_on_preemption() -> None:
    provider = _observing_provider()
    output = SimpleNamespace(num_scheduled_tokens={"r": 1})
    provider.begin_query_step(output)
    provider.capture_attention_query(
        "model.layers.0.self_attn", torch.tensor([[1.0, 2.0]])
    )
    provider.finish_query_step()

    provider.before_update_states(
        SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids={"r"},
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            num_scheduled_tokens={},
        )
    )

    assert provider.method.buffers == {}
    assert provider.method.discarded == ["r"]


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
