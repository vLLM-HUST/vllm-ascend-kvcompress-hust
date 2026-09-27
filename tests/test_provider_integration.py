# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
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

    assert _allowed_scheduler_block_sizes(config) == frozenset({128, 2048})


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
    provider.active["r"] = ActiveCompression(300, 128)

    provider.compress_scheduled_requests(_output_with_plan(428, (5, 9), (50,), 1))

    assert provider.pending["r"] == PendingCompression(428, 128, (50,))
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
