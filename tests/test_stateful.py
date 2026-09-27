# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)

from vllm_ascend_kvcompress.config import ProviderSelection
from vllm_ascend_kvcompress.provider import SCHEDULER_TRANSACTIONS_ATTRIBUTE
from vllm_ascend_kvcompress.stateful import (
    _GROUP_LENGTH_STATE_ATTRIBUTE,
    SchedulerActiveCompression,
    SchedulerCompressionState,
    SchedulerDeferredRelease,
    _install_coordinator_hooks,
    _install_manager_hooks,
)


def _selection() -> ProviderSelection:
    return ProviderSelection.from_mapping(
        {
            "method": "triattention",
            "method_config": {
                "stats_path": "/tmp/stats.pt",
                "kv_budget": 128,
                "recompute_window": 128,
                "score_chunk_size": 128,
            },
        }
    )


def _scheduler(
    *,
    scheduler_module="vllm.v1.core.sched.scheduler",
    balance=False,
    block_size=128,
    hybrid_model_type=None,
    attention_block_size=None,
    async_scheduling=False,
    max_concurrent_batches=None,
    prefix_caching=False,
    allocation_fails=False,
    speculative=False,
    mamba_cache_mode="none",
):
    if async_scheduling and scheduler_module == "vllm.v1.core.sched.scheduler":
        scheduler_module = "vllm.v1.core.sched.async_scheduler"
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=prefix_caching,
            mamba_cache_mode=mamba_cache_mode,
        ),
        model_config=SimpleNamespace(
            is_hybrid=hybrid_model_type is not None,
            hf_text_config=SimpleNamespace(model_type=hybrid_model_type),
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=2)
        if speculative
        else None,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        parallel_config=parallel,
    )
    if max_concurrent_batches is not None:
        vllm_config.max_concurrent_batches = max_concurrent_batches
    blocks = [SimpleNamespace(block_id=value) for value in (3, 4, 5)]
    if attention_block_size is None:
        attention_block_size = block_size
    attention_group = KVCacheGroupSpec(
        layer_names=["model.layers.0.self_attn"],
        kv_cache_spec=FullAttentionSpec(
            block_size=attention_block_size,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.float16,
        ),
    )
    single_manager = SimpleNamespace(
        block_size=attention_block_size,
        req_to_blocks={"r": blocks},
        num_cached_block={"r": 0},
    )
    groups = [attention_group]
    single_managers = [single_manager]
    if hybrid_model_type is not None:
        groups.append(
            KVCacheGroupSpec(
                layer_names=["model.layers.1.linear_attn"],
                kv_cache_spec=MambaSpec(
                    block_size=32768,
                    shapes=((1,),),
                    dtypes=(torch.float16,),
                    mamba_cache_mode=mamba_cache_mode,
                ),
            )
        )
        single_managers.append(
            SimpleNamespace(block_size=32768, req_to_blocks={}, num_cached_block={})
        )
    freed = []
    next_block_id = 100

    def get_new_blocks(count):
        if allocation_fails:
            raise ValueError("insufficient free blocks")
        return [
            SimpleNamespace(block_id=next_block_id + value, ref_cnt=1, block_hash=None)
            for value in range(count)
        ]

    class Coordinator(SimpleNamespace):
        def get_num_blocks_to_allocate(self, *args, **kwargs):
            return 0

        def allocate_new_blocks(self, *args, **kwargs):
            return ()

        def remove_skipped_blocks(self, *args, **kwargs):
            return None

        def cache_blocks(self, *args, **kwargs):
            return None

    coordinator = Coordinator(
        single_type_managers=tuple(single_managers), retention_interval=None
    )
    manager = SimpleNamespace(
        coordinator=coordinator,
        block_pool=SimpleNamespace(
            free_blocks=lambda values: freed.extend(values),
            get_new_blocks=get_new_blocks,
        ),
    )
    request = SimpleNamespace(
        request_id="r", num_computed_tokens=300, num_prompt_tokens=300, max_tokens=128
    )
    if "vllm_ascend" in scheduler_module:
        scheduler_name = "BalanceScheduler"
    elif scheduler_module.endswith("async_scheduler"):
        scheduler_name = "AsyncScheduler"
    else:
        scheduler_name = "Scheduler"
    scheduler_type = type(scheduler_name, (SimpleNamespace,), {})
    scheduler_type.__module__ = scheduler_module
    scheduler = scheduler_type(
        vllm_config=vllm_config,
        block_size=block_size,
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        kv_cache_manager=manager,
        requests={"r": request},
        finished_req_ids=set(),
        reset_preempted_req_ids=set(),
        _balance_enabled=balance,
    )
    return scheduler, single_manager, freed


def test_qwen35_scheduler_separates_alignment_from_attention_pages() -> None:
    scheduler, _, _ = _scheduler(
        block_size=32768,
        hybrid_model_type="qwen3_5_moe_text",
        attention_block_size=2048,
    )
    selection = ProviderSelection.from_mapping(
        {
            "method": "triattention",
            "method_config": {
                "stats_path": "/tmp/stats.pt",
                "kv_budget": 4096,
                "recompute_window": 128,
                "score_chunk_size": 128,
            },
        }
    )

    state = SchedulerCompressionState(scheduler, selection)

    assert state.scheduler_alignment_size == 32768
    assert state.scheduler_block_size == 2048


def test_qwen35_scheduler_rejects_wrong_group_lcm_alignment() -> None:
    scheduler, _, _ = _scheduler(
        block_size=16384,
        hybrid_model_type="qwen3_5_moe_text",
        attention_block_size=2048,
    )
    selection = ProviderSelection.from_mapping(
        {
            "method": "triattention",
            "method_config": {
                "stats_path": "/tmp/stats.pt",
                "kv_budget": 4096,
                "recompute_window": 128,
                "score_chunk_size": 128,
            },
        }
    )

    try:
        SchedulerCompressionState(scheduler, selection)
    except RuntimeError as error:
        assert "LCM of cache-group block sizes" in str(error)
    else:
        raise AssertionError("mismatched scheduler alignment should be rejected")


def test_qwen35_align_installs_group_specific_length_translation() -> None:
    scheduler, _, _ = _scheduler(
        block_size=32768,
        hybrid_model_type="qwen3_5_moe_text",
        attention_block_size=2048,
        mamba_cache_mode="align",
    )
    selection = ProviderSelection.from_mapping(
        {
            "method": "triattention",
            "method_config": {
                "stats_path": "/tmp/stats.pt",
                "kv_budget": 4096,
                "recompute_window": 128,
                "score_chunk_size": 128,
            },
        }
    )

    state = SchedulerCompressionState(scheduler, selection)

    assert state.group_specific_lengths
    assert (
        getattr(scheduler.kv_cache_manager.coordinator, _GROUP_LENGTH_STATE_ATTRIBUTE)
        is state
    )


def test_scheduler_arms_then_commits_tail_after_barrier() -> None:
    scheduler, manager, freed = _scheduler()
    state = SchedulerCompressionState(scheduler, _selection())
    output = SimpleNamespace(
        num_scheduled_tokens={"r": 44},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )

    state.after_schedule(output)
    state.before_schedule()

    assert [block.block_id for block in manager.req_to_blocks["r"]] == [3]
    assert [block.block_id for block in freed] == [5, 4]
    assert state.active["r"] == SchedulerActiveCompression(300, 128)


def test_async_scheduler_commits_on_output_and_defers_tail_release() -> None:
    scheduler, manager, freed = _scheduler(
        scheduler_module="vllm.v1.core.sched.async_scheduler",
        async_scheduling=True,
    )
    state = SchedulerCompressionState(scheduler, _selection())
    output = SimpleNamespace(
        num_scheduled_tokens={"r": 44},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )

    state.after_schedule(output)
    state.before_schedule()

    assert [block.block_id for block in manager.req_to_blocks["r"]] == [3, 4, 5]
    assert not freed

    state.after_model_output()

    assert [block.block_id for block in manager.req_to_blocks["r"]] == [3]
    assert state.active["r"] == SchedulerActiveCompression(300, 128)
    assert not freed

    state.after_model_output()

    assert [block.block_id for block in freed] == [5, 4]


def test_async_scheduler_honors_larger_batch_queue_before_release() -> None:
    scheduler, _, freed = _scheduler(async_scheduling=True, max_concurrent_batches=3)
    state = SchedulerCompressionState(scheduler, _selection())
    state.after_schedule(
        SimpleNamespace(
            num_scheduled_tokens={"r": 44},
            finished_req_ids=set(),
            preempted_req_ids=set(),
        )
    )

    state.after_model_output()
    state.after_model_output()
    assert not freed

    state.after_model_output()
    assert [block.block_id for block in freed] == [5, 4]


def test_prefix_caching_uses_private_destination_blocks() -> None:
    scheduler, manager, freed = _scheduler(prefix_caching=True)
    state = SchedulerCompressionState(scheduler, _selection())
    output = SimpleNamespace(
        num_scheduled_tokens={"r": 44},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )

    state.after_schedule(output)

    assert getattr(output, SCHEDULER_TRANSACTIONS_ATTRIBUTE) == {"r": (100,)}
    assert [block.block_id for block in manager.req_to_blocks["r"]] == [3, 4, 5]

    state.before_schedule()

    assert [block.block_id for block in manager.req_to_blocks["r"]] == [100]
    assert [block.block_id for block in freed] == [5, 4, 3]
    assert manager.num_cached_block["r"] == 0


def test_prefix_caching_skips_compression_when_private_allocation_fails() -> None:
    scheduler, manager, freed = _scheduler(prefix_caching=True, allocation_fails=True)
    state = SchedulerCompressionState(scheduler, _selection())
    output = SimpleNamespace(
        num_scheduled_tokens={"r": 44},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )

    state.after_schedule(output)

    assert getattr(output, SCHEDULER_TRANSACTIONS_ATTRIBUTE) == {}
    assert not state.pending
    assert [block.block_id for block in manager.req_to_blocks["r"]] == [3, 4, 5]
    assert not freed


def test_speculative_step_does_not_arm_compression() -> None:
    scheduler, manager, freed = _scheduler(speculative=True)
    state = SchedulerCompressionState(scheduler, _selection())
    output = SimpleNamespace(
        num_scheduled_tokens={"r": 44},
        scheduled_spec_decode_tokens={"r": [17, 18]},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )

    state.after_schedule(output)

    assert getattr(output, SCHEDULER_TRANSACTIONS_ATTRIBUTE) == {}
    assert not state.pending
    assert [block.block_id for block in manager.req_to_blocks["r"]] == [3, 4, 5]
    assert not freed


def test_mtp_launch_can_arm_on_non_speculative_prefill_step() -> None:
    scheduler, _, _ = _scheduler(speculative=True)
    state = SchedulerCompressionState(scheduler, _selection())
    output = SimpleNamespace(
        num_scheduled_tokens={"r": 44},
        scheduled_spec_decode_tokens={},
        finished_req_ids=set(),
        preempted_req_ids=set(),
    )

    state.after_schedule(output)

    assert getattr(output, SCHEDULER_TRANSACTIONS_ATTRIBUTE) == {"r": (3,)}
    assert "r" in state.pending


def test_prefix_caching_preemption_releases_uncommitted_destination() -> None:
    scheduler, _, freed = _scheduler(prefix_caching=True)
    state = SchedulerCompressionState(scheduler, _selection())
    state.after_schedule(
        SimpleNamespace(
            num_scheduled_tokens={"r": 44},
            finished_req_ids=set(),
            preempted_req_ids=set(),
        )
    )

    state.after_schedule(
        SimpleNamespace(
            num_scheduled_tokens={},
            finished_req_ids=set(),
            preempted_req_ids={"r"},
        )
    )

    assert not state.pending
    assert [block.block_id for block in freed] == [100]


def test_async_prefix_commit_releases_source_growth_after_output_barrier() -> None:
    scheduler, manager, freed = _scheduler(prefix_caching=True, async_scheduling=True)
    state = SchedulerCompressionState(scheduler, _selection())
    state.after_schedule(
        SimpleNamespace(
            num_scheduled_tokens={"r": 44},
            finished_req_ids=set(),
            preempted_req_ids=set(),
        )
    )
    manager.req_to_blocks["r"].append(SimpleNamespace(block_id=6))

    state.after_model_output()

    assert [block.block_id for block in manager.req_to_blocks["r"]] == [100]
    assert not freed

    state.after_model_output()

    assert [block.block_id for block in freed] == [6, 5, 4, 3]


def test_async_finished_request_discards_unacknowledged_transaction() -> None:
    scheduler, manager, freed = _scheduler(async_scheduling=True)
    state = SchedulerCompressionState(scheduler, _selection())
    state.after_schedule(
        SimpleNamespace(
            num_scheduled_tokens={"r": 44},
            finished_req_ids=set(),
            preempted_req_ids=set(),
        )
    )
    scheduler.requests.clear()

    state.after_model_output()

    assert not state.pending
    assert [block.block_id for block in manager.req_to_blocks["r"]] == [3, 4, 5]
    assert not freed


def test_manager_allocation_temporarily_uses_physical_length() -> None:
    class Manager:
        def allocate_slots(self, request):
            self.seen = request.num_computed_tokens
            return "allocated"

        def free(self, request):
            return request.request_id

    _install_manager_hooks(Manager)
    manager = Manager()
    request = SimpleNamespace(request_id="r", num_computed_tokens=300)
    manager._ascend_kvcompress_active_offsets_v3 = {
        "r": SchedulerActiveCompression(300, 128)
    }

    result = manager.allocate_slots(request)

    assert result == "allocated"
    assert manager.seen == 128
    assert request.num_computed_tokens == 300


def test_manager_free_releases_async_tail_on_abort() -> None:
    freed = []

    class Manager:
        block_pool = SimpleNamespace(free_blocks=lambda values: freed.extend(values))

        def allocate_slots(self, request):
            return None

        def free(self, request):
            return request.request_id

    _install_manager_hooks(Manager)
    manager = Manager()
    request = SimpleNamespace(request_id="r")
    tail = (SimpleNamespace(block_id=5), SimpleNamespace(block_id=4))
    manager._ascend_kvcompress_deferred_releases_v3 = {
        "r": [
            SchedulerDeferredRelease(request_id="r", blocks=tail, remaining_outputs=1)
        ]
    }

    assert manager.free(request) == "r"
    assert [block.block_id for block in freed] == [5, 4]
    assert not manager._ascend_kvcompress_deferred_releases_v3


def test_hybrid_coordinator_translates_only_full_attention_lengths() -> None:
    class RecordingManager:
        def __init__(self):
            self.count_calls = []
            self.allocate_calls = []
            self.remove_calls = []
            self.cache_calls = []

        def get_num_blocks_to_allocate(
            self,
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=False,
        ):
            self.count_calls.append(
                (
                    request_id,
                    num_tokens,
                    total_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap,
                )
            )
            return 1

        def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
            self.allocate_calls.append((request_id, num_tokens, num_tokens_main_model))
            return []

        def remove_skipped_blocks(
            self, request_id, total_computed_tokens, num_prompt_tokens=None
        ):
            self.remove_calls.append(
                (request_id, total_computed_tokens, num_prompt_tokens)
            )

        def cache_blocks(self, request, num_computed_tokens, retention_interval=None):
            self.cache_calls.append(
                (request.request_id, num_computed_tokens, retention_interval)
            )

    class Coordinator:
        def __init__(self):
            self.single_type_managers = (RecordingManager(), RecordingManager())
            self.retention_interval = 64

        def get_num_blocks_to_allocate(self, *args, **kwargs):
            raise AssertionError("active request must use group translation")

        def allocate_new_blocks(self, *args, **kwargs):
            raise AssertionError("active request must use group translation")

        def remove_skipped_blocks(self, *args, **kwargs):
            raise AssertionError("active request must use group translation")

        def cache_blocks(self, *args, **kwargs):
            raise AssertionError("active request must use group translation")

    _install_coordinator_hooks(Coordinator)
    coordinator = Coordinator()
    setattr(
        coordinator,
        _GROUP_LENGTH_STATE_ATTRIBUTE,
        SimpleNamespace(
            attention_group_index=0,
            active={"r": SchedulerActiveCompression(300, 128)},
        ),
    )

    assert (
        coordinator.get_num_blocks_to_allocate(
            "r", 305, ((), ()), 0, 300, 305, apply_admission_cap=True
        )
        == 2
    )
    coordinator.allocate_new_blocks("r", 305, 305)
    coordinator.remove_skipped_blocks("r", 300, 300)
    coordinator.cache_blocks(SimpleNamespace(request_id="r"), 300)

    attention, mamba = coordinator.single_type_managers
    assert attention.count_calls == [("r", 133, 128, 133, True)]
    assert mamba.count_calls == [("r", 305, 300, 305, True)]
    assert attention.allocate_calls == [("r", 133, 133)]
    assert mamba.allocate_calls == [("r", 305, 305)]
    assert attention.remove_calls == [("r", 128, 300)]
    assert mamba.remove_calls == [("r", 300, 300)]
    assert not attention.cache_calls
    assert mamba.cache_calls == [("r", 300, 64)]


def test_finished_request_discards_pending_state() -> None:
    scheduler, _, _ = _scheduler()
    state = SchedulerCompressionState(scheduler, _selection())
    state.after_schedule(
        SimpleNamespace(
            num_scheduled_tokens={"r": 44},
            finished_req_ids=set(),
            preempted_req_ids=set(),
        )
    )
    scheduler.finished_req_ids.add("r")

    state.before_schedule()

    assert not state.pending


def test_short_decode_request_is_not_armed_for_compression() -> None:
    scheduler, _, _ = _scheduler()
    selection = ProviderSelection.from_mapping(
        {
            "method": "triattention",
            "method_config": {
                "stats_path": "/tmp/stats.pt",
                "kv_budget": 128,
                "recompute_window": 128,
                "score_chunk_size": 128,
                "min_output_tokens_for_compression": 64,
            },
        }
    )
    scheduler.requests["r"].max_tokens = 32
    state = SchedulerCompressionState(scheduler, selection)

    state.after_schedule(
        SimpleNamespace(
            num_scheduled_tokens={"r": 44},
            finished_req_ids=set(),
            preempted_req_ids=set(),
        )
    )

    assert not state.pending


def test_preempted_request_discards_scheduler_state() -> None:
    scheduler, _, _ = _scheduler()
    state = SchedulerCompressionState(scheduler, _selection())
    state.active["r"] = SchedulerActiveCompression(300, 128)

    state.after_schedule(
        SimpleNamespace(
            num_scheduled_tokens={},
            finished_req_ids=set(),
            preempted_req_ids={"r"},
        )
    )

    assert not state.active


def test_current_ascend_balance_scheduler_wrapper_is_supported_when_inert() -> None:
    scheduler, _, _ = _scheduler(
        scheduler_module="vllm_ascend.patch.platform.patch_balance_schedule"
    )

    SchedulerCompressionState(scheduler, _selection())


def test_active_ascend_balance_scheduling_is_rejected() -> None:
    scheduler, _, _ = _scheduler(
        scheduler_module="vllm_ascend.patch.platform.patch_balance_schedule",
        balance=True,
    )

    try:
        SchedulerCompressionState(scheduler, _selection())
    except RuntimeError as error:
        assert "balance scheduling must be disabled" in str(error)
    else:
        raise AssertionError("active balance scheduling should be rejected")
