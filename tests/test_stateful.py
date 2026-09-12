# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_ascend_kvcompress.config import ProviderSelection
from vllm_ascend_kvcompress.stateful import (
    SchedulerActiveCompression,
    SchedulerCompressionState,
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


def _scheduler(*, scheduler_module="vllm.v1.core.sched.scheduler", balance=False):
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        speculative_config=None,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        parallel_config=parallel,
    )
    blocks = [SimpleNamespace(block_id=value) for value in (3, 4, 5)]
    single_manager = SimpleNamespace(
        req_to_blocks={"r": blocks}, num_cached_block={"r": 0}
    )
    freed = []
    manager = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=(single_manager,)),
        block_pool=SimpleNamespace(free_blocks=lambda values: freed.extend(values)),
    )
    request = SimpleNamespace(
        request_id="r", num_computed_tokens=300, num_prompt_tokens=300, max_tokens=128
    )
    scheduler_name = (
        "BalanceScheduler" if "vllm_ascend" in scheduler_module else "Scheduler"
    )
    scheduler_type = type(scheduler_name, (SimpleNamespace,), {})
    scheduler_type.__module__ = scheduler_module
    scheduler = scheduler_type(
        vllm_config=vllm_config,
        block_size=128,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
        kv_cache_manager=manager,
        requests={"r": request},
        finished_req_ids=set(),
        reset_preempted_req_ids=set(),
        _balance_enabled=balance,
    )
    return scheduler, single_manager, freed


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
