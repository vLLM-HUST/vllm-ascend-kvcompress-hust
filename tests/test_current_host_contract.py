# SPDX-License-Identifier: Apache-2.0
"""Contract checks against the installed upstream-aligned host packages."""

from __future__ import annotations

import inspect
from importlib.metadata import version

import pytest
from packaging.specifiers import SpecifierSet


def _parameter_names(callable_object: object) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_object).parameters)


def test_installed_host_versions_are_in_the_validated_lines() -> None:
    assert version("vllm") in SpecifierSet(">=0.28.1.post1.dev0,<0.29")
    assert version("vllm-ascend") in SpecifierSet(">=0.25.1rc2.dev0,<0.26")
    assert version("vllm-hust-ext") in SpecifierSet(">=0.2.0.dev0,<0.3")


def test_scheduler_and_cache_manager_seams_match_current_host() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler

    assert _parameter_names(Scheduler.schedule) == ("self", "throttle_prefills")
    assert _parameter_names(KVCacheManager.allocate_slots)[:3] == (
        "self",
        "request",
        "num_new_tokens",
    )
    assert _parameter_names(KVCacheManager.free) == ("self", "request")


def test_ascend_v1_worker_seams_match_current_host() -> None:
    try:
        from vllm_ascend.worker.block_table import BlockTable, MultiGroupBlockTable
        from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    except (ImportError, OSError, RuntimeError, SystemError) as error:
        pytest.skip(f"Ascend runtime is unavailable in this test environment: {error}")

    assert _parameter_names(BlockTable.compute_slot_mapping) == (
        "self",
        "num_reqs",
        "query_start_loc",
        "positions",
    )
    assert _parameter_names(BlockTable.add_row) == ("self", "block_ids", "row_idx")
    assert hasattr(MultiGroupBlockTable, "block_tables") is False
    assert _parameter_names(MultiGroupBlockTable.add_row) == (
        "self",
        "block_ids",
        "row_idx",
    )
    for method in (
        "initialize_kv_cache",
        "_update_states",
        "_build_attention_metadata",
        "sample_tokens",
    ):
        assert callable(getattr(NPUModelRunner, method))
