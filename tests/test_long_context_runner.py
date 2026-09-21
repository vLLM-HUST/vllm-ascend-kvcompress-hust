# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "kvcompress_long_context_run.py"
    spec = importlib.util.spec_from_file_location("kvcompress_long_context_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_comparator():
    path = Path(__file__).parents[1] / "scripts" / "kvcompress_acceptance_compare.py"
    spec = importlib.util.spec_from_file_location("kvcompress_acceptance_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_optional_compression_accepts_below_threshold_workload() -> None:
    runner = _load_runner()

    assert runner._compression_evidence_passes(
        {"scheduler_commits": [], "worker_acks": 0}, "optional", "B1"
    )


def test_optional_compression_still_requires_worker_acknowledgements() -> None:
    runner = _load_runner()

    assert not runner._compression_evidence_passes(
        {"scheduler_commits": [{}], "worker_acks": 0}, "optional", "B1"
    )


def test_auto_requires_compression_for_b1() -> None:
    runner = _load_runner()

    assert not runner._compression_evidence_passes(
        {"scheduler_commits": [], "worker_acks": 0}, "auto", "B1"
    )


def test_runner_freezes_every_sampling_control() -> None:
    runner = _load_runner()

    assert runner.FROZEN_SAMPLING_PARAMS == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "n": 1,
        "use_beam_search": False,
        "stop": [],
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": True,
    }


def test_optional_below_threshold_comparison_marks_m3_not_applicable() -> None:
    comparator = _load_comparator()
    rows = [
        {
            "compression_evidence_expectation": "optional",
            "compression": {"scheduler_commits": [], "worker_acks": 0},
        }
    ]

    applicable, passed = comparator._compression_reduction_gate(rows, None)

    assert not applicable
    assert passed


def test_optional_comparison_reenables_m3_after_a_commit() -> None:
    comparator = _load_comparator()
    rows = [
        {
            "compression_evidence_expectation": "optional",
            "compression": {"scheduler_commits": [{}], "worker_acks": 1},
        }
    ]

    applicable, passed = comparator._compression_reduction_gate(rows, 0.19)

    assert applicable
    assert not passed


def test_comparison_override_supports_legacy_evidence() -> None:
    comparator = _load_comparator()
    rows = [
        {
            "compression": {"scheduler_commits": [], "worker_acks": 0},
        },
        {
            "compression_evidence_expectation": "optional",
            "compression": {"scheduler_commits": [], "worker_acks": 0},
        },
    ]

    applicable, passed = comparator._compression_reduction_gate(rows, None, "optional")

    assert not applicable
    assert passed


def test_comparison_requires_exact_declared_tp_ack_multiplicity() -> None:
    comparator = _load_comparator()
    rows = [
        {
            "compression": {
                "scheduler_commits": [{}, {}],
                "worker_acks": 4,
            }
        }
    ]

    assert comparator._commit_ack_counts_match(rows, 2)
    assert not comparator._commit_ack_counts_match(rows, 1)
