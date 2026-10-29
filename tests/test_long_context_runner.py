# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts import kvcompress_acceptance_compare as comparator
from scripts import kvcompress_long_context_run as runner


def test_comparison_override_supports_legacy_evidence() -> None:
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


@pytest.mark.parametrize(
    ("expectation", "commits", "acks", "passed"),
    [("optional", 0, 0, True), ("optional", 1, 0, False), ("auto", 0, 0, False)],
)
def test_runner_compression_evidence_contract(expectation, commits, acks, passed):
    evidence = {"scheduler_commits": [{}] * commits, "worker_acks": acks}
    assert runner._compression_evidence_passes(evidence, expectation, "B1") is passed


@pytest.mark.parametrize(
    ("commits", "reduction", "expected"),
    [(0, None, (False, True)), (1, 0.19, (True, False)), (1, 0.20, (True, True))],
)
def test_optional_comparison_applies_reduction_gate_only_after_compression(
    commits, reduction, expected
):
    rows = [
        {
            "compression_evidence_expectation": "optional",
            "compression": {
                "scheduler_commits": [{}] * commits,
                "worker_acks": commits,
            },
        }
    ]
    assert comparator._compression_reduction_gate(rows, reduction) == expected
