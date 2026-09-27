"""CPU contract checks for the plugin-owned Qwen GDN output operator."""

from __future__ import annotations

import pytest
import torch

from vllm_ascend_kvcompress.qwen_gdn_ops import (
    _segments,
    _validate,
    causal_conv1d_spec_reference,
)


def test_packed_segment_mapping_uses_compact_nonempty_indices() -> None:
    q = torch.empty(1, 2, 97, 128)
    assert _segments(q, 64, (0, 64, 64, 97), (0, 0, 1, 0)) == [
        (0, 0, 64, 0),
        (0, 64, 0, 1),
        (0, 64, 33, 1),
    ]


def test_packed_segment_mapping_rejects_changed_chunk_order() -> None:
    q = torch.empty(1, 2, 97, 128)
    with pytest.raises(ValueError, match="chunk_indices"):
        _segments(q, 64, (0, 64, 97), (1, 0, 0, 0))


def test_qwen_output_contract_rejects_unsupported_layout() -> None:
    q = torch.empty(1, 2, 64, 128)
    v = torch.empty(1, 4, 64, 128)
    h = torch.empty(1, 4, 1, 128, 128)
    with pytest.raises(ValueError, match="transposed state"):
        _validate(q, q, v, h, None, None, None, None, 64, True)


def test_qwen_spec_reference_updates_output_and_state() -> None:
    x = torch.tensor([[1.0, -1.0], [2.0, 3.0]], dtype=torch.float32)
    weight = torch.ones(2, 3)
    state = torch.zeros(1, 2, 3)
    output = torch.empty_like(x)

    actual = causal_conv1d_spec_reference(
        output,
        x,
        weight,
        state,
        None,
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([[0, 7, 8]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        0,
        -1,
    )

    assert actual is output
    torch.testing.assert_close(actual, torch.tensor([[1.0, -1.0], [3.0, 2.0]]))
    torch.testing.assert_close(
        state[0], torch.tensor([[0.0, 1.0, 2.0], [0.0, -1.0, 3.0]])
    )


def test_qwen_spec_reference_uses_accepted_offset_and_extended_state() -> None:
    x = torch.tensor([[1.0], [2.0]])
    weight = torch.ones(1, 3)
    state = torch.tensor([[[10.0, 20.0, 30.0]]])
    output = torch.empty_like(x)

    causal_conv1d_spec_reference(
        output,
        x,
        weight,
        state,
        None,
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        0,
        -1,
    )

    torch.testing.assert_close(output, torch.tensor([[51.0], [33.0]]))
    torch.testing.assert_close(state, torch.tensor([[[30.0, 1.0, 2.0]]]))


def test_qwen_spec_reference_rolls_all_draft_positions() -> None:
    x = torch.tensor([[1.0], [2.0], [3.0]])
    weight = torch.ones(1, 4)
    state = torch.tensor([[[10.0, 20.0, 30.0, 40.0, 50.0]]])
    output = torch.empty_like(x)

    causal_conv1d_spec_reference(
        output,
        x,
        weight,
        state,
        None,
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([[0, 1, 2]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        0,
        -1,
    )

    torch.testing.assert_close(output, torch.tensor([[121.0], [93.0], [56.0]]))
    torch.testing.assert_close(state, torch.tensor([[[40.0, 50.0, 1.0, 2.0, 3.0]]]))
