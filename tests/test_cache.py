# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend_kvcompress.methods.triattention.cache import (
    gather_paged_tokens,
    materialize_selected_tokens,
)


def test_tuple_cache_materialization_gathers_before_overlapping_write() -> None:
    block_size = 4
    k_cache = torch.arange(4 * block_size, dtype=torch.float32).view(
        4, block_size, 1, 1
    )
    v_cache = k_cache.clone() + 100
    source_blocks = torch.tensor([2, 0, 3], dtype=torch.long)
    keep = torch.tensor([1, 4, 6, 9], dtype=torch.long)
    expected_k = gather_paged_tokens(k_cache, source_blocks, keep, block_size).clone()
    expected_v = gather_paged_tokens(v_cache, source_blocks, keep, block_size).clone()

    materialize_selected_tokens(
        k_cache,
        v_cache,
        source_blocks,
        torch.tensor([2], dtype=torch.long),
        keep,
        block_size,
    )

    torch.testing.assert_close(k_cache[2], expected_k)
    torch.testing.assert_close(v_cache[2], expected_v)


def test_gather_maps_logical_tokens_through_block_table() -> None:
    cache = torch.arange(3 * 4, dtype=torch.float32).view(3, 4, 1, 1)
    result = gather_paged_tokens(
        cache,
        torch.tensor([2, 0]),
        torch.tensor([0, 3, 4, 7]),
        4,
    )
    torch.testing.assert_close(result.flatten(), torch.tensor([8.0, 11.0, 0.0, 3.0]))
