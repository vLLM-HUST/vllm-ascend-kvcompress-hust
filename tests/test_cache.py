# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend_kvcompress.methods.base import ModelShape
from vllm_ascend_kvcompress.methods.triattention.cache import (
    gather_paged_tokens,
    materialize_selected_tokens,
    materialize_token_slots,
    token_slots,
)
from vllm_ascend_kvcompress.provider import _validate_cache_tensor


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


def test_padded_pages_preserve_separate_kv_and_unused_storage() -> None:
    block_size, num_blocks, heads, dim = 4, 4, 1, 2
    logical_page = block_size * heads * dim
    page_stride = 2 * logical_page + 4
    raw = torch.full((num_blocks * page_stride,), -1, dtype=torch.float16)
    strides = (page_stride, heads * dim, dim, 1)
    shape = (num_blocks, block_size, heads, dim)
    k_cache = torch.as_strided(raw, shape, strides)
    v_cache = torch.as_strided(raw, shape, strides, storage_offset=logical_page)
    k_cache.copy_(torch.arange(num_blocks * logical_page).view(shape))
    v_cache.copy_(torch.arange(num_blocks * logical_page).view(shape) + 100)
    model = ModelShape("qwen2", 1, 1, heads, dim, 10_000.0, False)
    _validate_cache_tensor("layer", "K", k_cache, model, expected_block_size=block_size)
    _validate_cache_tensor("layer", "V", v_cache, model, expected_block_size=block_size)

    source_blocks = torch.tensor([2, 0, 3])
    keep = torch.tensor([1, 4, 6, 9])
    expected_k = gather_paged_tokens(k_cache, source_blocks, keep, block_size).clone()
    expected_v = gather_paged_tokens(v_cache, source_blocks, keep, block_size).clone()
    materialize_selected_tokens(
        k_cache, v_cache, source_blocks, torch.tensor([2]), keep, block_size
    )
    torch.testing.assert_close(k_cache[2], expected_k)
    torch.testing.assert_close(v_cache[2], expected_v)

    source_slots = token_slots(
        torch.tensor([2]), torch.tensor([0, 1, 2, 3]), block_size
    )
    destination_slots = token_slots(
        torch.tensor([1]), torch.tensor([0, 1, 2, 3]), block_size
    )
    materialize_token_slots(
        k_cache,
        v_cache,
        source_slots,
        destination_slots,
        torch.empty_like(expected_k),
        torch.empty_like(expected_v),
    )
    torch.testing.assert_close(k_cache[1], expected_k)
    torch.testing.assert_close(v_cache[1], expected_v)
    torch.testing.assert_close(
        raw.view(num_blocks, page_stride)[:, -4:],
        torch.full((num_blocks, 4), -1, dtype=torch.float16),
    )
