# SPDX-License-Identifier: Apache-2.0
"""Measure Qwen3.5-shaped selection and KV movement on one Ascend NPU."""

from __future__ import annotations

import argparse
import time

import torch
import torch_npu  # noqa: F401

from vllm_ascend_kvcompress.methods.triattention.cache import (
    materialize_token_slots,
    token_slots,
)
from vllm_ascend_kvcompress.methods.triattention.selection import (
    select_keep_indices,
)


def elapsed_ms(function, *, iterations: int = 5) -> float:
    function()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    torch.npu.synchronize()
    return 1000 * (time.perf_counter() - start) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--budget", type=int, default=8192)
    args = parser.parse_args()
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    block_size = 128
    torch.manual_seed(20260927)
    scores = torch.randn(args.tokens, device=device, dtype=torch.float32)
    source_blocks = torch.arange(
        (args.tokens + block_size - 1) // block_size,
        device=device,
        dtype=torch.int64,
    )
    destination_blocks = source_blocks[: args.budget // block_size]
    destination_slots = token_slots(
        destination_blocks,
        torch.arange(args.budget, device=device, dtype=torch.int64),
        block_size,
    )
    cache = torch.randn(
        len(source_blocks), block_size, 1, 256, device=device, dtype=torch.bfloat16
    )
    value_cache = torch.randn_like(cache)
    key_workspace = torch.empty(
        args.budget, 1, 256, device=device, dtype=torch.bfloat16
    )
    value_workspace = torch.empty_like(key_workspace)

    def select() -> torch.Tensor:
        return select_keep_indices(
            scores,
            budget=args.budget,
            protected_prefix=128,
            protected_recent=512,
            segments=8,
            policy="v3",
        )

    def select_reference() -> torch.Tensor:
        prefix = 128
        recent = 512
        segments = 8
        middle_count = args.tokens - prefix - recent
        evict_total = args.tokens - args.budget
        mask = torch.zeros(args.tokens, device=device, dtype=torch.bool)
        mask[:prefix] = True
        mask[-recent:] = True
        for segment in range(segments):
            start = prefix + middle_count * segment // segments
            stop = prefix + middle_count * (segment + 1) // segments
            evict_before = evict_total * (start - prefix) // middle_count
            evict_after = evict_total * (stop - prefix) // middle_count
            keep_count = stop - start - (evict_after - evict_before)
            if keep_count:
                local = torch.topk(scores[start:stop], k=keep_count).indices
                mask[start + local] = True
        return torch.nonzero(mask).flatten()

    keep = select()
    torch.testing.assert_close(keep.cpu(), select_reference().cpu(), rtol=0, atol=0)
    source_slots = token_slots(source_blocks, keep, block_size)

    def move() -> None:
        materialize_token_slots(
            cache,
            value_cache,
            source_slots,
            destination_slots,
            key_workspace,
            value_workspace,
        )

    print(f"tokens={args.tokens} budget={args.budget}")
    print(f"reference_selection_ms={elapsed_ms(select_reference):.3f}")
    print(f"batched_selection_ms={elapsed_ms(select):.3f}")
    print(
        "slot_mapping_ms="
        f"{elapsed_ms(lambda: token_slots(source_blocks, keep, block_size)):.3f}"
    )
    print(f"one_layer_kv_move_ms={elapsed_ms(move):.3f}")
    print(f"ten_layer_kv_move_ms={elapsed_ms(lambda: [move() for _ in range(10)]):.3f}")


if __name__ == "__main__":
    main()
