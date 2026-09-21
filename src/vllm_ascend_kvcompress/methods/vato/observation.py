# SPDX-License-Identifier: Apache-2.0
"""Bounded per-request windows of attention output before the output projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class OutputWindow:
    semantic_end: int
    values: torch.Tensor


class OutputObserver:
    def __init__(
        self,
        runner: Any,
        *,
        window_size: int,
        num_kv_heads: int,
        head_dim: int,
        num_query_heads: int,
    ) -> None:
        self.runner = runner
        self.window_size = window_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.windows: dict[tuple[str, str], OutputWindow] = {}

    def hook(self, layer_name: str):
        def capture(module: Any, args: Any, output: torch.Tensor) -> None:
            from vllm.forward_context import get_forward_context

            # Profile/dummy runs have no request attention metadata.
            if get_forward_context().attn_metadata is not None:
                self.observe(layer_name, output)

        return capture

    def observe(self, layer_name: str, output: torch.Tensor) -> None:
        runner = self.runner
        num_reqs = runner.input_batch.num_reqs
        boundaries = runner.query_start_loc.cpu[: num_reqs + 1].tolist()
        if (
            len(boundaries) != num_reqs + 1
            or boundaries[0] != 0
            or boundaries[-1] > output.shape[0]
            or any(a >= b for a, b in zip(boundaries, boundaries[1:], strict=False))
        ):
            raise RuntimeError("V@O observation has invalid request token boundaries")
        if (
            output.ndim not in (2, 3)
            or output.shape[1:].numel() != self.num_query_heads * self.head_dim
        ):
            raise RuntimeError(
                "V@O observation has an unsupported attention output shape"
            )
        output = output.detach().reshape(-1, self.num_query_heads, self.head_dim)
        for row, request_id in enumerate(runner.input_batch.req_ids[:num_reqs]):
            request = runner.requests[request_id]
            start, end = boundaries[row : row + 2]
            semantic_start = int(request.num_computed_tokens)
            # Slice before casting so memory is bounded by the observation window.
            recent = output[max(start, end - self.window_size) : end].float()
            recent = recent.reshape(
                -1,
                self.num_kv_heads,
                self.num_query_heads // self.num_kv_heads,
                self.head_dim,
            ).sum(dim=2)
            key = (layer_name, request_id)
            previous = self.windows.get(key)
            if previous is not None and previous.semantic_end == semantic_start:
                recent = torch.cat((previous.values, recent), dim=0)
            self.windows[key] = OutputWindow(
                semantic_end=semantic_start + end - start,
                values=recent[-self.window_size :].clone(),
            )

    def mean(self, layer_name: str, request_id: str, semantic_end: int) -> torch.Tensor:
        window = self.windows.get((layer_name, request_id))
        if window is None or window.semantic_end != semantic_end:
            raise RuntimeError(
                "V@O requires a current attention output observation for "
                f"{request_id!r}, layer {layer_name!r}"
            )
        return window.values.mean(dim=0)

    def reset_requests(self, request_ids: set[str]) -> None:
        for key in tuple(self.windows):
            if key[1] in request_ids:
                del self.windows[key]
