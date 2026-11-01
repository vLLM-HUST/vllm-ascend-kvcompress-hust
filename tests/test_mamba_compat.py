# SPDX-License-Identifier: Apache-2.0
"""Preserve Mamba prefill admission and Ascend external-KV reservations."""

from types import SimpleNamespace

from vllm_ascend_kvcompress.host.compat import install_mamba_prefill_end_compat


def test_prefill_boundary_and_external_kv_reservation_are_preserved():
    calls = []

    class Core:
        def get_num_blocks_to_allocate(self, *args, prefill_end=0, **kwargs):
            calls.append((args, kwargs, prefill_end))
            return 3

    class Ascend(Core):
        block_size = 128

        def get_num_blocks_to_allocate(self, request_id):
            raise AssertionError("old adapter cannot forward the prefill boundary")

    assert install_mamba_prefill_end_compat(Ascend)
    assert not install_mamba_prefill_end_compat(Ascend)
    manager = Ascend()
    blocks = [SimpleNamespace(block_id=1)]
    assert (
        manager.get_num_blocks_to_allocate(
            "local", 512, blocks, 128, 128, 512, prefill_end=499
        )
        == 3
    )
    assert calls[-1][2] == 499
    assert calls[-1][0][-1] == 512
    assert (
        manager.get_num_blocks_to_allocate(
            "external",
            512,
            blocks,
            256,
            128,
            512,
            apply_admission_cap=True,
            prefill_end=500,
        )
        == 4
    )
    assert calls[-1][1]["apply_admission_cap"] is True
    assert calls[-1][2] == 500


def test_aligned_mamba_adapter_is_unchanged():
    class Aligned:
        def get_num_blocks_to_allocate(self, *, prefill_end=0):
            return prefill_end

    original = Aligned.get_num_blocks_to_allocate
    assert not install_mamba_prefill_end_compat(Aligned)
    assert Aligned.get_num_blocks_to_allocate is original
