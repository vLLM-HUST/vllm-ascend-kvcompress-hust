# SPDX-License-Identifier: Apache-2.0
"""worker 与调度适配器共享的宿主、模型和缓存约束。"""

from __future__ import annotations

import os
from typing import Any

import torch

from ..core.constants import KERNEL_CACHE_BLOCK_SIZE
from ..methods.base import ModelShape


# 这是实验路径的资格判断，不会自动开启投机解码。
# 必须同时满足显式开关、Qwen3.5 混合结构、固定两个草稿 token 及 align 缓存模式。
def supports_mtp2(vllm_config: Any) -> bool:
    """仅在显式启用实验选项时开放尚未完成验证的 MTP2 路径。

    只在精确的提示词边界安排压缩。后续推测解码步的乐观 KV 尾部可能
    在验收后回退，因此不能再次触发事务。注意力管理器使用物理长度，
    Mamba 状态仍使用语义长度。
    """
    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is None:
        return False
    if os.getenv("VLLM_ASCEND_KVCOMPRESS_EXPERIMENTAL_MTP2") != "1":
        return False
    model = vllm_config.model_config
    text_config = getattr(model, "hf_text_config", None)
    return (
        getattr(speculative, "method", None) == "mtp"
        and getattr(speculative, "num_speculative_tokens", None) == 2
        and not getattr(speculative, "num_speculative_tokens_per_batch_size", None)
        and bool(getattr(model, "is_hybrid", False))
        and getattr(text_config, "model_type", None)
        in {"qwen3_5_text", "qwen3_5_moe_text"}
        and getattr(vllm_config.cache_config, "mamba_cache_mode", None) == "align"
    )


# 只核对宿主运行条件，不读取 TriAttention 校准文件，也不执行 patch。
# provider 会再合并 method 给出的算法条件，一次报告启动不兼容原因。
def common_compatibility_reasons(vllm_config: Any, runner: Any) -> tuple[str, ...]:
    reasons: list[str] = []
    cache_config = vllm_config.cache_config
    # 前缀命中可能与其他请求共享只读源块。
    # 调度器预留独立目标块，待 worker 完成复制后才切换请求行；
    # 其 cache_blocks 钩子还会阻止私有压缩 KV
    # 被登记到语义前缀哈希下。
    block_size = int(cache_config.block_size)
    allowed_block_sizes = allowed_scheduler_block_sizes(vllm_config)
    if block_size not in allowed_block_sizes:
        reasons.append(
            f"logical KV block size must be one of {sorted(allowed_block_sizes)}, "
            f"got {block_size}"
        )
    if getattr(cache_config, "cache_dtype", "auto") not in {
        "auto",
        "bfloat16",
        "float16",
    }:
        reasons.append("quantized KV cache is unsupported")
    if vllm_config.speculative_config is not None and not supports_mtp2(vllm_config):
        reasons.append(
            "speculative decoding is unqualified; experimental Qwen3.5 MTP2 "
            "requires mamba_cache_mode='align' and explicit opt-in"
        )
    if vllm_config.kv_transfer_config is not None:
        reasons.append("KV transfer is unsupported")
    # 调度适配器在安排私有块复制后冻结请求。
    # 只有已被宿主接受的 worker 输出确认事务后，才切换块表，
    # 从而避免其他正在执行的批次
    # 从尚未写完的目标块进行解码。
    model_config = vllm_config.model_config
    if bool(getattr(model_config, "is_hybrid", False)):
        text_config = getattr(model_config, "hf_text_config", None)
        model_type = str(getattr(text_config, "model_type", ""))
        if model_type not in {"qwen3_5_text", "qwen3_5_moe_text"}:
            reasons.append(
                "only Qwen3.5-style full-attention/Gated-DeltaNet hybrids are supported"
            )
        if getattr(cache_config, "mamba_cache_mode", "none") not in {
            "none",
            "align",
        }:
            reasons.append(
                "hybrid compression requires mamba_cache_mode='none' or 'align'"
            )
    if bool(getattr(model_config, "use_mla", False)):
        reasons.append("MLA cache layouts are unsupported")
    if bool(getattr(model_config, "is_encoder_decoder", False)):
        reasons.append("encoder-decoder models are unsupported")
    aux_output_config = getattr(vllm_config, "aux_output_config", None)
    if bool(getattr(aux_output_config, "enable_return_routed_experts", False)):
        reasons.append("routed-expert output is unsupported on this host ABI")
    parallel = vllm_config.parallel_config
    tensor_parallel_size = int(getattr(parallel, "tensor_parallel_size", 1))
    text_config = getattr(model_config, "hf_text_config", None)
    num_kv_heads = int(getattr(text_config, "num_key_value_heads", 0))
    if tensor_parallel_size <= 0 or num_kv_heads % tensor_parallel_size:
        reasons.append(
            "tensor parallel size must be positive and divide the model KV heads"
        )
    for name, attr in (
        ("pipeline parallel", "pipeline_parallel_size"),
        ("data parallel", "data_parallel_size"),
        ("prefill context parallel", "prefill_context_parallel_size"),
        ("decode context parallel", "decode_context_parallel_size"),
    ):
        size = int(getattr(parallel, attr, 1))
        if size != 1:
            reasons.append(f"{name} size must be one, got {size}")
    runner_type = type(runner)
    if (
        runner_type.__module__ != "vllm_ascend.worker.model_runner_v1"
        or runner_type.__name__ != "NPUModelRunner"
    ):
        reasons.append("the standard v1 Ascend NPUModelRunner is required")
    if bool(getattr(runner, "use_sparse", False)):
        reasons.append("Ascend sparse attention is unsupported")
    if bool(getattr(runner, "use_compress", False)):
        reasons.append("model-native Ascend compression is unsupported")
    return tuple(reasons)


def allowed_scheduler_block_sizes(vllm_config: Any) -> frozenset[int]:
    """返回受支持模型家族允许的逻辑页大小，不支持的组合直接拒绝。

    Ascend 内核缓存块保持为 128 token。混合缓存适配器可能将全注意力
    管理器的逻辑页提升为 1152 或 2048 token，分别对应 9 或 16 个
    连续内核块。调度器跨缓存组的最小公倍数对齐约束另行校验。
    """
    model_config = getattr(vllm_config, "model_config", None)
    if not bool(getattr(model_config, "is_hybrid", False)):
        return frozenset({KERNEL_CACHE_BLOCK_SIZE})
    text_config = getattr(model_config, "hf_text_config", None)
    model_type = str(getattr(text_config, "model_type", ""))
    if model_type in {"qwen3_5_text", "qwen3_5_moe_text"}:
        return frozenset({KERNEL_CACHE_BLOCK_SIZE, 1152, 2048})
    return frozenset({KERNEL_CACHE_BLOCK_SIZE})


# 按 cache spec 判断组类型，而非假设第 0 组一定是注意力。
# 只有全注意力组会被压缩；其他已支持组必须全部是循环状态，不能混入未知布局。
def find_full_attention_group(groups: Any) -> int:
    """返回唯一的全注意力缓存组，允许同时存在 Mamba 缓存组。"""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        UniformTypeKVCacheSpecs,
    )

    full_groups: list[int] = []
    unsupported: list[str] = []
    for index, group in enumerate(groups):
        spec = group.kv_cache_spec
        # UniformTypeKVCacheSpecs 包装的是多层 spec，需逐项判断。
        # 一个缓存组内部若混有不同类型，不能只凭其中第一层推断整组可压缩。
        specs = (
            tuple(spec.kv_cache_specs.values())
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
        )
        if all(isinstance(item, FullAttentionSpec) for item in specs):
            full_groups.append(index)
        elif not all(isinstance(item, MambaSpec) for item in specs):
            unsupported.append(type(spec).__name__)
    if unsupported:
        raise RuntimeError(
            "Ascend KV compression supports only full-attention plus optional "
            "Mamba groups; found " + ", ".join(unsupported)
        )
    if len(full_groups) != 1:
        raise RuntimeError(
            "Ascend KV compression requires exactly one full-attention KV group"
        )
    return full_groups[0]


def unpack_layer_cache(
    layer_name: str, layer: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    cache = getattr(layer, "kv_cache", None)
    if isinstance(cache, list) and len(cache) == 1:
        cache = cache[0]
    impl = getattr(layer, "impl", None)
    # 优先让 Ascend 后端解包标准存储，取得 K/V view；旧的双 tensor 表示仍可直接使用。
    # 这里只借用存储，没有复制整份 cache，也不会改变宿主层的绑定。
    unpack = getattr(impl, "_unpack_kv_cache", None)
    if callable(unpack):
        k_cache, v_cache = unpack(cache)
    elif isinstance(cache, (tuple, list)) and len(cache) == 2:
        k_cache, v_cache = cache
    else:
        raise RuntimeError(
            f"attention layer {layer_name!r} does not expose dense Ascend K/V views"
        )
    if not isinstance(k_cache, torch.Tensor) or not isinstance(v_cache, torch.Tensor):
        raise RuntimeError(f"attention layer {layer_name!r} K/V bindings are invalid")
    return k_cache, v_cache


def validate_cache_tensor(
    layer_name: str,
    kind: str,
    cache: torch.Tensor,
    model: ModelShape,
    expected_kv_heads: int | None = None,
    expected_block_size: int = KERNEL_CACHE_BLOCK_SIZE,
) -> None:
    expected_tail = (
        expected_block_size,
        model.num_kv_heads if expected_kv_heads is None else expected_kv_heads,
        model.head_dim,
    )
    if cache.ndim != 4 or tuple(cache.shape[1:]) != expected_tail:
        raise RuntimeError(
            f"attention layer {layer_name!r} {kind} cache shape {tuple(cache.shape)} "
            f"does not end in {expected_tail}"
        )
    if cache.dtype not in {torch.bfloat16, torch.float16}:
        raise RuntimeError(
            f"attention layer {layer_name!r} {kind} cache dtype {cache.dtype} "
            "is unsupported"
        )
    # 允许页间 padding，但页内部的 token、头、特征维必须连续。
    # 这保证分页 gather 和直接读页的 kernel 可以使用同一套下标语义。
    inner_stride = (expected_tail[1] * expected_tail[2], expected_tail[2], 1)
    page_elements = expected_tail[0] * inner_stride[0]
    if cache.stride()[1:] != inner_stride or cache.stride(0) < page_elements:
        raise RuntimeError(
            f"attention layer {layer_name!r} {kind} cache must have contiguous "
            "inner blocks and a nonoverlapping page stride"
        )


# scheduler 与 worker 的请求对象形态不同，这里统一读取最大输出数。
# worker 缓存请求可能仅在 sampling_params 中保存该值；pooling 请求按一个输出处理。
def request_max_tokens(request: Any) -> int:
    value = getattr(request, "max_tokens", None)
    if value is None:
        sampling_params = getattr(request, "sampling_params", None)
        value = getattr(sampling_params, "max_tokens", None)
    if value is None and getattr(request, "pooling_params", None) is not None:
        value = 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(
            "current vLLM worker request must expose a non-negative output limit"
        )
    return value
