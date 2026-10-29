# SPDX-License-Identifier: Apache-2.0
"""独立于调度事务及观测器的算法数据契约。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelShape:
    """模型全局维度，层索引保留原始模型编号。"""

    # HF 文本模型类型标识，如 qwen3_5_text；用于判断模型兼容性。
    model_type: str

    # 层数和头数均为模型全局值；TP 本地头数在绑定设备 cache 时换算。
    # 文本模型的完整层数；混合模型中也包含 GDN 等非全注意力层。
    num_layers: int

    # 每个全注意力层的 Query 头数，尚未按张量并行划分。
    num_attention_heads: int

    # 每个全注意力层的 K/V 头数；GQA 中多个 Query 头共享一个 K/V 头。
    num_kv_heads: int

    # 单个注意力头的完整特征维度，包含旋转和未旋转部分。
    head_dim: int

    # RoPE 的基频底数 theta；非默认缩放下还需校准产物中的实际频率信息。
    rope_theta: float

    # 是否存在需额外处理的 RoPE 缩放或参数配置；用于校验校准频率信息。
    has_rope_scaling: bool

    # 每个 Q/K 头参与 RoPE 旋转的维度数；None 表示使用完整 head_dim。
    # 部分 RoPE 模型中该值小于 head_dim，剩余维度仍参与注意力点积。
    rotary_dim: int | None = None

    # 全注意力层在原始文本模型中的零起始层号，保留原编号，不重新连续编号。
    # None 表示所有层均为全注意力层；混合模型显式列出需压缩的层。
    attention_layer_indices: tuple[int, ...] | None = None

    @property
    def effective_rotary_dim(self) -> int:
        return self.head_dim if self.rotary_dim is None else self.rotary_dim

    @property
    # 未显式指定时视为所有层均为全注意力；混合模型返回原始层号。
    # 例如 Qwen3.5 的全注意力层可能为 3、7、…，不能重编号为 0、1、…。
    def full_attention_layer_indices(self) -> tuple[int, ...]:
        if self.attention_layer_indices is None:
            return tuple(range(self.num_layers))
        return self.attention_layer_indices

    @property
    def is_hybrid(self) -> bool:
        """是否为混合模型，即存在非全注意力层。"""
        return len(self.full_attention_layer_indices) != self.num_layers


@dataclass(frozen=True)
class MethodRuntimeSpec:
    """
    静态 token 长度约束，所有事务均使用私有目标块。

    在 backend 构造描述时统一验证，早于调度器分配与 worker 初始化，
    无需创建算法实例。
    """

    # 压缩触发阈值，单位为 token
    compression_threshold_tokens: int

    # 为算法保留的前向重算后缀长度，单位为 token，必须为正整数。
    # 符合压缩条件的请求命中 APC 时，限制可复用前缀，留出这段后缀执行 forward。
    # 它不表示压缩完成后立即重跑历史；TriAttention 将其设为 recompute_window。
    required_recompute_tokens: int

    # 每轮压缩后各全注意力层统一的物理 KV 目标长度，单位为 token。
    # 调度器据此预留私有目标块，worker 校验算法返回长度与之相等。
    # 必须为正且小于触发阈值；具体缓存页对齐约束在绑定缓存时校验。
    max_physical_num_tokens: int

    # 允许压缩的请求输出长度门槛，单位为 token；默认 0 表示不施加此门槛。
    # 比较请求配置的最大生成 token 数，而非已生成数、剩余数或实际输出数。
    # 低于门槛的请求跳过压缩，也不为压缩限制其 APC 命中长度。
    min_output_tokens_for_compression: int = 0

    def __post_init__(self) -> None:
        """
        在静态描述创建时集中验证预算，让 scheduler 与 worker 使用相同约束。
        更具体的页对齐要求仍由 cache 绑定阶段检查。
        """
        positive = (
            self.compression_threshold_tokens,
            self.required_recompute_tokens,
            self.max_physical_num_tokens,
        )
        if any(type(value) is not int or value <= 0 for value in positive):
            raise ValueError("compression runtime limits must be positive integers")
        if self.max_physical_num_tokens >= self.compression_threshold_tokens:
            raise ValueError("compression target must be below its threshold")
        minimum = self.min_output_tokens_for_compression
        if type(minimum) is not int or minimum < 0:
            raise ValueError("compression output threshold must be non-negative")


@dataclass(frozen=True)
class LayerCache:
    """借用的 K/V 视图，形状为 [page, block, local_kv_heads, head_dim]。

    内部维度连续，页之间可有填充。分配与引用计数由调度器负责，
    算法只能写入已授权的槽位。
    """

    name: str
    layer_index: int
    k_cache: torch.Tensor
    v_cache: torch.Tensor


@dataclass(frozen=True)
class CompressionRequest:
    """前向执行完成后的长度与设备端内核页表。

    语义位置保留原始历史与 RoPE 时间线，物理位置索引压缩后的缓存。
    CPU 块所有权与事务授权保留在 provider 中，不进入算法契约。
    """

    request_id: str
    semantic_num_tokens: int
    physical_num_tokens: int
    # 两张表都是设备上的 128-token 内核页编号，可能来自更大的逻辑页展开。
    # 算法据此索引 KV；CPU 块对象、引用计数和计划确认由通用运行时管理。
    source_block_ids_device: torch.Tensor
    destination_block_ids_device: torch.Tensor


@dataclass(frozen=True)
class CompressionResult:
    """各层统一的缓存长度，等于 backend 声明的目标长度。"""

    physical_num_tokens: int


# 阅读职责边界：Backend 决定怎样初始化，method 只负责检查、绑定和执行算法。
# Query 观测及逐层 metadata 不属于此接口，扩展状态归下游 provider 所有。
class KVCompressionMethod(ABC):
    """仅在 worker 侧使用的算法，模型准备完成后构造一次。"""

    @abstractmethod
    def compatibility_reasons(self, worker: Any) -> tuple[str, ...]:
        """在分配缓存前描述算法与模型之间的不兼容原因。"""

    @abstractmethod
    def bind_model_runner(
        self, runner: Any, layer_caches: tuple[LayerCache, ...]
    ) -> None:
        """绑定已验证的缓存视图，并分配算法工作区。"""

    @abstractmethod
    def compress(self, request: CompressionRequest) -> CompressionResult:
        """将选中的 K/V 写入目标块，不分配或释放缓存块。"""
