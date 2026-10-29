# SPDX-License-Identifier: Apache-2.0
"""已配置的 backend 与可覆盖的 runtime factory。

backend 只持有轻量的进程内配置。构造调度器时不得加载校准产物
或分配加速器内存。下游注册 backend factory，并覆盖 provider 或
scheduler factory 来实现自定义生命周期，避免向通用算法 API 添加能力标志。

* register_method 与入口点应返回 CompressionBackend，而非算法实例。
* 在 backend 构造器中解析配置并构造 runtime_spec。
* 将产物准备移入 prepare_model_load，由 create_method 创建 worker 算法。
* 覆盖 create_provider/create_scheduler 实现下游生命周期策略。
* Query 或逐层元数据消费者须显式选择临时 extensions.legacy 适配器；
  更推荐在下游仓库中实现这些适配器。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeAlias

from .base import KVCompressionMethod, MethodRuntimeSpec, ModelShape

if TYPE_CHECKING:
    from ..runtime.provider import AscendKVCompressionProvider
    from ..runtime.scheduler import SchedulerCompressionState


class CompressionBackend(ABC):
    """
    Backend 是每个进程各自持有的配置入口；scheduler 与 worker 不共享该对象。
    静态阶段只描述预算，worker 阶段再准备产物、创建算法和绑定 KV。
    """

    # name 对应 JSON 的 method 和注册表名称；
    # runtime_spec 不依赖设备或校准文件。
    # 下游实现构造函数时，应先完成这两个属性，供调度端直接读取。
    name: str
    runtime_spec: MethodRuntimeSpec

    def __init__(self, vllm_config: Any, model_shape: ModelShape) -> None:
        self.vllm_config = vllm_config
        self.model_shape = model_shape

    def prepare_model_load(self, runner: Any) -> None:
        """
        在宿主加载服务权重前准备算法产物。默认无需准备。
        TriAttention 在这里生成或验证离线统计。
        此时服务权重尚未加载，适合临时加载校准模型后释放其资源。
        """
        return None

    @abstractmethod
    def create_method(self) -> KVCompressionMethod:
        """在 prepare_model_load 完成后创建 worker 算法状态。"""

    def create_provider(self) -> AscendKVCompressionProvider:
        """
        由下游覆盖，以自行持有额外的 worker 状态。
        默认返回只管理统一长度的 provider
        """
        # 导入放在方法内，避免 Backend、provider 与 scheduler 的模块级循环依赖
        from ..runtime.provider import AscendKVCompressionProvider

        return AscendKVCompressionProvider(self)

    def create_scheduler(self, scheduler: Any) -> SchedulerCompressionState:
        """
        调度策略或事务状态需要扩展时，返回下游自己的状态类。
        该类接收同一份静态预算；不需要先创建持有校准和 workspace 的算法
        """
        from ..runtime.scheduler import SchedulerCompressionState

        return SchedulerCompressionState(scheduler, self)


# 工厂接收方法配置、宿主 vLLM 配置和模型描述，返回静态 backend。
# 放在类定义之后，因为类型别名赋值会在运行时求值。
BackendFactory: TypeAlias = Callable[
    [Mapping[str, Any], Any, ModelShape], CompressionBackend
]
