# SPDX-License-Identifier: Apache-2.0
"""已停止维护的 Query/FIA 图钩子，计划随 legacy.py 一并删除。"""

from __future__ import annotations

import inspect
from typing import Any

from vllm.logger import logger


# 选择 attention 自定义算子的后端边界观察 Q，避免模块 forward hook 只在追踪时运行。
# 只有层对象上挂了 provider 才采集；同一后端类的其他层仍调用原方法。
def install_attention_query_hook() -> None:
    """在图拆分的注意力自定义算子边界观察 query。"""
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

    from .legacy import ATTENTION_QUERY_PROVIDER_ATTRIBUTE

    marker = "_ascend_kvcompress_query_observer_v1"
    if AscendAttentionBackendImpl.__dict__.get(marker, False):
        return
    original = AscendAttentionBackendImpl.forward

    def forward(
        backend: Any,
        layer: Any,
        query: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        provider = getattr(layer, ATTENTION_QUERY_PROVIDER_ATTRIBUTE, None)
        if provider is not None:
            provider.capture_attention_query(layer.layer_name, query)
        return original(backend, layer, query, *args, **kwargs)

    AscendAttentionBackendImpl.forward = forward
    setattr(AscendAttentionBackendImpl, f"{marker}_original", original)
    setattr(AscendAttentionBackendImpl, marker, True)


def install_layer_aware_fia_graph_replay(vllm_config: Any) -> bool:
    """启用宿主已有的按层标识更新 FULL 图的路径。

    已验证的 Ascend 宿主会在每个捕获的 FIA 任务中记录注意力层名，
    并可根据当前逐层元数据重新绑定任务。宿主只为内置混合注意力
    模型家族启用此路径。外部压缩方法若各层物理长度不同，也需要
    相同的通用机制，因此仅在确认全部预期宿主接口存在后显式启用。
    """
    mode = getattr(vllm_config.compilation_config, "cudagraph_mode", None)
    mode_name = getattr(mode, "name", str(mode))
    if mode_name == "NONE":
        return False
    if mode_name != "FULL_AND_PIECEWISE":
        raise RuntimeError(
            "per-layer KV compression requires FULL_AND_PIECEWISE graph mode"
        )

    from vllm_ascend.attention import attention_v1
    from vllm_ascend.attention import utils as attention_utils

    impl = attention_v1.AscendAttentionBackendImpl
    marker = "_ascend_kvcompress_layer_aware_graph_v1"
    if getattr(impl, marker, False):
        return False
    required_impl_seams = (
        "_graph_metadata_layer_name",
        "update_graph_params",
        "full_graph_fia",
    )
    if any(not callable(getattr(impl, name, None)) for name in required_impl_seams):
        raise RuntimeError(
            "Ascend attention backend lacks layer-aware FIA graph replay seams"
        )
    original = getattr(attention_utils, "needs_layer_aware_fia_graph_replay", None)
    if not callable(original):
        raise RuntimeError("Ascend layer-aware graph capability probe is unavailable")
    if attention_v1.needs_layer_aware_fia_graph_replay is not original:
        raise RuntimeError("Ascend layer-aware graph capability probe changed")

    # 验证接口语义，避免误启用名称相似的旧版钩子。
    # 同名函数不保证能按层更新图参数；这里核对 layer_name 与 metadata 路由约定。
    # 安装后复用宿主现有逐层 FIA 重绑路径，不另行实现图执行器。
    update_source = inspect.getsource(impl.update_graph_params)
    if not all(
        token in update_source
        for token in ("layer_name", "metadata_key", "attn_metadata")
    ):
        raise RuntimeError(
            "Ascend FIA graph updater cannot consume layer-distinct metadata"
        )

    def layer_aware_replay_enabled() -> bool:
        return True

    attention_utils.needs_layer_aware_fia_graph_replay = layer_aware_replay_enabled
    attention_v1.needs_layer_aware_fia_graph_replay = layer_aware_replay_enabled
    setattr(impl, marker, True)
    logger.warning(
        "Enabled layer-aware FIA graph replay for external per-layer KV compression"
    )
    return True
