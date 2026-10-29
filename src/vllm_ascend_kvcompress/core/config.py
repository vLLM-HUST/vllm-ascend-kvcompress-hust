# SPDX-License-Identifier: Apache-2.0
"""provider 层配置与压缩方法选择。"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROVIDER_NAME = "ascend_kvcompress"
DEFAULT_METHOD = "triattention"
EXTENSION_ID = "org.vllm-hust.ascend-kvcompress"
ENABLE_ENV = "VLLM_ASCEND_KVCOMPRESS_ENABLED"
CONFIG_ENV = "VLLM_ASCEND_KVCOMPRESS_CONFIG"
MANAGER_ENABLED_ENV = "VLLMHUST_EXT_ENABLED_BUNDLES"


# 此层只读取 provider/method，忽略其他顶层字段，不解释算法参数。
# method_config 原样复制给方法解析器，由方法读取和校验自己使用的字段。
# 。
@dataclass(frozen=True)  # frozen=True表示实例创建后字段不能再修改
class PluginConfig:
    """
    用于读取并验证运行配置中的provider和method信息。忽略其他字段。
    method_config 原样复制给方法解析器，由方法读取和校验自己使用的字段。
    例如TriAttention 的预算、保护窗口、评分策略等由 methods/triattention/config.py 验证
    """

    provider_name: str            # 提供者(插件名)
    method: str                   # 启用的压缩方法名称
    method_config: dict[str, Any] # 压缩方法的具体配置，由各方法自行解析和验证

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PluginConfig:
        """从 Mapping 对象创建 PluginConfig 实例。"""
        provider_name = value.get("provider", PROVIDER_NAME)
        if provider_name != PROVIDER_NAME:
            raise ValueError(
                f"provider {provider_name!r} is unsupported; expected {PROVIDER_NAME!r}"
            )
        method = value.get("method", DEFAULT_METHOD)
        if not isinstance(method, str) or not method.strip():
            raise ValueError("'method' must be a non-empty string")
        raw_method_config = value.get("method_config", {})
        if not isinstance(raw_method_config, Mapping):
            raise ValueError("'method_config' must be an object")
        return cls(
            provider_name=str(provider_name),
            method=method.strip().lower(),
            method_config=dict(raw_method_config),
        )


def extension_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """仅在直接配置 ENABLE_ENV 对应的环境变量或 Extension Manager 明确启用插件时返回 True。"""
    env = os.environ if environment is None else environment
    direct = env.get(ENABLE_ENV, "").strip().lower()
    if direct in {"1", "true", "yes", "on"}:
        return True
    enabled = {
        item.strip()
        for item in env.get(MANAGER_ENABLED_ENV, "").split(",")
        if item.strip()
    }
    return EXTENSION_ID in enabled


def load_runtime_selection(
    environment: Mapping[str, str] | None = None,
) -> PluginConfig:
    """
    加载 CONFIG_ENV 对应的环境变量直接传入的 JSON、配置路径或管理器保存的配置。
    优先级：显式环境配置（指定的文件路径或 JSON 字符串）→ Extension Manager 配置。
    启用判断与配置读取分开：仅设置 CONFIG_ENV 不会自动启用插件。
    """
    env = os.environ if environment is None else environment
    direct = env.get(CONFIG_ENV)
    if direct:
        candidate = Path(direct).expanduser()
        payload = (
            json.loads(candidate.read_text(encoding="utf-8"))
            if candidate.is_file() # 文件路径且文件存在
            else json.loads(direct) # JSON字符串
        )
        if not isinstance(payload, Mapping):
            raise ValueError(f"{CONFIG_ENV} must resolve to a JSON object")
        return PluginConfig.from_mapping(payload)

    try:
        from vllm_hust_ext.config import load_config
    except ImportError as error:
        raise ValueError(
            f"{CONFIG_ENV} is required when Extension Manager is not installed"
        ) from error
    state = load_config().extension(EXTENSION_ID)
    if not state.enabled:
        raise ValueError(f"Extension Manager does not mark {EXTENSION_ID!r} enabled")
    return PluginConfig.from_mapping(state.configuration)