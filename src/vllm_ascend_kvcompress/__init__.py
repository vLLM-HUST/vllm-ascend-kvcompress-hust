# SPDX-License-Identifier: Apache-2.0
"""面向 vLLM-Ascend-HUST 的可扩展 KV 缓存压缩方法。"""

# 包级入口只导出 provider 名和方法注册/发现接口；宿主激活入口在 host.plugin.register。
# 内置方法只登记延迟工厂；算法实现和第三方 entry point 在后续按需加载。
# 职责划分：core 管理共享描述，methods 实现算法，runtime 管理压缩事务，
# host 对接宿主；extensions 提供旧扩展迁移支持（需主动启用）。
# 各子包初始化不集中导出运行时实现，避免公共 API 导入时加载宿主和设备依赖。
from .core.config import PROVIDER_NAME
from .methods import CompressionBackend, available_methods, register_method

__all__ = [
    "CompressionBackend",
    "PROVIDER_NAME",
    "available_methods",
    "register_method",
]
__version__ = "0.10.0"
