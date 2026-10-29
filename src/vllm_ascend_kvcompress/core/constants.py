# SPDX-License-Identifier: Apache-2.0
"""插件各模块共享的常量定义，具体含义与适用范围见各常量注释。"""

# 每个注意力内核缓存块容纳的 token 数，不是字节数。
# 这是插件当前支持的布局约束，不代表所有 Ascend 硬件的固定要求。
# 调度管理器的逻辑页可包含多个内核块；跨缓存组对齐粒度另行计算。
KERNEL_CACHE_BLOCK_SIZE = 128
