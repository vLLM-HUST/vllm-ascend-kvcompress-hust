# SPDX-License-Identifier: Apache-2.0
"""vLLM Ascend 接入、宿主约束与 ABI 兼容实现。

子模块按需导入，包初始化不安装钩子，也不提前加载设备依赖。
"""
