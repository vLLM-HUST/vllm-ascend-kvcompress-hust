# SPDX-License-Identifier: Apache-2.0
"""算法无关的压缩运行时，负责 worker、调度事务与坐标换算。

子模块按需导入，包初始化不安装钩子，也不提前加载设备依赖。
"""
