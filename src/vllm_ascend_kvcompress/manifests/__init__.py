"""静态 Extension Manager 清单包，可在不导入运行时代码时安全发现。"""

# 此包只承载 Extension Manager 静态 JSON；运行入口仍是 plugin.register。
# manifest 声明宿主范围、执行平面及 KV mutation 独占资源，不保存请求状态。
