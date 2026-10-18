# 用户使用指南

[English](README.md) | 简体中文

首次部署请按顺序阅读本组文档。选择与宿主匹配的包版本和配置；
仓库中的示例描述该源码树，可能与旧版发布包不同。
宿主栈和模型权重需要单独安装；插件包不提供这两者。

1. [安装并检查兼容性](installation.zh.md)。
2. [选择配置并启用插件](configuration.zh.md)。
3. [生成或复用校准统计](calibration.zh.md)。
4. [启动服务并确认压缩生效](serving.zh.md)。
5. [排查启动和运行问题](troubleshooting.zh.md)。

[Qwen3.5-35B-A3B 示例](../../examples/qwen3.5-35b-a3b-triattention.json)
针对已验证的 1,152-token 注意力页布局。使用其他宿主布局前，请阅读配置指南的
页对齐说明。性能结论需要选择适用的
[测试规程](../benchmark/README.zh.md)，并在[结果记录](../results/README.zh.md)
中保存证据；一次请求成功只能说明服务可用。
