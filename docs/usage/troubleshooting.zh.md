# 故障排查

[English](troubleshooting.md) | 简体中文 | [用户指南](README.zh.md)

| 现象 | 检查与处理 |
| --- | --- |
| `extension validate` 或 `extension check` 拒绝宿主 | 对照[已验证版本线](installation.zh.md)检查 vLLM、vLLM-Ascend、Manager、Python 和 Triton-Ascend。安装匹配的宿主栈，不要绕过 manifest。 |
| Qwen3.5 worker 无法导入 `fla_npu` | 按宿主版本安装与 NPU 架构匹配的 FLA NPU 产物，校验哈希，并在服务环境测试 `import fla_npu.ops.ascendc`。 |
| 插件已安装但未启用 | 检查 `extension status` 与 `extension check`；用 `vllm-hust-ext run --` 启动；限制插件发现时设置 `VLLM_PLUGINS=ascend,ascend_kvcompress`。切换状态后重启。 |
| 启动时报预算与页不对齐 | 查看实际全注意力页大小。预算须同时被该页大小和 128 整除；仓库中的 Qwen3.5 示例在 1152 页采用 4608。8192 可用于 2048 页对齐，但缺少配对性能证据。 |
| 校准无法加载模型或写文件 | 检查模型与 tokenizer 路径、revision、可见 NPU、目录权限和 `calibration_local_files_only`。Qwen3.5 BF16 需两张卡。 |
| 现有统计文件被拒绝 | 检查模型/权重绑定与文件完整性；为目标模型在新路径生成文件，不要复用不匹配的 `.pt`。 |
| 服务可用但没有压缩 commit | 确认插件已启用、上下文超过 `kv_budget + recompute_window`，且请求的 `max_tokens` 达到最少输出门槛。核对同一请求 ID 的 scheduler 与 worker 日志。 |
| NPU 显存不足 | 调低服务 KV 内存配额或并发，清理占卡的其他模型进程，并确认服务权重加载前临时校准模型已释放。 |
| 某个功能组合启动时被拒绝 | 查看[冲突矩阵](../../README.zh.md#冲突矩阵)。先使用简单服务配置，再逐项开启前缀缓存、异步调度或 MTP2。 |

报告问题时保留完整启动错误和 `extension check` 输出，附宿主及包版本、模型
revision、去除敏感信息后的配置和首个失败请求。运行边界见
[设计说明](../design/methods.zh.md)。
