# 项目归属与许可证

[English](ownership-and-licensing.md) | 简体中文

## 项目归属

- 学校：华中科技大学（HUST）
- 课题组：CGCL
- 指导教师：万瑶教授
- 主要负责人：刘思辰（[@Seas0](https://github.com/Seas0)）
- 当前维护者：张家万（[@Jiawan23](https://github.com/Jiawan23)）、
  韦若皓（[@kotoriqaq0](https://github.com/kotoriqaq0)）、刘思辰
  （[@Seas0](https://github.com/Seas0)）

维护团队同意持续维护 vLLM-HUST 和 vLLM-Ascend-HUST 的版本兼容性，并把
本项目作为独立 Extension Manager 插件发布。本项目不应被描述成内置于任一
宿主仓库的代码。

## 算法与实现来源

评分思想源自采用 Apache-2.0 许可证的
[TriAttention 仓库](https://github.com/WeianMao/triattention)；`NOTICE` 将
来源固定到提交 `a4bc3c8f709db60f016ef42c3feb290fd0c00c1b`，并列出对应论文。
本仓库针对 Ascend 分页 KV 存储重新实现运行时，没有复制上游 CUDA runtime
kernel。本地实现、打包、测试和文档均采用 Apache-2.0。

## 可再分发边界

| 材料 | 源码仓包含 | Python wheel 包含 | 再分发说明 |
| --- | --- | --- | --- |
| 插件源码、测试、示例、文档 | 是 | 仅源码和 manifest | Apache-2.0；须保留 `LICENSE` 与 `NOTICE` |
| Extension Manager manifest | 是 | 是 | Apache-2.0 |
| 已提交的历史 `.pt` 统计 | 是 | 否 | 模型派生聚合数据且溯源不完整；仓库许可证本身不授予原始模型/数据集的再分发权 |
| 模型权重、tokenizer 资产 | 否 | 否 | 按其上游条款另行获取 |
| 内置 bootstrap 校准文本 | 在生成器源码中 | 是 | 项目原创 Apache-2.0 文本，不代表生产质量保证 |
| 使用者校准或 benchmark 输入 | 否 | 否 | 使用者自行确认使用和再分发权利 |
| 新生成 `.pt` 统计 | 否 | 否 | 运行环境拥有的模型派生数据；分发需满足模型/数据条款并附完整溯源 |
| 原始服务日志、prompt、生成结果、profiler trace | 否 | 否 | 属于运行环境材料，本仓库未授予再分发权 |
| 公开 benchmark 脚本与聚合证据摘要 | 是 | 否 | 项目代码/文档采用 Apache-2.0；上游 benchmark 正文仍遵循各源数据集条款 |
| `docs/resuts*.md` 历史测量 | 是 | 仅作为文档 | 文档采用 Apache-2.0，但不是 0.4 版本验收证据 |

分发校准产物前，至少应记录 SHA-256、模型及其 revision、tokenizer revision、
生成器提交、输入数据来源、token 数、生成日期和责任维护者。现有两个遗留产物
不具备完整信息，应视为开发样例，而非可普遍再分发的生产资产。
