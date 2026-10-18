# Git 提交规范

[English](git-commit.md) | 简体中文

本指南依据本地 vLLM-HUST 历史和本仓库发布历史总结。vLLM 常用
`[Bugfix][Spec Decode]`、`[Perf][HiSparse]`、`[Attention]`、`[CI]`
等简短领域前缀，再接祈使句摘要。本仓库也使用过 `fix(runtime): ...`、
`docs(dataset-program): ...` 和 `[Release][Ascend] ...`。前缀应准确反映改动，
不要机械套用无关宿主代码的标签。

## 标题与正文

- 一个提交只处理一个逻辑改动。标题用祈使句说明效果，尽量不超过 72 个字符。
  例如 `perf(triattention): batch TP score reduction`、
  `docs(benchmark): index procedures and archive 0.9 history`。
- 优先使用带明确范围的 `fix`、`feat`、`perf`、`docs`、`test`、
  `refactor`、`build`、`ci`、`release`；若更清楚，也可使用 vLLM 风格的
  `[Perf][Ascend] ...`。
- 非简单修复或优化在标题后空一行写正文，说明原行为、精确改动、正确性检查、
  B0/B1 实测变化或不可变证据链接。初步结果须写清限制，不夸大加速效果。
- 适用时在正文关联 issue 或 PR。发布提交要标明精确版本和门槛证据，不能把无关
  功能修改藏进发布提交。

## 提交前

检查 `git status` 和完整暂存 diff，保留用户已有改动；运行与改动相关的检查和
`git diff --check`。模型文件、原始 prompt、日志、凭据及 `dev/` 不入库。
作性能结论前，从[测试规程索引](../benchmark/README.zh.md)选择适用规程，
按其对照规则测量，并在提交正文中链接结果证据。
作者身份必须来自维护者指令或本仓库 Git 配置，不得自行编造；未经授权不 amend
或 push。
