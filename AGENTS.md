# Repository working rules

## Documentation / 文档

- Keep authored `docs/**/*.md` documents in English and Chinese pairs:
  `name.md` and `name.zh.md`. The HTML leaderboard and machine-readable
  evidence are shared language-neutral artifacts. / `docs` 中自行编写的
  Markdown 文档必须保持 `name.md`、`name.zh.md` 中英成对；HTML 榜单和机器可读
  证据可共用。
- Put user-facing instructions in `docs/usage/`, benchmark protocols and their
  index in `docs/benchmark/`, design in
  `docs/design/`, dated optimization records in `docs/optimization/`,
  development guidance in `docs/development/`, current result
  summaries and aggregate evidence in `docs/results/`, issue notes in
  `docs/issue/`, and report templates in `docs/template/`. In `docs/archive/`,
  use `benchmark/` for retired protocols, `results/` for dated measurements,
  `design/` for superseded architecture, and `development/` for old procedures.
  / 用户指南放在 `docs/usage/`；当前测试、设计、优化记录、开发、结果、议题、模板分别归入上述目录；归档内仍按旧规程、
  旧结果、旧设计和旧开发流程分类。
- Keep `docs/README.md` and `docs/README.zh.md` as the navigation entry points.
  When moving a document, repair links in both languages, root READMEs, scripts,
  and tests. Preserve the original version, host, and model identity in dated
  evidence; never rewrite a negative result into a current claim. / 移动文档时
  同步修复链接，历史结果保留原始版本、宿主、模型和负结果。

## Development / 开发

- If present, read [AGENTS.dev.md](AGENTS.dev.md) for personal workspace
  instructions. It is intentionally ignored by Git and is not a repository
  requirement for other machines. / 若本机存在 `AGENTS.dev.md`，还需阅读其中
  个人工作区说明；它被 Git 忽略，不是其他机器必须具备的仓库文件。

- `dev/` holds personal temporary documents and data and is excluded from Git.
  Put weekly reports in `dev/reports/weekly/`, monthly reports in
  `dev/reports/monthly/`, and some run/test outputs in `dev/results/`. /
  `dev/` 存放个人临时文档与数据，不参与 Git 提交；周报和月报分别在上述目录，
  部分运行与测试结果放入 `dev/results/`。
- Preserve pre-existing worktree changes. For performance claims, choose the
  applicable protocol from [benchmark protocols](docs/benchmark/README.md)
  before measuring. Record its baseline and candidate conditions, primary
  metrics, and regressions. A kernel microbenchmark alone is not a service
  result. / 保留已有工作树改动；性能结论须先选定适用规程，记录基线与候选条件、
  主要指标及退化；单算子微基准不能替代服务结果。
- Follow [commit conventions](docs/development/git-commit.md) when a commit is
  authorized. Do not commit or publish merely because a version string was
  edited. / 获授权提交时遵循提交规范；改版本号不等于获授权提交或发布。
