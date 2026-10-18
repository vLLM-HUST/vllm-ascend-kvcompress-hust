# Git commit conventions

English | [简体中文](git-commit.zh.md)

This guide summarizes the style observed in the local vLLM-HUST history and
this repository's release history. vLLM commonly uses a short area prefix such
as `[Bugfix][Spec Decode]`, `[Perf][HiSparse]`, `[Attention]`, or `[CI]`, followed
by an imperative summary. This plugin has also used `fix(runtime): ...`,
`docs(dataset-program): ...`, and `[Release][Ascend] ...`. Keep a consistent,
reviewable subject instead of mechanically copying a prefix from unrelated
host code.

## Subject and body

- Use one logical change per commit. State the effect in the imperative mood,
  preferably within 72 characters. Examples: `perf(triattention): batch TP score
  reduction` and `docs(benchmark): index procedures and archive 0.9 history`.
- Use `fix`, `feat`, `perf`, `docs`, `test`, `refactor`, `build`, `ci`, or
  `release` with a focused scope. A vLLM-style `[Perf][Ascend] ...` subject is
  also acceptable when it makes the affected area clearer.
- For a nontrivial fix or optimization, put a blank line after the subject.
  Explain the previous behavior, the exact change, correctness checks, and
  measured B0/B1 effect or a link to immutable evidence. Include a benchmark
  limitation when the result is preliminary. Avoid unsupported speedup claims.
- Reference issues or PRs in the body when applicable. Release commits identify
  the exact version and gate evidence; they must not be used to hide unrelated
  feature work.

## Before creating a commit

Inspect `git status` and the complete staged diff. Preserve pre-existing user
changes. Run the checks relevant to changed code, then `git diff --check`.
Keep generated model files, raw prompts, logs, credentials, and `dev/` out of
Git. Before writing a performance claim, select the applicable protocol from
the [benchmark index](../benchmark/README.md), follow its comparison rules,
and link the result evidence in the commit body. Set the author identity from
the maintainer's instruction or repository-local Git configuration. Never
invent an author identity or amend/push a commit without authorization.
