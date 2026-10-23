# 打包与发布

[English](packaging-and-release.md) | 简体中文

本流程适用于各次候选发布。从 `pyproject.toml` 读取发行版本，上传前检查
软件包索引，并复核候选版本的[结果证据](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/README.zh.md)。
[0.9 历史发布配方与证据](../archive/development/packaging-and-release-0.9.zh.md)
仅供复核，后续版本不得沿用其中固定的文件名。

## 包契约

发行名为 `vllm-ascend-kvcompress-hust`，导入包为
`vllm_ascend_kvcompress`，稳定 Extension Manager ID 为
`org.vllm-hust.ascend-kvcompress`。`pyproject.toml`、
`src/vllm_ascend_kvcompress/__init__.py` 与 manifest 中的插件版本必须相同。
清单文件名 `manifests/vllm-hust-extension-v0.3.json` 中的 `0.3` 是 schema
版本，不是插件版本。匹配的加速宿主另行部署，因此 wheel 核心依赖保持为空；
可选 manager extra 面向 Extension Manager 0.3。

## 候选检查

准备候选版本时，冻结精确宿主、CANN/PyTorch 栈、模型 revision、校准
SHA-256、负载文件哈希、配置和源码 revision。完成 CPU/NPU 正确性测试、隔离
wheel 安装、manager 的发现／配置／启用／检查／禁用／忘记生命周期，以及声明的
服务测试。根据发布声明从[测试规程索引](../benchmark/README.zh.md)
选定所需规程，并在测量前记录验收指标。保留所有退化、
原始报告和服务日志，存于临时构建目录之外的持久存储。

验证完成后，从干净候选工作树构建：

```bash
uv build --no-sources --out-dir dist
python -m twine check dist/*.whl dist/*.tar.gz
python scripts/kvcompress_leaderboard.py --check
```

检查精确的 wheel/sdist，确认清单、入口点、`LICENSE`、`NOTICE` 齐全，
且不含模型权重、`.pt` 校准文件、原始 prompt、日志或 `dev/`。记录哈希，并在
匹配宿主中不加源码 `PYTHONPATH`、使用 `--no-deps` 安装 wheel。包索引不允许
覆盖已上传版本；后续代码变化需升版并重新测试。

只有明确授权的发布任务才能提交、推送、上传和验证正式 PyPI。Git 作者身份
按维护者指令设置，并遵循[提交规范](git-commit.zh.md)。PyPI token 不得出现在
命令行参数、日志、仓库文件或提交中。
