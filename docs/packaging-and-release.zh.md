# 打包与发布指南

[English](packaging-and-release.md) | 简体中文

本项目遵循 vLLM-HUST 的 BidKV 打包发布流程。PyPI 包名为
`vllm-ascend-kvcompress-hust`，Python 模块名为 `vllm_ascend_kvcompress`，稳定的
Extension Manager ID 为 `org.vllm-hust.ascend-kvcompress`。

## 发布门槛

只要[验收记录](validation.zh.md)中仍存在正确性/质量失败项或声明依赖栈尚未验收，
就不能发布 alpha。打 tag 前必须：

1. 保持 `[project].version`、包 `__version__` 与 manifest
   `extension_version` 完全一致；
2. 通过 CPU suite、Ruff、Ascend 数值 smoke 及验收规程中的完整配对服务矩阵；
3. 记录宿主精确 commit、模型/校准溯源、原始结果和发布 commit；
4. 确认工作树只包含本次发布需要的变更。

PyPI 文件不可覆盖；上传后若源码变化，必须增加版本号。

## 构建与检查

在仓库根目录禁用本地 source override，同时构建 wheel 和 sdist：

```bash
uv build --no-sources --out-dir dist
python -m zipfile -l \
  dist/vllm_ascend_kvcompress_hust-0.3.0-py3-none-any.whl
```

wheel 必须包含 Python 包、`LICENSE`、`NOTICE` 与
`manifests/vllm-hust-extension-v0.2.json`；wheel 和 sdist 都不得包含
`artifacts/*.pt`，因为这些校准文件不具备充分的再分发溯源。`entry_points.txt`
必须包含：

```ini
[vllm.general_plugins]
ascend_kvcompress = vllm_ascend_kvcompress.plugin:register

[vllm_hust.extension_bundles]
org.vllm-hust.ascend-kvcompress = vllm_ascend_kvcompress.manifests
```

环境中有相应工具时，执行发行元数据和哈希检查：

```bash
python -m twine check dist/*
sha256sum dist/*
```

## 隔离生命周期 smoke

在与宿主相同的干净环境安装 wheel 和 Extension Manager，确认发现后完成全部状态
生命周期：

```bash
python -m pip install \
  'vllm-hust-ext>=0.2.0.dev0,<0.3' \
  dist/vllm_ascend_kvcompress_hust-0.3.0-py3-none-any.whl

vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /absolute/path/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension status org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension forget org.vllm-hust.ascend-kvcompress
python -m pip uninstall vllm-ascend-kvcompress-hust
```

卸载后，`extension list` 中不能再出现本扩展。无宿主环境可能把兼容性标记为
unverified，并拒绝受信任进程内扩展的 `run --dry-run`；最终 render 和服务测试
必须在精确宿主环境完成。

## 发布与发布后验证

使用保存在 CI Secret 中、仅授权本项目的 PyPI Token。必须在同一受保护 tag/commit
完成构建、测试和上传，并显式列出本次 wheel 与 sdist：

```bash
export UV_PUBLISH_TOKEN='<从密码库读取>'
uv publish --check-url https://pypi.org/simple \
  dist/vllm_ascend_kvcompress_hust-0.3.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.3.0.tar.gz
unset UV_PUBLISH_TOKEN
```

最后从正式 PyPI 无缓存安装精确版本，确认 Manager 可发现和启用，启动新的
Manager 包装宿主进程，检查 `/health` 并复测一个代表性正确性用例。将正式发布
文件哈希写入验收记录和 release notes。
