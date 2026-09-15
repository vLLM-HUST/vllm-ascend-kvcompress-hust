# 打包与发布指南

[English](packaging-and-release.md) | 简体中文

本仓库遵循 BidKV 打包流程。PyPI distribution 为
`vllm-ascend-kvcompress-hust`，Python 包为 `vllm_ascend_kvcompress`，稳定的
Extension Manager ID 为 `org.vllm-hust.ascend-kvcompress`。

## 发布门槛

发布前必须保持项目版本、`__version__` 和 manifest 版本一致；完成 CPU、包、NPU、
生命周期及声明的服务测试；冻结宿主、模型、校准、数据来源；审核完整 diff。PyPI
文件不可覆盖，任何代码变更都必须提升版本。

0.6.0 是实验性发布版本。当前验收仅支持有边界的工程性能结论，不支持 V4.6 正式
通过或生产可用声明。

## 构建与检查

优先采用 BidKV 的 `uv` 流程：

```bash
uv build --no-sources --out-dir dist
```

若发布环境没有 `uv`，已验证的等价回退方式为：

```bash
python -m build --no-isolation --outdir dist
```

随后只检查本次候选文件：

```bash
python -m zipfile -l \
  dist/vllm_ascend_kvcompress_hust-0.6.0-py3-none-any.whl
python -m twine check \
  dist/vllm_ascend_kvcompress_hust-0.6.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.6.0.tar.gz
sha256sum dist/vllm_ascend_kvcompress_hust-0.6.0*
```

wheel 必须包含代码、`LICENSE`、`NOTICE` 和
`manifests/vllm-hust-extension-v0.2.json`；entry-point metadata 必须同时包含
`vllm.general_plugins`、`vllm_hust.extension_bundles` 和
`vllm-ascend-kvcompress-calibrate` 命令入口。两个发行包均不得包含
`artifacts/*.pt`、原始数据、服务日志或历史 `docs/dev/results`；sdist 只可包含
`docs/evidence` 下的聚合证据摘要。

## 隔离生命周期

在支持的宿主环境中安装 wheel，且不把源码 checkout 加入 `PYTHONPATH`：

```bash
python -m pip install --no-deps \
  dist/vllm_ascend_kvcompress_hust-0.6.0-py3-none-any.whl
vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /absolute/path/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension check org.vllm-hust.ascend-kvcompress
vllm-hust-ext run --dry-run -- python -c 'print("manager-run-ok")'
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension forget org.vllm-hust.ascend-kvcompress
python -m pip uninstall -y vllm-ascend-kvcompress-hust
```

wheel 的核心 `Requires-Dist` 必须为空，尤其不得把 `vllm`、`vllm-ascend`、
Triton-Ascend、NumPy 或 OpenCV 加入插件核心依赖。加速器宿主由另一套经过验证的
锁定清单部署，扩展清单负责声明兼容宿主范围。安装插件时重新解析宿主，可能把
仅支持 NumPy 1 的 Triton-Ascend 3.2.2 与当前 vLLM 中仅支持 NumPy 2 的 OpenCV
约束混在一起。

确认 `extension list` 不再发现插件。然后重装、配置并启用将要发布的精确 wheel。

## 上传与发布后验证

使用由 PyPI `intellistream` 组织/项目授权的项目级 token，不在命令行明文传递，也不
提交到仓库：

```bash
export UV_PUBLISH_TOKEN='<从密钥存储读取>'
uv publish --check-url https://pypi.org/simple \
  dist/vllm_ascend_kvcompress_hust-0.6.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.6.0.tar.gz
unset UV_PUBLISH_TOKEN
```

若受保护发布器使用 Twine，则从密钥存储设置 `TWINE_USERNAME=__token__` 和
`TWINE_PASSWORD`，上传同样两个明确文件。最后从正式 PyPI 无缓存安装 0.6.0，
核对哈希、Manager 发现/启用、`/health` 和一个正确性用例。
