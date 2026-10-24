# 0.9 历史打包与发布指南

[English](packaging-and-release-0.9.md) | 简体中文

完整历史测量、发布门槛及失败记录见 [benchmark 原文](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/5eeccc2d69b2c4a634e273a9f926584c8d252cf7/archive/mod-targeted-20261023/compress-release/snapshot/docs/archive/development/packaging-and-release-0.9.zh.md)。
此页保留历史安装配方；当前发布流程见[现行指南](../../development/packaging-and-release.zh.md)。
旧结果仅为单次工程观测，不代表当前正式成绩；0.8 wheel 的 eager 短测不代表图编译通过，
原编译失败保持不变。原 JSON 中的 B0/B1 是当时配对标签，不替代固定 0.18/0.18 B0 定义。

## 原始证据直链

以下为当时保存的配对 JSON，不冒充逐请求日志。JSON 中仅记录哈希的外部运行日志及
0.8 wheel 短测原始日志，在此已下载集合中没有对应原件；保留该可用性限制。

- [docs/results/evidence/kvcompress-qwen35-20260927-agentx-postmerge-positive-pair.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20260927-agentx-postmerge-positive-pair.json)
- [docs/results/evidence/kvcompress-qwen35-20260927-swe-postmerge-negative-pair.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20260927-swe-postmerge-negative-pair.json)
- [docs/results/evidence/kvcompress-qwen35-20261008-v090-agentx-pair.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261008-v090-agentx-pair.json)
- [docs/results/evidence/kvcompress-qwen35-20261008-v090-swe-pair.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/results/evidence/kvcompress-qwen35-20261008-v090-swe-pair.json)

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
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl
python -m twine check \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.9.0.tar.gz
sha256sum dist/vllm_ascend_kvcompress_hust-0.9.0*
```

wheel 必须包含代码、`LICENSE`、`NOTICE` 和
`manifests/vllm-hust-extension-v0.3.json`；entry-point metadata 必须同时包含
`vllm.general_plugins`、`vllm_hust.extension_bundles` 和
`vllm-ascend-kvcompress-calibrate` 命令入口。两个发行包均不得包含
`artifacts/*.pt`、原始数据、服务日志或历史 `docs/dev/results`。原 0.9 配方曾随 sdist 携带榜单数据；这不是当前打包要求。历史数据现由 benchmark 保管，
不得重新嵌入发布包；当前流程以[现行指南](../../development/packaging-and-release.zh.md)为准。

## 隔离生命周期

在支持的宿主环境中安装 wheel，且不把源码 checkout 加入 `PYTHONPATH`：

```bash
python -m pip install --no-deps \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl
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
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.9.0.tar.gz
unset UV_PUBLISH_TOKEN
```

若受保护发布器使用 Twine，则从密钥存储设置 `TWINE_USERNAME=__token__` 和
`TWINE_PASSWORD`，上传同样两个明确文件。最后从正式 PyPI 无缓存安装 0.9.0，
核对哈希、Manager 发现/启用、`/health` 和一个正确性用例。
