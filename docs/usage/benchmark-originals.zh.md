# benchmark 原件

## 榜单与历史混合结果

[历史清单](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/manifest.json)
保留原结果目录。HTML 读取 benchmark 固定 JSON；本地校验使用：

```bash
python scripts/fetch_benchmark_originals.py /absolute/external/originals
export VLLM_HUST_BENCHMARK_ORIGINALS=/absolute/external/originals
python -m pytest tests/test_leaderboard.py tests/test_benchmark_inputs.py
python scripts/kvcompress_leaderboard.py --check
```

重新生成须显式指定外部新文件 `--output` 和 `--frontier-output`；缺文件或哈希变化会失败。
仅有方法代码引用的历史算子行仍标明原始计时未提供，不伪称已恢复。

本次仅迁移指定历史集合；原始版本、失败、负收益和诊断身份不变，不代表新的运行验收。
固定清单与原件入口见[英文配对说明](benchmark-originals.md)。其他历史集合仍单列处理。

下载到源码仓库之外的空目录：

```bash
python scripts/fetch_benchmark_originals.py /absolute/external/originals
```

工具核验固定清单、逐文件长度与 SHA256，拒绝不安全路径，不执行归档代码。
CI 持续回验并阻止已迁路径重新进入 Git。副本可从 benchmark 固定原件或历史 Git 恢复。
