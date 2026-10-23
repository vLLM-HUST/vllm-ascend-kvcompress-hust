# benchmark 原件

本次仅迁移指定历史集合；原始版本、失败、负收益和诊断身份不变，不代表新的运行验收。
固定清单与原件入口见[英文配对说明](benchmark-originals.md)。其他历史集合仍单列处理。

下载到源码仓库之外的空目录：

```bash
python scripts/fetch_benchmark_originals.py /absolute/external/originals
```

工具核验固定清单、逐文件长度与 SHA256，拒绝不安全路径，不执行归档代码。
CI 持续回验并阻止已迁路径重新进入 Git。副本可从 benchmark 固定原件或历史 Git 恢复。
