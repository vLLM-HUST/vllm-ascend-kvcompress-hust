# 贡献指南

[English](CONTRIBUTING.md) | 简体中文

感谢你改进 vLLM Ascend KV Cache Compression。请保持修改聚焦，对不支持的
运行时组合采用 fail-closed 原则，并维护与具体压缩方法无关的 provider 边界。

## 开发环境

先安装匹配的 editable vLLM-HUST 和 vLLM-Ascend-HUST 包，然后执行：

```bash
uv pip install -e '.[test]'
```

## 架构规则

- 公共 vLLM/昇腾生命周期逻辑放在 `provider.py` 或 `plugin.py`。
- 算法专用配置、产物、兼容性检查、评分和物化逻辑放在
  独立的 `methods/<method>/` 包中。
- 每种方法的公开 factory 放在 `methods/<method>/__init__.py`；按需将其配置、
  运行时适配、评分、统计和缓存操作拆分到职责单一的模块。
- 新算法必须实现 `KVCompressionMethod`；不要在公共 provider 中加入基于
  方法名称的条件分支。
- provider 配置保持扁平 JSON 标量映射，这是 vLLM-HUST schema v1 协议。

详见[压缩方法架构](docs/methods.zh.md)。

## 必需检查

```bash
python -m pytest -q
ruff check .
ruff format --check .
python -m compileall -q src tests
git diff --check
```

运行时改动需要在空闲且已分配的 NPU 上执行长上下文压缩关闭/开启匹配测试。
确认所有服务进程退出并释放 NPU。不得为了让本插件通过测试而修改相邻的 HUST
仓库。

## 文档

公开、可发布的行为放在 README 或 `docs/` 下的公开文件中。中间测试报告和
机器专用开发日志放在被 git ignore 的 `docs/dev/`；生成的校准产物放在被
git ignore 的 `artifacts/`。每次修改公开英文文档时，必须同步更新对应的
`.zh.md` 中文版本。
