# 压缩方法架构

[English](methods.md) | 简体中文

## 分层

插件包含三个明确层次：

1. `plugin.py` 安装幂等的 vLLM-Ascend hook。
2. `provider.py` 负责公共昇腾缓存协议和 vLLM-HUST 事务生命周期。
3. 每个 `methods/<name>/` 包负责一种算法的配置、兼容性、状态、评分和 KV
   物化。

公共包根目录不得放置内置方法专用的评分、校准或缓存操作模块。方法 registry
只依赖 `methods/base.py` 中的稳定协议，因此新增或修改算法不会扩大公共
provider 的依赖面。

内置 TriAttention 包展示了推荐组织方式：

| 路径 | 职责 |
| --- | --- |
| `methods/triattention/__init__.py` | 方法的公开导出和 factory |
| `methods/triattention/config.py` | 方法专用选项解析和校验 |
| `methods/triattention/method.py` | `KVCompressionMethod` 实现 |
| `methods/triattention/scoring.py` | 向量化 token 评分 |
| `methods/triattention/stats.py` | 校准产物加载和校验 |
| `methods/triattention/cache.py` | 分页缓存 gather 和物化 |

外层 vLLM provider 固定为 `ascend_kvcompress`。扁平的
`provider_config.method` 标量选择已注册的方法，其余标量选项原样传递给该
方法的 factory。

## 方法协议

实现 `methods/base.py` 中的 `KVCompressionMethod`：

- `name`：稳定的配置和 registry 名称。
- `runtime_spec`：scheduler 可见的阈值、recompute window、最大物理长度和
  私有目标要求。
- `compatibility_reasons(worker)`：无副作用的方法专用检查。
- `bind_model_runner(runner, layer_caches)`：公共缓存布局检查和分配完成后
  初始化状态。
- `compress(request)`：物化一次最终 prefill 事务，返回物理长度以及可选的
  逐层长度。

公共 provider 会在创建 scheduler plan 前验证方法结果。方法不得修改
scheduler 所有权或 model-runner block table。
如果方法返回逐层物理长度，必须完整且不重复地报告每个已绑定层；每个值都必须
为正数，且不能超过全局物理长度。

## 进程内注册

```python
from collections.abc import Mapping
from typing import Any

from vllm_ascend_kvcompress import register_method
from vllm_ascend_kvcompress.config import JsonScalar
from vllm_ascend_kvcompress.methods.base import KVCompressionMethod, ModelShape


def create_method(
    options: Mapping[str, JsonScalar],
    vllm_config: Any,
    model_shape: ModelShape,
) -> KVCompressionMethod:
    return MyCompressionMethod(options, vllm_config, model_shape)


register_method("my_method", create_method)
```

## 第三方 Entry Point

外部包无需提前导入插件即可注册：

```toml
[project.entry-points."vllm_ascend_kvcompress.methods"]
my_method = "my_package.method:create_method"
```

使用配置：

```json
{
  "schema_version": 1,
  "provider": "ascend_kvcompress",
  "provider_config": {
    "method": "my_method",
    "option_owned_by_my_method": 128
  }
}
```

名称会规范为小写，可包含字母、数字、连字符和下划线。重复名称、未知方法、
无效 factory 以及 factory/方法名称不一致都会按 fail-closed 原则失败。

## 方法检查清单

- 只解析自身拥有的选项，并拒绝未知键。
- 在 KV 分配前返回确定的正数 scheduler 限制。
- 在 `compatibility_reasons` 中验证模型和校准约束。
- 只使用 provider 传入且已经验证的 `LayerCache` binding。
- 返回前将结果物化到提供的目标 block。
- 返回不超过声明最大值的有效物理长度。
- 添加 registry、配置、兼容性、物化和事务测试。
- 运行长上下文压缩关闭/开启匹配 benchmark；稳定公开行为写入发布文档，原始
  机器专用日志放入 `docs/dev/`。
