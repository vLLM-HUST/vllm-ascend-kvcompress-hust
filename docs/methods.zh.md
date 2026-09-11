# 压缩方法架构

[English](methods.md) | 简体中文

## 运行时分层

0.4 是自包含插件，不再使用已删除的 vLLM-HUST 压缩生命周期：

1. `plugin.py` 是显式启用的 `vllm.general_plugins` 入口。调度侧立即挂接，
   Ascend worker 在对应模块真正加载时才挂接，因此管理器和纯 API 进程不会
   提前导入 NPU runner。
2. `stateful.py` 管理 scheduler/KV manager 状态。在同步调度边界提交已完成的
   压缩，并释放旧 block table 尾部。
3. `provider.py` 校验当前宿主对象、在 worker 镜像事务、把语义位置转换为物理
   slot，并把选择和物化委托给具体方法。
4. `methods/<name>/` 负责算法选项、兼容性检查、评分、校准和 KV 物化。

适配层明确依赖主 README 列出的宿主内部符号。这些接口尚未冻结，所以只能在
已测试版本线上提供支持；接口变化时必须拒绝启动并重新验收。

## 配置契约

Extension Manager 保存一个 JSON 对象，必须包含 `schema_version`、`provider`、
`method` 和 `method_config`。完整示例见
[`examples/triattention.json`](../examples/triattention.json)。未知字段、未知方法、
不满足 block 对齐的值及不存在的校准文件都会报错，不会被静默忽略。

## 方法契约

实现 `methods/base.py` 中的 `KVCompressionMethod`：

- `name`：稳定的注册和配置名称；
- `runtime_spec`：供 scheduler 适配层使用的 block 对齐阈值、重算窗口、最大
  物理长度和目标区要求；
- `compatibility_reasons(worker)`：无副作用的算法兼容检查；
- `bind_model_runner(runner, layer_caches)`：公共缓存校验后分配状态；
- `compress(request)`：同步完成一次物化并返回新物理长度，以及可选的逐层长度。

方法只能使用已校验的 cache binding，不能修改 scheduler 拥有的 request 或
block table。返回长度必须为正、不超过声明上限；使用逐层长度时必须覆盖全部层。

## 注册其他方法

仓库内部可调用：

```python
from vllm_ascend_kvcompress import register_method

register_method("my_method", create_method)
```

外部包可声明：

```toml
[project.entry-points."vllm_ascend_kvcompress.methods"]
my_method = "my_package.method:create_method"
```

名称只允许小写字母、数字、连字符和下划线；重复名称、非法 factory 或名称不匹配
都会失败关闭。

## 验收清单

- 只解析本方法拥有的选项，拒绝未知值；
- scheduler 限制必须确定且满足 block 对齐；
- 服务启动前校验模型、RoPE、校准、dtype 和 cache layout；
- `compress` 返回前完成所有层的物化；
- 补齐注册、配置、事务和数值测试；
- 在精确宿主 revision 上完成压缩关闭/开启的正确性、质量、吞吐、时延和 HBM
  对照验收。
