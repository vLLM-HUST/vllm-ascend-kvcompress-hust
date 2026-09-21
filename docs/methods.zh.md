# 压缩方法架构

[English](methods.md) | 简体中文

## 运行时分层

0.6 是自包含插件，不再使用已删除的 vLLM-HUST 压缩生命周期：

1. `plugin.py` 是显式启用的 `vllm.general_plugins` 入口。调度侧立即挂接，
   Ascend worker 在对应模块真正加载时才挂接，因此管理器和纯 API 进程不会
   提前导入 NPU runner。
2. `calibration.py` 在服务权重加载前生成缺失的模型绑定产物，也提供相同生成器的
   命令行入口。
3. `stateful.py` 管理 scheduler/KV manager 状态。在同步调度边界提交已完成的
   压缩，并释放旧 block table 尾部。
4. `provider.py` 校验当前宿主对象、在 worker 镜像事务、把语义位置转换为物理
   slot，并把选择和物化委托给具体方法。
5. `methods/<name>/` 负责算法选项、兼容性检查、评分、校准和 KV 物化。

适配层明确依赖主 README 列出的宿主内部符号。这些接口尚未冻结，所以只能在
已测试版本线上提供支持；接口变化时必须拒绝启动并重新验收。

## 配置契约

Extension Manager 保存一个 JSON 对象，必须包含 `schema_version`、`provider`、
`method` 和 `method_config`。完整示例见
[`examples/triattention.json`](../examples/triattention.json)。未知字段、未知方法、
不满足 block 对齐的值及无效校准文件都会报错，不会被静默忽略。仅在
`auto_calibrate` 启用时生成缺失文件，详见校准产物文档。

`min_output_tokens_for_compression` 是非负的请求输出门槛。当请求的最大生成长度低于
该值时，scheduler 和 worker 都跳过事务。`0` 保持原有的始终可压缩行为；公开
benchmark 推荐值为 `64`，可消除 32-token 检索负载上的无效压缩开销。

标准运行时的 scheduler、全注意力 manager 与内核缓存块均为 128 token。对于
明确支持的 Qwen3.5 混合模型，Ascend 使用 32,768-token 跨组 scheduler 对齐，
把全注意力 manager 页提升为 2,048 token，同时保留 128-token 内核块。适配层先
校验 scheduler 对齐等于所有 manager block size 的最小公倍数，再在调用 method 前
按 16:1 展开注意力 block ID。method 的最大物理 token 数必须能被 2,048-token
注意力页整除，但无需被跨组对齐整除。

`score_layer_stride=8` 会从 48 个已校准层中均匀抽取 6 层用于选择，但仍会物化
全部 48 层 K/V。在冻结的公开 Qasper 与 LongBench-v2 集合上，该配置保持了基线
质量，并比原值 `4` 获得更好的同机端到端结果。

## V3 位置策略

`position_policy=global` 保留旧的全局 top-k 行为；显式选择 `v3` 时执行论文的
三段式位置策略：

1. 开头 `protected_prefix_window` 个 token 强制保留；
2. 末尾 `protected_recent_window` 个 token 强制保留；
3. 中间上下文切成 `position_segments` 个等长区间，把全局驱逐数按比例精确分配
   到各区间，再在区间内保留得分最高的 token。

实现使用整数累计配额，所以即使区间长度不完全相同，最终也严格保留
`kv_budget` 个位置。

## Qwen3.5 混合架构路径

Qwen3.5-35B-A3B 的 40 个文本层由 10 组“3 个 Gated-DeltaNet 层 + 1 个全注意力
层”组成。方法只绑定第 3、7、…、39 层的全注意力缓存，循环状态组保持不变。
其 256 维注意力头只有 64 维使用 RoPE，因此校准产物会为剩余 192 维保存
`q_pass_mean`，Torch 回退评分再把这部分直接内容点积加入三角评分；该布局不会误用
只支持完整 RoPE 的融合评分算子。

张量并行时，校准行按本地 KV 头切分，并在选择前对逐层分数执行最大值 all-reduce，
保证各 rank 选择相同位置。TP 大小必须整除 KV 头数。其他混合布局、`none` 之外的
Mamba mode 以及 PP/DP/DCP/PCP 仍会拒绝启动。

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
