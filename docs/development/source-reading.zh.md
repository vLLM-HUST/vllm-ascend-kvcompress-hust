# 源码阅读指南

[English](source-reading.md) | 简体中文

## 先建立整体认识

源码根目录只保留公共入口 `__init__.py`，实现按职责归入子包：

```text
vllm_ascend_kvcompress/
├── __init__.py             # 公共 API 与版本
├── core/                   # 共享配置、常量与校验工具
├── methods/                # backend/算法契约与注册表
│   └── triattention/       # 内置算法、校准与内核
├── runtime/                # worker/调度事务及坐标换算
├── host/                   # 模型配置转换、vLLM Ascend 接入与 ABI 兼容
├── extensions/             # 旧扩展迁移支持（需主动启用）
└── manifests/              # Extension Manager 静态声明
```

`CompressionBackend` 描述静态预算并选择运行时工厂；`KVCompressionMethod`
负责 worker 侧算法执行。统一长度的核心 provider 与调度器负责块所有权、
输出确认和长度映射。Query 观测与逐层物理长度由下游实现负责，临时的
`extensions/legacy.py` 适配器将这些功能与默认 TriAttention 路径分开保存。

压缩不会缩短请求的语义历史。假设在语义长度 S=8192 时保留 P=2048 个 KV，
则累计偏移为 S-P=6144；之后新增 token 的语义位置 8192 写到物理位置 2048，
RoPE 仍使用 8192。保留下来的 K 已带有原始 RoPE，不会按密集物理位置重新旋转。
Mamba/GDN 的循环状态继续使用语义长度。

还需区分三种块尺度：scheduler 跨缓存组的 LCM 对齐、全注意力管理器的逻辑页、
设备缓存的 128-token 内核页。混合模型中它们可能不同。

## 文件地图

下表路径均相对 `src/vllm_ascend_kvcompress/`，文件名链接到当前源码位置。

| 文件 | 作用与阅读重点 |
| --- | --- |
| [__init__.py](../../src/vllm_ascend_kvcompress/__init__.py) | 包级公共导出：版本、provider 名、backend 基类以及方法注册与发现接口。 |
| [core/__init__.py](../../src/vllm_ascend_kvcompress/core/__init__.py) | 共享配置、常量与校验工具的包入口，不提前初始化宿主或算法。 |
| [core/config.py](../../src/vllm_ascend_kvcompress/core/config.py) | 显式启用、环境变量或 Extension Manager 配置读取、provider/method 选择；忽略顶层未知字段，将方法选项交给 backend。 |
| [core/constants.py](../../src/vllm_ascend_kvcompress/core/constants.py) | 插件各模块共享的常量定义，具体含义与适用范围见各常量注释。 |
| [core/validation.py](../../src/vllm_ascend_kvcompress/core/validation.py) | 配置选项的通用校验工具，检查整数、布尔类型和字符串允许值。 |
| [methods/__init__.py](../../src/vllm_ascend_kvcompress/methods/__init__.py) | 公共接口 导出与内置工厂注册；具体算法实现和第三方入口点按需加载。 |
| [methods/base.py](../../src/vllm_ascend_kvcompress/methods/base.py) | 模型、静态预算、缓存、请求和结果结构，以及 worker 算法契约；核心结果只描述各层统一的缓存长度。 |
| [methods/backend.py](../../src/vllm_ascend_kvcompress/methods/backend.py) | 静态运行描述，以及可覆盖的产物准备、算法创建、provider 与调度器工厂。 |
| [methods/registry.py](../../src/vllm_ascend_kvcompress/methods/registry.py) | 名称规范化、backend 工厂注册、第三方入口点发现及 backend 类型/名称校验；拒绝注册冲突。 |
| [methods/builtins.py](../../src/vllm_ascend_kvcompress/methods/builtins.py) | 内置方法的延迟工厂与注册，由 methods/__init__.py 装配；registry.py 不包含内置算法名称。 |
| [methods/triattention/__init__.py](../../src/vllm_ascend_kvcompress/methods/triattention/__init__.py) | 内置配置与 backend 导出；worker 实现按需加载。 |
| [methods/triattention/config.py](../../src/vllm_ascend_kvcompress/methods/triattention/config.py) | 预算、重算后缀、保护窗口、评分与选取、层抽样及校准选项；threshold = budget + recompute_window。 |
| [methods/triattention/backend.py](../../src/vllm_ascend_kvcompress/methods/triattention/backend.py) | 解析配置并描述静态预算，不加载校准或分配设备状态；在后续生命周期准备产物并创建 worker 算法。 |
| [methods/triattention/method.py](../../src/vllm_ascend_kvcompress/methods/triattention/method.py) | 绑定 TP 本地统计与缓存、分配工作区、抽样评分层、跨 TP 聚合，并用同一组索引搬运所有层。 |
| [methods/triattention/stats.py](../../src/vllm_ascend_kvcompress/methods/triattention/stats.py) | 加载校准格式、校验模型/RoPE/数值、按 GQA 分组、切分 TP 统计并构造设备值。 |
| [methods/triattention/scoring.py](../../src/vllm_ascend_kvcompress/methods/triattention/scoring.py) | PyTorch 评分公式：未来位置几何采样、post-RoPE key、幅度补偿、部分 RoPE 内容项与每头归一化。 |
| [methods/triattention/selection.py](../../src/vllm_ascend_kvcompress/methods/triattention/selection.py) | global/v3 策略、前缀与近期保护、分段整数配额、批量 top-k 及有序索引输出。 |
| [methods/triattention/cache.py](../../src/vllm_ascend_kvcompress/methods/triattention/cache.py) | token 到槽位映射、分页读取，以及通过可处理源目标重叠的工作区写入所选 token。 |
| [methods/triattention/calibration.py](../../src/vllm_ascend_kvcompress/methods/triattention/calibration.py) | CLI 与自动校准：临时 HF 模型、RoPE 前 Q 捕获、统计验证、进程锁及原子发布。 |
| [methods/triattention/kernels.py](../../src/vllm_ascend_kvcompress/methods/triattention/kernels.py) | 平均相位预计算、直接分页评分，以及融合归一化/query 头最大值/跨层累加。 |
| [runtime/__init__.py](../../src/vllm_ascend_kvcompress/runtime/__init__.py) | 运行时包入口，导入该包不会提前加载 worker 或调度实现。 |
| [runtime/transaction.py](../../src/vllm_ascend_kvcompress/runtime/transaction.py) | 可序列化的 CompressionPlan、共享 CompressionAnchor 和 SchedulerOutput 计划属性；授权与逻辑块号不进入算法输入。 |
| [runtime/coordinates.py](../../src/vllm_ascend_kvcompress/runtime/coordinates.py) | 块数向上取整、逻辑页到内核页展开，以及语义位置到物理位置的偏移换算。 |
| [runtime/provider.py](../../src/vllm_ascend_kvcompress/runtime/provider.py) | worker 缓存绑定、执行授权事务、pending/active 镜像、请求清理、统一注意力长度与槽位换算；提供可覆盖的生命周期方法。 |
| [runtime/scheduler.py](../../src/vllm_ascend_kvcompress/runtime/scheduler.py) | 预留私有目标块、确认、提交块表和释放源块；处理异步冻结、APC、缓存组长度及清理。 |
| [host/__init__.py](../../src/vllm_ascend_kvcompress/host/__init__.py) | 宿主接入包入口，导入时不安装钩子或加载设备依赖。 |
| [host/model_config.py](../../src/vllm_ascend_kvcompress/host/model_config.py) | 提取全局模型维度、原始层号、全注意力层集合和部分 RoPE 维度。 |
| [host/contracts.py](../../src/vllm_ascend_kvcompress/host/contracts.py) | worker 与调度器共用的模型和缓存约束、支持布局、MTP2 实验开关及缓存张量校验。 |
| [host/plugin.py](../../src/vllm_ascend_kvcompress/host/plugin.py) | 注册、延迟加载 runner 与生命周期分发；将 provider/调度状态接入宿主执行、元数据及槽位映射。 |
| [host/compat.py](../../src/vllm_ascend_kvcompress/host/compat.py) | 路由、已删除符号/输出字段、GDN 列表元数据及 MTP2 缓存绑定的宿主 ABI 桥接；与算法准备分离。 |
| [host/qwen_gdn_ops.py](../../src/vllm_ascend_kvcompress/host/qwen_gdn_ops.py) | 缺失 GDN chunk 输出算子的 Triton 实现、Meta 注册及实验 MTP2 卷积参考路径。 |
| [extensions/__init__.py](../../src/vllm_ascend_kvcompress/extensions/__init__.py) | 可选适配器入口；默认运行时不导入其中的实现。 |
| [extensions/legacy.py](../../src/vllm_ascend_kvcompress/extensions/legacy.py) | 需显式选择、已停止维护且计划删除的 迁移适配器；自行持有 Query/逐层请求、结果、pending、active 与元数据状态。 |
| [extensions/legacy_hooks.py](../../src/vllm_ascend_kvcompress/extensions/legacy_hooks.py) | 迁移适配器使用的 Query 自定义算子观测与逐层 FIA 图回放钩子。 |
| [manifests/__init__.py](../../src/vllm_ascend_kvcompress/manifests/__init__.py) | Extension Manager 静态包；manifests/*.json 声明宿主范围、激活、执行平面及 KV 修改独占资源。 |

仓库外围目录：

| 路径 | 作用 |
| --- | --- |
| `pyproject.toml`、`MANIFEST.in`、`MOD_METADATA.json` | 打包、插件/校准/扩展包 entry point 与发行元数据。 |
| `examples/` | 可直接对照解析流程的 TriAttention JSON 配置。 |
| `tests/test_*.py` | CPU 可运行的算法、事务、hook、模型/宿主契约、工具和文档规则测试；部分测试依赖已安装宿主。 |
| `tests/run_npu_*.py` | 显式运行的设备 smoke/benchmark，覆盖 kernel、partial RoPE、GDN 和 Qwen3.5 事务。 |
| `scripts/` | 输入准备、服务压测、质量评分、配对比较、榜单与证据校验，见下面的工具地图；不在插件推理调用链中。 |
| `docs/` | 中英双语用户指南、设计、规程、开发说明和历史入口。 |
| `LICENSE`、`NOTICE`、`LICENSES/` | 许可与派生代码归属，尤其注意 GDN 算子的来源边界。 |
| `dev/`、`dist/`、本地 cache/构建目录 | 忽略的个人资料和生成产物，不作为实现主线阅读。 |

`scripts/` 工具地图：

| 文件 | 作用 |
| --- | --- |
| `kvcompress_ascend_env.sh` | 配置本地 Ascend 环境并启动命令。 |
| `benchmark_inputs.py`、`fetch_benchmark_originals.py` | 解析外置输入与下载固定版本、可校验的历史原件。 |
| `kvcompress_prepare_dataset.py`、`kvcompress_benchmark_data.py` | 准备验收输入、获取并冻结公开长上下文请求。 |
| `kvcompress_dataset_applicability.py` | 判断固定的 tokenized 数据集能否触发压缩。 |
| `kvcompress_benchmark_run.py`、`kvcompress_benchmark_score.py` | 调用 OpenAI 兼容服务并评分公开 LongBench 系列结果。 |
| `kvcompress_long_context_run.py`、`kvcompress_acceptance_compare.py` | 确定性的流式长上下文运行与三次冷启动 B0/B1 验收比较。 |
| `kvcompress_m3_quality.py`、`kvcompress_m3_analyze.py` | M3 质量护栏与三次生命周期聚合。 |
| `kvcompress_frontier_preflight.py` | Frontier 固定测试条件的只读就绪检查。 |
| `kvcompress_frontier_swe_pair.py`、`kvcompress_frontier_agentx_pair.py` | SWE / AgentX 固定运行的配对校验与比较。 |
| `kvcompress_leaderboard.py`、`verify_leaderboard_browser.py` | 构建/校验榜单及浏览器可用性、原件链接与失败展示。 |

## 推荐阅读顺序

按“算法契约 → 内置算法 → 事务运行时 → 宿主接入”阅读。
下列包内路径对应上方文件地图；`examples/` 与 `tests/` 路径从仓库根目录开始。

1. `__init__.py` → `examples/triattention.json` → `core/config.py` → `host/model_config.py`：了解公共导出、启用方式、方法选择和模型维度。
2. `methods/base.py`：先理解 ModelShape、MethodRuntimeSpec、LayerCache、CompressionRequest/Result 和 KVCompressionMethod，再追执行流程。
3. `methods/backend.py` → `methods/registry.py` → `methods/builtins.py`：区分静态 backend 与 worker 算法实例，理解注册、延迟发现和下游工厂覆盖入口。
4. `methods/triattention/config.py` → 同目录的 `backend.py` → `method.py`：串起配置、静态预算、产物准备、实例构造、缓存绑定与压缩，第一遍暂时跳过内核细节。
5. TriAttention 的 `stats.py` → `scoring.py` → `selection.py` → `cache.py`：沿校准统计 → token 分数 → 保留索引 → K/V 搬运阅读。
6. `runtime/transaction.py` → `runtime/coordinates.py`：理解事务计划、锚点、块所有权及语义/物理坐标。
7. `runtime/provider.py`：沿 `bind_model_runner` → `compress_scheduled_requests` → pending 暂存 → `before_update_states` → 行偏移/槽位/注意力长度更新阅读，关注请求构造、算法执行和状态激活的覆盖点。
8. `runtime/scheduler.py`：沿 `after_schedule` → `after_output` → `_commit_pending` 阅读，再看请求清理、延迟释放、异步冻结、APC 和缓存组长度钩子。
9. `host/contracts.py` → `host/plugin.py`：先理解支持的宿主与缓存布局，再回到 `register`，串起调度器和 provider 如何接入真实宿主。
10. `host/compat.py` → `host/qwen_gdn_ops.py`：理解主事务链后，再看宿主 ABI 与 GDN 兼容。

第二遍深入时，读 `methods/triattention/calibration.py` 理解校准产物生成，
读 `methods/triattention/kernels.py` 理解 NPU 实现细节。
最后再读 `extensions/legacy.py` 与 `legacy_hooks.py`，将其可选协议与核心 API 区分开。
插件发现与激活声明则看 `manifests/` 和 `pyproject.toml`。

若重点是后续架构维护，优先精读 `methods/base.py` → `methods/backend.py` →
`runtime/transaction.py` → `runtime/provider.py` → `runtime/scheduler.py`。
这五个文件直接决定算法策略、事务所有权和宿主状态之间的边界。

### 对照测试阅读

可将以下文件作为行为说明阅读，无需执行。部分测试需要已安装宿主包或加速器支持。

| 主题 | `tests/` 下的文件 |
| --- | --- |
| 注册与初始化边界 | `test_method_registry.py`、`test_lifecycle.py` |
| worker 事务 | `runtime_fakes.py`、`test_provider_integration.py` |
| 调度、确认与缓存释放 | `test_stateful.py` |
| 宿主约束与接入 | `test_host_contract.py`、`test_current_host_contract.py`、`test_plugin.py`、`test_host_compat.py` |
| 宿主路由与 GDN ABI | `test_router_compat.py`、`test_qwen_gdn_ops.py` |
| TriAttention 统计与算法 | `test_stats.py`、`test_scoring.py`、`test_selection.py`、`test_cache.py` |
| 校准 | `test_calibration.py` |
| 可选迁移适配器 | `test_legacy_extensions.py` |
| 模块归属与入口声明 | `test_package_layout.py`、`test_manifest.py` |

## 一次事务的实际顺序

1. 原宿主完成本步调度并推进 scheduler 计数；`after_schedule` 判断资格，预留私有目的块，把计划附到 SchedulerOutput。异步模式冻结该请求的后续调度。
2. worker 使用源块完成本步 forward/采样，`compress_scheduled_requests` 验证计划，调用 method 写目的块，记录 worker pending。
3. 宿主接受该输出后，`after_output` 把 scheduler pending 标为 ready。同步模式在下次调度前提交；异步模式在输出确认后提交。
4. 提交替换 scheduler 的 full-attention 块表，记录 active，释放或延迟释放源块，并恢复异步请求的可调度状态。
5. 请求再次被调度时，worker 的 `before_update_states` 用 pending 切换本地块表并建立 active；之后每步按当前 batch 行同步偏移、调整 attention 长度和写入 slot。

这里的屏障是宿主输出被接受的生命周期边界，不是显式的逐事务 `torch.npu.synchronize()`。
计划没有额外携带“压缩成功”响应对象；两端按计划、状态核对和输出接受路径配合。

## 当前扩展与状态边界

以下描述已经实现的 接口边界，不再是尚未实施的重构提案。

| 关注点 | 当前归属与契约 |
| --- | --- |
| 算法策略 | `methods/triattention/` 自行负责校准、评分、保护/分段选取与所选 token 搬运；其他方法在各自仓库实现算法。 |
| 静态构造 | CompressionBackend 提供 runtime_spec，无需创建 worker 算法；prepare_model_load 准备产物，create_method 构造 worker 状态。 |
| 运行时定制 | 下游 backend 覆盖 create_provider/create_scheduler；provider 可覆盖生命周期、请求构造、算法执行和 pending 激活，不必向核心添加下游策略字段。 |
| 统一算法契约 | CompressionRequest 包含前向完成后的长度及设备页表，不包含调度计划或 CPU 块所有权；CompressionResult 返回与 backend 目标一致的单一长度。核心事务均使用私有目标块。 |
| 必要事务状态 | runtime 负责目标块预留、输出接受确认、pending/active 镜像、坐标映射、APC 保护及源块释放；这些机制不因无需 Query 观测而消失。 |
| Query/逐层扩展 | 仅显式选择的 extensions.legacy 适配器保留相应协议和缓冲；它已停止维护并计划删除，下游应自行持有适配实现。TriAttention 不加载该适配器。 |
| 通用工具 | runtime.coordinates 保存坐标运算，host.contracts 保存共享宿主/缓存校验；核心 provider 不导入 TriAttention 私有内核。 |
| 宿主与模型 ABI | host 负责 GDN、MTP2 和路由兼容，属于宿主接入而非稀疏化策略。 |
| 导入路径 | 实现已归入 core/runtime/host 或 methods/triattention；不提供旧顶层模块转发别名，包级公共导出继续保留。 |


### 迁移后的入口

- vLLM 插件：`vllm_ascend_kvcompress.host.plugin:register`。
- 校准 CLI：`vllm_ascend_kvcompress.methods.triattention.calibration:main`；命令名仍为 `vllm-ascend-kvcompress-calibrate`。
- 扩展包：`vllm_ascend_kvcompress.manifests`。

`pyproject.toml` 与静态清单已使用这些位置。更新源码后，已有安装需重新安装以刷新入口元数据；
直接导入旧实现模块的下游代码需更新导入路径。

更多背景可参考[方法架构](../design/methods.zh.md)；旧描述若与当前实现不同，
以实际源码和测试为准。
