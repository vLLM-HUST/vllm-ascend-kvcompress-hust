# 浦江指定数据集：准备、smoke 与 B0/B1 对比

[English](pujiang.md) | 简体中文 | [规程索引](README.zh.md)

统一入口为 `scripts/run_benchmark.sh` 和 `scripts/set_env.sh`。
当前实现 MMLU-Pro、HLE-Verified、FrontierScience；SWE-bench-Pro 和 Terminal-Bench 文件留空，选择它们会明确报错。
脚本调用[官网指定客户端](https://vllm-hust.sage.org.ai/dataset-introductions.html)，不复制评分器，不修改外部仓库。

## 目录与客户端安装

下面的命令在插件仓库根目录运行。源码默认放在插件仓库旁的 `../pujiang-benchmark/`；源码根目录可以通过 `--client-root` 覆盖。
客户端环境独立于模型服务环境，默认位于工作区 `.envs/kvcompress-mmlu-pro`、
`.envs/kvcompress-hle-verified`、`.envs/kvcompress-frontierscience`。

```bash
# 安装对应的仓库源码
scripts/run_benchmark.sh mmlu-pro download-client
scripts/run_benchmark.sh hle-verified download-client
scripts/run_benchmark.sh frontier-science download-client

# 安装客户端环境
# 这里的python也可不指定，使用本机默认的python
scripts/set_env.sh install client --dataset mmlu-pro --python /path/to/python3.11
scripts/set_env.sh install client --dataset hle-verified --python /path/to/python3.11
scripts/set_env.sh install client --dataset frontier-science --python /path/to/python3.11

scripts/set_env.sh check client --dataset mmlu-pro
```

安装复用本地源码。已有仓库必须匹配固定提交且干净，否则拒绝覆盖、切换或重置；可复用已有 Git 缓存对象。
安装器直接安装已校验的本地客户端源码：MMLU-Pro 使用 `[prepare]` extra，准备依赖以外部客户端
`pyproject.toml` 为准；HLE/Frontier 安装基础包，不再读取额外的 requirements 文件。
MMLU 环境检查校验声明的准备依赖，并执行 `pip check`。上游未锁定的依赖由 pip 解析，
实际安装版本记录在 `kvcompress-client.json` 中。
已有 Python 环境只有在安装源码身份及准备依赖匹配时才复用。首次配置失败后保留诊断目录，使用新 `--prefix` 重试。
自定义客户端环境时，后续评测传 `--client-python /path/to/env/bin/python`；自定义源码位置时各步骤使用同一 `--client-root`。

| 数据集 | 客户端仓库 | 固定提交 |
| --- | --- | --- |
| MMLU-Pro | `vLLM-HUST/hust-eval-client-mmlu-pro` | `b6215bf76808b1e9d351295e2ecb401542317372` |
| HLE-Verified | `vLLM-HUST/hust-eval-client-hle-verified` | `534c22cd2d826bed2e8cbcf674bbebf4c3004bc1` |
| FrontierScience | `vLLM-HUST/hust-eval-client-frontierscience` | `ff7b4315876258b5d0ac9b9413ef79ae27146fa4` |

## 同基座环境与插件开关

B0/B1 共用[锁定的 HUST 基座](../../requirements/ascend-host.json)、Python 环境、工具链和兼容桥，
默认环境均为 `../.envs/kvcompress-host`。B0 关闭压缩插件，B1 使用指定配置启用插件；
两侧共享环境中均保留插件安装包。安装前按[环境指南](../development/environment-installation.zh.md)
准备已有 HUST 基座和 SDK，并设置 `KVCOMPRESS_BASE_PYTHON`。

```bash
scripts/set_env.sh install b1
scripts/set_env.sh check b1
# b0 的安装/检查是同一个宿主环境的别名。
scripts/set_env.sh check b0
```

首次安装或复用已有基座的选项见[环境安装指南](../development/environment-installation.zh.md)。
`install/check b0` 与 `install/check b1` 接受相同安装参数，包括 `--prefix`，无需另建服务环境。
两个 `run` 模式使用相同的宿主及工具链启动器：

```bash
scripts/set_env.sh run b0 -- python scripts/env/benchmark_host.py \
  -m vllm.entrypoints.cli.main serve /path/to/model --served-model-name qwen35-eval
VLLM_ASCEND_KVCOMPRESS_CONFIG=examples/qwen3.5-35b-a3b-triattention.json \
  scripts/set_env.sh run b1 -- python scripts/env/benchmark_host.py \
  -m vllm.entrypoints.cli.main serve /path/to/model --served-model-name qwen35-eval
```

两侧使用相同服务参数和设备设置。B0 清除继承的压缩及 Manager 配置，仅加载 Ascend 插件；
B1 要求配置文件存在，显式启用压缩并加载 Ascend 和压缩插件。
两个模式均清除继承的 vLLM 设置，仅保留 `VLLM_API_KEY`；服务设置通过命令行参数传入。

## 数据与冻结计划

数据和计划默认位于 `../.benchmarks/pujiang/<dataset>/`，本地运行结果默认位于
`../.benchmarks/results/<dataset>/<action>-<arm-or-pair>/`。默认路径根据插件仓库位置定位，
不受命令执行目录影响；显式传入的 CLI 相对路径以命令执行目录为基准。
已移动到 `../.benchmarks/` 的数据可通过 `--data-root` 和 `--plan` 指定复用；
已有结果记录中的历史路径保持原样，不自动改写。`download-client` 下载代码，`download` 下载固定数据，互不混用。
准备阶段只读取 tokenizer/model 小文件，不读取完整权重，不使用评测题作压缩校准。

```bash
scripts/run_benchmark.sh mmlu-pro download
scripts/run_benchmark.sh mmlu-pro prepare --model-path /path/to/model

scripts/run_benchmark.sh hle-verified download
scripts/run_benchmark.sh hle-verified prepare --model-path /path/to/model \
  --solver-config examples/benchmark/hle-verified-solver.json \
  --judge-config examples/benchmark/hle-verified-judge.json

scripts/run_benchmark.sh frontier-science download
scripts/run_benchmark.sh frontier-science prepare --model-path /path/to/model \
  --solver-config examples/benchmark/frontier-science-solver.json \
  --judge-config examples/benchmark/frontier-science-judge.json
```

**先编辑示例 JSON**：设置真实 endpoint、served model、model artifact、预算和鉴权环境变量名。
Solver 示例使用 18180；固定独立 Judge 示例使用 18181，Judge 由用户预先启动并保持两侧配置一致。
密钥放入环境变量，不写入配置。HLE 图片要求真实图文模型服务，Judge 要求 Structured Outputs。
默认并发 1；HLE 使用 `strict_json`，Frontier 使用 `strict_final`，保留官方指定客户端的提示和评分策略。
HLE Gold/Revision 分开报告，默认不含 Uncertain；Frontier Olympiad/Research 分开报告。
Research 0–10 分、≥7 算成功；未判定仍按上游策略报告，不赋造假的零分或准确率。
MMLU 本地答案提取失败记错，不随机猜测；五个 validation few-shot 示例保持不变。

已有计划拒绝覆盖。若两侧 URL、模型别名或 Solver 配置不同，使用 `--plan` 分别准备两份计划，
配对器核对题目、模型小文件、请求预算及固定 Judge；其他设置不匹配就拒绝比较。
同一地址顺序启动且请求配置一致时可以共用计划。
可传 `--runtime-contract` 和 `--judge-runtime-contract` 绑定操作者提供的真实回执；未知服务身份保持未知。

外部服务通过 `--runtime-contract` 提供运行身份记录。自动配对会记录基座身份、插件状态和启动设置。
比较时，完整匹配记录标为 `runtime_verification.status=verified`；缺少完整记录标为 `unverified`，
已知的基座或启动设置差异会拒绝比较。外部服务可在相同 `set_env.sh run` 模式及环境中运行
`scripts/env/runtime_identity.py`，补充包含 `model_path`、`served_model`、`server_args`、`server_env`
的 `server` 对象后作为记录传入；评测客户端无法独立证明外部服务进程的实际设置。

## 固定 smoke 和重新采样

```bash
# 连接已启动的 B0；URL 不含 /v1。B1 用新 output 重复同一命令并改 arm。
scripts/run_benchmark.sh mmlu-pro run --arm b0 \
  --base-url http://127.0.0.1:18180 --served-model qwen35-eval \
  --smoke --output ../.benchmarks/results/mmlu/b0
scripts/run_benchmark.sh mmlu-pro run --arm b1 \
  --base-url http://127.0.0.1:18180 --served-model qwen35-eval \
  --smoke --output ../.benchmarks/results/mmlu/b1
scripts/run_benchmark.sh mmlu-pro compare \
  --baseline ../.benchmarks/results/mmlu/b0 --candidate ../.benchmarks/results/mmlu/b1 \
  --output ../.benchmarks/results/mmlu/comparison
```

HLE/Frontier 使用相同 `run` 参数，且服务 URL/别名必须匹配计划中的 Solver 配置。
脚本按顺序执行上游 Solver、独立 Judge、summary，保留错误、截断和未判定。

`--smoke` 默认每子集随机选最多 100 题，`--sample-size N` 更改数量，首次 seed=17。
MMLU 从 test 抽样；HLE 每个 Gold/Revision、Frontier 每个 Olympiad/Research 分别抽样。
因此默认 Frontier 实际为 100+60 题，模式仍标为 smoke。执行顺序保持冻结计划顺序。
样本保存在 `samples/selection-*.json`，`current.json` 指向当前版本；两侧和后续优化均复用。
可通过 `--selection-dir` 指定共同清单目录。数据、数量或显式 seed 不匹配时要求重新采样。

```bash
# 创建新的采样版本，旧清单保留；随后 B1 不再传 --resample。
scripts/run_benchmark.sh mmlu-pro run --arm b0 \
  --base-url http://127.0.0.1:18180 --served-model qwen35-eval \
  --smoke --sample-size 50 --resample --seed 23 \
  --output ../.benchmarks/results/mmlu/b0-new-sample
```

不指定 `--seed` 的重新采样会生成并记录新 seed。不要在配对中途重新采样。
省略 `--smoke` 才是完整选定范围；`--track` 可显式选择子集。
外部服务模式支持同一计划和选题的 `--resume`；报告另存，旧报告保留。不能把续跑称为独立冷生命周期。

## 自动冷启动配对

编辑 `examples/benchmark/<dataset>-pair.json`，填写模型路径、设备、服务参数和 B1 插件配置。
配置中的文件路径相对于配置文件目录：示例使用 `../../../.benchmarks/pujiang/<dataset>/plan`
指向仓库旁的数据目录。源码/客户端默认位置仍相对于插件仓库的上级目录。
`server_args` 必须是参数数组。`server_env` 可设置可见设备及线程参数，不应存放密钥值。
两侧必须使用相同的宿主环境路径、模型别名、服务参数和有效服务环境；仅 B1 指定 `plugin_config`。
三个示例通过共享的 `host_prefix` 指定宿主环境，路径相对于 JSON 文件解析；默认压测客户端也使用该环境。
运行前检查两侧基座包、宿主配置、兼容桥和插件开关，
每侧启动前再次核对身份；环境发生变化则终止配对。

```bash
scripts/run_benchmark.sh mmlu-pro pair \
  --config examples/benchmark/mmlu-pro-pair.json \
  --smoke --pairs 1 --with-bench \
  --output ../.benchmarks/results/mmlu-pair
```

默认一对，可用 `--pairs 3` 重复。两侧在启动前冻结同一选题，检查计划和环境，分别冷启动并等待健康。
依次执行质量评测及可选性能压测，Judge 不计入性能窗口。只清理本次创建的进程组；端口已有健康服务则拒绝启动。
启动失败、超时、中断、客户端失败均保留日志并清理所属服务。
结果目录包括 `pair-N/{b0,b1}/` 下的 `runtime.json`、`server.log`、`lifecycle.json`、
`telemetry.jsonl.gz`、`accuracy/`、可选 `bench/` 及每对 `comparison/comparison.json`。
运行器不擅自增加质量预热；性能预热遵循固定上游客户端，HLE 额外 warmups=0，仍有 vLLM 计时外首题测试请求。

## 独立性能压测

```bash
scripts/run_benchmark.sh hle-verified bench --arm b1 \
  --base-url http://127.0.0.1:18180 --served-model qwen35-eval \
  --model-path /path/to/model --smoke \
  --server-log /path/to/actual-server.log \
  --output ../.benchmarks/results/hle-bench
```

默认压测入口是 `scripts/env/bench_vllm.sh`：使用 B1 的固定 vLLM **客户端**同时测试 B0/B1 服务，不在客户端启用压缩。
可用 `--bench-vllm-bin` 指定另一个固定入口；先检查所需 CLI 功能，不能随服务切换压测端版本。
包括 HLE 所需 `custom_image` 在内的客户端能力会在发送请求前校验。
入口通过宿主 Python 模块启动，无需单独安装 `vllm` 命令行脚本。
客户端身份使用完整 CLI 帮助中 `usage:` 之后的内容，规范化随机请求 ID 和无序选项列表，同时保留实际选项及默认值变化；包含启动诊断的原始帮助保存在 `bench-client-help.txt`。
`--dry-run` 导出输入及命令、检查功能，不发送压测；实际执行使用新的输出目录。

质量和性能不共用请求语义：MMLU 性能发送重建文本；Frontier 使用服务默认模板/thinking；
HLE 性能只有 user 消息和原图、无准确率 system prompt，显式 thinking=false、模板只套一次。
默认输出上限256、temperature0、seed0、request-rate=inf，没有客户端并发上限；TTFT 包含排队。
服务端 max_num_seqs 不等于客户端并发。质量 `--concurrency` 不会偷偷改变原性能协议。

非 smoke 性能默认：MMLU 均匀取200题，HLE 每子集文本200及全部图片，Frontier 全部160题。
smoke 性能复用固定随机样本，HLE 再按文本/图片分组，空组不补题，明确报告零覆盖。
性能适配从当前验证计划导出并保存来源；不是从历史 sealed accuracy 偷绕源码哈希检查导入。
图片使用上游 `image_info` 和输入验证，逐图保留字节哈希，再将 detailed input_lens 与服务 `/tokenize` 核对。
不能核实时保留原始结果，但比较汇总不把估算总 token 吞吐作为可信指标。

**压缩是否真正发生必须另查。** 当前 Qwen 示例输出门槛为2048，而默认性能预算256会绕过压缩。
工具不自动改门槛；可显式通过 `--output-len` 和预先选定的插件配置构造优化检查负载，并记录协议差异。
提供 `--server-log` 时保存测量区间的 scheduler 提交及 worker 回执；未观察到与未提供证据分别标记。
HBM/NPU 由配对遥测采样，采样值不是精确的分配器高水位。

每次使用新输出目录；保留 upstream 原始 JSON、请求、明细、错误及哈希。
smoke 是开发检查，不是正式全量成绩。正式证据按[原件归属](../usage/benchmark-originals.zh.md)放入 benchmark 仓库，不能将私有题目和回答直接发布。

独立性能运行也可以使用 `compare --baseline ... --candidate ... --output ...`，传入两侧 bench 输出目录；压测端、tokenizer、输入或预算不同会拒绝比较。

HLE/Frontier 的并发和请求超时来自冻结的 Solver 配置；若在命令行显式传入不同的 `--concurrency` 或 `--timeout`，会要求重新准备计划，不会静默覆盖。MMLU 的默认值分别为1和1800秒。MMLU 续跑前会保存原始 Solver 输出快照；上游报告中的 complete 仅表示选定计划完成，是否 smoke 以适配器报告为准。

## 本地 Judge 分阶段评分

当 Solver 与本地 Judge 共用设备时，可先完成答案生成，再释放 Solver、启动固定 Judge。
HLE/Frontier 的 `run --stage solver` 仅生成答案并记录 `awaiting_judge`；
随后用相同计划、选题、arm、运行身份和输出目录执行 `run --stage judge --resume`。
Judge 阶段验证答案文件哈希，不重新调用 Solver；完成后可使用原有 `compare`。
`--stage all` 是默认值，保持 Solver 后立即评分的流程。MMLU 使用本地判分，不接受分阶段选项。

自动配对使用 `pair --stage solver` 延后评分，不要求 Judge 服务提前在线，也不生成提前的正确率比较。
两侧服务停止后，分别对 `pair-1/b0/accuracy`、`pair-1/b1/accuracy` 执行 Judge 阶段，
传回原来的 `--plan`、`--arm`、`--runtime-contract`、Solver URL/模型名和 smoke 参数。
最后比较两侧目录；缺失评分或评分失败会拒绝比较。
固定 Judge 必须关闭压缩，两侧共用相同模型、请求预算及评分策略。
本地 Judge 成绩用于开发检查，不等同于其他 Judge 模型给出的正式成绩。
